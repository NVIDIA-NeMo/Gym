# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Attribute pinned Codex rollouts without counting copied parent history.

The live adapter supplies complete records with host observation times and the
root ID emitted by ``codex exec --json``. Request metadata selects a candidate;
canonical owner, spawn linkage, and turn records must independently confirm it.
"""

import json
from dataclasses import dataclass

from environments.skill_trigger.scoring import EvidenceError, Session, Spawn, ToolResult
from environments.skill_trigger.tool_results import extract_tool_results
from environments.skill_trigger.trajectory import object_value


class PendingSession(LookupError):
    """Required canonical records have not arrived yet; do not forward inference."""


class ModelRouteBoundary(EvidenceError):
    """A proven child explicitly selected another model; do not forward it."""


@dataclass(frozen=True)
class CapturedRecord:
    event: dict[str, object]
    at: float


@dataclass(frozen=True)
class CodexSession:
    session: Session
    spawn: Spawn | None
    own: tuple[CapturedRecord, ...]


class CodexSessions:
    """Keep append-only, host-timed records for one CLI root and its children."""

    def __init__(self, *, model: str) -> None:
        self.model = model
        self.root_id: str | None = None
        self._files: dict[str, list[CapturedRecord]] = {}
        self._owners: dict[str, str] = {}

    def select_root(self, event: dict[str, object]) -> None:
        """Bind the root to the CLI event, never the newest rollout filename."""
        if event.get("type") != "thread.started":
            raise EvidenceError("Root selection requires Codex's thread.started event")
        root = event.get("thread_id")
        if not isinstance(root, str) or not root or self.root_id is not None:
            raise EvidenceError("Expected exactly one original Codex thread")
        self.root_id = root

    def append(self, *, path: str, event: dict[str, object], at: float) -> None:
        """Accept one complete record from a continuously monitored rollout."""
        if at < 0:
            raise EvidenceError("Codex execution records must follow prompt release")
        rows = self._files.setdefault(path, [])
        if rows and at < rows[-1].at:
            raise EvidenceError("Codex host observation clock went backwards")
        if not rows:
            owner = object_value(event.get("payload"))
            sid = owner.get("id")
            if event.get("type") != "session_meta" or not isinstance(sid, str) or not sid:
                raise EvidenceError("Codex rollout lacks its owning metadata first")
            if sid in self._owners:
                raise EvidenceError("Multiple Codex files claim the same session")
            if owner.get("cli_version") != "0.156.1":
                raise EvidenceError("Codex trace version differs from the pinned profile")
            self._owners[sid] = path
        rows.append(CapturedRecord(event=event, at=at))

    def session(self, sid: str, *, visiting: tuple[str, ...] = ()) -> CodexSession:
        """Prove lineage and remove an exactly matched copied parent prefix."""
        if sid in visiting:
            raise EvidenceError("Cyclic Codex session lineage")
        if self.root_id is None or sid not in self._owners:
            raise PendingSession("Owning Codex trace has not arrived")
        rows = self._files[self._owners[sid]]
        owner = object_value(rows[0].event.get("payload"))
        parent_id = owner.get("parent_thread_id")
        own = rows[1:]
        spawn = None
        inherited = []
        spawn_model = self.model
        if sid == self.root_id:
            if parent_id is not None or owner.get("source") != "exec":
                raise EvidenceError("Selected root is not a fresh Codex exec session")
        else:
            if not isinstance(parent_id, str) or not parent_id:
                raise EvidenceError("Child Codex trace lacks its parent")
            parent = self.session(parent_id, visiting=(*visiting, sid))
            calls = {}
            links = []
            for row in parent.own:
                value = object_value(row.event.get("payload"))
                if row.event.get("type") != "response_item":
                    continue
                if value.get("type") == "function_call" and value.get("name") == "spawn_agent":
                    call_id = value.get("call_id")
                    if not isinstance(call_id, str) or call_id in calls:
                        raise EvidenceError("Missing or duplicate Codex spawn call ID")
                    calls[call_id] = value
                if value.get("type") == "function_call_output" and value.get("call_id") in calls:
                    try:
                        result = json.loads(value.get("output", ""))
                    except (TypeError, ValueError):
                        continue
                    if isinstance(result, dict) and result.get("agent_id") == sid:
                        links.append((calls[value["call_id"]], row.at))
            if not links:
                raise PendingSession("Canonical parent spawn output has not arrived")
            if len(links) != 1:
                raise EvidenceError("Ambiguous Codex spawn linkage")
            call, spawn_at = links[0]
            try:
                args = object_value(json.loads(call["arguments"]))
            except (KeyError, TypeError, ValueError) as error:
                raise EvidenceError("Malformed Codex spawn arguments") from error
            spawn_model = args.get("model") or self.model
            if not isinstance(spawn_model, str):
                raise EvidenceError("Invalid selected child model")
            spawn = Spawn(parent_id=parent_id, child_id=sid, call_id=call["call_id"], at=spawn_at)
            if own and own[0].event.get("type") == "session_meta":
                if args.get("fork_context") is not True:
                    raise EvidenceError("Child copied history without a fork_context request")
                boundary = owner.get("subagent_history_start_ordinal")
                if type(boundary) is not int or boundary < 2 or owner.get("forked_from_id") != parent_id:
                    raise EvidenceError("Fork lacks Codex's explicit parent-history boundary")
                copied = boundary - 1  # ordinal zero is this child's owning metadata
                parent_rows = self._files[self._owners[parent_id]]
                if len(own) < copied or len(parent_rows) < copied:
                    raise PendingSession("The declared parent-history prefix has not arrived")
                for left, right in zip(own[:copied], parent_rows[:copied], strict=True):
                    if (left.event.get("type"), left.event.get("payload")) != (
                        right.event.get("type"),
                        right.event.get("payload"),
                    ):
                        raise EvidenceError("Inherited Codex record differs from its parent")
                if not any(r.event.get("type") == "turn_context" for r in own[:copied]):
                    raise EvidenceError("Copied Codex history lacks its parent model context")
                inherited = sorted(
                    {
                        r.event["payload"]["call_id"]
                        for r in own[:copied]
                        if r.event.get("type") == "response_item"
                        and isinstance(r.event.get("payload"), dict)
                        and isinstance(r.event["payload"].get("call_id"), str)
                    }
                )
                own = own[copied:]
            elif args.get("fork_context") is True:
                if not own:
                    raise PendingSession("Forked Codex history has not arrived")
                raise EvidenceError("Forked child is missing its copied parent prefix")
        if any(row.event.get("type") == "session_meta" for row in own):
            raise EvidenceError("Unexpected metadata in the session's own actions")
        contexts = [object_value(row.event.get("payload")) for row in own if row.event.get("type") == "turn_context"]
        if not contexts:
            raise PendingSession("Session's own model context has not arrived")
        if any(context.get("model") != spawn_model for context in contexts):
            raise EvidenceError("Codex turn model differs from its proven selection")
        if spawn_model != self.model:
            raise ModelRouteBoundary("Codex explicitly selected another model for its child")
        return CodexSession(
            session=Session(
                id=sid,
                model=self.model,
                parent_id=parent_id,
                spawn_call_id=spawn.call_id if spawn else None,
                inherited_call_ids=inherited,
            ),
            spawn=spawn,
            own=tuple(own),
        )

    def resolve_request(self, payload: dict[str, object]) -> CodexSession:
        """Corroborate a request's thread ID against captured canonical evidence."""
        metadata = object_value(payload.get("client_metadata"))
        sid = metadata.get("thread_id")
        if not isinstance(sid, str) or not sid:
            raise EvidenceError("Codex request lacks a thread identity")
        selected = self.session(sid)
        if payload.get("model") != selected.session.model:
            raise EvidenceError("Request model differs from its proven Codex session")
        return selected

    def own_trace(self, sid: str) -> list[dict[str, object]]:
        """Return the actual owner and new records for ATIF, excluding copied history."""
        selected = self.session(sid)
        return [self._files[self._owners[sid]][0].event, *(row.event for row in selected.own)]

    def outputs(self, sid: str, *, skill_paths: dict[str, str]) -> list[ToolResult]:
        """Extract only this session's newly completed canonical tool outputs."""
        selected = self.session(sid)
        completed = {}
        events = []
        inherited = set(selected.session.inherited_call_ids)
        for row in selected.own:
            events.append(row.event)
            value = object_value(row.event.get("payload"))
            if row.event.get("type") == "response_item" and value.get("type") == "function_call_output":
                call_id = value.get("call_id")
                if not isinstance(call_id, str) or call_id in completed or call_id in inherited:
                    raise EvidenceError("Duplicate, missing, or inherited Codex output ID")
                completed[call_id] = row.at
        return extract_tool_results(
            events, harness="codex", session_id=sid, completed_at=completed, skill_paths=skill_paths
        )
