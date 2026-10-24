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
"""Score returned skill instructions using evidence collected outside the agent.

This module does not establish trust in supplied JSON. The adapter must collect
canonical tool results, file observations, and request/session records on the
host. An EvidenceError is an unscored infrastructure/evidence error, never zero.
"""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EvidenceError(ValueError):
    """The evidence cannot support a valid benchmark result."""


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Identity(Record):
    task_id: str
    prompt_sha256: str
    catalog_sha256: str
    model: str
    harness: Literal["pi", "codex", "codex_guidance", "hermes", "hermes_guidance"]


class Session(Record):
    id: str
    model: str
    parent_id: str | None = None
    spawn_call_id: str | None = None
    # Calls copied into a fork are history, not new actions by that child.
    inherited_call_ids: list[str] = Field(default_factory=list)


class Request(Record):
    session_id: str
    model: str
    sent_at: float = Field(ge=0)
    max_output_tokens: int


class Access(Record):
    skill: str
    at: float = Field(ge=0)


class ReadResult(Record):
    """Returned tool content, without claiming session or timing validation."""

    tool: str
    text: str
    success: bool
    # Set only for a read of the exact staged SKILL.md, or skill_view with
    # matching requested AND returned skill names and no support-file request.
    resolved_skill: str | None = None


class ToolResult(ReadResult):
    session_id: str
    call_id: str
    at: float = Field(ge=0)


class Spawn(Record):
    parent_id: str
    child_id: str
    call_id: str
    at: float = Field(ge=0)


class TrialEvidence(Record):
    identity: Identity
    root_session_id: str
    sessions: list[Session]
    spawns: list[Spawn] = Field(default_factory=list)
    requests: list[Request]
    # All times use one host clock with prompt release at zero. Observation
    # starts after catalog staging and before release; adapters enforce this.
    catalog_staged_at: float
    observer_ready_at: float
    observer_stopped_at: float
    finished_at: float = Field(ge=0)
    active_work_stopped: bool
    outcome: Literal[
        "completed",
        "routing_horizon",
        "timeout",
        "output_limit",
        "routing_boundary",
        "provider_error",
        "setup_error",
        "observation_error",
        "trace_error",
        "cancelled",
    ]
    accesses: list[Access]
    outputs: list[ToolResult]


class ReadEvidence(Record):
    skill: str
    session_id: str
    call_id: str


class Score(Record):
    reward: Literal[0, 1]
    loaded_skills: list[str]
    reads: list[ReadEvidence]


def normalize(text: str, *, javascript: bool = False) -> str:
    """Match the original whitespace and rendered-line-number normalization."""
    if javascript:
        # ECMAScript whitespace and digits differ from Python's Unicode \s/\d.
        space = r"[\t\n\v\f\r \u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff]"
        start = r"(?:^|(?<=[\n\r\u2028\u2029]))"
        text = re.sub(start + space + r"*[0-9]+[|→]" + space + "?", "", text)
        return re.sub(space + "+", " ", text).strip(" ")
    return re.sub(r"\s+", " ", re.sub(r"^\s*\d+[|→]\s?", "", text, flags=re.MULTILINE)).strip()


def skill_body(text: str, *, javascript: bool = False) -> str:
    return normalize(re.sub(r"^---\r?\n[\s\S]*?\r?\n---\r?\n", "", text), javascript=javascript)


def character_units(text: str, *, javascript: bool) -> str:
    """Pi's reference counts UTF-16 units; the other scorers count code points."""
    if not javascript:
        return text
    data = text.encode("utf-16-le", errors="surrogatepass")
    return "".join(chr(int.from_bytes(data[i : i + 2], "little")) for i in range(0, len(data), 2))


def has_excerpt(output: str, source: str, *, competing: list[str], javascript: bool = False) -> bool:
    """Require 120 body characters (or the whole shorter body), excluding shared text."""
    text = character_units(normalize(output, javascript=javascript), javascript=javascript)
    body = character_units(skill_body(source, javascript=javascript), javascript=javascript)
    others = [character_units(skill_body(s, javascript=javascript), javascript=javascript) for s in competing]
    size = min(120, len(body))
    return bool(size) and any(
        body[i : i + size] in text and not any(body[i : i + size] in other for other in others)
        for i in range(len(body) - size + 1)
    )


def validate_evidence(evidence: TrialEvidence, expected: Identity) -> dict[str, Session]:
    """Check identity, lineage, complete observation, and shared frozen limits."""
    if evidence.identity != expected:
        raise EvidenceError("Trial identity differs from the requested task, catalog, model, or harness")
    if evidence.outcome not in {"completed", "routing_horizon", "timeout", "output_limit", "routing_boundary"}:
        raise EvidenceError(f"Unscored trial outcome: {evidence.outcome}")
    if not evidence.active_work_stopped:
        raise EvidenceError("Active work has not stopped before grading")
    if (
        not evidence.catalog_staged_at <= evidence.observer_ready_at <= 0
        or evidence.observer_stopped_at < evidence.finished_at
    ):
        raise EvidenceError("Independent observation does not cover the complete trial")
    if evidence.finished_at > 120 and evidence.outcome != "timeout":
        raise EvidenceError("Trial exceeded its deadline without recording a timeout")
    sessions = {s.id: s for s in evidence.sessions}
    if len(sessions) != len(evidence.sessions) or evidence.root_session_id not in sessions:
        raise EvidenceError("Missing root session or duplicate session identity")
    root = sessions[evidence.root_session_id]
    if root.parent_id is not None or root.spawn_call_id is not None or root.model != expected.model:
        raise EvidenceError("Invalid original session identity")
    if root.inherited_call_ids:
        raise EvidenceError("A fresh root session cannot inherit tool calls")
    for session in sessions.values():
        if session.model != expected.model:
            raise EvidenceError("Different-model session cannot contribute to this trial")
        ancestors = set()
        current = session
        while current.id != root.id:
            if current.id in ancestors or current.parent_id not in sessions:
                raise EvidenceError("Unproven or cyclic child-session lineage")
            ancestors.add(current.id)
            matches = [
                s
                for s in evidence.spawns
                if (s.parent_id, s.child_id, s.call_id) == (current.parent_id, current.id, current.spawn_call_id)
            ]
            if len(matches) != 1 or matches[0].at > min(evidence.finished_at, 120):
                raise EvidenceError("Child session lacks a unique in-budget parent spawn record")
            current = sessions[current.parent_id]
    if not 1 <= len(evidence.requests) <= 3:
        raise EvidenceError("Forwarded request count is outside the shared allowance")
    for request in evidence.requests:
        if (
            request.session_id not in sessions
            or request.model != expected.model
            or request.sent_at > min(evidence.finished_at, 120)
            or request.max_output_tokens != 8192
        ):
            raise EvidenceError("Forwarded request violates session, model, deadline, or output-token limits")
        session = sessions[request.session_id]
        if session.parent_id is not None:
            spawn = next(s for s in evidence.spawns if s.child_id == session.id and s.call_id == session.spawn_call_id)
            if request.sent_at < spawn.at:
                raise EvidenceError("Child request precedes its parent spawn")
    output_keys = [(o.session_id, o.call_id) for o in evidence.outputs]
    if len(output_keys) != len(set(output_keys)):
        raise EvidenceError("Duplicate canonical tool result")
    for output in evidence.outputs:
        if output.session_id not in sessions or output.at > evidence.finished_at:
            raise EvidenceError("Tool result lacks a known session or is outside the recorded trial")
        own_requests = [r for r in evidence.requests if r.session_id == output.session_id and r.sent_at <= output.at]
        if output.call_id not in sessions[output.session_id].inherited_call_ids and not own_requests:
            raise EvidenceError("Tool result lacks a preceding model request in its owning session")
    return sessions


def matches_skill_read(output: ReadResult, *, skill: str, sources: dict[str, str], harness: str) -> bool:
    """Classify returned instructions; caller must prove access, lineage and limits.

    Kept separate for replaying older transcripts without inventing the timing
    records required by score_trial. This predicate alone is not a trial score.
    """
    if not output.success:
        return False
    if harness == "pi":
        if output.tool not in {"read", "bash"} or output.tool == "read" and output.resolved_skill != skill:
            return False
    elif harness in {"codex", "codex_guidance"}:
        if output.tool != "terminal":
            return False
    elif harness in {"hermes", "hermes_guidance"}:
        if output.tool not in {"terminal", "skill_view", "read_file"}:
            return False
    else:
        raise EvidenceError(f"Unsupported harness: {harness}")
    direct = output.resolved_skill == skill and output.tool in {"read", "read_file", "skill_view"}
    competitors = [] if direct else [text for other, text in sources.items() if other != skill]
    return has_excerpt(output.text, sources[skill], competing=competitors, javascript=harness == "pi")


def score_trial(evidence: TrialEvidence, *, expected: Identity, sources: dict[str, str], target: str) -> Score:
    """Score positive skill opening; invalid evidence raises instead of inventing a miss."""
    sessions = validate_evidence(evidence, expected)
    if target not in sources or not skill_body(sources[target]):
        raise EvidenceError("Target instructions are missing or empty")
    accessed = {a.skill for a in evidence.accesses if a.at <= min(evidence.finished_at, 120)}
    if any(a.skill not in sources for a in evidence.accesses):
        raise EvidenceError("Observer recorded a skill outside the frozen catalog")
    reads = []
    for name in sources:
        if name not in accessed:
            continue
        for output in evidence.outputs:
            if (
                not output.success
                or output.at > 120
                or output.call_id in sessions[output.session_id].inherited_call_ids
            ):
                continue
            if matches_skill_read(output, skill=name, sources=sources, harness=expected.harness):
                reads.append(ReadEvidence(skill=name, session_id=output.session_id, call_id=output.call_id))
                break
    loaded = sorted(read.skill for read in reads)
    return Score(reward=int(target in loaded), loaded_skills=loaded, reads=reads)
