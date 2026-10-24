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
"""Connect Codex's live rollout stream to shared budgets, scoring, and ATIF."""

import asyncio
import json
from pathlib import Path

from harbor.models.trajectories import SubagentTrajectoryRef, Trajectory

from environments.skill_trigger.budget import BudgetReached, TrialBudget
from environments.skill_trigger.codex import CodexSessions, ModelRouteBoundary, PendingSession
from environments.skill_trigger.container import TrialContainer
from environments.skill_trigger.scoring import EvidenceError, Session, Spawn, ToolResult
from environments.skill_trigger.trajectory import convert_trace


class CodexRuntime:
    """Capture native records outside the agent and admit only proven sessions."""

    def __init__(self, *, container: TrialContainer, budget: TrialBudget, evidence_dir: Path, model: str) -> None:
        self.container = container
        self.budget = budget
        self.path = evidence_dir
        self.model = model
        self.records = CodexSessions(model=model)
        self.process: asyncio.subprocess.Process | None = None
        self.pid: int | None = None
        self.reader: asyncio.Task | None = None
        self._registered: set[str] = set()
        self._stopped = False
        self._closed = False

    def accept_cli(self, event: dict[str, object]) -> None:
        """Bind the root only from the CLI's actual startup event."""
        if event.get("type") == "thread.started":
            self.records.select_root(event)
            self.budget.bind_root_session(self.records.root_id)

    async def start(self) -> None:
        """Start root-owned capture before releasing the prompt."""
        self.process = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            "-u",
            "0",
            self.container.id,
            "python3",
            "-u",
            "-c",
            Path(__file__).with_name("codex_watch.py").read_text(),
            "/home/agent/native-codex/config/sessions",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=32_000_000,
        )
        ready = json.loads(await asyncio.wait_for(self.process.stdout.readline(), 10))
        if ready.get("ready") is not True or type(ready.get("pid")) is not int:
            raise EvidenceError("Codex rollout watcher did not become ready")
        self.pid = ready["pid"]

    async def collect(self) -> None:
        """Copy complete records to private host storage with host completion times."""
        sequence = {}
        with (self.path / "codex-rollouts.jsonl").open("x") as file:
            while line := await self.process.stdout.readline():
                record = json.loads(line)
                if record == {"rollouts_stopped": True}:
                    self._stopped = True
                    return
                path = record.get("path")
                if not isinstance(path, str) or record.get("sequence") != sequence.get(path, 0):
                    raise EvidenceError("Missing or reordered Codex rollout records")
                at = self.budget.elapsed()
                file.write(json.dumps(record | {"captured_at": at}) + "\n")
                file.flush()
                self.records.append(path=path, event=record["event"], at=at)
                sequence[path] = sequence.get(path, 0) + 1
        raise EvidenceError("Codex rollout capture ended without its stop marker")

    def _register(self, sid: str) -> None:
        if sid in self._registered:
            return
        selected = self.records.session(sid)
        if selected.session.parent_id:
            self._register(selected.session.parent_id)
            self.budget.add_session(session_id=sid, parent_id=selected.session.parent_id)
        self._registered.add(sid)

    async def resolve_request(self, payload: dict[str, object]) -> str:
        """Wait for canonical attribution; the gateway owns the same trial deadline."""
        while True:
            if self.reader is not None and self.reader.done():
                self.reader.result()
                raise EvidenceError("Codex inference attempted after rollout collection ended")
            try:
                selected = self.records.resolve_request(payload)
                self._register(selected.session.id)
                return selected.session.id
            except PendingSession:
                await asyncio.sleep(0.01)
            except ModelRouteBoundary as error:
                raise BudgetReached("routing_boundary") from error

    async def stop(self) -> None:
        """Drain records after agent work stops, then verify watcher termination."""
        if self.process is None or self._closed:
            return
        try:
            if self.process.returncode is None and self.pid is not None:
                await self.container.stop_observer(pid=self.pid)
            if self.reader is not None:
                await asyncio.wait_for(asyncio.shield(self.reader), 10)
            await asyncio.wait_for(self.process.wait(), 10)
        finally:
            try:
                if self.process.returncode is None:
                    try:
                        await asyncio.wait_for(self.process.wait(), 10)
                    except TimeoutError:
                        self.process.kill()
                        await self.process.wait()
                errors = await self.process.stderr.read()
                (self.path / "codex-watch.stderr").write_bytes(errors)
            finally:
                self._closed = True
        if self.process.returncode != 0 or (self.reader is not None and not self._stopped):
            raise EvidenceError("Codex rollout watcher failed or did not finish draining")

    def evidence(self, *, skill_paths: dict[str, str]) -> tuple[str, list[Session], list[Spawn], list[ToolResult]]:
        """Revalidate all contributing sessions against the final retained records."""
        root = self.records.root_id
        if root is None:
            raise EvidenceError("Codex never established its original session")
        ids = {root, *(request.session_id for request in self.budget.requests)}
        selected = [self.records.session(sid) for sid in sorted(ids)]
        return (
            root,
            [item.session for item in selected],
            [item.spawn for item in selected if item.spawn is not None],
            [output for sid in sorted(ids) for output in self.records.outputs(sid, skill_paths=skill_paths)],
        )

    def trajectory(self) -> Trajectory:
        """Embed actual child trajectories and link them to canonical parent spawn results."""
        ids = {self.records.root_id, *(request.session_id for request in self.budget.requests)}
        trajectories = {}
        for index, sid in enumerate(sorted(ids)):
            path = self.path / f"codex-session-{index}.jsonl"
            path.write_text("".join(json.dumps(event) + "\n" for event in self.records.own_trace(sid)))
            trajectories[sid] = convert_trace(path, harness="codex", session_id=sid, model=self.model)
        for sid in ids - {self.records.root_id}:
            selected = self.records.session(sid)
            parent = trajectories[selected.session.parent_id]
            links = [
                result
                for step in parent.steps
                if step.observation
                for result in step.observation.results
                if result.source_call_id == selected.spawn.call_id
            ]
            if len(links) != 1:
                raise EvidenceError("ATIF lacks the canonical parent spawn result")
            links[0].subagent_trajectory_ref = [SubagentTrajectoryRef(trajectory_id=sid)]
        root = trajectories.pop(self.records.root_id)
        root.subagent_trajectories = list(trajectories.values()) or None
        return Trajectory.model_validate(root.model_dump())
