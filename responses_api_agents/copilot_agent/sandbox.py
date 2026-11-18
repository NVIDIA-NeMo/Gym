# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Copilot artifacts and request state over the shared sandbox lifecycle."""

import asyncio
import json
import logging
import shlex
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, ValidationError

from nemo_gym.agents import (
    HARNESS_RUNNER_FILE,
    AgentSessionState,
    SandboxCommand,
    SandboxSession,
    shell_assignments,
)
from nemo_gym.core import FailureReport, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import read_text, upload_text
from nemo_gym.trajectory import AgentObservationBundle


LOG = logging.getLogger(__name__)


class HarnessProcessInfo(BaseModel):
    """Optional runner identity, independent of supervisor cleanup."""

    model_config = ConfigDict(extra="forbid", strict=True)
    hostname: str
    pid: int


class CopilotRuntime(BaseModel):
    """The installed runtime that ``install_copilot_runtime.sh`` reports."""

    model_config = ConfigDict(extra="ignore")
    binary: str
    dist_dir: str
    package: str
    version: str


class CopilotArtifacts(BaseModel):
    """Files collected after harness termination; each is ``None`` when the run did not write it."""

    events: str
    otel: str | None = None
    stderr: str | None = None
    compaction_state: bool | None = None


def parse_compaction_state(logs: list[str]) -> bool | None:
    """Whether Copilot's persisted session events record a compaction; ``None`` when they cannot tell.

    They include background compaction that is absent from stdout and the chat spans.
    """
    observed = False
    for log in logs:
        for line in log.split("\n"):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except (ValueError, RecursionError):
                return None
            if not isinstance(event, dict) or not isinstance(event.get("type"), str):
                return None
            observed = True
            if event["type"] in ("session.compaction_start", "session.compaction_complete"):
                return True
    return False if observed else None


def parse_runtime_info(payload: object) -> HarnessProcessInfo | None:
    """Read runner diagnostics without failing an otherwise valid episode."""
    try:
        return HarnessProcessInfo.model_validate(payload)
    except ValidationError:
        LOG.warning("Copilot runtime metadata is missing or malformed")
        return None


@dataclass
class CopilotSandboxSession(AgentSessionState):
    """Copilot request/output state; SandboxSession owns execution and teardown."""

    session: SandboxSession[CopilotArtifacts]
    # Correlates model calls with this invocation (x-session-id). Copilot masks provider header values wherever they
    # appear in its output, so this must be an ID that task text cannot contain, not the short capture key.
    invocation_id: str
    runtime: CopilotRuntime | None = None
    sandbox_id: str | None = None
    task: asyncio.Task[NeMoGymResponse] | None = None
    runtime_info: HarnessProcessInfo | None = None
    observations: AgentObservationBundle | None = None
    failure_report: FailureReport = field(default_factory=FailureReport)
    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None

    async def read_text(self, name: str) -> str:
        """Read adapter output using file transfer."""
        return await read_text(self.session.sandbox, path=f"{self.session.session_dir}/{name}")

    async def read_optional(self, name: str) -> str | None:
        try:
            return await self.read_text(name)
        except Exception:
            return None

    async def close(self, timeout: float) -> None:
        """Capture and release before cancelling the HTTP activation."""
        await self.session.close(timeout=timeout)
        if self.task is not None and not self.task.done() and not self.task.cancelling():
            self.task.cancel()
        if self.task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self.task), timeout=timeout)
            except asyncio.CancelledError:
                if not self.task.cancelled():
                    raise
            except Exception:
                if not self.task.done():
                    raise
                # The agent base replays the activation error independently of close.

    async def execute(
        self, *, command: list[str], env: dict[str, str], timeout: float, close_timeout: float
    ) -> CopilotArtifacts:
        """Run through the common lifecycle; the adapter validates the terminal events."""
        return await self.session.execute(
            stage_activation=lambda: self.stage_activation(command=command, env=env),
            collect=self.collect_artifacts,
            timeout=timeout,
            close_timeout=close_timeout,
        )

    async def stage_activation(self, *, command: list[str], env: dict[str, str]) -> SandboxCommand:
        """Stage the invocation and return the shared runner command."""
        directory = self.session.session_dir
        params = {
            "cwd": self.session.workdir,
            "command": command,
            "env": env,
            "stdin": "/dev/null",
            "stdout": f"{directory}/stdout.jsonl",
            "stderr": f"{directory}/stderr.log",
            "events": f"{directory}/events.jsonl",
        }
        await upload_text(self.session.sandbox, path=f"{directory}/params.sh", text=shell_assignments(params))
        return SandboxCommand(argv=["bash", f"{directory}/{HARNESS_RUNNER_FILE}", f"{directory}/params.sh"])

    async def read_compaction_state(self) -> bool | None:
        """Read the private native event store after termination, before session cleanup."""
        store = shlex.quote(f"{self.session.session_dir}/copilot-home/session-state")
        try:
            listed = await self.session.sandbox.exec(f"ls -1 -- {store}/*/events.jsonl", timeout_s=30)
            if listed.return_code != 0 or listed.error_type:
                return None  # No event store, or it cannot be listed.
            paths = [path for path in (listed.stdout or "").split("\n") if path]
            logs = [await read_text(self.session.sandbox, path=path) for path in paths]
        except Exception:
            LOG.warning("Could not inspect Copilot native compaction state", exc_info=True)
            return None
        return parse_compaction_state(logs)

    async def collect_artifacts(self) -> CopilotArtifacts:
        """Keep partial events even when runner metadata or Copilot's own files are missing."""
        try:
            events = await self.read_text("events.jsonl")
        except Exception as error:
            logs = await self.session.read_output_log()
            raise RuntimeError(f"Copilot sandbox runner returned no valid result: {logs}") from error
        runtime = await self.read_optional("runtime.json")
        try:
            runtime_payload = json.loads(runtime) if runtime is not None else None
        except ValueError:
            runtime_payload = None
        self.runtime_info = parse_runtime_info(runtime_payload)
        return CopilotArtifacts(
            events=events,
            otel=await self.read_optional("otel.jsonl"),
            stderr=await self.read_optional("stderr.log"),
            compaction_state=await self.read_compaction_state(),
        )
