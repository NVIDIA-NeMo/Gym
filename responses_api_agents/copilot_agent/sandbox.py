# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Copilot artifacts and request state over the shared sandbox lifecycle."""

import asyncio
import json
import logging
import shlex
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from nemo_gym.agent_utils.sandbox_session import SandboxCommand, SandboxSession
from nemo_gym.base_responses_api_agent import AgentSessionState
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentObservationBundle
from nemo_gym.sandbox.utils import read_text, upload_text


LOG = logging.getLogger(__name__)


class HarnessProcessInfo(BaseModel):
    """Optional runner identity, independent of supervisor cleanup."""

    model_config = ConfigDict(extra="forbid", strict=True)
    hostname: str
    pid: int
    python: str | None = None


class CopilotRuntime(BaseModel):
    """The installed runtime that ``install_copilot_runtime.py`` reports."""

    model_config = ConfigDict(extra="ignore")
    binary: str
    dist_dir: str
    package: str
    version: str


class CopilotArtifacts(BaseModel):
    """Files collected after harness termination; each is ``None`` when the run did not write it."""

    events: str
    otel: str | None = None
    usage: str | None = None
    compaction_state: bool | None = None


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
    runtime: CopilotRuntime | None = None
    sandbox_id: str | None = None
    task: asyncio.Task[NeMoGymResponse] | None = None
    runtime_info: HarnessProcessInfo | None = None
    observations: AgentObservationBundle | None = None
    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None

    async def upload_json(self, name: str, payload: JsonValue) -> None:
        """Stage adapter input using the shared text-transfer utility."""
        await upload_text(self.session.sandbox, path=f"{self.session.session_dir}/{name}", text=json.dumps(payload))

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
        self, payload: dict[str, JsonValue], *, timeout: float, close_timeout: float
    ) -> CopilotArtifacts:
        """Run through the common lifecycle; the adapter validates the terminal events."""
        return await self.session.execute(
            stage_activation=lambda: self.stage_activation(payload),
            collect=self.collect_artifacts,
            timeout=timeout,
            close_timeout=close_timeout,
        )

    async def stage_activation(self, payload: dict[str, JsonValue]) -> SandboxCommand:
        """Stage the invocation and return the runner command."""
        await self.upload_json("input.json", payload)
        return SandboxCommand(
            python="python3",
            argv=[
                "python3",
                "-I",
                f"{self.session.session_dir}/sandbox_runner.py",
                f"{self.session.session_dir}/input.json",
            ],
        )

    async def read_compaction_state(self) -> bool | None:
        """Inspect the private native event store after termination, before session cleanup."""
        directory = self.session.session_dir
        try:
            result = await self.session.sandbox.exec(
                shlex.join(["python3", "-I", f"{directory}/sandbox_runner.py", "--compaction-state", directory]),
                timeout_s=30,
            )
            if result.return_code == 0 and not result.error_type:
                value = json.loads(result.stdout)
                if isinstance(value, bool):
                    return value
        except Exception:
            LOG.warning("Could not inspect Copilot native compaction state", exc_info=True)
        return None

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
            usage=await self.read_optional("usage.json"),
            compaction_state=await self.read_compaction_state(),
        )
