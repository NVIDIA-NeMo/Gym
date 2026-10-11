# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenCode artifacts and request state over the shared sandbox lifecycle."""

import asyncio
import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from shlex import quote

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from nemo_gym.agent_utils.sandbox_session import SandboxCommand, SandboxSession
from nemo_gym.base_responses_api_agent import AgentSessionState
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentObservationBundle, ObservationGap
from nemo_gym.sandbox.utils import read_text, upload_text
from responses_api_agents.opencode_agent.artifacts import parse_opencode_observations


LOG = logging.getLogger(__name__)


def format_sandbox_error(message: str, *, output: str = "", stderr: str = "") -> str:
    """Bound the combined error to 16k characters, preserving context and log tails."""
    parts = [message]
    if output:
        parts.append(f"output: {output}")
    if stderr and stderr not in message and stderr not in output:
        parts.append(f"stderr: {stderr}")
    budget = 16000 // len(parts) - 1
    marker = "\n...[truncated]...\n"
    return "\n".join(
        part if len(part) <= budget else part[:256] + marker + part[-(budget - 256 - len(marker)) :] for part in parts
    )


class HarnessProcessInfo(BaseModel):
    """Optional OpenCode shim identity, independent of supervisor cleanup."""

    model_config = ConfigDict(extra="forbid", strict=True)
    hostname: str
    pid: int
    python: str | None = None


def parse_runtime_info(payload: object) -> HarnessProcessInfo | None:
    """Read OpenCode diagnostics without failing an otherwise valid episode."""
    try:
        return HarnessProcessInfo.model_validate(payload)
    except ValidationError:
        LOG.warning("OpenCode runtime metadata is missing or malformed")
        return None


@dataclass
class OpenCodeSandboxSession(AgentSessionState):
    """OpenCode request/output state; SandboxSession owns execution and teardown."""

    session: SandboxSession[str]
    runtime: str
    model_ref: ModelServerRef | None = None
    task: asyncio.Task[NeMoGymResponse] | None = None
    runtime_info: HarnessProcessInfo | None = None
    observations: AgentObservationBundle | None = None
    stderr: str = ""
    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None

    async def upload_json(self, name: str, payload: JsonValue) -> None:
        """Stage adapter input using the shared text-transfer utility."""
        await upload_text(self.session.sandbox, path=f"{self.session.session_dir}/{name}", text=json.dumps(payload))

    async def read_text(self, name: str) -> str:
        """Read adapter output using file transfer."""
        return await read_text(self.session.sandbox, path=f"{self.session.session_dir}/{name}")

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

    async def execute(self, payload: dict[str, JsonValue], *, timeout: float, close_timeout: float) -> str:
        """Run through the common lifecycle; the adapter validates terminal events."""
        artifacts = await self.session.execute(
            stage_activation=lambda: self.stage_activation(payload),
            collect=lambda: self.collect_artifacts(timeout=close_timeout),
            timeout=timeout,
            close_timeout=close_timeout,
        )
        if self.session.closing:
            raise asyncio.CancelledError
        return artifacts

    async def stage_activation(self, payload: dict[str, JsonValue]) -> SandboxCommand:
        """Stage the invocation and return the OpenCode worker command."""
        if payload.get("instructions"):
            await upload_text(
                self.session.sandbox, path=f"{self.session.session_dir}/instructions.md", text=payload["instructions"]
            )
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

    async def collect_artifacts(self, *, timeout: float) -> str:
        """Snapshot only after cleanup; keep observations before sandbox release."""
        try:
            self.stderr = (await self.read_text("stderr.log"))[-16000:]
        except Exception:
            LOG.warning("OpenCode stderr log unavailable", exc_info=True)
        try:
            await self.snapshot(timeout)
        except Exception as error:
            if self.stderr:
                raise RuntimeError(format_sandbox_error(str(error), stderr=self.stderr)) from error
            raise
        try:
            with tempfile.TemporaryDirectory(prefix="opencode-observations-") as directory:
                path = Path(directory) / "observations.db"
                await self.session.sandbox.download(f"{self.session.session_dir}/observations.db", path)
                self.observations = await asyncio.to_thread(
                    parse_opencode_observations,
                    path,
                    self.request.episode_id.capture_key,
                    require_terminal_finish=True,
                    model_ref=self.model_ref,
                )
        except Exception:
            LOG.exception("Failed to parse OpenCode observations")
            self.observations = AgentObservationBundle(
                source="opencode",
                gaps=[
                    ObservationGap(code="agent_artifact_unavailable"),
                    ObservationGap(code="observation_capture_failed"),
                ],
            )
        try:
            runtime = json.loads(await self.read_text("runtime.json"))
        except Exception:
            LOG.exception("Failed to read OpenCode runtime metadata")
            runtime = None
        self.runtime_info = parse_runtime_info(runtime)
        try:
            return await self.read_text("export.json")
        except Exception as error:
            logs = await self.session.read_output_log()
            raise RuntimeError(
                format_sandbox_error(
                    "OpenCode sandbox runner returned no valid result", output=logs, stderr=self.stderr
                )
            ) from error

    async def snapshot(self, timeout: float) -> None:
        """Copy SQLite output after all harness database writers have stopped."""
        if self.session.cleanup is None or not self.session.cleanup["cleanup_confirmed"]:
            raise RuntimeError("OpenCode transcript capture requires confirmed cleanup")
        captured = await self.session.sandbox.exec(
            f"python3 -I {quote(self.session.session_dir + '/sandbox_runner.py')} --snapshot {quote(self.session.session_dir)}",
            cwd=self.session.workdir,
            timeout_s=timeout,
        )
        if captured.return_code != 0 or captured.error_type:
            raise RuntimeError(f"OpenCode transcript capture failed: {captured.stderr}")
