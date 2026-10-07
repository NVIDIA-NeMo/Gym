# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenCode artifacts and request state over the shared sandbox lifecycle."""

import asyncio
import json
import logging
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from shlex import quote
from time import monotonic

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from nemo_gym.agent_utils.sandbox_session import SandboxCommand, SandboxSession
from nemo_gym.agent_utils.supervisor_client import remove_session_directory
from nemo_gym.base_responses_api_agent import AgentSessionState
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentObservationBundle, ObservationGap, TrajectoryRecord
from nemo_gym.sandbox.utils import read_text, upload_text
from responses_api_agents.opencode_agent.artifacts import parse_opencode_observations


LOG = logging.getLogger(__name__)


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
    python: str = "python3"
    model_ref: ModelServerRef | None = None
    task: asyncio.Task[NeMoGymResponse] | None = None
    runtime_info: HarnessProcessInfo | None = None
    ripgrep_info: dict[str, str] | None = None
    opencode_config: dict[str, JsonValue] = field(default_factory=dict)
    observations: AgentObservationBundle | None = None
    trajectory: TrajectoryRecord | None = None
    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None
    session_directory: str | None = None
    native_session_id: str | None = None
    message_ids: list[str] = field(default_factory=list)
    activation_observations: AgentObservationBundle | None = None
    activation_log: str = ""
    event_stream_gaps: list[ObservationGap] = field(default_factory=list)
    started_at: float = field(default_factory=monotonic)
    execution_started_at: float | None = None
    system_instructions: str | None = None

    @property
    def persistent_directory(self) -> str:
        """The private native conversation store survives supervised activations."""
        return self.session_directory or self.session.session_dir

    async def prepare_activation(self, activation_id: int) -> None:
        """Use distinct supervisor state only after the preceding process was reaped."""
        if self.session.launch_started and (
            self.session.cleanup is None or not self.session.cleanup["cleanup_confirmed"]
        ):
            raise RuntimeError("OpenCode cannot resume before confirmed previous activation cleanup")
        if self.session.closing:
            raise RuntimeError("OpenCode session is closing")
        self.session_directory = self.persistent_directory
        directory = f"{self.session_directory}/activation-{activation_id}"
        result = await self.session.sandbox.exec(f"mkdir -p -- {quote(directory)}", cwd="/", timeout_s=30)
        if result.return_code != 0 or result.error_type:
            raise RuntimeError(f"Cannot prepare OpenCode activation directory: {result.stderr}")
        self.session = SandboxSession(
            sandbox=self.session.sandbox,
            session_dir=directory,
            workdir=self.session.workdir,
            harness="OpenCode",
            owns_sandbox=self.session.owns_sandbox,
        )
        self.activation_log = ""
        self.activation_observations = None

    async def upload_json(self, name: str, payload: JsonValue) -> None:
        """Stage adapter input using the shared text-transfer utility."""
        await upload_text(self.session.sandbox, path=f"{self.session.session_dir}/{name}", text=json.dumps(payload))

    async def read_text(self, name: str) -> str:
        """Read adapter output using file transfer."""
        return await read_text(self.session.sandbox, path=f"{self.session.session_dir}/{name}")

    async def close(self, timeout: float) -> None:
        """Capture and release before cancelling the HTTP activation."""
        # The session root includes earlier activation receipts and the native database.
        # Remove it while the borrowed connection is still live, after final capture.
        if self.session_directory and not self.session.closed and not self.session.owns_sandbox:
            await self.session._finish(timeout=timeout)
            await remove_session_directory(
                self.session.sandbox,
                session_dir=self.session_directory,
                workdir=self.session.workdir,
                timeout=timeout,
                harness="OpenCode",
            )
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
        self,
        payload: dict[str, JsonValue],
        *,
        timeout: float,
        close_timeout: float,
        timeout_resolver: Callable[[], float] | None = None,
    ) -> str:
        """Run through the common lifecycle; the adapter validates terminal events."""
        artifacts = await self.session.execute(
            stage_activation=lambda: self.stage_activation(payload),
            collect=lambda: self.collect_artifacts(timeout=close_timeout),
            timeout=timeout,
            close_timeout=close_timeout,
            timeout_resolver=timeout_resolver,
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
            python=self.python,
            argv=[
                self.python,
                "-I",
                f"{self.persistent_directory}/sandbox_runner.py",
                f"{self.session.session_dir}/input.json",
            ],
        )

    async def collect_artifacts(self, *, timeout: float) -> str:
        """Snapshot only after cleanup; keep observations before sandbox release."""
        await self.snapshot(timeout)
        try:
            export_text = await self.read_text("export.json")
            export = json.loads(export_text)
        except Exception as error:
            logs = await self.session.read_output_log()
            raise RuntimeError(f"OpenCode sandbox runner returned no valid result: {logs}") from error
        if self.session_directory:
            native_id = export.get("session_id")
            if not isinstance(native_id, str) or not native_id:
                raise RuntimeError("OpenCode did not persist a resumable session ID")
            if self.native_session_id is not None and native_id != self.native_session_id:
                raise RuntimeError("OpenCode resumed a different native session")
            self.native_session_id = native_id
            self.message_ids = export["message_ids"]
        try:
            self.activation_log = await self.read_text("stdout.jsonl")
        except Exception:
            self.activation_log = ""
        try:
            with tempfile.TemporaryDirectory(prefix="opencode-observations-") as directory:
                path = Path(directory) / "observations.db"
                await self.session.sandbox.download(f"{self.session.session_dir}/observations.db", path)
                trajectory = TrajectoryRecord(
                    task_id=self.request.task_id.task_id, rollout_id=self.request.episode_id.capture_key
                )
                self.observations = parse_opencode_observations(
                    path,
                    self.request.episode_id.capture_key,
                    trajectory,
                    require_terminal_finish=True,
                    model_ref=self.model_ref,
                )
                self.trajectory = trajectory
                self.activation_observations = parse_opencode_observations(
                    path,
                    self.request.episode_id.capture_key,
                    require_terminal_finish=True,
                    model_ref=self.model_ref,
                    message_ids=set(export.get("activation_message_ids", [])) if self.session_directory else None,
                )
        except Exception:
            if self.trajectory is not None:
                self.trajectory.gaps.append(ObservationGap(code="turns_unavailable"))
            missing = AgentObservationBundle(
                source="opencode",
                gaps=[
                    ObservationGap(code="agent_artifact_unavailable"),
                    ObservationGap(code="observation_capture_failed"),
                ],
            )
            self.activation_observations = missing
            if self.observations is None:
                self.observations = missing.model_copy(deep=True)
            else:
                self.observations.gaps.extend(missing.gaps)
        try:
            runtime = json.loads(await self.read_text("runtime.json"))
        except Exception:
            runtime = None
        self.runtime_info = parse_runtime_info(runtime)
        return export_text

    async def snapshot(self, timeout: float) -> None:
        """Copy SQLite output after all harness database writers have stopped."""
        if self.session.cleanup is None or not self.session.cleanup["cleanup_confirmed"]:
            raise RuntimeError("OpenCode transcript capture requires confirmed cleanup")
        captured = await self.session.sandbox.exec(
            f"{quote(self.python)} -I {quote(self.persistent_directory + '/sandbox_runner.py')} "
            f"--snapshot {quote(self.session.session_dir)} {quote(self.persistent_directory)}",
            cwd=self.session.workdir,
            timeout_s=timeout,
        )
        if captured.return_code != 0 or captured.error_type:
            raise RuntimeError(f"OpenCode transcript capture failed: {captured.stderr}")
