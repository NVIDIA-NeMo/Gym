# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenCode artifacts and request state over the shared sandbox lifecycle."""

import asyncio
import json
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from shlex import quote

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from nemo_gym.agent_utils.sandbox_session import SandboxCommand, SandboxSession
from nemo_gym.base_responses_api_agent import AgentSessionState
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentObservationBundle, ObservationGap, TrajectoryRecord
from nemo_gym.sandbox.utils import read_text, upload_text
from responses_api_agents.opencode_agent.artifacts import parse_opencode_observations
from responses_api_agents.opencode_agent.observability import scope_opencode_activation


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
    """Keep native HOME/history under ``conversation_directory`` across launches.

    ``session.session_dir`` holds only the current supervisor and input/output
    files. ``persistent_directory`` selects that original session root for legacy
    one-shot calls and the retained conversation root for continuation calls.
    SandboxSession owns finalization and removal of the entire directory tree.
    """

    session: SandboxSession[str]
    runtime: str
    model_ref: ModelServerRef | None = None
    task: asyncio.Task[NeMoGymResponse] | None = None
    runtime_info: HarnessProcessInfo | None = None
    observations: AgentObservationBundle | None = None
    stderr: str = ""
    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None
    conversation_directory: str | None = None
    native_session_id: str | None = None
    message_ids: list[str] = field(default_factory=list)
    system_instructions: str | None = None
    activation_log: str = ""
    activation_observations: AgentObservationBundle | None = None
    trajectory: TrajectoryRecord | None = None
    activation_trajectory: TrajectoryRecord | None = None
    activation_id: int | None = None

    @property
    def invocation_prefix(self) -> str:
        """Unique process-activation identity, also passed to the native header plugin."""
        return f"{self.request.agent_session_id}:activation-{self.activation_id}"

    @property
    def persistent_directory(self) -> str:
        """Private native conversation storage that survives process activations."""
        return self.conversation_directory or self.session.session_dir

    async def prepare_activation(self, activation_id: int, *, timeout: float) -> None:
        """Keep one conversation while rearming the supervised execution lifecycle."""
        self.activation_id = activation_id
        self.activation_log = ""
        self.activation_observations = None
        self.activation_trajectory = None
        self.runtime_info = None
        self.stderr = ""
        self.conversation_directory = self.persistent_directory
        directory = f"{self.conversation_directory}/activation-{activation_id}"
        await self.session.begin_activation(session_dir=directory, timeout=timeout)
        result = await self.session.sandbox.exec(f"mkdir -p -- {quote(directory)}", cwd="/", timeout_s=timeout)
        if result.error_type == "timeout":
            raise TimeoutError("OpenCode activation directory preparation timed out")
        if result.return_code != 0 or result.error_type:
            raise RuntimeError(f"Cannot prepare OpenCode activation directory: {result.stderr}")

    def retain_activation(self) -> None:
        """Accumulate completed launch evidence once; retries replay above the adapter."""
        if self.activation_observations is None:
            return
        if self.observations is None:
            self.observations = AgentObservationBundle(source="opencode")
        self.observations.records.extend(self.activation_observations.records)
        self.observations.gaps.extend(self.activation_observations.gaps)
        if self.activation_trajectory is not None:
            if self.trajectory is None:
                self.trajectory = TrajectoryRecord(
                    task_id=self.request.task_id.task_id, rollout_id=self.request.episode_id.capture_key
                )
            for field in ("invocations", "turns", "tool_calls", "gaps"):
                getattr(self.trajectory, field).extend(getattr(self.activation_trajectory, field))

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

    async def execute(
        self,
        payload: dict[str, JsonValue],
        *,
        timeout: float,
        close_timeout: float,
        deadline: float | None = None,
    ) -> str:
        """Run through the common lifecycle; the adapter validates terminal events."""
        artifacts = await self.session.execute(
            stage_activation=lambda: self.stage_activation(payload),
            collect=lambda: self.collect_artifacts(timeout=close_timeout),
            timeout=timeout,
            close_timeout=close_timeout,
            deadline=deadline,
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
        if self.conversation_directory is not None:
            await self.session.sandbox.upload(
                Path(__file__).with_name("task_environment.mjs"), f"{self.session.session_dir}/task_environment.mjs"
            )
        await self.upload_json("input.json", payload)
        return SandboxCommand(
            python="python3",
            argv=[
                "python3",
                "-I",
                f"{self.persistent_directory}/sandbox_runner.py",
                f"{self.session.session_dir}/input.json",
            ],
        )

    async def collect_artifacts(self, *, timeout: float) -> str:
        """Snapshot only after cleanup; keep observations before sandbox release."""
        previous_session_id = self.native_session_id
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
        if self.conversation_directory is None:
            await self._check_task_environment()
        try:
            export_text = await self.read_text("export.json")
            export = json.loads(export_text)
        except Exception as error:
            logs = await self.session.read_output_log()
            raise RuntimeError(
                format_sandbox_error(
                    "OpenCode sandbox runner returned no valid result", output=logs, stderr=self.stderr
                )
            ) from error
        if self.conversation_directory:
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
            LOG.warning("OpenCode event log unavailable", exc_info=True)
        try:
            with tempfile.TemporaryDirectory(prefix="opencode-observations-") as directory:
                path = Path(directory) / "observations.db"
                await self.session.sandbox.download(f"{self.session.session_dir}/observations.db", path)
                trajectory = TrajectoryRecord(
                    task_id=self.request.task_id.task_id, rollout_id=self.request.episode_id.capture_key
                )
                observations = await asyncio.to_thread(
                    parse_opencode_observations,
                    path,
                    self.request.episode_id.capture_key,
                    trajectory,
                    require_terminal_finish=True,
                    model_ref=self.model_ref,
                    message_ids=set(export.get("activation_message_ids", []))
                    if self.conversation_directory is not None
                    else None,
                )
                if self.activation_id is not None:
                    scope_opencode_activation(
                        observations,
                        trajectory,
                        invocation_prefix=self.invocation_prefix,
                        activation_id=self.activation_id,
                    )
                    self.activation_observations = observations
                    self.activation_trajectory = trajectory
                else:
                    self.observations = observations
        except Exception:
            LOG.exception("Failed to parse OpenCode observations")
            observations = AgentObservationBundle(
                source="opencode",
                gaps=[
                    ObservationGap(code="agent_artifact_unavailable"),
                    ObservationGap(code="observation_capture_failed"),
                ],
            )
            if self.activation_id is not None:
                self.activation_observations = observations
            else:
                self.observations = observations
        try:
            runtime = json.loads(await self.read_text("runtime.json"))
        except Exception:
            LOG.exception("Failed to read OpenCode runtime metadata")
            runtime = None
        self.runtime_info = parse_runtime_info(runtime)
        try:
            if self.conversation_directory is not None:
                await self._check_task_environment()
        except Exception as error:
            # A resumed process can exhaust its relative budget before plugins load.
            # Accept only an unchanged native checkpoint and confirmed timeout;
            # any newly accepted input still requires the task-environment receipt.
            if (
                previous_session_id is not None
                and self.native_session_id == previous_session_id
                and self.session.cleanup is not None
                and self.session.cleanup["cleanup_confirmed"]
                and self.session.cleanup["timed_out"]
                and export.get("activation_message_ids") == []
                and export.get("messages") == []
            ):
                return export_text
            raise RuntimeError(
                format_sandbox_error("OpenCode task environment initialization was not confirmed", stderr=self.stderr)
            ) from error
        return export_text

    async def _check_task_environment(self) -> None:
        try:
            if await self.read_text("task-environment.ready") != "ready\n":
                raise ValueError("Invalid task environment initialization receipt")
        except Exception as error:
            raise RuntimeError(
                format_sandbox_error("OpenCode task environment initialization was not confirmed", stderr=self.stderr)
            ) from error

    async def snapshot(self, timeout: float) -> None:
        """Capture SQLite after CLI completion; borrowed task services may remain alive."""
        if self.session.cleanup is None or not self.session.cleanup["cleanup_confirmed"]:
            raise RuntimeError("OpenCode transcript capture requires confirmed cleanup")
        captured = await self.session.sandbox.exec(
            f"python3 -I {quote(self.persistent_directory + '/sandbox_runner.py')} "
            f"--snapshot {quote(self.session.session_dir)} {quote(self.persistent_directory)}",
            cwd=self.session.workdir,
            timeout_s=timeout,
        )
        if captured.return_code != 0 or captured.error_type:
            raise RuntimeError(f"OpenCode transcript capture failed: {captured.stderr}")
