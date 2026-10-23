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
from time import monotonic

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError

from nemo_gym.agent_utils.sandbox_session import SandboxCommand, SandboxSession
from nemo_gym.base_responses_api_agent import AgentSessionState
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentInvocation, AgentObservationBundle, ObservationGap, TrajectoryRecord
from nemo_gym.sandbox import SandboxExecResult
from nemo_gym.sandbox.utils import read_text, upload_text
from responses_api_agents.opencode_agent.artifacts import parse_opencode_observations
from responses_api_agents.opencode_agent.observability import scope_opencode_activation
from responses_api_agents.opencode_agent.sandbox_runner import ExportStatus, export_failures


LOG = logging.getLogger(__name__)


def format_error_message(message: str, *, output: str = "", stderr: str = "") -> str:
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


_EXPORT_STATUS = TypeAdapter(ExportStatus)


def _export_status(result: SandboxExecResult) -> ExportStatus | None:
    """The runner's one-line export status, from stdout or wherever the provider put the command's output."""
    for stream in (result.stdout, result.stderr):
        for line in reversed((stream or "").splitlines()):
            if line.startswith("{"):
                try:
                    return _EXPORT_STATUS.validate_json(line)
                except ValidationError:
                    continue
    return None


def _export_failure(result: SandboxExecResult, *, budget: float) -> str:
    """Why ``--export`` failed: the runner's status first, then what the provider kept of its output.

    OpenSandbox's background exec, for one, returns stderr merged into stdout and only the exit status as stderr.
    """
    parts = [f"exit={result.return_code}, error={result.error_type}, exec budget {budget:.1f} s"]
    status = _export_status(result)
    if status is None:
        parts.append(f"no export status; stdout: {(result.stdout or '')[-1000:]}")
    else:
        exported = sum(outcome == "ok" for outcome in status["sessions"].values())
        parts.append(
            f"{exported} of {len(status['sessions'])} sessions exported in {status['elapsed_seconds']} of "
            f"{status['budget_seconds']} s{', deadline exhausted' if status['deadline_exhausted'] else ''}"
        )
        parts.extend(export_failures(status))
    if result.stderr:
        parts.append(f"stderr: {result.stderr[-1000:]}")
    return "; ".join(parts)


@dataclass
class OpenCodeSandboxSession(AgentSessionState):
    """Keep OpenCode HOME/history under ``conversation_directory`` across launches.

    ``session.session_dir`` holds only the current supervisor and input/output
    files. ``persistent_directory`` selects that original session root for legacy
    one-shot calls and the retained conversation root for continuation calls.
    SandboxSession owns finalization and removal of the entire directory tree.
    """

    session: SandboxSession[str]
    runtime: str
    sandbox_id: str | None = None
    python_executable: str = "python3"
    model_ref: ModelServerRef | None = None
    task: asyncio.Task[NeMoGymResponse] | None = None
    runtime_info: HarnessProcessInfo | None = None
    observations: AgentObservationBundle | None = None
    stderr: str = ""
    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None
    conversation_directory: str | None = None
    harness_session_id: str | None = None
    message_ids: list[str] = field(default_factory=list)
    system_instructions: str | None = None
    activation_log: str = ""
    activation_observations: AgentObservationBundle | None = None
    trajectory: TrajectoryRecord | None = None
    activation_trajectory: TrajectoryRecord | None = None
    activation_id: int | None = None
    export_dir: str | None = None
    _export_gaps: list[ObservationGap] = field(default_factory=list, init=False, repr=False)
    _retained_activations: dict[int, tuple[AgentObservationBundle, TrajectoryRecord | None]] = field(
        default_factory=dict, init=False, repr=False
    )

    @property
    def is_continuation(self) -> bool:
        """Whether the seed opted into multi-turn continuation, including before its first launch."""
        return self.request.continuation is not None

    @property
    def invocation_prefix(self) -> str:
        """Unique process-activation identity, also passed to the OpenCode header plugin."""
        assert self.activation_id is not None
        return f"{self.request.agent_session_id}:activation-{self.activation_id}"

    @property
    def persistent_directory(self) -> str:
        """Private OpenCode session storage that survives process activations."""
        return self.conversation_directory or self.session.session_dir

    async def prepare_activation(self, activation_id: int, *, timeout: float) -> None:
        """Keep one conversation while rearming the supervised execution lifecycle."""
        assert self.is_continuation
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
        """Upsert activation evidence without changing an outcome already returned to its caller."""
        if self.activation_observations is None:
            return
        assert self.activation_id is not None
        previous = self._retained_activations.get(self.activation_id)
        if previous is not None and previous[0] is not self.activation_observations:
            original = next(
                (r for r in previous[0].records if isinstance(r, AgentInvocation) and r.parent_invocation_id is None),
                None,
            )
            if original is not None:
                if not any(
                    isinstance(record, AgentInvocation) and record.parent_invocation_id is None
                    for record in self.activation_observations.records
                ):
                    # Failed artifact recovery must not erase the activation's outcome
                    # or previously captured evidence. Only add the new capture gaps.
                    gaps = list(previous[0].gaps)
                    for gap in self.activation_observations.gaps:
                        if gap not in gaps:
                            gaps.append(gap)
                    self.activation_observations = previous[0].model_copy(update={"gaps": gaps})
                    self.activation_trajectory = previous[1]
                # A close retry can discover the OpenCode session ID and transcript after
                # a failed activation retained only a synthetic root invocation.
                recovered = list(self.activation_observations.records)
                if self.activation_trajectory is not None:
                    recovered.extend(self.activation_trajectory.invocations)
                for record in recovered:
                    if isinstance(record, AgentInvocation) and record.parent_invocation_id is None:
                        record.status = original.status
                        record.error_type = original.error_type
                        record.stop_reason = original.stop_reason
                        record.duration_ms = original.duration_ms
        self._retained_activations[self.activation_id] = (self.activation_observations, self.activation_trajectory)
        bundles = [bundle for bundle, _ in self._retained_activations.values()]
        self.observations = AgentObservationBundle(
            source="opencode",
            records=[record for bundle in bundles for record in bundle.records],
            gaps=[gap for bundle in bundles for gap in bundle.gaps],
        )
        trajectories = [trajectory for _, trajectory in self._retained_activations.values() if trajectory is not None]
        if trajectories:
            self.trajectory = TrajectoryRecord(
                task_id=self.request.task_id.task_id,
                rollout_id=self.request.episode_id.capture_key,
                **{
                    name: [record for trajectory in trajectories for record in getattr(trajectory, name)]
                    for name in ("invocations", "turns", "model_calls", "tool_calls", "gaps")
                },
            )

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
        if self.is_continuation:
            self.retain_activation()

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
        if self.is_continuation:
            await self.session.sandbox.upload(
                Path(__file__).with_name("task_environment.mjs"), f"{self.session.session_dir}/task_environment.mjs"
            )
        await self.upload_json("input.json", payload)
        return SandboxCommand(
            python=self.python_executable,
            argv=[
                self.python_executable,
                "-I",
                f"{self.persistent_directory}/sandbox_runner.py",
                f"{self.session.session_dir}/input.json",
            ],
        )

    def take_export_gaps(self) -> list[ObservationGap]:
        """Hand over export failures recorded since the last call, so each is reported once."""
        gaps, self._export_gaps = self._export_gaps, []
        return gaps

    async def collect_artifacts(self, *, timeout: float) -> str:
        """Snapshot only after cleanup; keep observations before sandbox release.

        With ``export_dir``, the owner's copies are written after the capture, whether or not it
        succeeded, and inside this step's budget, so exporting can neither fail the capture nor delay
        release. A failed export becomes an observation gap.
        """
        if self.export_dir is None:
            return await self._capture_artifacts(timeout=timeout)
        # Leave a tenth of the budget so the capture result returns before the step's own timeout.
        export_deadline = monotonic() + timeout * 0.9
        try:
            return await self._capture_artifacts(timeout=timeout)
        finally:
            await self.write_exports(deadline=export_deadline)

    async def write_exports(self, *, deadline: float) -> None:
        """Write every OpenCode session's ``opencode export`` and the event log to ``export_dir``.

        Runs only after confirmed termination, before the session directory is removed. Failures,
        including an exhausted budget, are recorded as an ``export_dir_write_failed`` gap that names
        each failed session from the runner's status line.
        """
        assert self.export_dir is not None
        try:
            budget = deadline - monotonic()
            if budget <= 0:
                raise TimeoutError("no time left in the artifact collection budget")
            # The runner stops first, so it reports each session's outcome before its exec times out.
            runner_budget = budget - min(5.0, 0.1 * budget)
            exec_timeout = asyncio.timeout(budget)
            try:
                async with exec_timeout:
                    result = await self.session.sandbox.exec(
                        f"python3 -I {quote(self.persistent_directory + '/sandbox_runner.py')} --export "
                        f"{quote(self.session.session_dir)} {quote(self.persistent_directory)} "
                        f"{quote(self.export_dir)} {runner_budget:.3f}",
                        cwd=self.session.workdir,
                        timeout_s=budget,
                    )
            except TimeoutError:
                if not exec_timeout.expired():
                    raise
                raise TimeoutError(
                    f"the export exec did not return within {budget:.1f} s (runner deadline {runner_budget:.1f} s)"
                ) from None
            if result.return_code != 0 or result.error_type:
                raise RuntimeError(_export_failure(result, budget=budget))
        except Exception as error:
            LOG.warning("OpenCode session exports for %s are incomplete", self.export_dir, exc_info=True)
            self._export_gaps.append(
                ObservationGap(code="export_dir_write_failed", detail=f"{type(error).__name__}: {error}"[:2500])
            )

    async def _capture_artifacts(self, *, timeout: float) -> str:
        previous_session_id = self.harness_session_id
        try:
            self.stderr = (await self.read_text("stderr.log"))[-16000:]
        except Exception:
            LOG.warning("OpenCode stderr log unavailable", exc_info=True)
        try:
            await self.snapshot(timeout)
        except Exception as error:
            if self.stderr:
                raise RuntimeError(format_error_message(str(error), stderr=self.stderr)) from error
            raise
        # Messages that predate this activation (earlier activations, an imported prefix) are history, not output.
        scoped = self.is_continuation or bool(self.message_ids)
        if self.is_continuation:
            export_text = await self._read_export()
            export = json.loads(export_text)
            harness_session_id = export.get("session_id")
            if not isinstance(harness_session_id, str) or not harness_session_id:
                raise RuntimeError("OpenCode did not persist a resumable session ID")
            if self.harness_session_id is not None and harness_session_id != self.harness_session_id:
                raise RuntimeError("OpenCode resumed a different OpenCode session")
            self.harness_session_id = harness_session_id
            self.message_ids = export["message_ids"]
        elif scoped:
            export = json.loads(await self._read_export())
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
                    message_ids=set(export.get("activation_message_ids", [])) if scoped else None,
                )
                if self.is_continuation:
                    assert self.activation_id is not None
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
            if self.is_continuation:
                self.activation_observations = observations
            else:
                self.observations = observations
        try:
            runtime = json.loads(await self.read_text("runtime.json"))
        except Exception:
            LOG.exception("Failed to read OpenCode runtime metadata")
            runtime = None
        self.runtime_info = parse_runtime_info(runtime)
        if not self.is_continuation:
            # Preserve the single-turn evidence/error order. The response layer
            # parses this raw export and translates malformed content to 502.
            await self._check_task_environment()
            return await self._read_export()
        try:
            await self._check_task_environment()
        except Exception as error:
            # A process can exhaust its relative budget before plugins load.
            # Accept only an empty OpenCode checkpoint and confirmed timeout;
            # any newly accepted input still requires the task-environment receipt.
            if (
                self.harness_session_id is not None
                and (previous_session_id is None or self.harness_session_id == previous_session_id)
                and self.session.cleanup is not None
                and self.session.cleanup["cleanup_confirmed"]
                and self.session.cleanup["timed_out"]
                and export.get("activation_message_ids") == []
                and export.get("messages") == []
            ):
                return export_text
            raise RuntimeError(
                format_error_message("OpenCode task environment initialization was not confirmed", stderr=self.stderr)
            ) from error
        return export_text

    async def _read_export(self) -> str:
        try:
            return await self.read_text("export.json")
        except Exception as error:
            logs = await self.session.read_output_log()
            raise RuntimeError(
                format_error_message(
                    "OpenCode sandbox runner returned no valid result", output=logs, stderr=self.stderr
                )
            ) from error

    async def _check_task_environment(self) -> None:
        try:
            if await self.read_text("task-environment.ready") != "ready\n":
                raise ValueError("Invalid task environment initialization receipt")
        except Exception as error:
            raise RuntimeError(
                format_error_message("OpenCode task environment initialization was not confirmed", stderr=self.stderr)
            ) from error

    async def snapshot(self, timeout: float) -> None:
        """Capture SQLite after CLI completion; borrowed task services may remain alive."""
        if self.session.cleanup is None or not self.session.cleanup["cleanup_confirmed"]:
            raise RuntimeError("OpenCode transcript capture requires confirmed cleanup")
        captured = await self.session.sandbox.exec(
            f"{quote(self.python_executable)} -I {quote(self.persistent_directory + '/sandbox_runner.py')} "
            f"--snapshot {quote(self.session.session_dir)} {quote(self.persistent_directory)}",
            cwd=self.session.workdir,
            timeout_s=timeout,
        )
        if captured.return_code != 0 or captured.error_type:
            raise RuntimeError(f"OpenCode transcript capture failed: {captured.stderr}")
