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

from nemo_gym.agents import HARNESS_RUNNER_FILE, AgentSessionState, SandboxCommand, SandboxSession, shell_assignments
from nemo_gym.config import ModelServerRef
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import read_text, upload_text
from nemo_gym.trajectory import AgentInvocation, AgentObservationBundle, ObservationGap, TrajectoryRecord
from responses_api_agents.opencode_agent.artifacts import parse_opencode_observations
from responses_api_agents.opencode_agent.observability import scope_opencode_activation
from responses_api_agents.opencode_agent.snapshot import snapshot


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


def parse_runtime_info(payload: object) -> HarnessProcessInfo | None:
    """Read OpenCode diagnostics without failing an otherwise valid episode."""
    try:
        return HarnessProcessInfo.model_validate(payload)
    except ValidationError:
        LOG.warning("OpenCode runtime metadata is missing or malformed")
        return None


_TASK_DIRECTORY_KEYS = ("HOME", "XDG_DATA_HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME")
# Record the task's home and XDG directories, before OpenCode's overrides, for the
# task_environment.mjs shell.env hook. Defaults follow the XDG specification.
_TASK_ENV_SETUP = r"""json() {
    local value=${1//\\/\\\\} code octal character escaped
    value=${value//\"/\\\"}
    # JSON forbids raw control characters; a bash string cannot hold NUL.
    for ((code = 1; code < 32; code++)); do
        printf -v octal '%03o' "$code"
        printf -v character "\\$octal"
        printf -v escaped '\\u%04x' "$code"
        value=${value//"$character"/"$escaped"}
    done
    printf '"%s"' "$value"
}
home=${HOME:-~}
printf '{"HOME": %s, "XDG_DATA_HOME": %s, "XDG_CONFIG_HOME": %s, "XDG_CACHE_HOME": %s, "XDG_STATE_HOME": %s}' \
    "$(json "$home")" \
    "$(json "${XDG_DATA_HOME:-${home%/}/.local/share}")" \
    "$(json "${XDG_CONFIG_HOME:-${home%/}/.config}")" \
    "$(json "${XDG_CACHE_HOME:-${home%/}/.cache}")" \
    "$(json "${XDG_STATE_HOME:-${home%/}/.local/state}")" >"$task_env" || exit 1
"""


def runner_setup(directory: str, private_directories: list[str]) -> str:
    """Bash run in the sandbox before the CLI: fresh receipt, task directories, private directories."""
    return (
        # Never accept an initialization receipt left by an earlier invocation.
        f"rm -f -- {quote(f'{directory}/task-environment.ready')} || exit 1\n"
        f"task_env={quote(f'{directory}/task-env.json')}\n"
        + _TASK_ENV_SETUP
        + f"mkdir -p -- {' '.join(quote(path) for path in private_directories)}\n"
    )


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
        directory = self.session.session_dir
        await upload_text(self.session.sandbox, path=f"{directory}/prompt.txt", text=payload["prompt"])
        env = payload["env"]
        params = {
            "cwd": payload["cwd"],
            "command": payload["command"],
            "env": env,
            "stdin": f"{directory}/prompt.txt",
            "stdout": f"{directory}/stdout.jsonl",
            "stderr": f"{directory}/stderr.log",
            "setup": runner_setup(directory, [env[key] for key in _TASK_DIRECTORY_KEYS]),
        }
        if payload.get("path_prefix"):
            # Preinstalled runtime tools come first; the task's own PATH still follows.
            params["setup"] += f"PATH={quote(':'.join(payload['path_prefix']))}:$PATH\n"
        await upload_text(self.session.sandbox, path=f"{directory}/params.sh", text=shell_assignments(params))
        return SandboxCommand(
            argv=["bash", f"{self.persistent_directory}/{HARNESS_RUNNER_FILE}", f"{directory}/params.sh"]
        )

    async def collect_artifacts(self, *, timeout: float) -> str:
        """Snapshot only after cleanup; keep observations before sandbox release."""
        with tempfile.TemporaryDirectory(prefix="opencode-snapshot-") as directory:
            return await self._collect_artifacts(Path(directory), timeout=timeout)

    async def _collect_artifacts(self, directory: Path, *, timeout: float) -> str:
        previous_session_id = self.harness_session_id
        try:
            self.stderr = (await self.read_text("stderr.log"))[-16000:]
        except Exception:
            LOG.warning("OpenCode stderr log unavailable", exc_info=True)
        try:
            await self.snapshot(directory, timeout)
        except Exception as error:
            if self.stderr:
                raise RuntimeError(format_error_message(str(error), stderr=self.stderr)) from error
            raise
        if self.is_continuation:
            export_text = await self._read_export(directory)
            export = json.loads(export_text)
            harness_session_id = export.get("session_id")
            if not isinstance(harness_session_id, str) or not harness_session_id:
                raise RuntimeError("OpenCode did not persist a resumable session ID")
            if self.harness_session_id is not None and harness_session_id != self.harness_session_id:
                raise RuntimeError("OpenCode resumed a different OpenCode session")
            self.harness_session_id = harness_session_id
            self.message_ids = export["message_ids"]
        try:
            self.activation_log = await self.read_text("stdout.jsonl")
        except Exception:
            LOG.warning("OpenCode event log unavailable", exc_info=True)
        try:
            trajectory = TrajectoryRecord(
                task_id=self.request.task_id.task_id, rollout_id=self.request.episode_id.capture_key
            )
            database = directory / "observations.db"
            if not database.is_file():
                raise FileNotFoundError(f"OpenCode observations were not captured: {database}")
            observations = await asyncio.to_thread(
                parse_opencode_observations,
                database,
                self.request.episode_id.capture_key,
                trajectory,
                require_terminal_finish=True,
                model_ref=self.model_ref,
                message_ids=set(export.get("activation_message_ids", [])) if self.is_continuation else None,
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
            return await self._read_export(directory)
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

    async def _read_export(self, directory: Path) -> str:
        try:
            return (directory / "export.json").read_text()
        except OSError as error:
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

    async def snapshot(self, directory: Path, timeout: float) -> None:
        """Copy SQLite out after CLI completion and export it here; borrowed task services may remain alive."""
        if self.session.cleanup is None or not self.session.cleanup["cleanup_confirmed"]:
            raise RuntimeError("OpenCode transcript capture requires confirmed cleanup")
        database = f"{self.persistent_directory}/data/opencode"
        deadline = asyncio.get_running_loop().time() + timeout
        try:
            async with asyncio.timeout_at(deadline):
                # A stopped CLI may leave its write-ahead log behind; it holds the latest messages.
                listed = await self.session.sandbox.exec(
                    "for name in opencode.db opencode.db-wal opencode.db-shm; "
                    f'do [ ! -e {quote(database)}/"$name" ] || echo "$name"; done',
                    cwd=self.session.workdir,
                    timeout_s=timeout,
                )
            if listed.return_code != 0 or listed.error_type:
                raise RuntimeError(listed.stderr)
        except Exception as error:
            raise RuntimeError(f"OpenCode transcript capture failed: {error}") from error
        names = (listed.stdout or "").split()
        if "opencode.db" not in names:
            return  # No export: collection reports the missing result with the CLI logs.
        try:
            async with asyncio.timeout_at(deadline):
                for name in names:
                    await self.session.sandbox.download(f"{database}/{name}", directory / name)
        except Exception:
            LOG.warning("OpenCode session database unavailable", exc_info=True)
            return
        try:
            await asyncio.to_thread(
                snapshot,
                directory / "opencode.db",
                directory,
                harness_session_id=self.harness_session_id,
                previous_message_ids=self.message_ids,
            )
        except Exception as error:
            raise RuntimeError(f"OpenCode transcript capture failed: {error}") from error
