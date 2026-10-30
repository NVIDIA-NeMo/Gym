# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenCode artifacts and request state over the shared sandbox lifecycle."""

import asyncio
import json
import logging
import re
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from shlex import quote
from time import monotonic
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from nemo_gym.agents import (
    HARNESS_RUNNER_FAILED,
    HARNESS_RUNNER_FILE,
    HARNESS_RUNNER_SCRIPT,
    AgentSessionState,
    NothingCapturedError,
    RestoredLaunch,
    SandboxCommand,
    SandboxSession,
    exec_idempotent,
    shell_assignments,
)
from nemo_gym.config import ModelServerRef
from nemo_gym.core import MediaFile, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import SandboxExecResult, read_text, stage_media, upload_text
from nemo_gym.trajectory import AgentInvocation, AgentObservationBundle, ObservationGap, TrajectoryRecord
from responses_api_agents.opencode_agent.artifacts import parse_opencode_observations
from responses_api_agents.opencode_agent.observability import scope_opencode_activation
from responses_api_agents.opencode_agent.runtime import (
    REQUIRED_MCP_PLUGIN_FILE,
    REQUIRED_MCP_PLUGIN_SOURCE,
    TASK_ENVIRONMENT_PLUGIN_FILE,
    TASK_ENVIRONMENT_PLUGIN_SOURCE,
)
from responses_api_agents.opencode_agent.snapshot import NoRootSessionError, snapshot, stored_sessions


LOG = logging.getLogger(__name__)

# The model relay a launch runs on a sandbox whose snapshots keep processes (harness_runner.sh starts it): the
# script in the session root, and per launch directory its endpoint (``{"gen", "upstream"}``) and status files.
MODEL_RELAY_SCRIPT = HARNESS_RUNNER_SCRIPT.with_name("model_relay.py")
RELAY_ENDPOINT_FILE = "relay-endpoint.json"
RELAY_STATUS_FILE = "relay-status.json"


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


def cli_environment(*, workdir: str, env: Mapping[str, str], path_prefix: Sequence[str]) -> str:
    """POSIX sh that gives the commands after it the environment of OpenCode's ``import`` and ``export``.

    Like the run, they start in the task workdir with ``PWD`` naming it, since OpenCode binds a session to that
    directory, and find the configured tools first on ``PATH``. ``env`` holds the session's private directories and
    isolation settings. The run's configuration is dropped, so no plugin loads again. A failure here exits
    ``HARNESS_RUNNER_FAILED``, as the harness runner's own does.
    """
    lines = [f"cd {quote(workdir)} || exit {HARNESS_RUNNER_FAILED}", "unset OPENCODE_CONFIG_CONTENT"]
    if path_prefix:
        lines.append(f"PATH={quote(':'.join(path_prefix))}:$PATH")
    lines.append("export " + " ".join(f"{key}={quote(value)}" for key, value in {**env, "PWD": workdir}.items()))
    return "".join(f"{line}\n" for line in lines)


# OpenCode session IDs become file names in the owner's export directory.
_SESSION_ID = re.compile(r"[A-Za-z0-9_-]+")
# Replaces "$2" with a copy of "$1" made beside it, so the owner never reads a partial file.
_PUBLISH = """partial="${2%/*}/.${2##*/}.$$.tmp"
cp -- "$1" "$partial" && mv -f -- "$partial" "$2" || { rm -f -- "$partial"; exit 1; }
"""
# The event log step's exit status when the run left no event log.
_NO_EVENT_LOG = 2
_PUBLISH_EVENT_LOG = f'[ -f "$1" ] || exit {_NO_EVENT_LOG}\n' + _PUBLISH
# Writes the ``opencode export`` of session "$3", made by the CLI "$1", to "$2/$3.json", or prints why not and exits 1.
# The CLI writes "$4.json" and "$4.stderr" in the session directory, so an export stopped at the deadline leaves nothing
# in the owner's. OpenCode (Bun) can exit before a pipe drains, which cut exports to their first 8 KB; it writes a
# regular file synchronously.
_EXPORT_SESSION = (
    """rm -f -- "$2/$3.json" || exit 1
"$1" export "$3" </dev/null >"$4.json" 2>"$4.stderr"
status=$?
if [ "$status" -ne 0 ]; then
    printf 'exit %s: %s' "$status" "$(tail -c 1000 -- "$4.stderr")"
    exit 1
fi
if [ "$(head -c 1 -- "$4.json")" != "{" ]; then
    printf 'export (%s bytes) is not a JSON object' "$(wc -c <"$4.json" | tr -d ' ')"
    exit 1
fi
set -- "$4.json" "$2/$3.json"
"""
    + _PUBLISH
    + 'rm -f -- "$1"\n'
)


@dataclass
class OpenCodeSandboxSession(AgentSessionState):
    """Keep OpenCode HOME/history under ``conversation_directory`` across launches.

    ``session.session_dir`` holds only the current supervisor and input/output
    files. ``persistent_directory`` selects that original session root for legacy
    one-shot calls and the retained conversation root for continuation calls.
    SandboxSession owns finalization and removal of the entire directory tree.

    A ``checkpointed`` session pauses its harness in place at a checkpoint, and relaunches it in
    ``launch-K`` directories under the same root when a pause failed or a crash interrupted it,
    resuming the conversation kept in the root. A session a restore installed has no sandbox
    connection (``session`` is ``None``) until its replacement attempt's seed re-binds it;
    ``restored`` keeps what the checkpoint exported until then.

    On a sandbox whose snapshots keep processes (``keeps_processes``), the pause holds the harness's model calls
    instead of stopping it, and OpenCode reaches the model through a relay in the sandbox: a restore brings the
    launch back running, ``adoption`` names the held pause it is taken over from, and ``relay_gen`` is the
    generation of the relay's endpoint, raised when a restore points it at the replacement attempt.
    """

    session: SandboxSession[str] | None
    runtime: str
    sandbox_id: str | None = None
    model_ref: ModelServerRef | None = None
    task: asyncio.Task[NeMoGymResponse] | None = None
    runtime_info: HarnessProcessInfo | None = None
    # All retained activations use this same sandbox clock. A new session gets a
    # separate domain; neither model-server clocks nor imported DBs are equated.
    agent_instance: str = field(default_factory=lambda: f"opencode-agent:{uuid4().hex}", init=False)
    tool_clock_domain: str = field(default_factory=lambda: f"opencode-wall:{uuid4().hex}", init=False)
    observations: AgentObservationBundle | None = None
    stderr: str = ""
    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None
    # The latest validated continuation request with its prompt, system text and decoded media.
    # Activation takes it, so it holds at most one request's media.
    validated_input: tuple[NeMoGymResponseCreateParamsNonStreaming, str, str, list[MediaFile]] | None = field(
        default=None, repr=False
    )
    conversation_directory: str | None = None
    # Whether an activation of this session has declared image input; later activations keep declaring it.
    declares_image_input: bool = False
    harness_session_id: str | None = None
    message_ids: list[str] = field(default_factory=list)
    system_instructions: str | None = None
    activation_log: str = ""
    activation_observations: AgentObservationBundle | None = None
    trajectory: TrajectoryRecord | None = None
    activation_trajectory: TrajectoryRecord | None = None
    activation_id: int | None = None
    close_requested_at: float | None = None
    export_dir: str | None = None
    # POSIX sh that gives OpenCode's import and export their environment (see ``cli_environment``).
    opencode_environment: str = ""
    _export_gaps: list[ObservationGap] = field(default_factory=list, init=False, repr=False)
    checkpointed: bool = False
    # Launches of the harness so far, and what each ended with, for the checkpoint and the reply's evidence.
    launch_count: int = 0
    launch_records: list[dict[str, JsonValue]] = field(default_factory=list)
    # The activation this session completed and its reply, given again to a caller that sends it again.
    completed_activation: int = 0
    completed_reply: dict[str, JsonValue] | None = None
    restored: dict[str, JsonValue] | None = None
    keeps_processes: bool = False
    relay_gen: int = 0
    adoption: RestoredLaunch | None = None
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
        if self.conversation_directory:
            return self.conversation_directory
        # A checkpointed session's launches each have a directory under the root, which keeps the conversation.
        return self.session.session_root if self.checkpointed else self.session.session_dir

    def require_session(self) -> SandboxSession[str]:
        """The sandbox session; a restored session gets one when its replacement attempt's seed re-binds it."""
        if self.session is None:
            raise RuntimeError(
                "OpenCode session was restored from a checkpoint and has not been re-bound to a sandbox"
            )
        return self.session

    async def prepare_launch(self, launch: int, *, timeout: float) -> None:
        """Rearm the supervisor in ``launch-K`` under the session root, keeping the conversation in the root."""
        assert self.checkpointed and launch > 1
        self.conversation_directory = self.session.session_root
        directory = f"{self.conversation_directory}/launch-{launch}"
        await self.session.begin_activation(session_dir=directory, timeout=timeout)
        result = await self.session.sandbox.exec(f"mkdir -p -- {quote(directory)}", cwd="/", timeout_s=timeout)
        if result.error_type == "timeout":
            raise TimeoutError("OpenCode launch directory preparation timed out")
        if result.return_code != 0 or result.error_type:
            raise RuntimeError(f"Cannot prepare OpenCode launch directory: {result.stderr}")

    async def inspect_conversation(self, *, timeout: float) -> dict[str, JsonValue] | None:
        """Export the conversation OpenCode saved in this session's root, or ``None`` when there is none."""
        with tempfile.TemporaryDirectory(prefix="opencode-resume-") as directory:
            if not await self._download_database(Path(directory), timeout):
                return None
            try:
                await asyncio.to_thread(
                    snapshot,
                    Path(directory) / "opencode.db",
                    Path(directory),
                    harness_session_id=None,
                    previous_message_ids=(),
                )
                return json.loads((Path(directory) / "export.json").read_text())
            except NoRootSessionError:
                LOG.info("OpenCode had saved no conversation yet in %s", self.persistent_directory)
                return None
            except Exception:
                LOG.warning("The saved OpenCode conversation cannot be read", exc_info=True)
                return None

    async def discard_conversation(self, *, timeout: float) -> None:
        """Set the saved conversation aside, so the next launch starts afresh in an empty data directory."""
        data = f"{self.persistent_directory}/data"
        # Moving it only if present, so running it again is safe.
        result = await exec_idempotent(
            self.session.sandbox,
            f"if [ -e {quote(data)} ]; then mv -- {quote(data)} {quote(data)}.discarded-{self.launch_count}; fi",
            what="Setting the saved OpenCode conversation aside",
            cwd="/",
            timeout_s=timeout,
        )
        if result.return_code != 0 or result.error_type:
            raise RuntimeError(f"Cannot set the saved OpenCode conversation aside: {result.stderr}")

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
            schema_version="2.0",
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
        if self.close_requested_at is None:
            self.close_requested_at = time.time()
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
        media: Sequence[MediaFile] = (),
        timeout: float,
        close_timeout: float,
        deadline: float | None = None,
        pause_allowance_s: float = 0.0,
    ) -> str:
        """Run through the common lifecycle; the adapter validates terminal events."""
        artifacts = await self.session.execute(
            stage_activation=lambda: self.stage_activation(payload, media=media),
            collect=lambda: self.collect_artifacts(timeout=close_timeout),
            timeout=timeout,
            close_timeout=close_timeout,
            deadline=deadline,
            pause_allowance_s=pause_allowance_s,
        )
        if self.session.closing:
            raise asyncio.CancelledError
        return artifacts

    async def adopt(
        self, *, pause_seq: int, timeout: float, close_timeout: float, pause_allowance_s: float = 0.0
    ) -> str:
        """Take over the launch a restore brought back holding pause ``pause_seq``, as ``execute`` finishes one."""
        artifacts = await self.session.adopt(
            pause_seq=pause_seq,
            collect=lambda: self.collect_artifacts(timeout=close_timeout),
            timeout=timeout,
            close_timeout=close_timeout,
            pause_allowance_s=pause_allowance_s,
        )
        if self.session.closing:
            raise asyncio.CancelledError
        return artifacts

    async def point_relay(self, upstream: str) -> None:
        """Point the current launch's model relay at ``upstream`` as its next endpoint generation."""
        self.relay_gen += 1
        await upload_text(
            self.session.sandbox,
            path=f"{self.session.session_dir}/{RELAY_ENDPOINT_FILE}",
            text=json.dumps({"gen": self.relay_gen, "upstream": upstream}),
        )

    async def stage_activation(
        self, payload: dict[str, JsonValue], *, media: Sequence[MediaFile] = ()
    ) -> SandboxCommand:
        """Stage the invocation and return the OpenCode worker command.

        Staging runs inside the session's execute, so close fences it. A continuation activation's deadline also
        bounds it.
        """
        if payload.get("instructions"):
            await upload_text(
                self.session.sandbox, path=f"{self.session.session_dir}/instructions.md", text=payload["instructions"]
            )
        if self.is_continuation or self.session.session_dir != self.persistent_directory:
            await self.session.sandbox.upload(
                TASK_ENVIRONMENT_PLUGIN_SOURCE, f"{self.session.session_dir}/{TASK_ENVIRONMENT_PLUGIN_FILE}"
            )
        directory = self.session.session_dir
        if payload.get("mcp_grants") is not None:
            await self.session.sandbox.upload(REQUIRED_MCP_PLUGIN_SOURCE, f"{directory}/{REQUIRED_MCP_PLUGIN_FILE}")
            await upload_text(
                self.session.sandbox, path=f"{directory}/mcp-grants.json", text=json.dumps(payload["mcp_grants"])
            )
            # null until the plugin checks the grants before the first model call of this activation.
            await upload_text(self.session.sandbox, path=f"{directory}/mcp-status.json", text=json.dumps(None))
        await upload_text(self.session.sandbox, path=f"{directory}/prompt.txt", text=payload["prompt"])
        command = list(payload["command"])
        # OpenCode shows each attached file's path to the model, so stage_media names files by content:
        # rollouts of one task then get the same prompt.
        if media:
            for path in await stage_media(self.session.sandbox, media):
                command.extend(["--file", path])
        if payload.get("resume") is not None:
            # The request body resume_opencode.sh re-posts: the saved last user message, with no new parts.
            await upload_text(
                self.session.sandbox, path=f"{directory}/resume.json", text=json.dumps(payload["resume"])
            )
        env = payload["env"]
        params = {
            "cwd": payload["cwd"],
            "command": command,
            "env": env,
            "stdin": f"{directory}/prompt.txt",
            "stdout": f"{directory}/stdout.jsonl",
            "stderr": f"{directory}/stderr.log",
            "setup": runner_setup(directory, [env[key] for key in _TASK_DIRECTORY_KEYS]),
        }
        if payload.get("path_prefix"):
            # Preinstalled runtime tools come first; the task's own PATH still follows.
            params["setup"] += f"PATH={quote(':'.join(payload['path_prefix']))}:$PATH\n"
        relay = payload.get("relay")
        if relay is not None:
            # The runner starts the relay; without one (no Python 3.6+ in the image) OpenCode calls the model
            # directly, under the fallback environment.
            await upload_text(
                self.session.sandbox,
                path=f"{directory}/{RELAY_ENDPOINT_FILE}",
                text=json.dumps({"gen": self.relay_gen, "upstream": relay["upstream"]}),
            )
            params |= {
                "MODEL_RELAY_PORT": str(relay["port"]),
                "MODEL_RELAY_ENDPOINT": f"{directory}/{RELAY_ENDPOINT_FILE}",
                "MODEL_RELAY_STATUS": f"{directory}/{RELAY_STATUS_FILE}",
                "MODEL_RELAY_SCRIPT": f"{self.persistent_directory}/{MODEL_RELAY_SCRIPT.name}",
                "MODEL_RELAY_FALLBACK_ENV": relay["fallback_env"],
            }
        await upload_text(self.session.sandbox, path=f"{directory}/params.sh", text=shell_assignments(params))
        return SandboxCommand(
            argv=["bash", f"{self.persistent_directory}/{HARNESS_RUNNER_FILE}", f"{directory}/params.sh"]
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
        # Leave a tenth of the budget so the capture result returns before the step's own timeout.
        export_deadline = monotonic() + timeout * 0.9
        with tempfile.TemporaryDirectory(prefix="opencode-snapshot-") as directory:
            try:
                return await self._collect_artifacts(Path(directory), timeout=timeout)
            finally:
                if self.export_dir is not None:
                    await self.write_exports(Path(directory), deadline=export_deadline)

    async def write_exports(self, directory: Path, *, deadline: float) -> None:
        """Write every OpenCode session's ``opencode export`` and this run's event log to ``export_dir``.

        Runs once the CLI has stopped, before the session directory is removed. ``sessions.json`` lists the stored
        session tree first, so the owner can tell a failed export from an absent session; the tree comes from the
        session store that the capture copied into ``directory``. Each session is exported by a command of its own,
        which the provider stops at ``deadline``, with the session's private directories but without the run's
        configuration. Failures, including an exhausted budget, are recorded as one ``export_dir_write_failed`` gap
        that names each failed session.
        """
        assert self.export_dir is not None
        started = monotonic()
        outcomes: dict[str, str] = {}
        event_log = "not attempted"
        exhausted = False
        error = None
        try:
            if self.session.cleanup is None or not self.session.cleanup["cleanup_confirmed"]:
                raise RuntimeError("OpenCode session export requires confirmed cleanup")
            # Never leave an earlier activation's inventory in place of this one.
            inventory = f"{self.export_dir}/sessions.json"
            await self._export_step(f"mkdir -p -- {quote(self.export_dir)} && rm -f -- {quote(inventory)}", deadline)
            sessions = await self._stored_sessions(directory, deadline)
            staged = f"{self.session.session_dir}/sessions.json"
            async with asyncio.timeout(deadline - monotonic()):
                await upload_text(self.session.sandbox, path=staged, text=json.dumps(sessions))
            exportable = [
                session["id"]
                for session in sessions
                if isinstance(session["id"], str) and _SESSION_ID.fullmatch(session["id"])
            ]
            # Each session's earlier export goes first, so a session that is not exported again is never left stale.
            stale = " ".join(quote(f"{self.export_dir}/{session_id}.json") for session_id in exportable)
            await self._export_step(
                f"rm -f -- {stale}\nset -- {quote(staged)} {quote(inventory)}\n{_PUBLISH}", deadline
            )
            for session in sessions:
                session_id = session["id"]
                if session_id not in exportable:
                    outcomes[str(session_id)] = "unsafe session ID"
                    continue
                command = self.opencode_environment + (
                    f"set -- {quote(f'{self.runtime}/opencode')} {quote(self.export_dir)} {session_id} "
                    f"{quote(f'{self.session.session_dir}/export-{session_id}')}\n{_EXPORT_SESSION}"
                )
                result = await self._exec_until(command, deadline)
                if result is None:
                    exhausted = True
                    outcomes[session_id] = "export deadline exceeded"
                elif result.return_code == 0 and result.error_type is None:
                    outcomes[session_id] = "ok"
                else:
                    outcomes[session_id] = (result.stdout or "").strip() or (
                        f"exit {result.return_code}, error {result.error_type}: {(result.stderr or '')[-1000:]}"
                    )
            roots = [session["id"] for session in sessions if session["parent_id"] is None]
            if len(roots) == 1 and roots[0] in exportable:
                suffix = "" if self.activation_id is None else f"activation-{self.activation_id}."
                events = f"{self.export_dir}/{roots[0]}.{suffix}events.jsonl"
                result = await self._exec_until(
                    f"set -- {quote(f'{self.session.session_dir}/stdout.jsonl')} {quote(events)}\n"
                    + _PUBLISH_EVENT_LOG,
                    deadline,
                )
                if result is None:
                    exhausted = True
                    event_log = "export deadline exceeded"
                elif result.return_code == _NO_EVENT_LOG and result.error_type is None:
                    event_log = "absent"
                elif result.return_code == 0 and result.error_type is None:
                    event_log = "ok"
                else:
                    event_log = (
                        f"exit {result.return_code}, error {result.error_type}: {(result.stderr or '')[-1000:]}"
                    )
            else:
                event_log = f"no single root session to belong to: {roots!r}"
        except Exception as caught:
            exhausted = exhausted or isinstance(caught, TimeoutError)
            error = f"{type(caught).__name__}: {caught}"
        failures = [f"{session_id}: {outcome}" for session_id, outcome in outcomes.items() if outcome != "ok"]
        if event_log not in ("ok", "absent"):
            failures.append(f"event log: {event_log}")
        if error is not None:
            failures.append(error)
        if failures:
            LOG.warning("OpenCode session exports for %s are incomplete: %s", self.export_dir, failures)
            exported = sum(outcome == "ok" for outcome in outcomes.values())
            summary = (
                f"{exported} of {len(outcomes)} sessions exported in {monotonic() - started:.1f} of "
                f"{deadline - started:.1f} s{', deadline exhausted' if exhausted else ''}"
            )
            self._export_gaps.append(
                ObservationGap(code="export_dir_write_failed", detail="; ".join([summary, *failures])[:2500])
            )

    async def _exec_until(self, command: str, deadline: float) -> SandboxExecResult | None:
        """Run ``command`` from the task workdir; None if ``deadline`` passes first."""
        remaining = deadline - monotonic()
        if remaining <= 0:
            return None
        try:
            async with asyncio.timeout(remaining):
                result = await self.session.sandbox.exec(command, cwd=self.session.workdir, timeout_s=remaining)
        except TimeoutError:
            return None
        return None if result.error_type == "timeout" else result

    async def _export_step(self, command: str, deadline: float) -> None:
        """Run a step that every export needs, raising if it fails."""
        result = await self._exec_until(command, deadline)
        if result is None:
            raise TimeoutError("no time left in the artifact collection budget")
        if result.return_code != 0 or result.error_type:
            raise RuntimeError(
                f"exit {result.return_code}, error {result.error_type}: {(result.stderr or result.stdout or '')[-1000:]}"
            )

    async def _stored_sessions(self, directory: Path, deadline: float) -> list[dict[str, str | None]]:
        """The stored session tree, from the store the capture copied out or, if it did not, a fresh copy."""
        database = directory / "opencode.db"
        if not database.is_file():
            remaining = deadline - monotonic()
            async with asyncio.timeout(remaining):
                names = await self.list_store(timeout=remaining)
                if "opencode.db" not in names:
                    raise RuntimeError("OpenCode left no session store")
                await self.download_store(names, directory)
        return await asyncio.to_thread(stored_sessions, database)

    async def _collect_artifacts(self, directory: Path, *, timeout: float) -> str:
        capture_started_at = time.time()
        previous_session_id = self.harness_session_id
        try:
            self.stderr = (await self.read_text("stderr.log"))[-16000:]
        except Exception:
            LOG.warning("OpenCode stderr log unavailable", exc_info=True)
        try:
            await self.snapshot(directory, timeout)
        except NoRootSessionError as error:
            if self._launch_failed():
                raise await self._nothing_captured(str(error)) from None
            error = RuntimeError(f"OpenCode transcript capture failed: {error}")
            if self.stderr:
                raise RuntimeError(format_error_message(str(error), stderr=self.stderr)) from error
            raise error
        except Exception as error:
            if self.stderr:
                raise RuntimeError(format_error_message(str(error), stderr=self.stderr)) from error
            raise
        if self._launch_failed() and not (directory / "export.json").exists():
            raise await self._nothing_captured("OpenCode sandbox runner returned no valid result: no session database")
        # Messages that predate this activation (earlier activations, an imported prefix) are history, not output.
        scoped = self.is_continuation or bool(self.message_ids)
        if self.is_continuation or self.checkpointed:
            export_text = await self._read_export(directory)
            export = json.loads(export_text)
            harness_session_id = export.get("session_id")
            if not isinstance(harness_session_id, str) or not harness_session_id:
                raise RuntimeError("OpenCode did not persist a resumable session ID")
            if self.harness_session_id is not None and harness_session_id != self.harness_session_id:
                raise RuntimeError("OpenCode resumed a different OpenCode session")
            self.harness_session_id = harness_session_id
            if self.is_continuation:
                # A one-activation session's reply covers the whole conversation, across every launch.
                self.message_ids = export["message_ids"]
        elif scoped:
            export = json.loads(await self._read_export(directory))
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
                observation_schema_version="2.0",
                model_ref=self.model_ref,
                clock_domain=self.tool_clock_domain,
                agent_instance=self.agent_instance,
                message_ids=set(export.get("activation_message_ids", [])) if scoped else None,
            )
            lifecycle = json.dumps(
                {
                    "capture_started_at": capture_started_at,
                    "parse_completed_at": time.time(),
                    "close_requested_at": self.close_requested_at,
                    "cleanup": {
                        key: self.session.cleanup.get(key) for key in ("cleanup_confirmed", "timed_out", "return_code")
                    }
                    if self.session.cleanup is not None
                    else None,
                },
                separators=(",", ":"),
            )
            for gap in trajectory.gaps:
                if gap.code == "turns_unavailable":
                    gap.detail = (gap.detail or "") + "; capture_lifecycle=" + lifecycle
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
                self.trajectory = trajectory
        except Exception:
            LOG.exception("Failed to parse OpenCode observations")
            observations = AgentObservationBundle(
                schema_version="2.0",
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
                and (self.session.cleanup["timed_out"] or self.session.cleanup.get("stopped", False))
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

    @property
    def store_directory(self) -> str:
        """Where OpenCode keeps its session database in private storage."""
        return f"{self.persistent_directory}/data/opencode"

    async def list_store(self, *, timeout: float) -> list[str]:
        """The session database files in private storage.

        A stopped CLI may leave its write-ahead log behind; it holds the latest messages.
        """
        listed = await self.session.sandbox.exec(
            "for name in opencode.db opencode.db-wal opencode.db-shm; "
            f'do [ ! -e {quote(self.store_directory)}/"$name" ] || echo "$name"; done',
            cwd=self.session.workdir,
            timeout_s=timeout,
        )
        if listed.return_code != 0 or listed.error_type:
            raise RuntimeError(listed.stderr)
        return (listed.stdout or "").split()

    async def download_store(self, names: list[str], directory: Path) -> None:
        """Copy the session database files ``names`` into ``directory``."""
        for name in names:
            await self.session.sandbox.download(f"{self.store_directory}/{name}", directory / name)

    async def snapshot(self, directory: Path, timeout: float) -> None:
        """Copy SQLite out after CLI completion and export it here; borrowed task services may remain alive."""
        if self.session.cleanup is None or not self.session.cleanup["cleanup_confirmed"]:
            raise RuntimeError("OpenCode transcript capture requires confirmed cleanup")
        if not await self._download_database(directory, timeout):
            return  # No export: collection reports the missing result with the CLI logs.
        try:
            await asyncio.to_thread(
                snapshot,
                directory / "opencode.db",
                directory,
                harness_session_id=self.harness_session_id,
                previous_message_ids=self.message_ids,
            )
        except NoRootSessionError:
            raise
        except Exception as error:
            raise RuntimeError(f"OpenCode transcript capture failed: {error}") from error

    def _launch_failed(self) -> bool:
        """Whether the launch ended without OpenCode finishing: a failure, a stop, or no receipt at all."""
        cleanup = self.session.cleanup
        return cleanup is None or cleanup["return_code"] != 0 or bool(cleanup.get("stopped"))

    async def _nothing_captured(self, detail: str) -> NothingCapturedError:
        """The capture error of a launch that ended before OpenCode saved anything, with the runner's log for why."""
        cleanup = self.session.cleanup or {}
        message = (
            f"OpenCode left nothing to capture after its launch ended without finishing "
            f"(exit {cleanup.get('return_code')}, stopped {bool(cleanup.get('stopped'))}): {detail}"
        )
        return NothingCapturedError(
            format_error_message(message, output=await self.session.read_output_log(), stderr=self.stderr)
        )

    async def _download_database(self, directory: Path, timeout: float) -> bool:
        """Copy OpenCode's database, with its write-ahead log, into ``directory``; ``False`` when there is none."""
        deadline = asyncio.get_running_loop().time() + timeout
        try:
            async with asyncio.timeout_at(deadline):
                names = await self.list_store(timeout=timeout)
        except Exception as error:
            raise RuntimeError(f"OpenCode transcript capture failed: {error}") from error
        if "opencode.db" not in names:
            return False
        try:
            async with asyncio.timeout_at(deadline):
                await self.download_store(names, directory)
        except Exception:
            LOG.warning("OpenCode session database unavailable", exc_info=True)
            return False
        return True
