# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kilo-specific runtime and artifacts for the shared task-sandbox lifecycle."""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from shlex import quote
from tempfile import TemporaryDirectory
from time import perf_counter

from nemo_gym.agent_utils.sandbox_session import (
    HARNESS_COMMAND_NOT_STARTED,
    HARNESS_RUNNER_FAILED,
    HARNESS_RUNNER_FILE,
    HARNESS_RUNNER_SCRIPT,
    SandboxCommand,
    SandboxSession,
    shell_assignments,
)
from nemo_gym.base_responses_api_agent import AgentSessionState
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming, NeMoGymResponseUsage
from nemo_gym.rollout_observability import AgentObservationBundle
from nemo_gym.sandbox import read_text, upload_text
from responses_api_agents.kilocode_agent.observability import read_kilo_snapshot


LOG = logging.getLogger(__name__)
LOG_TAIL_BYTES = 4096
TASK_ENVIRONMENT_PLUGIN = "task_environment.mjs"
# Per-run files the runner and plugin write next to the runtime; a rearmed activation must not read the preflight's.
RUN_OUTPUTS = ("stdout.jsonl", "stderr.log", "runtime.json", "task-environment.ready", "required-mcp.ready")


# Runs in the sandbox before Kilo's private overrides: stale plugin receipts must not count,
# and the plugin's shell.env hook needs the task's own HOME and XDG directories.
_ACTIVATION_SETUP = r"""rm -f -- "$session/task-environment.ready" "$session/required-mcp.ready" || exit 1
mkdir -p -- "$session/home" "$session/data" "$session/config" "$session/cache" || exit 1
json_string() {
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
home=${HOME:-$(unset HOME; echo ~)}
printf '{"HOME": %s, "XDG_DATA_HOME": %s, "XDG_CONFIG_HOME": %s, "XDG_CACHE_HOME": %s, "XDG_STATE_HOME": %s}\n' \
    "$(json_string "$home")" "$(json_string "${XDG_DATA_HOME:-$home/.local/share}")" \
    "$(json_string "${XDG_CONFIG_HOME:-$home/.config}")" "$(json_string "${XDG_CACHE_HOME:-$home/.cache}")" \
    "$(json_string "${XDG_STATE_HOME:-$home/.local/state}")" >"$session/task-env.json"
"""

# Resolves symlinks in the longest existing prefix, since the session root may not exist yet.
_RESOLVE = r"""resolve() {
    path=$1 rest=
    while [ -n "$path" ] && [ ! -d "$path" ]; do rest=/${path##*/}$rest path=${path%/*}; done
    path=$(cd -- "${path:-/}" && pwd -P) && printf '%s%s\n' "${path%/}" "$rest"
}
"""


def _sandbox_prepare_command(workdir: str, directory: str) -> str:
    # Bootstrap with POSIX sh; the runtime installer and the harness runner require Bash.
    bootstrap = """set -eu
set --
command -v bash >/dev/null 2>&1 || set -- "$@" bash
if [ "$#" -gt 0 ]; then
    [ "$(id -u)" = 0 ] || { echo "Kilo sandbox sessions require $*: preinstall these tools or use a root image." >&2; exit 1; }
    if command -v apk >/dev/null 2>&1; then
        apk add --no-cache "$@"
    elif command -v apt-get >/dev/null 2>&1; then
        apt-get update
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$@"
    else
        echo "Kilo sandbox sessions require $*: preinstall these tools (automatic installation requires apt-get or apk)." >&2
        exit 1
    fi
fi
"""
    root = quote(str(PurePosixPath(directory).parent))
    validate_paths = (
        f"workdir=$(cd -- {quote(workdir)} && pwd -P) || {{ echo 'Kilo task workdir is missing' >&2; exit 1; }}\n"
        f"root=$(resolve {root})\n"
        # Each pattern also matches equal paths.
        'case "$root/" in "${workdir%/}"/*) overlap=1 ;; esac\n'
        'case "$workdir/" in "${root%/}"/*) overlap=1 ;; esac\n'
        '[ -z "${overlap-}" ] || { echo "Kilo session storage overlaps the task workdir" >&2; exit 1; }\n'
    )
    return bootstrap + _RESOLVE + validate_paths + f"mkdir -p -- {quote(directory)}"


@dataclass(frozen=True, kw_only=True)
class KiloArtifacts:
    """CLI events and diagnostics retained before releasing the sandbox."""

    stdout: str
    stderr: str
    exit_code: int | None
    observations: AgentObservationBundle | None = None
    usage: NeMoGymResponseUsage | None = None
    wall_time_s: float | None = None
    # Kilo logs model-connection errors only to its own log file, never to stdout or stderr.
    log_tail: str = ""
    # Runner identity in the sandbox, for checking that the harness did not run on the host.
    runner_hostname: str | None = None
    runner_pid: int | None = None
    # Whether the plugin that gives tool shells the task's HOME and XDG directories initialized.
    task_environment_ready: bool = False
    # False only when a required MCP server was not confirmed; both collectors set it from the receipt.
    required_mcp_ready: bool = True


@dataclass
class KiloSandboxSession(AgentSessionState):
    """Kilo input/output state; SandboxSession owns process cleanup and release."""

    session: SandboxSession[KiloArtifacts]
    runtime_dir: str | None = None
    required_mcp_servers: tuple[str, ...] = ()
    mcp_header_values: tuple[str, ...] = ()
    provider_name: str | None = None
    sandbox_id: str | None = None
    started_at: float | None = None
    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None
    task: asyncio.Task[NeMoGymResponse] | None = None

    @property
    def directory(self) -> str:
        """Keep the CLI runtime outside the fresh supervisor directory used after seed preflight."""
        return self.runtime_dir or self.session.session_dir

    def redact_diagnostics(self, text: str) -> str:
        """Keep grant credentials out of returned and logged CLI diagnostics."""
        for value in sorted(self._credential_forms(), key=len, reverse=True):
            text = text.replace(value, "[REDACTED]")
        return text

    def _credential_forms(self) -> set[str]:
        return {
            form
            for value in self.mcp_header_values
            if value
            for form in (value, json.dumps(value)[1:-1], json.dumps(value, ensure_ascii=False)[1:-1])
        }

    @property
    def kilo_argv(self) -> list[str]:
        """Invoke the CLI through the private Node, so the task's PATH stays untouched."""
        runtime = f"{self.directory}/runtime"
        return [f"{runtime}/node/bin/node", f"{runtime}/kilo/node_modules/@kilocode/cli/bin/kilo"]

    @property
    def task_environment_plugin(self) -> str:
        """The plugin, as a kilo.json entry, that restores the task's HOME and XDG directories for tools."""
        return PurePosixPath(self.directory, TASK_ENVIRONMENT_PLUGIN).as_uri()

    async def install_runtime(self, *, version: str, timeout: float) -> None:
        """Install pinned Node and Kilo into the session directory with the bundled installer."""
        directory = self.directory
        sandbox = self.session.sandbox
        prepared = await sandbox.exec(
            _sandbox_prepare_command(self.session.workdir, directory), cwd="/", timeout_s=timeout
        )
        if prepared.return_code != 0 or prepared.error_type:
            details = "\n".join(part for part in (prepared.stderr, prepared.stdout) if part)
            raise RuntimeError(f"Cannot prepare the Kilo session (exit {prepared.return_code}): {details[-16000:]}")
        installer = f"{directory}/install_kilo_runtime.sh"
        await sandbox.upload(Path(__file__).with_name("install_kilo_runtime.sh"), installer)
        installed = await sandbox.exec(
            f"bash {quote(installer)} {quote(directory + '/runtime')} {quote(version)}",
            cwd=self.session.workdir,
            timeout_s=timeout,
        )
        if installed.return_code != 0 or installed.error_type:
            # Background execution can leave only a generic message in stderr; keep both streams.
            details = "\n".join(part for part in (installed.stderr, installed.stdout) if part)
            raise RuntimeError(
                f"Kilo sandbox installation failed (exit {installed.return_code}, "
                f"error_type={installed.error_type}): {details[-16000:]}"
            )
        await sandbox.upload(HARNESS_RUNNER_SCRIPT, f"{directory}/{HARNESS_RUNNER_FILE}")
        await sandbox.upload(
            Path(__file__).with_name(TASK_ENVIRONMENT_PLUGIN), f"{directory}/{TASK_ENVIRONMENT_PLUGIN}"
        )

    async def stage_activation(
        self, *, command: list[str], config: str, mcp_preflight: bool = False
    ) -> SandboxCommand:
        """Stage the isolated provider config and CLI argv without altering task files."""
        directory = self.directory
        if self.session.session_dir != directory:
            stale = " ".join(quote(f"{directory}/{name}") for name in RUN_OUTPUTS)
            result = await self.session.sandbox.exec(
                f"mkdir -p -- {quote(self.session.session_dir)} && rm -f -- {stale}", cwd="/", timeout_s=30
            )
            if result.error_type or result.return_code != 0:
                raise RuntimeError("Cannot create Kilo activation directory")
        await upload_text(self.session.sandbox, path=f"{directory}/kilo.json", text=config)
        # The task-environment plugin reads which MCP servers it must confirm, and when.
        await upload_text(
            self.session.sandbox,
            path=f"{directory}/input.json",
            text=json.dumps({"required_mcp_servers": self.required_mcp_servers, "mcp_preflight": mcp_preflight}),
        )
        params = {
            "cwd": self.session.workdir,
            "command": command,
            "env": {
                "HOME": f"{directory}/home",
                "XDG_DATA_HOME": f"{directory}/data",
                "XDG_CONFIG_HOME": f"{directory}/config",
                "XDG_CACHE_HOME": f"{directory}/cache",
                "KILO_NO_DAEMON": "1",
                "KILO_DB": f"{directory}/kilo.db",
                "KILO_CONFIG": f"{directory}/kilo.json",
                "KILO_DISABLE_PROJECT_CONFIG": "1",
            },
            # Do not inherit provider credentials or configuration overrides from the image.
            "unset_env": ["KILO_*", "OPENCODE_*", "OPENAI_BASE_URL", "OPENAI_API_KEY"],
            "setup": f"session={quote(directory)}\n{_ACTIVATION_SETUP}",
            "stdin": "/dev/null",
            # File-backed stdout survives supervisor termination.
            "stdout": f"{directory}/stdout.jsonl",
            "stderr": f"{directory}/stderr.log",
        }
        await upload_text(self.session.sandbox, path=f"{directory}/params.sh", text=shell_assignments(params))
        return SandboxCommand(argv=["bash", f"{directory}/{HARNESS_RUNNER_FILE}", f"{directory}/params.sh"])

    async def prepare_mcp(self, *, config: str, timeout: float, close_timeout: float) -> None:
        """Check Kilo's real MCP clients without inference, under the normal process supervisor."""

        async def collect() -> KiloArtifacts:
            try:
                stderr = await read_text(self.session.sandbox, path=f"{self.directory}/stderr.log")
            except Exception:
                # A runner that failed before starting Kilo leaves its reason in the supervisor log.
                stderr = await self.session.read_output_log()
            cleanup = self.session.cleanup
            return KiloArtifacts(
                stdout="",
                stderr=self.redact_diagnostics(stderr),
                # The runner execs Kilo, so the receipt holds Kilo's exit status; a timed-out Kilo has none.
                exit_code=cleanup["return_code"] if cleanup is not None and not cleanup["timed_out"] else None,
                log_tail=await self._read_log_tail(),
                required_mcp_ready=await self._mcp_ready(),
            )

        artifacts = await self.session.execute(
            stage_activation=lambda: self.stage_activation(
                command=[*self.kilo_argv, "debug", "agent", "build"], config=config, mcp_preflight=True
            ),
            collect=collect,
            timeout=timeout,
            close_timeout=close_timeout,
        )
        if (
            artifacts.exit_code != 0
            or not artifacts.required_mcp_ready
            or not self.session.cleanup
            or not self.session.cleanup["cleanup_confirmed"]
            or self.session.cleanup["return_code"] not in (0, None)
            or self.session.cleanup["error"]
            or self.session.cleanup["timed_out"]
        ):
            # Seed cleanup then removes the runtime directory, so return Kilo's redacted reason now.
            diagnostics = artifacts.log_tail or artifacts.stderr[-LOG_TAIL_BYTES:]
            raise RuntimeError(
                "Kilo could not initialize required MCP servers: "
                + ", ".join(self.required_mcp_servers)
                + (f"; {diagnostics}" if diagnostics else "")
            )
        await self.session.begin_activation(session_dir=f"{self.directory}/activation", timeout=close_timeout)

    async def _mcp_ready(self) -> bool:
        if not self.required_mcp_servers:
            return True
        try:
            ready = json.loads(await read_text(self.session.sandbox, path=f"{self.directory}/required-mcp.ready"))
        except Exception:
            return False
        return ready == list(self.required_mcp_servers)

    async def collect_artifacts(self) -> KiloArtifacts:
        """Retain flushed CLI output, including partial events after a supervised stop."""
        wall_time_s = perf_counter() - self.started_at if self.started_at is not None else None
        directory = self.directory
        try:
            stdout = await read_text(self.session.sandbox, path=f"{directory}/stdout.jsonl")
        except Exception as error:
            logs = await self.session.read_output_log()
            raise RuntimeError(
                f"Kilo runner did not produce readable artifacts: {self.redact_diagnostics(logs)}"
            ) from error
        try:
            stderr = await read_text(self.session.sandbox, path=f"{directory}/stderr.log")
        except Exception:
            stderr = await self.session.read_output_log()
        stderr = self.redact_diagnostics(stderr)
        # The runner execs Kilo, so the receipt holds Kilo's exit status; a timed-out Kilo has none of its
        # own, nor does a Kilo that the runner never started.
        cleanup = self.session.cleanup
        exit_code = cleanup["return_code"] if cleanup is not None and not cleanup["timed_out"] else None
        if exit_code == HARNESS_RUNNER_FAILED or exit_code in HARNESS_COMMAND_NOT_STARTED:
            exit_code = None
        try:
            runtime = json.loads(await read_text(self.session.sandbox, path=f"{directory}/runtime.json"))
        except Exception:
            runtime = None
        try:
            ready = await read_text(self.session.sandbox, path=f"{directory}/task-environment.ready")
        except Exception:
            ready = None
        hostname = runtime.get("hostname") if isinstance(runtime, dict) else None
        pid = runtime.get("pid") if isinstance(runtime, dict) else None
        log_tail = ""
        if self.session.cleanup is not None and self.session.cleanup["timed_out"]:
            log_tail = await self._read_log_tail()
        observations, usage = None, None
        try:
            if self.session.cleanup is None or not self.session.cleanup["cleanup_confirmed"]:
                raise RuntimeError("Kilo database capture requires confirmed cleanup")
            # Kilo has stopped, so its database files are quiescent; the WAL holds recent commits.
            listed = await self.session.sandbox.exec(
                f"cd {quote(directory)} && [ -f kilo.db ] && ls -- kilo.db*",
                cwd="/",
                timeout_s=30,
            )
            if listed.error_type or listed.return_code != 0:
                raise RuntimeError(f"Kilo database unavailable: {listed.stderr}")
            with TemporaryDirectory(prefix="nemo-gym-kilo-observations-") as temporary:
                for name in {"kilo.db", "kilo.db-wal", "kilo.db-shm"} & set((listed.stdout or "").split()):
                    await self.session.sandbox.download(f"{directory}/{name}", Path(temporary) / name)
                observations, usage = await asyncio.to_thread(
                    read_kilo_snapshot, Path(temporary), fallback_invocation_id=self.request.agent_session_id
                )
        except Exception:
            # Partial stdout remains usable even if a killed worker never initialized its database.
            LOG.warning("Kilo persisted observations unavailable; retaining CLI output", exc_info=True)
        if stderr and (exit_code != 0 or (self.session.cleanup and self.session.cleanup["timed_out"])):
            LOG.warning(
                "Kilo diagnostics: episode=%s session=%s stderr=%s",
                self.request.episode_id.capture_key,
                self.request.agent_session_id,
                stderr[-4096:],
            )
        return KiloArtifacts(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            observations=observations,
            usage=usage,
            wall_time_s=wall_time_s,
            log_tail=log_tail,
            runner_hostname=hostname if isinstance(hostname, str) else None,
            runner_pid=pid if type(pid) is int else None,
            task_environment_ready=ready == "ready\n",
            required_mcp_ready=await self._mcp_ready(),
        )

    async def _read_log_tail(self) -> str:
        """Return the end of Kilo's newest log, which names errors such as an unreachable model."""
        logs = quote(f"{self.directory}/data/kilo/log")
        # Include enough preceding bytes to redact a credential crossing the final tail boundary.
        padding = max((len(value.encode()) for value in self._credential_forms()), default=0)
        try:
            result = await self.session.sandbox.exec(
                f'log=$(ls -t -- {logs}/*.log 2>/dev/null | head -n 1) && [ -n "$log" ] '
                f'&& tail -c {LOG_TAIL_BYTES + padding} -- "$log"',
                cwd="/",
                timeout_s=30,
            )
        except Exception:
            LOG.warning("Kilo log unavailable", exc_info=True)
            return ""
        if result.return_code != 0 or result.error_type:
            return ""
        raw = (result.stdout or "").encode()
        boundary = max(0, len(raw) - LOG_TAIL_BYTES)
        start = boundary
        # Select the byte tail before replacement shrinks it. Include complete credentials
        # crossing that boundary so redaction never receives only their trailing fragment.
        for value in self._credential_forms():
            for match in re.finditer(re.escape(value.encode()), raw):
                if match.start() < boundary < match.end():
                    start = min(start, match.start())
        redacted = self.redact_diagnostics(raw[start:].decode(errors="replace"))
        return redacted.encode()[-LOG_TAIL_BYTES:].decode(errors="replace")

    async def close(self, *, timeout: float) -> None:
        """Confirm remote cleanup and capture output before cancelling activation waiters."""
        await self.session.close(timeout=timeout)
        if self.task is not None:
            # A retried close must not cancel again: that would interrupt the activation's own cleanup.
            if not self.task.done() and not self.task.cancelling():
                self.task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(self.task), timeout=timeout)
            except asyncio.CancelledError:
                if not self.task.cancelled():
                    raise
            except Exception:
                if not self.task.done():
                    raise
