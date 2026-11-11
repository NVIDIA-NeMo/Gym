# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen Code runtime, input staging and artifact capture over Gym's shared sandbox lifecycle."""

import asyncio
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from shlex import quote

from nemo_gym.agents import (
    HARNESS_RUNNER_FILE,
    HARNESS_RUNNER_SCRIPT,
    AgentSeedRejectedError,
    AgentSessionState,
    SandboxCommand,
    SandboxSession,
    shell_assignments,
)
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import read_text, upload_text
from nemo_gym.trajectory import AgentObservationBundle


LOG = logging.getLogger(__name__)

QWEN_VERSION = "0.25.1"
NODE_VERSION = "22.23.2"
# sha256 of the official Node.js builds, from https://nodejs.org/dist/v22.23.2/SHASUMS256.txt, keyed by `uname -m`.
NODE_SHA256 = {
    "x86_64": "b294a556e639d64338823920e5866c21c02741742d2e1529ee1a225c1ec9252a",  # pragma: allowlist secret
    "aarch64": "013b59cfd2819703a6f4a14ab891fc46fc2a4e3f5bcd92de3fb4929b43e35b30",  # pragma: allowlist secret
}
NODE_ARCH = {"x86_64": "x64", "aarch64": "arm64"}
CLI = "lib/node_modules/@qwen-code/qwen-code/cli-entry.js"
SESSION_ROOT = "/tmp/nemo-gym-qwen-sessions"
RUNTIME_ROOT = "/tmp/nemo-gym-qwen-runtime"
INSTALLER = "install_qwen_runtime.sh"
# Bounds the stderr tail copied back to the agent server.
STDERR_TAIL_BYTES = 1024 * 1024


@dataclass(frozen=True)
class QwenArtifacts:
    """Qwen Code output copied before the sandbox connection is released, including partial runs."""

    stdout: str
    stderr: str
    sessions: bytes | None


@dataclass
class QwenSession(AgentSessionState):
    """Per-episode Qwen Code state; SandboxSession owns supervision, termination and release."""

    session: SandboxSession[QwenArtifacts]
    provider_name: str
    sandbox_id: str | None
    runtime: str = f"{RUNTIME_ROOT}-{QWEN_VERSION}-node-{NODE_VERSION}"
    task: asyncio.Task[NeMoGymResponse] | None = None
    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None
    observations: AgentObservationBundle | None = None

    @property
    def qwen_home(self) -> str:
        """QWEN_HOME: Qwen Code's settings, agents and session transcripts, outside the task repository and home."""
        return f"{self.session.session_dir}/qwen"

    @property
    def settings_path(self) -> str:
        """Generated settings, loaded at Qwen Code's system scope through QWEN_CODE_SYSTEM_SETTINGS_PATH."""
        return f"{self.session.session_dir}/system-settings.json"

    async def install(self, *, runtime_archive: str | None, timeout: float) -> None:
        """Create the session directory and install (or reuse) Node.js and the pinned Qwen Code package.

        The agent server downloads and checks the Node.js build once and uploads it, so the sandbox needs no download
        tools; npm then installs Qwen Code from the registry. A runtime_archive replaces both, for offline sandboxes.
        """
        # Shared host-side download cache for pinned sandbox assets.
        from responses_api_agents.miniswe_sandboxed_agent.bootstrap import cached_download

        sandbox, directory = self.session.sandbox, self.session.session_dir
        prepared = await sandbox.exec(
            f"test -d {quote(self.session.workdir)} && umask 077 && mkdir -p {quote(self.qwen_home)}",
            cwd="/",
            timeout_s=timeout,
        )
        if prepared.return_code != 0 or prepared.error_type:
            raise RuntimeError(f"Cannot prepare the Qwen Code session directory: {prepared.stderr or prepared.stdout}")
        await sandbox.upload(HARNESS_RUNNER_SCRIPT, f"{directory}/{HARNESS_RUNNER_FILE}")
        # The runtime path names both versions, and ready is written only after the version check.
        reuse = await sandbox.exec(f"test -f {quote(self.runtime)}/ready", cwd="/", timeout_s=60)
        if reuse.return_code == 0 and not reuse.error_type:
            return
        await sandbox.upload(Path(__file__).with_name(INSTALLER), f"{directory}/{INSTALLER}")
        # Each line is printed even when its check fails, so a musl image still yields three lines.
        probe = await sandbox.exec(
            "uname -m; getconf GNU_LIBC_VERSION || echo; command -v flock || echo", cwd="/", timeout_s=60
        )
        if probe.error_type:
            raise RuntimeError(f"Cannot probe the sandbox platform: {probe.error_type}")
        arch, libc, flock = [*(probe.stdout or "").splitlines(), "", "", ""][:3]
        arch_name = {"amd64": "x86_64", "arm64": "aarch64"}.get(arch, arch)
        if arch_name not in NODE_SHA256:
            raise AgentSeedRejectedError(f"No Node.js {NODE_VERSION} build for sandbox architecture {arch_name!r}")
        # The official Node.js 22 Linux builds need glibc 2.28 or newer.
        version = libc.removeprefix("glibc ").split(".")
        if not libc.startswith("glibc ") or tuple(map(int, version[:2])) < (2, 28):
            raise AgentSeedRejectedError(f"Qwen Code's Node.js runtime needs glibc 2.28 or newer, not {libc!r}")
        if not flock:
            raise AgentSeedRejectedError("Qwen Code's installer needs flock in the sandbox image")
        node, archive = "", ""
        if runtime_archive:
            # For sandboxes without internet access; the installer checks the version it reports.
            archive = f"{directory}/qwen-runtime.tar.gz"
            await sandbox.upload(Path(runtime_archive), archive)
        else:
            name = f"node-v{NODE_VERSION}-linux-{NODE_ARCH[arch_name]}.tar.gz"
            local = await cached_download(
                f"https://nodejs.org/dist/v{NODE_VERSION}/{name}", sha256=NODE_SHA256[arch_name], timeout=timeout
            )
            node = f"{directory}/{name}"
            await sandbox.upload(local, node)
        result = await sandbox.exec(
            f"bash {quote(f'{directory}/{INSTALLER}')} {quote(self.runtime)} {quote(QWEN_VERSION)} "
            f"{quote(NODE_SHA256[arch_name])} {quote(node)} {quote(archive)}",
            cwd="/",
            timeout_s=timeout,
        )
        if result.return_code != 0 or result.error_type:
            details = "\n".join(part for part in (result.stderr, result.stdout) if part)
            raise RuntimeError(
                f"Qwen Code {QWEN_VERSION} installation failed (exit {result.return_code}, "
                f"error_type={result.error_type}): {details[-4096:]}"
            )

    async def stage(
        self, *, command: list[str], env: dict[str, str], prompt: str, settings: str, subagent: str | None
    ) -> SandboxCommand:
        """Upload the task and Qwen Code's settings as files, never through shell interpolation."""
        directory = self.session.session_dir
        await upload_text(self.session.sandbox, path=f"{directory}/prompt.txt", text=prompt)
        await upload_text(self.session.sandbox, path=self.settings_path, text=settings)
        if subagent:
            await upload_text(self.session.sandbox, path=f"{self.qwen_home}/agents/general-purpose.md", text=subagent)
        params = {
            "cwd": self.session.workdir,
            "command": command,
            "env": env,
            # Task-image settings must not redirect Qwen Code's configuration, model calls or private Node.
            "unset_env": ["QWEN_*", "OPENAI_*", "NEMO_GYM_MODEL_API_KEY", "NODE_OPTIONS", "NODE_PATH"],
            "stdin": f"{directory}/prompt.txt",
            "stdout": f"{directory}/stdout.jsonl",
            "stderr": f"{directory}/stderr.log",
        }
        await upload_text(self.session.sandbox, path=f"{directory}/params.sh", text=shell_assignments(params))
        return SandboxCommand(argv=["bash", f"{directory}/{HARNESS_RUNNER_FILE}", f"{directory}/params.sh"])

    async def collect(self) -> QwenArtifacts:
        """Copy stdout, the stderr tail and the session transcripts after the supervisor confirms termination."""
        sandbox, directory = self.session.sandbox, self.session.session_dir
        try:
            stdout = await read_text(sandbox, path=f"{directory}/stdout.jsonl")
        except Exception:
            LOG.warning("Qwen Code stdout unavailable", exc_info=True)
            stdout = ""
        stderr = ""
        tail = await sandbox.exec(
            f"tail -c {STDERR_TAIL_BYTES} {quote(f'{directory}/stderr.log')} > {quote(f'{directory}/stderr.tail')}",
            cwd="/",
            timeout_s=30,
        )
        if tail.return_code == 0 and not tail.error_type:
            stderr = await read_text(sandbox, path=f"{directory}/stderr.tail")
        sessions = None
        archive = f"{directory}/sessions.tgz"
        packed = await sandbox.exec(
            f"tar -czf {quote(archive)} -C {quote(self.qwen_home)} projects", cwd="/", timeout_s=60
        )
        if packed.return_code == 0 and not packed.error_type:
            with tempfile.TemporaryDirectory(prefix="gym-qwen-capture-") as temporary:
                local = Path(temporary) / "sessions.tgz"
                await sandbox.download(archive, local)
                sessions = await asyncio.to_thread(local.read_bytes)
        else:
            LOG.warning("Qwen Code session transcripts unavailable: %s", packed.stderr or packed.stdout)
        return QwenArtifacts(stdout=stdout, stderr=stderr, sessions=sessions)

    async def close(self, *, timeout: float) -> None:
        """Capture and release the sandbox before cancelling the shared activation task."""
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
