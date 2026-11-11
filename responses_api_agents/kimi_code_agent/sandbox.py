# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi Code runtime, input staging and artifact capture over Gym's shared sandbox lifecycle."""

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

KIMI_VERSION = "2.1.1"
# sha256 of the release binaries, from https://code.kimi.com/kimi-code/binaries/2.1.1/manifest.json, keyed by `uname -m`.
# Kimi ships glibc builds only.
KIMI_BINARY_SHA256 = {
    "x86_64": "66f47536e40b02bb1d577cdd324768f73d39b8b9472e0ca140e73d8e225cd7de",  # pragma: allowlist secret
    "aarch64": "8bcc0a2267a0ecb1ff48cad284f7e364528a01598f38d696ec1baa6b069a88c8",  # pragma: allowlist secret
}
KIMI_TARGETS = {"x86_64": "linux-x64", "aarch64": "linux-arm64"}
KIMI_RELEASE_URL = f"https://code.kimi.com/kimi-code/binaries/{KIMI_VERSION}"
SESSION_ROOT = "/tmp/nemo-gym-kimi-sessions"
RUNTIME_ROOT = "/tmp/nemo-gym-kimi-runtime"
INSTALLER = "install_kimi_runtime.sh"
# install_kimi_runtime.sh exit status for an image without glibc.
NO_GLIBC_EXIT = 3
# Bounds the stderr tail copied back to the agent server.
STDERR_TAIL_BYTES = 1024 * 1024


def kimi_binary_name(arch: str) -> str:
    """Release binary name for a sandbox architecture reported by ``uname -m``."""
    return f"kimi-code-{KIMI_TARGETS[arch]}"


@dataclass(frozen=True)
class KimiArtifacts:
    """Kimi output copied before the sandbox connection is released, including partial runs."""

    stdout: str
    stderr: str
    sessions: bytes | None


@dataclass
class KimiSession(AgentSessionState):
    """Per-episode Kimi Code state; SandboxSession owns supervision, termination and release."""

    session: SandboxSession[KimiArtifacts]
    provider_name: str
    sandbox_id: str | None
    runtime: str = f"{RUNTIME_ROOT}-{KIMI_VERSION}"
    task: asyncio.Task[NeMoGymResponse] | None = None
    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None
    observations: AgentObservationBundle | None = None

    @property
    def home(self) -> str:
        """KIMI_CODE_HOME: Kimi's config, sessions and logs, outside the task repository and home."""
        return f"{self.session.session_dir}/home"

    async def install(self, *, timeout: float) -> None:
        """Create the session directory and install (or reuse) the pinned Kimi Code release.

        The agent server downloads and checks the release binary once, then uploads it, so the sandbox needs no
        download tools or internet access.
        """
        # Shared host-side download cache for pinned sandbox assets.
        from responses_api_agents.miniswe_sandboxed_agent.bootstrap import cached_download

        sandbox, directory = self.session.sandbox, self.session.session_dir
        prepared = await sandbox.exec(
            f"test -d {quote(self.session.workdir)} && umask 077 && mkdir -p {quote(self.home)}",
            cwd="/",
            timeout_s=timeout,
        )
        if prepared.return_code != 0 or prepared.error_type:
            raise RuntimeError(f"Cannot prepare the Kimi session directory: {prepared.stderr or prepared.stdout}")
        await sandbox.upload(HARNESS_RUNNER_SCRIPT, f"{directory}/{HARNESS_RUNNER_FILE}")
        await sandbox.upload(Path(__file__).with_name(INSTALLER), f"{directory}/{INSTALLER}")
        arch = await sandbox.exec("uname -m", cwd="/", timeout_s=60)
        if arch.return_code != 0 or arch.error_type:
            raise RuntimeError(f"Cannot probe the sandbox architecture: {arch.error_type or arch.stderr}")
        arch_name = {"amd64": "x86_64", "arm64": "aarch64"}.get(
            (arch.stdout or "").strip(), (arch.stdout or "").strip()
        )
        if arch_name not in KIMI_BINARY_SHA256:
            raise AgentSeedRejectedError(f"No Kimi Code {KIMI_VERSION} release for sandbox architecture {arch_name!r}")
        name = kimi_binary_name(arch_name)
        local = await cached_download(
            f"{KIMI_RELEASE_URL}/{name}", sha256=KIMI_BINARY_SHA256[arch_name], timeout=timeout
        )
        binary = f"{directory}/{name}"
        await sandbox.upload(local, binary)
        result = await sandbox.exec(
            f"bash {quote(f'{directory}/{INSTALLER}')} {quote(self.runtime)} {quote(KIMI_VERSION)} "
            f"{quote(KIMI_TARGETS[arch_name])} {quote(KIMI_BINARY_SHA256[arch_name])} {quote(binary)}",
            cwd="/",
            timeout_s=timeout,
        )
        if result.return_code == NO_GLIBC_EXIT and not result.error_type:
            raise AgentSeedRejectedError(f"Kimi Code {KIMI_VERSION} needs a glibc sandbox image")
        if result.return_code != 0 or result.error_type:
            details = "\n".join(part for part in (result.stderr, result.stdout) if part)
            raise RuntimeError(
                f"Kimi Code {KIMI_VERSION} installation failed (exit {result.return_code}, "
                f"error_type={result.error_type}): {details[-4096:]}"
            )

    async def stage(
        self, *, command: list[str], env: dict[str, str], config: str, agents_md: str | None
    ) -> SandboxCommand:
        """Upload Kimi's config (and global instructions) as files and return the runner command."""
        directory = self.session.session_dir
        await upload_text(self.session.sandbox, path=f"{self.home}/config.toml", text=config)
        if agents_md:
            await upload_text(self.session.sandbox, path=f"{self.home}/AGENTS.md", text=agents_md)
        params = {
            "cwd": self.session.workdir,
            "command": command,
            "env": env,
            # Task-image settings must not redirect Kimi's configuration, model calls or credentials.
            "unset_env": ["KIMI_*", "OPENAI_*", "MOONSHOT_*"],
            "stdin": "/dev/null",
            "stdout": f"{directory}/stdout.jsonl",
            "stderr": f"{directory}/stderr.log",
        }
        await upload_text(self.session.sandbox, path=f"{directory}/params.sh", text=shell_assignments(params))
        return SandboxCommand(argv=["bash", f"{directory}/{HARNESS_RUNNER_FILE}", f"{directory}/params.sh"])

    async def collect(self) -> KimiArtifacts:
        """Copy stdout, the stderr tail and the session store after the supervisor confirms termination."""
        if self.session.expired_before_launch:
            return KimiArtifacts("", "", None)  # A rollout stop arrived before launch, so Kimi never ran.
        sandbox, directory = self.session.sandbox, self.session.session_dir
        # A failed read raises, so a transient capture error is retried instead of graded as an empty answer.
        stdout = await read_text(sandbox, path=f"{directory}/stdout.jsonl")
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
        packed = await sandbox.exec(f"tar -czf {quote(archive)} -C {quote(self.home)} sessions", cwd="/", timeout_s=60)
        if packed.return_code == 0 and not packed.error_type:
            with tempfile.TemporaryDirectory(prefix="gym-kimi-capture-") as temporary:
                local = Path(temporary) / "sessions.tgz"
                await sandbox.download(archive, local)
                sessions = await asyncio.to_thread(local.read_bytes)
        else:
            LOG.warning("Kimi session store unavailable: %s", packed.stderr or packed.stdout)
        return KimiArtifacts(stdout=stdout, stderr=stderr, sessions=sessions)

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
