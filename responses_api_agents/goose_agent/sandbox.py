# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Goose runtime, input staging and artifact capture over Gym's shared sandbox lifecycle."""

import asyncio
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from shlex import quote

import certifi

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

GOOSE_VERSION = "1.54.0"
# sha256 of the official static (musl) release archives, keyed by `uname -m`.
GOOSE_ARCHIVE_SHA256 = {
    "x86_64": "777e49b6ae128fae06600f94bf51ff05c81572fce63c1581e9ea94bc49d64e42",  # pragma: allowlist secret
    "aarch64": "baeb3e1b188fce75b444af35b5955f255e68fde87f60a5ba01f032227618a4ed",  # pragma: allowlist secret
}
GOOSE_RELEASE_URL = f"https://github.com/aaif-goose/goose/releases/download/v{GOOSE_VERSION}"
SESSION_ROOT = "/tmp/nemo-gym-goose-sessions"
RUNTIME_ROOT = "/tmp/nemo-gym-goose-runtime"
INSTALLER = "install_goose_runtime.sh"
# Bounds the stderr tail copied back to the agent server.
STDERR_TAIL_BYTES = 1024 * 1024
# CA bundles the common distributions install. An image with one of these, or with SSL_CERT_FILE or
# SSL_CERT_DIR already set, keeps its own trust store.
SYSTEM_CA_BUNDLES = (
    "/etc/ssl/certs/ca-certificates.crt",  # Debian, Ubuntu, Alpine, Arch
    "/etc/pki/tls/certs/ca-bundle.crt",  # RHEL, Fedora
    "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem",  # RHEL 9+
    "/etc/ssl/ca-bundle.pem",  # openSUSE
    "/etc/ssl/cert.pem",  # Alpine, LibreSSL
)


def goose_archive_name(arch: str) -> str:
    """Release archive name for a sandbox architecture reported by ``uname -m``."""
    return f"goose-{arch}-unknown-linux-musl.tar.gz"


@dataclass(frozen=True)
class GooseArtifacts:
    """Goose output copied before the sandbox connection is released, including partial runs."""

    stdout: str
    stderr: str
    sessions: bytes | None


@dataclass
class GooseSession(AgentSessionState):
    """Per-episode Goose state; SandboxSession owns supervision, termination and release."""

    session: SandboxSession[GooseArtifacts]
    provider_name: str
    sandbox_id: str | None
    runtime: str = f"{RUNTIME_ROOT}-{GOOSE_VERSION}"
    task: asyncio.Task[NeMoGymResponse] | None = None
    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None
    observations: AgentObservationBundle | None = None
    # Certifi's bundle in the session directory, set only when the task image has no trust store.
    ca_bundle: str | None = None

    @property
    def root(self) -> str:
        """GOOSE_PATH_ROOT: Goose's config, sessions and state, outside the task repository and home."""
        return f"{self.session.session_dir}/root"

    async def install(self, *, timeout: float) -> None:
        """Create the session directory and install (or reuse) the pinned Goose release.

        The agent server downloads and checks the release archive once, then uploads it, so the sandbox needs no
        download tools or internet access.
        """
        # Shared host-side download cache for pinned sandbox assets.
        from responses_api_agents.miniswe_sandboxed_agent.bootstrap import cached_download

        sandbox, directory = self.session.sandbox, self.session.session_dir
        prepared = await sandbox.exec(
            f"test -d {quote(self.session.workdir)} && umask 077 && mkdir -p {quote(self.root + '/config')}",
            cwd="/",
            timeout_s=timeout,
        )
        if prepared.return_code != 0 or prepared.error_type:
            raise RuntimeError(f"Cannot prepare the Goose session directory: {prepared.stderr or prepared.stdout}")
        await sandbox.upload(HARNESS_RUNNER_SCRIPT, f"{directory}/{HARNESS_RUNNER_FILE}")
        await sandbox.upload(Path(__file__).with_name(INSTALLER), f"{directory}/{INSTALLER}")
        # The static release needs CA roots to construct its HTTP client, even for an HTTP model endpoint.
        # Supply Certifi's only when the image has none: SSL_CERT_FILE also reaches the task's tool shells,
        # and there it would replace a trust store the task relies on (or one the agent extends).
        found = " || ".join(
            ['test -n "${SSL_CERT_FILE:-}"', 'test -n "${SSL_CERT_DIR:-}"']
            + [f"test -s {quote(bundle)}" for bundle in SYSTEM_CA_BUNDLES]
        )
        trust_store = await sandbox.exec(found, cwd="/", timeout_s=60)
        if trust_store.error_type:
            raise RuntimeError(f"Cannot check the sandbox trust store: {trust_store.stderr or trust_store.error_type}")
        if trust_store.return_code != 0:
            self.ca_bundle = f"{self.root}/ca-certificates.pem"
            await sandbox.upload(Path(certifi.where()), self.ca_bundle)
        arch = await sandbox.exec("uname -m", cwd="/", timeout_s=60)
        if arch.return_code != 0 or arch.error_type:
            raise RuntimeError(f"Cannot read the sandbox architecture: {arch.stderr or arch.error_type}")
        arch_name = {"amd64": "x86_64", "arm64": "aarch64"}.get(
            (arch.stdout or "").strip(), (arch.stdout or "").strip()
        )
        if arch_name not in GOOSE_ARCHIVE_SHA256:
            raise AgentSeedRejectedError(f"No Goose {GOOSE_VERSION} release for sandbox architecture {arch_name!r}")
        name = goose_archive_name(arch_name)
        local = await cached_download(
            f"{GOOSE_RELEASE_URL}/{name}", sha256=GOOSE_ARCHIVE_SHA256[arch_name], timeout=timeout
        )
        archive = f"{directory}/{name}"
        await sandbox.upload(local, archive)
        result = await sandbox.exec(
            f"bash {quote(f'{directory}/{INSTALLER}')} {quote(self.runtime)} {quote(GOOSE_VERSION)} "
            f"{quote(arch_name)} {quote(GOOSE_ARCHIVE_SHA256[arch_name])} {quote(archive)}",
            cwd="/",
            timeout_s=timeout,
        )
        if result.return_code != 0 or result.error_type:
            details = "\n".join(part for part in (result.stderr, result.stdout) if part)
            raise RuntimeError(
                f"Goose {GOOSE_VERSION} installation failed (exit {result.return_code}, "
                f"error_type={result.error_type}): {details[-4096:]}"
            )

    async def stage(self, *, command: list[str], env: dict[str, str], prompt: str, config: str) -> SandboxCommand:
        """Upload the task and Goose's config as files, never through shell interpolation; return the runner command."""
        directory = self.session.session_dir
        await upload_text(self.session.sandbox, path=f"{directory}/prompt.txt", text=prompt)
        await upload_text(self.session.sandbox, path=f"{self.root}/config/config.yaml", text=config)
        params = {
            "cwd": self.session.workdir,
            "command": command,
            "env": env,
            # Task-image settings must not redirect Goose's model calls, configuration or traces.
            "unset_env": ["GOOSE_*", "OPENAI_*", "OTEL_*", "LANGFUSE_*"],
            "stdin": f"{directory}/prompt.txt",
            "stdout": f"{directory}/stdout.jsonl",
            "stderr": f"{directory}/stderr.log",
        }
        await upload_text(self.session.sandbox, path=f"{directory}/params.sh", text=shell_assignments(params))
        return SandboxCommand(argv=["bash", f"{directory}/{HARNESS_RUNNER_FILE}", f"{directory}/params.sh"])

    async def collect(self) -> GooseArtifacts:
        """Copy stdout, the stderr tail and the session store after the supervisor confirms termination."""
        if self.session.expired_before_launch:
            return GooseArtifacts(stdout="", stderr="", sessions=None)  # A rollout stop kept Goose from starting.
        sandbox, directory = self.session.sandbox, self.session.session_dir
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
        packed = await sandbox.exec(
            f"tar -czf {quote(archive)} -C {quote(self.root + '/data')} sessions", cwd="/", timeout_s=60
        )
        if packed.return_code == 0 and not packed.error_type:
            with tempfile.TemporaryDirectory(prefix="gym-goose-capture-") as temporary:
                local = Path(temporary) / "sessions.tgz"
                await sandbox.download(archive, local)
                sessions = await asyncio.to_thread(local.read_bytes)
        else:
            LOG.warning("Goose session store unavailable: %s", packed.stderr or packed.stdout)
        return GooseArtifacts(stdout=stdout, stderr=stderr, sessions=sessions)

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
