# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prime Agent runtime, input staging and artifact capture over Gym's shared sandbox lifecycle."""

import asyncio
import hashlib
import io
import json
import logging
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from shlex import quote

from nemo_gym.agents import (
    HARNESS_RUNNER_FILE,
    HARNESS_RUNNER_SCRIPT,
    AgentSessionState,
    SandboxCommand,
    SandboxSession,
    shell_assignments,
)
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import read_text, upload_text
from nemo_gym.trajectory import AgentObservationBundle


LOG = logging.getLogger(__name__)

PRIME_AGENT_VERSION = "0.7.0"
# The GitHub release package; its sha256 matches the SHA256SUMS published with the release on Prime's download host.
PRIME_AGENT_PACKAGE_URL = (
    "https://github.com/PrimeIntellect-ai/prime-agent/releases/download/v0.7.0/prime-agent-0.7.0.tgz"
)
PRIME_AGENT_PACKAGE_SHA256 = (
    "88b6578518c72cd51a825bc80f28e0fef9a64c67de4a7d6fd7afd7ca1b34da0b"  # pragma: allowlist secret
)
# Release dependencies use an R2 host in their manifests. Fetch the identical, pinned GitHub release assets and
# rewrite only those URLs to uploaded local archives, including the core package's transitive AI dependency.
PRIME_RELEASE_PACKAGES = {
    "prime-agent": PRIME_AGENT_PACKAGE_SHA256,
    "prime-agent-ai": "7cdbb3e835f48dd103325f7a351ce540b27af4d161aeb9c7b9bdcc12fe7909af",  # pragma: allowlist secret
    "prime-agent-core": "0313373089831d9a2ce06e874fab8b9c05762c0094ff9fc202908cf7db7f99cd",  # pragma: allowlist secret
    "prime-agent-tui": "3225f7f92e87db80fe2c9005d1f7770735ae625c32935ef2283688fc9bd33951",  # pragma: allowlist secret
}
PRIME_RELEASE_URL = PRIME_AGENT_PACKAGE_URL.rsplit("/", 1)[0]
PRIME_DEPENDENCY_URL = "https://pub-728493de92a943e2a9b2d17b4719f318.r2.dev/releases/v0.7.0"


NODE_VERSION = "22.23.2"
# sha256 of the official Node.js builds, from https://nodejs.org/dist/v22.23.2/SHASUMS256.txt, keyed by `uname -m`.
NODE_SHA256 = {
    "x86_64": "b294a556e639d64338823920e5866c21c02741742d2e1529ee1a225c1ec9252a",  # pragma: allowlist secret
    "aarch64": "013b59cfd2819703a6f4a14ab891fc46fc2a4e3f5bcd92de3fb4929b43e35b30",  # pragma: allowlist secret
}
NODE_ARCH = {"x86_64": "x64", "aarch64": "arm64"}
CLI = "lib/node_modules/prime-agent/dist/bundle/cli.js"
SESSION_ROOT = "/tmp/nemo-gym-prime-sessions"
RUNTIME_ROOT = "/tmp/nemo-gym-prime-runtime"
INSTALLER = "install_prime_runtime.sh"
# Bounds the stderr tail copied back to the agent server.
STDERR_TAIL_BYTES = 1024 * 1024


def _localize_release_package(source: Path, target: Path, replacements: dict[str, str]) -> str:
    # Stream archive members without extracting paths or modifying the verified download cache.
    found_manifest = False
    with tarfile.open(source, "r:gz") as original, tarfile.open(target, "w:gz") as localized:
        for member in original:
            data = original.extractfile(member) if member.isfile() else None
            if member.name == "package/package.json":
                if data is None:
                    raise ValueError("Prime release package.json must be a regular file")
                manifest = json.load(data)
                for name, value in manifest.get("dependencies", {}).items():
                    if value in replacements:
                        manifest["dependencies"][name] = replacements[value]
                content = json.dumps(manifest).encode()
                member.size = len(content)
                data = io.BytesIO(content)
                found_manifest = True
            localized.addfile(member, data)
    if not found_manifest:
        raise ValueError("Prime release archive has no package/package.json")
    return hashlib.sha256(target.read_bytes()).hexdigest()


@dataclass(frozen=True)
class PrimeArtifacts:
    """Prime Agent output copied before the sandbox connection is released, including partial runs."""

    stdout: str
    stderr: str


@dataclass
class PrimeSession(AgentSessionState):
    """Per-episode Prime Agent state; SandboxSession owns supervision, termination and release."""

    session: SandboxSession[PrimeArtifacts]
    provider_name: str
    sandbox_id: str | None
    runtime: str = f"{RUNTIME_ROOT}-{PRIME_AGENT_VERSION}-node-{NODE_VERSION}"
    task: asyncio.Task[NeMoGymResponse] | None = None
    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None
    observations: AgentObservationBundle | None = None

    @property
    def agent_dir(self) -> str:
        """PRIME_AGENT_CODING_AGENT_DIR: models, tools and state, outside the task repository and home."""
        return f"{self.session.session_dir}/agent"

    async def install(self, *, timeout: float) -> None:
        """Create the session directory and install (or reuse) Node.js and the pinned Prime Agent package.

        The agent server downloads and checks the Node.js build, the release package and the kernel's Python and uv
        once, then uploads them, so the sandbox needs no download tools. Prime release dependencies are also uploaded;
        npm and uv install remaining dependencies from their package indexes.
        """
        # Shared host-side download cache for pinned sandbox assets.
        from responses_api_agents.miniswe_sandboxed_agent.bootstrap import (
            PYTHON_SHA256,
            bootstrap_assets,
            cached_download,
        )

        sandbox, directory = self.session.sandbox, self.session.session_dir
        prepared = await sandbox.exec(
            f"test -d {quote(self.session.workdir)} && umask 077 && mkdir -p {quote(self.agent_dir)}",
            cwd="/",
            timeout_s=timeout,
        )
        if prepared.return_code != 0 or prepared.error_type:
            raise RuntimeError(
                f"Cannot prepare the Prime Agent session directory: {prepared.stderr or prepared.stdout}"
            )
        await sandbox.upload(HARNESS_RUNNER_SCRIPT, f"{directory}/{HARNESS_RUNNER_FILE}")
        await sandbox.upload(Path(__file__).with_name(INSTALLER), f"{directory}/{INSTALLER}")
        arch = await sandbox.exec("uname -m", cwd="/", timeout_s=60)
        arch_name = {"amd64": "x86_64", "arm64": "aarch64"}.get(
            (arch.stdout or "").strip(), (arch.stdout or "").strip()
        )
        if arch_name not in NODE_SHA256:
            raise RuntimeError(f"No Node.js {NODE_VERSION} build for sandbox architecture {arch_name!r}")
        name = f"node-v{NODE_VERSION}-linux-{NODE_ARCH[arch_name]}.tar.gz"
        node = await cached_download(
            f"https://nodejs.org/dist/v{NODE_VERSION}/{name}", sha256=NODE_SHA256[arch_name], timeout=timeout
        )
        # The installer requires glibc, so the kernel uses the glibc Python build.
        uv, python = await bootstrap_assets(arch_name, "gnu")
        await sandbox.upload(node, f"{directory}/{name}")
        replacements = {
            f"{PRIME_DEPENDENCY_URL}/{name}-{PRIME_AGENT_VERSION}.tgz": f"file:{directory}/{name}.tgz"
            for name in PRIME_RELEASE_PACKAGES
        }
        package_checksums = {}
        with tempfile.TemporaryDirectory(prefix="gym-prime-release-") as temporary:
            for package_name, checksum in PRIME_RELEASE_PACKAGES.items():
                package = await cached_download(
                    f"{PRIME_RELEASE_URL}/{package_name}-{PRIME_AGENT_VERSION}.tgz", sha256=checksum, timeout=timeout
                )
                localized = Path(temporary) / f"{package_name}.tgz"
                package_checksums[package_name] = await asyncio.to_thread(
                    _localize_release_package, package, localized, replacements
                )
                await sandbox.upload(localized, f"{directory}/{package_name}.tgz")
        dependency_checks = " ".join(
            f"{quote(f'{directory}/{name}.tgz')} {quote(checksum)}"
            for name, checksum in package_checksums.items()
            if name != "prime-agent"
        )
        await sandbox.upload(python, f"{directory}/python.tar.gz")
        await sandbox.upload(uv, f"{directory}/uv.tar.gz")
        result = await sandbox.exec(
            f"bash {quote(f'{directory}/{INSTALLER}')} {quote(self.runtime)} {quote(PRIME_AGENT_VERSION)} "
            f"{quote(f'{directory}/{name}')} {quote(NODE_SHA256[arch_name])} "
            f"{quote(f'{directory}/prime-agent.tgz')} {quote(package_checksums['prime-agent'])} "
            f"{quote(f'{directory}/python.tar.gz')} {quote(PYTHON_SHA256[arch_name, 'gnu'])} "
            f"{quote(f'{directory}/uv.tar.gz')} {dependency_checks}",
            cwd="/",
            timeout_s=timeout,
        )
        if result.return_code != 0 or result.error_type:
            details = "\n".join(part for part in (result.stderr, result.stdout) if part)
            raise RuntimeError(
                f"Prime Agent {PRIME_AGENT_VERSION} installation failed (exit {result.return_code}, "
                f"error_type={result.error_type}): {details[-4096:]}"
            )

    async def stage(self, *, command: list[str], env: dict[str, str], models: str) -> SandboxCommand:
        """Upload Prime Agent's model configuration as a file and return the runner command."""
        directory = self.session.session_dir
        await upload_text(self.session.sandbox, path=f"{self.agent_dir}/models.json", text=models)
        params = {
            "cwd": self.session.workdir,
            "command": command,
            "env": env,
            # Task-image settings must not redirect Prime Agent's configuration or model calls.
            "unset_env": ["PRIME_AGENT_*", "PI_*", "OPENAI_*"],
            "stdin": "/dev/null",
            "stdout": f"{directory}/stdout.jsonl",
            "stderr": f"{directory}/stderr.log",
        }
        await upload_text(self.session.sandbox, path=f"{directory}/params.sh", text=shell_assignments(params))
        # The runner creates stdout only at exec, so a rollout stop that lands before then still leaves an empty file.
        await upload_text(self.session.sandbox, path=params["stdout"], text="")
        return SandboxCommand(argv=["bash", f"{directory}/{HARNESS_RUNNER_FILE}", f"{directory}/params.sh"])

    async def collect(self) -> PrimeArtifacts:
        """Copy stdout and the stderr tail after the supervisor confirms termination."""
        if self.session.expired_before_launch:
            return PrimeArtifacts(stdout="", stderr="")  # A rollout stop kept Prime Agent from starting.
        sandbox, directory = self.session.sandbox, self.session.session_dir
        # A failed stdout read must raise so the run is retried, not scored as an empty answer.
        stdout = await read_text(sandbox, path=f"{directory}/stdout.jsonl")
        # stderr is optional diagnostics: failing to fetch it must not discard a completed run's stdout.
        stderr = ""
        try:
            tail = await sandbox.exec(
                f"tail -c {STDERR_TAIL_BYTES} {quote(f'{directory}/stderr.log')} > {quote(f'{directory}/stderr.tail')}",
                cwd="/",
                timeout_s=30,
            )
            if tail.return_code == 0 and not tail.error_type:
                stderr = await read_text(sandbox, path=f"{directory}/stderr.tail")
        except Exception:
            LOG.warning("Prime Agent stderr unavailable", exc_info=True)
        return PrimeArtifacts(stdout=stdout, stderr=stderr)

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
