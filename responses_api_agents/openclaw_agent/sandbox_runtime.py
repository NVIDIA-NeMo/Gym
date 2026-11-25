# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenClaw's sandbox runtime, prepared on the Agent Server host and uploaded into the task sandbox.

With ``sandbox_runtime_upload`` installing needs only Bash, ``tar`` and ``gzip`` in the sandbox: no network, no
root, no package manager and no ``curl``. The image must hold the shared libraries the official Node.js build links
to, such as ``libstdc++.so.6``. For each sandbox platform the host downloads the pinned Node.js build and checks its
SHA-256, then installs ``openclaw`` for that platform with that build's own npm, the one the in-sandbox installer
would run. The package tree is not portable: npm selects per-platform packages that hold prebuilt binaries, so
``--os``, ``--cpu`` and ``--libc`` name the sandbox, never the host, and no install script runs on the host. Both
archives are cached on the host under ``openclaw_sandbox_runtime`` in Gym's ``cache_dir``.

npm is JavaScript and needs a Node.js that runs on the host. The sandbox's build is used where it runs, a glibc
Linux host of the sandbox's architecture; otherwise the pinned build for the host is downloaded as well.

Only glibc sandboxes are supported. A musl sandbox is refused: the in-sandbox installer adds or replaces the C++
runtime its Node.js build needs, as root, and nothing is installed into the image here.

In the sandbox, ``install_uploaded_openclaw_runtime.sh`` unpacks both archives in a private directory, runs the
packages' install scripts there with ``npm rebuild --offline``, which the task image's npm settings do not reach,
checks that the CLI runs and renames the runtime into place in one step, so a concurrent seed never sees a partial
runtime.
"""

import asyncio
import functools
import hashlib
import json
import logging
import os
import shlex
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from collections.abc import Awaitable, Callable
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from aiohttp import ClientTimeout

from nemo_gym import CACHE_DIR
from nemo_gym.agents import AgentSeedRejectedError, SandboxPlatform, probe_sandbox_platform
from nemo_gym.client import request
from nemo_gym.config import CACHE_DIR_KEY_NAME, maybe_get_global_config_dict
from nemo_gym.sandbox import AsyncSandbox
from responses_api_agents.openclaw_agent.setup_openclaw import _node_platform, _node_reported_version


LOG = logging.getLogger(__name__)

#: The Node.js that runs OpenClaw in a sandbox; ``install_openclaw_runtime.sh`` installs the same version.
SANDBOX_NODE_VERSION = "22.19.0"
# Published in https://nodejs.org/dist/v22.19.0/SHASUMS256.txt. The macOS builds only ever run npm on a macOS host.
_NODE_SHA256 = {
    "linux-x64": "d36e56998220085782c0ca965f9d51b7726335aed2f5fc7321c6c0ad233aa96d",  # pragma: allowlist secret
    "linux-arm64": "d32817b937219b8f131a28546035183d79e7fd17a86e38ccb8772901a7cd9009",  # pragma: allowlist secret
    "darwin-x64": "3cfed4795cd97277559763c5f56e711852d2cc2420bda1cea30c8aa9ac77ce0c",  # pragma: allowlist secret
    "darwin-arm64": "c59006db713c770d6ec63ae16cb3edc11f49ee093b5c415d667bb4f436c6526d",  # pragma: allowlist secret
}
_NODE_DOWNLOADS = "https://nodejs.org/dist"
# The official Linux builds of Node.js 22 are linked against glibc 2.28.
_NODE_MIN_GLIBC_MINOR = 28
# `uname -m` spellings of the CPUs the Linux builds run on, and the name npm and nodejs.org use for each.
_NPM_CPU = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}
# Bounds one host download or build, which seeds share, so a stalled one ends and the next seed starts afresh.
_HOST_BUILD_TIMEOUT_S = 1800
# The host's npm_config_* settings that reach the build: where packages come from, never which are installed. One
# that selects packages, such as omit=optional, would be cached as this platform's runtime.
_NPM_HOST_SETTINGS = frozenset({"registry", "proxy", "https_proxy", "noproxy", "cafile", "strict_ssl"})
_INSTALLER = Path(__file__).with_name("install_uploaded_openclaw_runtime.sh")
# The build in progress for each cached archive, so concurrent cold-cache seeds in this process share one build.
_BUILDS: dict[Path, asyncio.Task[None]] = {}


def uploaded_runtime_dir(openclaw_version: str) -> str:
    """Sandbox directory an uploaded runtime is published to: ``node/`` and ``openclaw/``, whole or absent."""
    return f"/tmp/nemo-gym-openclaw-runtime-node-{SANDBOX_NODE_VERSION}-openclaw-{openclaw_version}"


def sandbox_node_build(platform: SandboxPlatform) -> str:
    """The Node.js build that runs on ``platform``, as nodejs.org names it, for example ``linux-x64``.

    Raises ``AgentSeedRejectedError`` for a platform no pinned build runs on, which every attempt on its image
    would repeat.
    """
    if platform.libc == "musl":
        raise AgentSeedRejectedError(
            "sandbox_runtime_upload supports glibc sandboxes only, and this sandbox's C library is musl. Use the "
            "in-sandbox installer or sandbox_runtime_dir for this image."
        )
    if platform.glibc_minor is not None and platform.glibc_minor < _NODE_MIN_GLIBC_MINOR:
        raise AgentSeedRejectedError(
            f"OpenClaw's pinned Node.js {SANDBOX_NODE_VERSION} needs glibc 2.{_NODE_MIN_GLIBC_MINOR} or newer; the "
            f"sandbox has glibc 2.{platform.glibc_minor}"
        )
    return f"linux-{_NPM_CPU[platform.arch]}"


def _cache_dir() -> Path:
    """Where the host keeps the archives: ``openclaw_sandbox_runtime`` under Gym's ``cache_dir``."""
    configured = (maybe_get_global_config_dict() or {}).get(CACHE_DIR_KEY_NAME)
    return Path(configured or CACHE_DIR) / "openclaw_sandbox_runtime"


def _environment_proxy(url: str) -> str | None:
    """The proxy the host's environment names for ``url``. Gym's shared HTTP client does not read it itself."""
    if urllib.request.proxy_bypass(urlsplit(url).hostname or ""):
        return None
    return urllib.request.getproxies().get(urlsplit(url).scheme)


def _forget_build(path: Path, build: asyncio.Task[None]) -> None:
    _BUILDS.pop(path, None)
    # Every waiter may have been cancelled; the outcome is still consumed here.
    if not build.cancelled():
        build.exception()


async def _cached(path: Path, build: Callable[[], Awaitable[None]]) -> Path:
    """``path``, built unless the host cache holds it. Builders write atomically, so it is whole or absent.

    Concurrent callers in this process share one build and its outcome, and a caller that is cancelled leaves it
    running for the others. Another worker process with a cold cache runs its own build. A build that fails, or
    that outlasts ``_HOST_BUILD_TIMEOUT_S`` and is stopped, is retried by the next call.
    """
    if not path.is_file():
        task = _BUILDS.get(path)
        if task is None:
            task = _BUILDS[path] = asyncio.create_task(_bounded(path, build))
            task.add_done_callback(functools.partial(_forget_build, path))
        await asyncio.shield(task)
    return path


async def _bounded(path: Path, build: Callable[[], Awaitable[None]]) -> None:
    """Run ``build`` under the host build deadline. A timeout in it is the build's failure, not the seed's."""
    try:
        async with asyncio.timeout(_HOST_BUILD_TIMEOUT_S) as deadline:
            await build()
    except TimeoutError as error:
        reason = f"took longer than {_HOST_BUILD_TIMEOUT_S} s" if deadline.expired() else "timed out"
        raise RuntimeError(f"Preparing {path.name} on the Agent Server host {reason}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


async def _node_archive(build: str) -> Path:
    """Return the host-cache path to the pinned Node.js archive for ``build`` after checking its SHA-256."""
    expected = _NODE_SHA256.get(build)
    if expected is None:
        raise RuntimeError(f"No pinned Node.js {SANDBOX_NODE_VERSION} build for {build}")
    name = f"node-v{SANDBOX_NODE_VERSION}-{build}.tar.gz"
    url = f"{_NODE_DOWNLOADS}/v{SANDBOX_NODE_VERSION}/{name}"
    # A host that cannot download it can be given the file here.
    path = _cache_dir() / name

    async def download() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f".{path.name}.{uuid4().hex}")
        try:
            async with await request(
                "GET", url, _max_num_tries=3, proxy=_environment_proxy(url), timeout=ClientTimeout(total=600)
            ) as response:
                if not response.ok:
                    raise RuntimeError(f"Downloading {url} failed with HTTP {response.status}")
                with partial.open("wb") as output:
                    async for chunk in response.content.iter_chunked(1 << 20):
                        await asyncio.to_thread(output.write, chunk)
            if await asyncio.to_thread(_sha256, partial) != expected:
                raise RuntimeError(f"Checksum mismatch for {url}")
            os.replace(partial, path)
        finally:
            partial.unlink(missing_ok=True)

    await _cached(path, download)
    # The cache outlives this process, so check the bytes that are about to be unpacked or uploaded.
    if await asyncio.to_thread(_sha256, path) != expected:
        raise RuntimeError(f"Cached Node.js build {path} does not match its pinned checksum; delete it to retry")
    return path


def _unpack_node(archive: Path, destination: Path) -> Path:
    """Unpack a Node.js archive into ``destination`` and return the directory that holds ``bin/`` and ``lib/``."""
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(destination, filter="data")
    (root,) = destination.iterdir()
    return root


async def _host_node(sandbox_build: str, sandbox_node: Path, scratch: Path) -> Path:
    """Return a ``node`` that runs on this host: the sandbox's own where it does, else the host's pinned build.

    Raises ``RuntimeError`` when neither runs, for example on a host no pinned build exists for.
    """
    candidate = sandbox_node / "bin" / "node"
    if await asyncio.to_thread(_node_reported_version, str(candidate)) == SANDBOX_NODE_VERSION:
        return candidate
    try:
        host_build = "-".join(_node_platform())
    except RuntimeError:
        # nodejs.org publishes no build for this host at all.
        host_build = sys.platform
    if host_build != sandbox_build and host_build in _NODE_SHA256:
        archive = await _node_archive(host_build)
        candidate = await asyncio.to_thread(_unpack_node, archive, scratch / "host-node") / "bin" / "node"
        if await asyncio.to_thread(_node_reported_version, str(candidate)) == SANDBOX_NODE_VERSION:
            return candidate
    raise RuntimeError(
        f"Building the OpenClaw sandbox runtime for {sandbox_build} needs Node.js {SANDBOX_NODE_VERSION} to run npm "
        f"on the Agent Server host, and no pinned build runs on this host ({host_build}). Build the runtime on a "
        f"supported host and copy its archive into {_cache_dir()}, or use the in-sandbox installer."
    )


async def _run(argv: list[str], *, env: dict[str, str]) -> None:
    process = await asyncio.create_subprocess_exec(
        *argv, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        stdout, stderr = await process.communicate()
    except BaseException:
        if process.returncode is None:
            process.kill()
            await process.wait()
        raise
    if process.returncode != 0:
        output = (stderr or stdout).decode(errors="replace")
        raise RuntimeError(f"npm {argv[2]} exited with {process.returncode}: {output[-4000:]}")


def _npm_environment(node: Path) -> dict[str, str]:
    """This process's environment for npm, with ``node`` first on ``PATH`` and only ``_NPM_HOST_SETTINGS`` of npm's."""
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.lower().startswith("npm_config_")
        or name.lower().removeprefix("npm_config_").replace("-", "_") in _NPM_HOST_SETTINGS
    }
    env["PATH"] = f"{node.parent}{os.pathsep}{env.get('PATH', '')}"
    return env


def _write_archive(source: Path, archive: Path) -> None:
    with tarfile.open(archive, "w:gz", compresslevel=6) as tar:
        tar.add(source, arcname="openclaw")


async def _openclaw_archive(openclaw_version: str, platform: SandboxPlatform) -> Path:
    """Return the host-cache path to the ``openclaw`` package tree installed for ``platform``, building it if needed.

    The archive unpacks to ``openclaw/``, as ``npm install --prefix`` leaves it in the in-sandbox installer.
    """
    build = sandbox_node_build(platform)
    path = _cache_dir() / f"openclaw-{openclaw_version}-node-{SANDBOX_NODE_VERSION}-{build}.tar.gz"

    async def install() -> None:
        node_archive = await _node_archive(build)
        LOG.info("Building the OpenClaw %s sandbox runtime for %s into %s", openclaw_version, build, path)
        path.parent.mkdir(parents=True, exist_ok=True)
        scratch = Path(tempfile.mkdtemp(prefix=".build-", dir=path.parent))
        try:
            sandbox_node = await asyncio.to_thread(_unpack_node, node_archive, scratch / "node")
            node = await _host_node(build, sandbox_node, scratch)
            tree = scratch / "openclaw"
            argv = [
                str(node),
                # The sandbox build's npm, whichever Node.js runs it: the same resolver the in-sandbox installer uses.
                str(sandbox_node / "lib/node_modules/npm/bin/npm-cli.js"),
                "install",
                "--prefix",
                str(tree),
                # Select the per-platform packages of the sandbox, not of this host.
                "--os",
                "linux",
                "--cpu",
                _NPM_CPU[platform.arch],
                "--libc",
                "glibc",
                # An install script would run on this host and for it; the sandbox runs them with npm rebuild.
                "--ignore-scripts",
                # npmrc files never shape the install; of the host's npm_config_* environment, only where packages
                # come from, such as a registry mirror, still applies.
                "--userconfig",
                str(scratch / "user-npmrc"),
                "--globalconfig",
                str(scratch / "global-npmrc"),
                "--cache",
                str(scratch / "npm-cache"),
                "--no-audit",
                "--no-fund",
                "--no-update-notifier",
                f"openclaw@{openclaw_version}",
            ]
            await _run(argv, env=_npm_environment(node))
            package = tree / "node_modules/openclaw/package.json"
            installed = json.loads(await asyncio.to_thread(package.read_text)).get("version")
            if installed != openclaw_version:
                raise RuntimeError(f"npm installed openclaw {installed}, not the requested {openclaw_version}")
            archive = scratch / "openclaw.tar.gz"
            await asyncio.to_thread(_write_archive, tree, archive)
            os.replace(archive, path)
        finally:
            # Tens of thousands of files; removing them on the event loop would stall every session on this server.
            await asyncio.to_thread(shutil.rmtree, scratch, ignore_errors=True)

    return await _cached(path, install)


def _readable_in_sandbox(archive: Path) -> Path:
    """Let every user read ``archive``, whatever umask wrote it or whoever put it in the cache.

    A provider may upload a file with its host mode and another owner than the sandbox's user, who unpacks it.
    """
    mode = archive.stat().st_mode
    if mode & 0o044 != 0o044:
        archive.chmod(mode | 0o044)
    return archive


async def _refuse_other_machines(sandbox: AsyncSandbox) -> None:
    """Refuse a sandbox whose kernel or CPU no pinned Node.js build runs on, which every attempt would repeat.

    The shared platform probe raises the same retryable error for such a sandbox as for a probe that failed.
    """
    named = await sandbox.exec("uname -s; uname -m", cwd="/", timeout_s=30)
    names = (named.stdout or "").split()
    if named.return_code != 0 or named.error_type is not None or len(names) != 2:
        # The platform probe that follows reports a sandbox that cannot be asked.
        return
    system, machine = names
    if system != "Linux" or machine not in _NPM_CPU:
        raise AgentSeedRejectedError(
            f"OpenClaw's pinned Node.js {SANDBOX_NODE_VERSION} has builds for Linux on x86_64 and aarch64 only; the "
            f"sandbox is {system} on {machine}"
        )


async def _discard(sandbox: AsyncSandbox, path: str) -> None:
    """Remove ``path`` from the sandbox, best effort. Cancelling the caller does not stop the removal."""
    removal = asyncio.ensure_future(sandbox.exec(f"rm -rf {shlex.quote(path)}", cwd="/", timeout_s=30))
    try:
        result = await asyncio.shield(removal)
    except Exception:
        LOG.warning("Could not remove %s from the sandbox", path, exc_info=True)
        return
    if result.return_code != 0 or result.error_type is not None:
        LOG.warning("Could not remove %s from the sandbox: %s", path, result.error_type or result.stderr)


async def upload_openclaw_runtime(
    sandbox: AsyncSandbox, *, runtime_dir: str, openclaw_version: str, timeout_s: float
) -> None:
    """Publish the runtime to ``runtime_dir`` in the sandbox unless one is there already.

    Builds both archives for the sandbox's platform on this host if needed, uploads them and unpacks them. A runtime
    directory that exists was published whole by an earlier seed and is left alone; the caller checks that it runs
    ``openclaw_version``. ``timeout_s`` bounds the whole installation, including a build that another seed started.

    Raises ``AgentSeedRejectedError`` when no pinned Node.js build runs on the sandbox's platform, ``RuntimeError``
    when the host build, a download or the installation in the sandbox fails or the host build outlasts its own
    deadline, and ``TimeoutError`` when ``timeout_s`` runs out.
    """
    try:
        async with asyncio.timeout(timeout_s):
            present = await sandbox.exec(f"test -e {shlex.quote(runtime_dir)}", cwd="/", timeout_s=30)
            if present.error_type is not None:
                raise RuntimeError(f"Could not look for an OpenClaw runtime in the sandbox: {present.error_type}")
            if present.return_code == 0:
                return
            await _refuse_other_machines(sandbox)
            platform = await probe_sandbox_platform(sandbox)
            node_archive = _readable_in_sandbox(await _node_archive(sandbox_node_build(platform)))
            openclaw_archive = _readable_in_sandbox(await _openclaw_archive(openclaw_version, platform))
            # Beside the runtime directory, on its filesystem, so publishing is one rename.
            setup = f"{runtime_dir}.setup-{uuid4().hex}"
            try:
                prepared = await sandbox.exec(f"mkdir -p {shlex.quote(setup)}", cwd="/", timeout_s=30)
                if prepared.return_code != 0 or prepared.error_type is not None:
                    raise RuntimeError(
                        prepared.stderr or prepared.stdout or f"Could not create {setup} in the sandbox"
                    )
                await sandbox.upload(node_archive, f"{setup}/node.tar.gz")
                await sandbox.upload(openclaw_archive, f"{setup}/openclaw.tar.gz")
                await sandbox.upload(_INSTALLER, f"{setup}/{_INSTALLER.name}")
            except BaseException:
                await _discard(sandbox, setup)
                raise
            command = shlex.join(["bash", f"{setup}/{_INSTALLER.name}", runtime_dir, setup, openclaw_version])
            result = None
            try:
                result = await sandbox.exec(command, cwd="/", timeout_s=timeout_s)
            finally:
                # LocalProvider kills the process group on timeout or cancellation, so the installer's EXIT trap
                # cannot run; once exec unwinds, no installer remains to use this setup directory. A returned
                # command failure is safe to clean up on any provider, even if the installer exited before setting
                # its trap. A remote timeout or provider error can leave the installer running, so it keeps the
                # directory.
                if sandbox.provider_name == "local" or (
                    result is not None and result.error_type is None and result.return_code != 0
                ):
                    await _discard(sandbox, setup)
    except TimeoutError as error:
        raise TimeoutError(f"Installing the uploaded OpenClaw runtime took longer than {timeout_s} s") from error
    if result.return_code != 0 or result.error_type is not None:
        raise RuntimeError(
            f"Installing the uploaded OpenClaw runtime failed with {result.error_type or result.return_code}: "
            f"{(result.stderr or '')[-2000:]}{(result.stdout or '')[-2000:]}"
        )
