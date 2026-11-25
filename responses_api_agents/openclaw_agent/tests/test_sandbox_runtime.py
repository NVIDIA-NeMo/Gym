# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The uploaded OpenClaw runtime: platform selection, the build on the agent server host, and publication."""

import asyncio
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemo_gym.agents import AgentSeedRejectedError, SandboxPlatform
from nemo_gym.sandbox import AsyncSandbox, LocalProvider, SandboxSpec
from responses_api_agents.openclaw_agent import sandbox_runtime
from responses_api_agents.openclaw_agent.sandbox_runtime import (
    SANDBOX_NODE_VERSION,
    sandbox_node_build,
    upload_openclaw_runtime,
    uploaded_runtime_dir,
)


VERSION = "2026.6.11"
X86_64_GLIBC = SandboxPlatform("x86_64", "gnu", 36)

# Stands in for node on every test host: it reports the pinned version, plays npm, and does what the installer's
# two JavaScript snippets do. The snippets themselves run under a real node in the tests that require one.
_FAKE_NODE = r"""
import errno, json, os, sys, time
from pathlib import Path

LOGGED = ("NPM_CONFIG_", "OPENCLAW_", "STATE_DIRECTORY", "TMPDIR", "NODE_COMPILE_CACHE")
args = sys.argv[1:]
if os.environ.get("FAKE_NODE_LOG"):
    with open(os.environ["FAKE_NODE_LOG"], "a") as log:
        env = {name: value for name, value in os.environ.items() if name.upper().startswith(LOGGED)}
        log.write(json.dumps({"node": sys.argv[0], "argv": args, "home": os.environ.get("HOME"), "env": env}) + "\n")
if args == ["--version"]:
    print("v%s" % os.environ.get("FAKE_NODE_VERSION", "@PINNED@"))
elif args[0].endswith("npm-cli.js") and args[1] == "install":
    package = Path(args[args.index("--prefix") + 1]) / "node_modules/openclaw"
    package.mkdir(parents=True)
    version = os.environ.get("FAKE_NPM_INSTALLS") or args[-1].rpartition("@")[2]
    (package / "package.json").write_text(json.dumps({"version": version}))
    (package / "openclaw.mjs").write_text("")
    if os.environ.get("FAKE_NPM_PID_FILE"):
        Path(os.environ["FAKE_NPM_PID_FILE"]).write_text(str(os.getpid()))
    time.sleep(float(os.environ.get("FAKE_NPM_SLEEP", "0")))
    sys.exit(int(os.environ.get("FAKE_NPM_EXIT", "0")))
elif args[0].endswith("npm-cli.js"):
    sys.exit(int(os.environ.get("FAKE_NPM_EXIT", "0")))
elif args[0] == "-e" and "renameSync" in args[1]:
    try:
        os.rename(args[2], args[3])
    except OSError as error:
        if error.errno not in (errno.ENOTEMPTY, errno.EEXIST):
            raise
elif args[0] == "-e":
    declared = json.loads(Path(args[2]).read_text())["version"]
    if declared != args[3]:
        sys.exit("OpenClaw version mismatch: the runtime holds %s, not %s" % (declared, args[3]))
elif args[0].endswith("openclaw.mjs"):
    print("OpenClaw fake")
""".replace("@PINNED@", SANDBOX_NODE_VERSION)


def _add(tar: tarfile.TarFile, name: str, text: str = "", mode: int = 0o644) -> None:
    entry = tarfile.TarInfo(name)
    entry.size, entry.mode = len(text.encode()), mode
    tar.addfile(entry, io.BytesIO(text.encode()))


def _node_archive(path: Path, build: str, *, node: str | None = None, npm: str = "") -> str:
    """Write a stand-in Node.js archive laid out like nodejs.org's and return its SHA-256."""
    root = f"node-v{SANDBOX_NODE_VERSION}-{build}"
    launcher = f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$(dirname "$0")/node.py" "$@"\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as tar:
        _add(tar, f"{root}/bin/node", launcher if node is None else node, mode=0o755)
        _add(tar, f"{root}/bin/node.py", _FAKE_NODE)
        _add(tar, f"{root}/lib/node_modules/npm/bin/npm-cli.js", npm)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _calls(log: Path) -> list[dict]:
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def _npm(log: Path, command: str) -> list[dict]:
    return [call for call in _calls(log) if call["argv"][0].endswith("npm-cli.js") and call["argv"][1] == command]


@pytest.mark.parametrize(
    "platform, build",
    [
        (SandboxPlatform("x86_64", "gnu", 36), "linux-x64"),
        (SandboxPlatform("aarch64", "gnu", 28), "linux-arm64"),
        # A sandbox that does not report its glibc version is not refused for it.
        (SandboxPlatform("aarch64", "gnu", None), "linux-arm64"),
    ],
)
def test_the_node_build_matches_the_sandbox_platform(platform: SandboxPlatform, build: str) -> None:
    assert sandbox_node_build(platform) == build
    assert build in sandbox_runtime._NODE_SHA256


@pytest.mark.parametrize(
    "platform, match",
    [
        (SandboxPlatform("x86_64", "musl"), "supports glibc sandboxes only"),
        (SandboxPlatform("aarch64", "musl"), "supports glibc sandboxes only"),
        (SandboxPlatform("x86_64", "gnu", 27), "needs glibc 2.28 or newer; the sandbox has glibc 2.27"),
    ],
)
def test_a_platform_without_a_node_build_is_refused(platform: SandboxPlatform, match: str) -> None:
    with pytest.raises(AgentSeedRejectedError, match=match):
        sandbox_node_build(platform)


def test_the_uploaded_runtime_is_the_node_the_in_sandbox_installer_installs() -> None:
    installer = Path(sandbox_runtime.__file__).with_name("install_openclaw_runtime.sh").read_text()
    assert f"\nnode_version={SANDBOX_NODE_VERSION}\n" in installer
    # The two runtimes never share a directory: only the in-sandbox installer's has a lock and a ready marker.
    assert (
        uploaded_runtime_dir(VERSION)
        == f"/tmp/nemo-gym-openclaw-runtime-node-{SANDBOX_NODE_VERSION}-openclaw-{VERSION}"
    )
    assert uploaded_runtime_dir(VERSION) != f"/tmp/nemo-gym-openclaw-node-{SANDBOX_NODE_VERSION}-{VERSION}"


class _Content:
    def __init__(self, body: bytes) -> None:
        self.body = body

    async def iter_chunked(self, size: int):
        for start in range(0, len(self.body), size):
            yield self.body[start : start + size]


class _Response:
    def __init__(self, body: bytes, status: int = 200) -> None:
        self.content = _Content(body)
        self.status = status
        self.ok = status < 400

    async def __aenter__(self) -> "_Response":
        return self

    async def __aexit__(self, *exc_info) -> None:
        return None


@pytest.fixture
def download(monkeypatch, tmp_path):
    """A download of ``body`` for the Node.js builds, whose pinned checksum is ``pinned``."""
    served = SimpleNamespace(
        body=b"node build", status=200, pinned=hashlib.sha256(b"node build").hexdigest(), urls=[], requests=[]
    )
    monkeypatch.setattr(sandbox_runtime, "_cache_dir", lambda: tmp_path)

    async def request(method: str, url: str, **kwargs) -> _Response:
        served.urls.append(url)
        served.requests.append(kwargs)
        return _Response(served.body, served.status)

    monkeypatch.setattr(sandbox_runtime, "request", request)
    monkeypatch.setattr(sandbox_runtime, "_NODE_SHA256", {"linux-x64": served.pinned})
    return served


async def test_the_node_build_is_downloaded_once_and_checked_on_every_use(download, tmp_path) -> None:
    archive = await sandbox_runtime._node_archive("linux-x64")
    assert await sandbox_runtime._node_archive("linux-x64") == archive

    assert archive == tmp_path / f"node-v{SANDBOX_NODE_VERSION}-linux-x64.tar.gz"
    assert archive.read_bytes() == b"node build"
    assert download.urls == [
        f"https://nodejs.org/dist/v{SANDBOX_NODE_VERSION}/node-v{SANDBOX_NODE_VERSION}-linux-x64.tar.gz"
    ]

    archive.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="does not match its pinned checksum"):
        await sandbox_runtime._node_archive("linux-x64")


async def test_a_node_build_that_does_not_match_its_pin_is_not_cached(download, tmp_path) -> None:
    download.body = b"another build"

    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        await sandbox_runtime._node_archive("linux-x64")

    assert os.listdir(tmp_path) == []


async def test_a_failed_download_is_not_cached_and_the_next_seed_retries(download, tmp_path) -> None:
    download.status = 503
    with pytest.raises(RuntimeError, match="HTTP 503"):
        await sandbox_runtime._node_archive("linux-x64")
    assert os.listdir(tmp_path) == []

    download.status = 200
    assert (await sandbox_runtime._node_archive("linux-x64")).read_bytes() == b"node build"


async def test_a_build_without_a_pinned_checksum_is_never_downloaded(download) -> None:
    with pytest.raises(RuntimeError, match="No pinned Node.js"):
        await sandbox_runtime._node_archive("linux-riscv64")

    assert download.urls == []


def test_the_host_cache_lives_in_gyms_cache_dir(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(sandbox_runtime, "maybe_get_global_config_dict", lambda: {"cache_dir": str(tmp_path)})
    assert sandbox_runtime._cache_dir() == tmp_path / "openclaw_sandbox_runtime"

    monkeypatch.setattr(sandbox_runtime, "maybe_get_global_config_dict", lambda: None)
    assert sandbox_runtime._cache_dir() == sandbox_runtime.CACHE_DIR / "openclaw_sandbox_runtime"


@pytest.fixture
def host(monkeypatch, tmp_path):
    """A host cache holding stand-in Node.js builds for every sandbox platform, and a log of what they ran."""
    cache, log = tmp_path / "cache", tmp_path / "node-calls"
    pins = {
        build: _node_archive(cache / f"node-v{SANDBOX_NODE_VERSION}-{build}.tar.gz", build)
        for build in ("linux-x64", "linux-arm64")
    }
    monkeypatch.setattr(sandbox_runtime, "_cache_dir", lambda: cache)
    monkeypatch.setattr(sandbox_runtime, "_NODE_SHA256", pins)
    monkeypatch.setattr(sandbox_runtime, "request", AsyncMock(side_effect=AssertionError("the cache is warm")))
    monkeypatch.setenv("FAKE_NODE_LOG", str(log))
    return SimpleNamespace(cache=cache, log=log, pins=pins)


@pytest.mark.parametrize(
    "platform, build, cpu",
    [
        (SandboxPlatform("x86_64", "gnu", 36), "linux-x64", "x64"),
        (SandboxPlatform("aarch64", "gnu", 36), "linux-arm64", "arm64"),
    ],
)
async def test_the_host_installs_openclaw_for_the_sandbox_platform(host, platform, build, cpu) -> None:
    archive = await sandbox_runtime._openclaw_archive(VERSION, platform)

    assert archive == host.cache / f"openclaw-{VERSION}-node-{SANDBOX_NODE_VERSION}-{build}.tar.gz"
    (install,) = _npm(host.log, "install")
    argv = install["argv"]
    # The sandbox build's own npm resolves the tree, for the sandbox's platform and without running any script here.
    assert argv[0].endswith(f"node-v{SANDBOX_NODE_VERSION}-{build}/lib/node_modules/npm/bin/npm-cli.js")
    for option, value in (("--os", "linux"), ("--cpu", cpu), ("--libc", "glibc")):
        assert argv[argv.index(option) + 1] == value
    assert "--ignore-scripts" in argv
    assert argv[-1] == f"openclaw@{VERSION}"
    # No npmrc file of the host shapes the tree: both are files that do not exist, which npm requires to differ.
    npmrcs = {argv[argv.index(option) + 1] for option in ("--userconfig", "--globalconfig")}
    assert len(npmrcs) == 2 and all(str(host.cache) in npmrc for npmrc in npmrcs)
    with tarfile.open(archive) as tar:
        package = json.load(tar.extractfile("openclaw/node_modules/openclaw/package.json"))
    assert package == {"version": VERSION}
    # Only archives remain: the scratch directory with the unpacked Node.js and npm's cache is gone.
    assert sorted(path.name for path in host.cache.iterdir() if path.name.startswith(".")) == []

    assert await sandbox_runtime._openclaw_archive(VERSION, platform) == archive
    assert len(_npm(host.log, "install")) == 1


async def test_host_npm_settings_that_select_packages_never_reach_the_build(host, monkeypatch) -> None:
    """Such a tree, for example one without the optional per-platform packages, would be cached as the runtime."""
    monkeypatch.setenv("npm_config_omit", "optional")
    monkeypatch.setenv("NPM_CONFIG_OS", "darwin")
    monkeypatch.setenv("npm_config_ignore-scripts", "false")
    monkeypatch.setenv("npm_config_registry", "https://mirror.example/npm/")
    monkeypatch.setenv("NPM_CONFIG_HTTPS_PROXY", "http://proxy.example:3128")

    await sandbox_runtime._openclaw_archive(VERSION, X86_64_GLIBC)

    (install,) = _npm(host.log, "install")
    assert {name: value for name, value in install["env"].items() if name.upper().startswith("NPM_CONFIG_")} == {
        "npm_config_registry": "https://mirror.example/npm/",
        "NPM_CONFIG_HTTPS_PROXY": "http://proxy.example:3128",
    }


async def test_concurrent_cold_seeds_share_one_host_build(host) -> None:
    archives = await asyncio.gather(*(sandbox_runtime._openclaw_archive(VERSION, X86_64_GLIBC) for _ in range(4)))

    assert len(set(archives)) == 1
    assert len(_npm(host.log, "install")) == 1


async def test_a_tree_that_holds_another_version_is_not_cached(host, monkeypatch) -> None:
    monkeypatch.setenv("FAKE_NPM_INSTALLS", "2026.5.1")

    with pytest.raises(RuntimeError, match=f"npm installed openclaw 2026.5.1, not the requested {VERSION}"):
        await sandbox_runtime._openclaw_archive(VERSION, X86_64_GLIBC)

    assert not any(path.name.startswith(("openclaw-", ".")) for path in host.cache.iterdir())


async def test_a_failed_npm_install_is_not_cached_and_the_next_seed_retries(host, monkeypatch) -> None:
    monkeypatch.setenv("FAKE_NPM_EXIT", "7")
    with pytest.raises(RuntimeError, match="npm install exited with 7"):
        await sandbox_runtime._openclaw_archive(VERSION, X86_64_GLIBC)
    assert not any(path.name.startswith(("openclaw-", ".")) for path in host.cache.iterdir())

    monkeypatch.delenv("FAKE_NPM_EXIT")
    assert (await sandbox_runtime._openclaw_archive(VERSION, X86_64_GLIBC)).is_file()


async def test_a_stalled_host_build_is_stopped_and_the_next_seed_builds_afresh(monkeypatch, tmp_path) -> None:
    """Seeds share one build; without its own deadline, a stalled one would hold every later seed to its timeout."""
    monkeypatch.setattr(sandbox_runtime, "_HOST_BUILD_TIMEOUT_S", 0.05)
    archive, started, stopped = tmp_path / "archive", asyncio.Event(), asyncio.Event()

    async def stall() -> None:
        started.set()
        try:
            await asyncio.sleep(3600)
        finally:
            stopped.set()

    waiter = asyncio.ensure_future(sandbox_runtime._cached(archive, stall))
    await started.wait()
    # A seed that gives up first leaves the shared build to its own deadline.
    leaver = asyncio.ensure_future(sandbox_runtime._cached(archive, stall))
    await asyncio.sleep(0)
    leaver.cancel()

    with pytest.raises(RuntimeError, match="Preparing archive on the Agent Server host took longer than 0.05 s"):
        await waiter

    assert stopped.is_set() and sandbox_runtime._BUILDS == {}

    async def build() -> None:
        archive.write_text("built")

    assert (await sandbox_runtime._cached(archive, build)).read_text() == "built"


async def test_stopping_a_host_build_kills_its_npm(host, monkeypatch, tmp_path) -> None:
    pid_file = tmp_path / "npm.pid"
    monkeypatch.setenv("FAKE_NPM_PID_FILE", str(pid_file))
    monkeypatch.setenv("FAKE_NPM_SLEEP", "3600")
    seed = asyncio.ensure_future(sandbox_runtime._openclaw_archive(VERSION, X86_64_GLIBC))
    async with asyncio.timeout(60):
        while not pid_file.exists() or not pid_file.read_text():
            await asyncio.sleep(0.02)
    (build,) = sandbox_runtime._BUILDS.values()

    # What the build deadline does to a build that outlasts it.
    build.cancel()

    with pytest.raises(asyncio.CancelledError):
        await seed
    with pytest.raises(ProcessLookupError):
        async with asyncio.timeout(30):
            while True:
                os.kill(int(pid_file.read_text()), 0)
                await asyncio.sleep(0.02)
    assert not any(path.name.startswith(("openclaw-", ".")) for path in host.cache.iterdir())


async def test_a_download_that_times_out_fails_the_build_not_the_seed_deadline(download, monkeypatch) -> None:
    async def request(method: str, url: str, **kwargs):
        raise TimeoutError("read timed out")

    monkeypatch.setattr(sandbox_runtime, "request", request)

    with pytest.raises(RuntimeError, match="on the Agent Server host timed out"):
        await sandbox_runtime._node_archive("linux-x64")


async def test_a_tampered_cached_node_build_is_never_unpacked(host) -> None:
    (host.cache / f"node-v{SANDBOX_NODE_VERSION}-linux-x64.tar.gz").write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="does not match its pinned checksum"):
        await sandbox_runtime._openclaw_archive(VERSION, X86_64_GLIBC)

    assert _calls(host.log) == []


async def test_a_host_that_cannot_run_the_sandbox_node_runs_npm_with_its_own_pinned_build(host, monkeypatch) -> None:
    """As on a macOS host, or a Linux host of another architecture or C library than the sandbox."""
    name = f"node-v{SANDBOX_NODE_VERSION}"
    pins = {
        "linux-x64": _node_archive(host.cache / f"{name}-linux-x64.tar.gz", "linux-x64", node="#!/bin/sh\nexit 126\n"),
        "darwin-arm64": _node_archive(host.cache / f"{name}-darwin-arm64.tar.gz", "darwin-arm64"),
    }
    monkeypatch.setattr(sandbox_runtime, "_NODE_SHA256", pins)
    monkeypatch.setattr(sandbox_runtime, "_node_platform", lambda: ("darwin", "arm64"))

    archive = await sandbox_runtime._openclaw_archive(VERSION, X86_64_GLIBC)

    (install,) = _npm(host.log, "install")
    # The host's Node.js runs the sandbox build's npm, and the tree is still the sandbox's.
    assert f"{name}-darwin-arm64/bin/node" in install["node"]
    assert install["argv"][0].endswith(f"{name}-linux-x64/lib/node_modules/npm/bin/npm-cli.js")
    assert install["argv"][install["argv"].index("--os") + 1] == "linux"
    assert install["argv"][install["argv"].index("--cpu") + 1] == "x64"
    assert archive.name == f"openclaw-{VERSION}-node-{SANDBOX_NODE_VERSION}-linux-x64.tar.gz"


@pytest.mark.parametrize("host_platform", [("linux", "x64"), ("win", "x64"), None])
async def test_a_host_no_pinned_node_runs_on_fails_instead_of_building(host, monkeypatch, host_platform) -> None:
    """A host of the sandbox's own build name that cannot run it (another C library), or one with no build."""
    name = f"node-v{SANDBOX_NODE_VERSION}-linux-x64.tar.gz"
    pins = {"linux-x64": _node_archive(host.cache / name, "linux-x64", node="#!/bin/sh\nexit 126\n")}
    monkeypatch.setattr(sandbox_runtime, "_NODE_SHA256", pins)

    def node_platform() -> tuple[str, str]:
        if host_platform is None:
            raise RuntimeError("no Node.js build is published for platform 'sunos5'")
        return host_platform

    monkeypatch.setattr(sandbox_runtime, "_node_platform", node_platform)

    with pytest.raises(RuntimeError, match="no pinned build runs on this host"):
        await sandbox_runtime._openclaw_archive(VERSION, X86_64_GLIBC)

    assert _npm(host.log, "install") == []
    assert not any(path.name.startswith(("openclaw-", ".")) for path in host.cache.iterdir())


@pytest.fixture
def image(host, tmp_path):
    """A task's working directory and HOME, and the runtime directory a seed publishes to."""
    workdir, home, bin_dir = tmp_path / "task", tmp_path / "home", tmp_path / "bin"
    for directory in (workdir, home, bin_dir):
        directory.mkdir()
    return SimpleNamespace(
        root=tmp_path, workdir=workdir, home=home, bin=bin_dir, runtime=tmp_path / "rt", log=host.log, cache=host.cache
    )


def _uname(image, system: str, machine: str, *, status: int = 0) -> None:
    """Make the sandbox report ``system`` on ``machine``, whatever host the test runs on."""
    uname = image.bin / "uname"
    uname.write_text(f'#!/bin/sh\n[ "$1" = -s ] && echo {system} || echo {machine}\nexit {status}\n')
    uname.chmod(0o755)


@pytest.fixture
async def task(image, monkeypatch):
    """``image`` behind a local sandbox that reports an x86_64 glibc platform."""
    _uname(image, "Linux", "x86_64")
    monkeypatch.setattr(sandbox_runtime, "probe_sandbox_platform", AsyncMock(return_value=X86_64_GLIBC))
    image.sandbox = AsyncSandbox(LocalProvider())
    await image.sandbox.start(
        SandboxSpec(
            workdir=str(image.workdir),
            env={"HOME": str(image.home), "PATH": f"{image.bin}{os.pathsep}{os.environ['PATH']}"},
        )
    )
    yield image
    await image.sandbox.stop()


def _tree(root: Path) -> list[str]:
    return sorted(str(path.relative_to(root)) for path in root.rglob("*"))


def _leftovers(task) -> list[str]:
    return [path.name for path in task.root.iterdir() if path.name.startswith("rt.")]


async def _upload(task, *, version: str = VERSION) -> None:
    await upload_openclaw_runtime(task.sandbox, runtime_dir=str(task.runtime), openclaw_version=version, timeout_s=60)


@pytest.mark.sandbox
async def test_seeding_publishes_the_runtime_and_leaves_the_task_untouched(task) -> None:
    await _upload(task)

    assert sorted(os.listdir(task.runtime)) == ["node", "openclaw"]
    assert os.access(task.runtime / "node/bin/node", os.X_OK)
    assert json.loads((task.runtime / "openclaw/node_modules/openclaw/package.json").read_text()) == {
        "version": VERSION
    }
    assert _leftovers(task) == []
    assert _tree(task.workdir) == [] and _tree(task.home) == []
    # The packages' install scripts run in the sandbox, offline, with HOME in the setup directory.
    (rebuild,) = _npm(task.log, "rebuild")
    prefix = rebuild["argv"][rebuild["argv"].index("--prefix") + 1]
    assert re.fullmatch(rf"{re.escape(str(task.runtime))}\.setup-[0-9a-f]{{32}}/runtime/openclaw", prefix)
    assert "--offline" in rebuild["argv"]
    assert rebuild["home"] == prefix.removesuffix("/runtime/openclaw") + "/home"


@pytest.mark.sandbox
async def test_a_runtime_already_in_the_sandbox_is_kept_and_nothing_is_uploaded(task, monkeypatch) -> None:
    await _upload(task)
    (task.runtime / "session-file").write_text("in use")
    calls = len(_calls(task.log))
    upload = AsyncMock(side_effect=AssertionError("nothing is uploaded"))
    monkeypatch.setattr(task.sandbox, "upload", upload)
    # Neither the platform nor the host cache is needed for a runtime that is already there.
    monkeypatch.setattr(sandbox_runtime, "probe_sandbox_platform", AsyncMock(side_effect=AssertionError("unused")))

    await _upload(task)

    assert (task.runtime / "session-file").read_text() == "in use"
    assert len(_calls(task.log)) == calls


@pytest.mark.sandbox
async def test_concurrent_seeds_publish_one_runtime(task) -> None:
    await asyncio.wait_for(asyncio.gather(*(_upload(task) for _ in range(4))), 60)

    assert sorted(os.listdir(task.runtime)) == ["node", "openclaw"]
    assert _leftovers(task) == []
    assert len(_npm(task.log, "install")) == 1


@pytest.mark.sandbox
@pytest.mark.parametrize(
    "failure, match",
    [
        ({"FAKE_NPM_INSTALLS": "2026.5.1"}, "npm installed openclaw 2026.5.1"),
        ({"FAKE_NODE_VERSION": "0.0.0"}, "no pinned build runs on this host"),
    ],
)
async def test_a_failed_host_build_uploads_nothing(task, monkeypatch, failure, match) -> None:
    for name, value in failure.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(sandbox_runtime, "_node_platform", lambda: ("linux", "x64"))
    upload = AsyncMock(side_effect=AssertionError("nothing is uploaded"))
    monkeypatch.setattr(task.sandbox, "upload", upload)

    with pytest.raises(RuntimeError, match=match):
        await _upload(task)

    assert not task.runtime.exists() and _leftovers(task) == []


@pytest.mark.sandbox
async def test_a_musl_sandbox_is_refused_before_any_build_or_upload(task, monkeypatch) -> None:
    monkeypatch.setattr(
        sandbox_runtime, "probe_sandbox_platform", AsyncMock(return_value=SandboxPlatform("x86_64", "musl"))
    )
    monkeypatch.setattr(task.sandbox, "upload", AsyncMock(side_effect=AssertionError("nothing is uploaded")))

    with pytest.raises(AgentSeedRejectedError, match="supports glibc sandboxes only"):
        await _upload(task)

    assert not task.runtime.exists() and _leftovers(task) == []
    assert _calls(task.log) == []


@pytest.mark.sandbox
@pytest.mark.parametrize("system, machine", [("Linux", "riscv64"), ("Darwin", "arm64")])
async def test_a_kernel_or_cpu_without_a_node_build_is_refused_not_retried(task, monkeypatch, system, machine) -> None:
    """The shared platform probe fails such a sandbox with the error of a probe that could not run."""
    _uname(task, system, machine)
    probe = AsyncMock(side_effect=RuntimeError(f"No sandbox Python runtime is available for {system} on {machine}"))
    monkeypatch.setattr(sandbox_runtime, "probe_sandbox_platform", probe)
    monkeypatch.setattr(task.sandbox, "upload", AsyncMock(side_effect=AssertionError("nothing is uploaded")))

    with pytest.raises(AgentSeedRejectedError, match=f"x86_64 and aarch64 only; the sandbox is {system} on {machine}"):
        await _upload(task)

    probe.assert_not_awaited()
    assert not task.runtime.exists() and _leftovers(task) == []


@pytest.mark.sandbox
@pytest.mark.parametrize("machine", ["x86_64", "amd64", "aarch64", "arm64"])
async def test_every_spelling_of_a_supported_cpu_reaches_the_platform_probe(task, machine) -> None:
    _uname(task, "Linux", machine)

    await _upload(task)

    assert sorted(os.listdir(task.runtime)) == ["node", "openclaw"]


@pytest.mark.sandbox
async def test_a_sandbox_that_cannot_name_its_platform_stays_retryable(task, monkeypatch) -> None:
    _uname(task, "Linux", "x86_64", status=1)
    probe = AsyncMock(side_effect=RuntimeError("Could not determine the sandbox platform: timeout"))
    monkeypatch.setattr(sandbox_runtime, "probe_sandbox_platform", probe)

    with pytest.raises(RuntimeError, match="Could not determine the sandbox platform") as raised:
        await _upload(task)

    assert not isinstance(raised.value, AgentSeedRejectedError)


@pytest.mark.sandbox
@pytest.mark.parametrize("cached", ["built", "supplied"])
async def test_archives_written_under_a_private_umask_are_uploaded_readable(task, monkeypatch, cached) -> None:
    """A provider may upload a file with its host mode and another owner than the user who unpacks it."""
    modes = {}
    upload = task.sandbox.upload

    async def record_mode(source, destination):
        modes[Path(destination).name] = Path(source).stat().st_mode & 0o777
        await upload(source, destination)

    monkeypatch.setattr(task.sandbox, "upload", record_mode)
    node = task.cache / f"node-v{SANDBOX_NODE_VERSION}-linux-x64.tar.gz"
    previous = os.umask(0o077)
    try:
        if cached == "supplied":
            # An operator filled the cache: the build finds both archives and writes neither.
            await sandbox_runtime._openclaw_archive(VERSION, X86_64_GLIBC)
        for archive in task.cache.glob("*.tar.gz"):
            archive.chmod(0o600)
        await _upload(task)
    finally:
        os.umask(previous)

    assert modes["node.tar.gz"] & 0o044 == 0o044 and modes["openclaw.tar.gz"] & 0o044 == 0o044
    assert node.stat().st_mode & 0o044 == 0o044
    assert sorted(os.listdir(task.runtime)) == ["node", "openclaw"]


@pytest.mark.sandbox
async def test_a_failed_upload_leaves_no_setup_directory_and_the_next_seed_installs(task, monkeypatch) -> None:
    upload = task.sandbox.upload
    monkeypatch.setattr(task.sandbox, "upload", AsyncMock(side_effect=RuntimeError("upload failed")))
    with pytest.raises(RuntimeError, match="upload failed"):
        await _upload(task)
    assert not task.runtime.exists() and _leftovers(task) == []

    monkeypatch.setattr(task.sandbox, "upload", upload)
    await _upload(task)
    assert sorted(os.listdir(task.runtime)) == ["node", "openclaw"]


def _setup(image, node_archive: Path, *, version: str = VERSION) -> Path:
    """A setup directory as a seed uploads it, for running the installer directly."""
    setup = image.root / "rt.setup-direct"
    setup.mkdir()
    shutil.copy(node_archive, setup / "node.tar.gz")
    with tarfile.open(setup / "openclaw.tar.gz", "w:gz") as tar:
        _add(tar, "openclaw/node_modules/openclaw/package.json", json.dumps({"version": version}))
        _add(tar, "openclaw/node_modules/openclaw/openclaw.mjs", "console.log('OpenClaw CLI');\n")
    return setup


def _install(image, setup: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(sandbox_runtime._INSTALLER), str(image.runtime), str(setup), VERSION],
        env={"PATH": os.environ["PATH"], "HOME": str(image.home), **(env or {})},
        cwd="/",
        capture_output=True,
        text=True,
        errors="replace",
        timeout=60,
    )


def _fake_node(image) -> Path:
    return image.cache / f"node-v{SANDBOX_NODE_VERSION}-linux-x64.tar.gz"


def _real_node(image) -> Path:
    """A Node.js archive whose ``node`` is the host's, so the installer's own JavaScript runs."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("needs node")
    archive = image.root / "real-node.tar.gz"
    _node_archive(archive, "linux-x64", node=f'#!/bin/sh\nexec {shlex.quote(node)} "$@"\n', npm="process.exit(0);\n")
    return archive


@pytest.fixture(params=[_fake_node, _real_node], ids=["stand_in_node", "real_node"])
def node_archive(request, image) -> Path:
    return request.param(image)


def test_the_installer_publishes_a_checked_runtime_in_one_rename(image, node_archive) -> None:
    setup = _setup(image, node_archive)

    done = _install(image, setup)

    assert done.returncode == 0, done.stderr
    assert sorted(os.listdir(image.runtime)) == ["node", "openclaw"]
    assert not setup.exists()
    assert _tree(image.home) == []


def test_the_installer_keeps_openclaw_out_of_directories_the_task_names(image) -> None:
    """OpenClaw's install script removes stale files from its state directory, which these variables set."""
    setup = _setup(image, _fake_node(image))
    task_env = {
        "OPENCLAW_STATE_DIR": str(image.workdir / "state"),
        "OPENCLAW_CONFIG_PATH": str(image.workdir / "openclaw.json"),
        "OPENCLAW_HOME": str(image.workdir),
        "STATE_DIRECTORY": str(image.workdir / "state"),
        "TMPDIR": str(image.workdir / "tmp"),
        "NODE_COMPILE_CACHE": str(image.workdir / "compile-cache"),
        "FAKE_NODE_LOG": str(image.log),
    }

    done = _install(image, setup, task_env)

    assert done.returncode == 0, done.stderr
    calls = [call for call in _calls(image.log) if call["argv"][0].endswith(("npm-cli.js", "openclaw.mjs"))]
    assert [call["argv"][0].rsplit("/", 1)[-1] for call in calls] == ["npm-cli.js", "openclaw.mjs"]
    for call in calls:
        assert call["home"] == f"{setup}/home"
        assert call["env"] == {
            "npm_config_cache": f"{setup}/cache/npm",
            "TMPDIR": f"{setup}/tmp",
            "NODE_COMPILE_CACHE": f"{setup}/cache/node-compile",
        }
    assert _tree(image.workdir) == [] and _tree(image.home) == []


def test_the_installer_never_publishes_a_runtime_that_holds_another_version(image, node_archive) -> None:
    setup = _setup(image, node_archive, version="2026.5.1")

    done = _install(image, setup)

    assert done.returncode != 0
    assert f"OpenClaw version mismatch: the runtime holds 2026.5.1, not {VERSION}" in done.stderr
    assert not image.runtime.exists() and not setup.exists()


def test_the_installer_keeps_a_runtime_another_seed_published(image, node_archive) -> None:
    """The other seed published between this seed's look and its rename; a session may already run from it."""
    first = _install(image, _setup(image, node_archive))
    assert first.returncode == 0, first.stderr
    (image.runtime / "session-file").write_text("in use")
    setup = _setup(image, node_archive)

    done = _install(image, setup)

    assert done.returncode == 0, done.stderr
    assert (image.runtime / "session-file").read_text() == "in use"
    assert sorted(os.listdir(image.runtime)) == ["node", "openclaw", "session-file"]
    assert not setup.exists()


def test_the_installer_reports_a_runtime_in_place_that_holds_another_version(image, node_archive) -> None:
    first = _install(image, _setup(image, node_archive))
    assert first.returncode == 0, first.stderr
    package = image.runtime / "openclaw/node_modules/openclaw/package.json"
    package.write_text(json.dumps({"version": "2026.5.1"}))

    done = _install(image, _setup(image, node_archive))

    # It is reported, not replaced: a session may be running from it.
    assert done.returncode != 0
    assert "OpenClaw version mismatch: the runtime holds 2026.5.1" in done.stderr
    assert json.loads(package.read_text()) == {"version": "2026.5.1"}


def test_the_installer_never_publishes_when_an_install_script_fails(image) -> None:
    setup = _setup(image, _fake_node(image))

    done = _install(image, setup, {"FAKE_NPM_EXIT": "9"})

    assert done.returncode == 9
    assert not image.runtime.exists() and not setup.exists()


def test_the_installer_never_publishes_a_node_that_does_not_run_in_the_sandbox(image) -> None:
    archive = image.root / "broken-node.tar.gz"
    _node_archive(archive, "linux-x64", node="#!/bin/sh\necho 'libstdc++.so.6: not found' >&2\nexit 127\n")
    setup = _setup(image, archive)

    done = _install(image, setup)

    assert done.returncode == 127
    assert "libstdc++.so.6: not found" in done.stderr
    assert not image.runtime.exists() and not setup.exists()
