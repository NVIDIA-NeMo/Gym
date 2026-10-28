# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sandbox Python runtimes: platform selection, the build on the Agent Server host, and publication in the sandbox."""

import asyncio
import hashlib
import io
import os
import shlex
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemo_gym.agent_utils import sandbox_python
from nemo_gym.agent_utils.sandbox_python import SandboxPlatform, SandboxPythonRuntime, probe_sandbox_platform
from nemo_gym.sandbox import AsyncSandbox, SandboxSpec
from nemo_gym.sandbox.providers.base import SandboxExecResult
from nemo_gym.sandbox.providers.local import LocalProvider


_SITE_PACKAGES = "python/lib/python3.13/site-packages"
_HERMES = "hermes-agent[mcp] @ https://github.com/owner/hermes-agent/archive/abc.tar.gz"


class _ProbedSandbox:
    def __init__(self, result: SandboxExecResult) -> None:
        self.result = result
        self.cwds: list[str | None] = []

    async def exec(self, command: str, *, cwd: str | None = None, timeout_s: float | None = None):
        self.cwds.append(cwd)
        return self.result


@pytest.mark.parametrize(
    "libc, expected",
    [
        ("glibc 2.35", SandboxPlatform("aarch64", "gnu", 35)),
        ("ldd (Ubuntu GLIBC 2.35-0ubuntu3) 2.35", SandboxPlatform("aarch64", "gnu", 35)),
        ("bash: ldd: command not found", SandboxPlatform("aarch64", "gnu", None)),
        ("musl", SandboxPlatform("aarch64", "musl")),
    ],
)
async def test_the_probe_reads_the_architecture_and_c_library(libc: str, expected: SandboxPlatform) -> None:
    sandbox = _ProbedSandbox(SandboxExecResult(stdout=f"Linux\narm64\n{libc}\n", stderr="", return_code=0))

    assert await probe_sandbox_platform(sandbox) == expected
    assert sandbox.cwds == ["/"]


@pytest.mark.parametrize(
    "stdout, return_code, match",
    [
        ("Darwin\narm64\nglibc 2.35\n", 0, "Darwin on arm64"),
        ("Linux\nriscv64\nglibc 2.35\n", 0, "riscv64"),
        ("", 127, "Could not determine the sandbox platform"),
    ],
)
async def test_the_probe_rejects_platforms_without_a_runtime(stdout: str, return_code: int, match: str) -> None:
    sandbox = _ProbedSandbox(SandboxExecResult(stdout=stdout, stderr="", return_code=return_code))

    with pytest.raises(RuntimeError, match=match):
        await probe_sandbox_platform(sandbox)


@pytest.mark.parametrize(
    "platform, expected",
    [
        (SandboxPlatform("x86_64", "gnu", 41), "x86_64-manylinux_2_40"),
        (SandboxPlatform("aarch64", "gnu", 35), "aarch64-manylinux_2_35"),
        (SandboxPlatform("x86_64", "gnu", 30), "x86_64-manylinux_2_28"),
        (SandboxPlatform("x86_64", "gnu", 27), "x86_64-manylinux_2_17"),
        (SandboxPlatform("x86_64", "gnu", None), "x86_64-manylinux_2_17"),
        (SandboxPlatform("aarch64", "musl"), "aarch64-unknown-linux-musl"),
    ],
)
def test_wheels_never_need_a_newer_glibc_than_the_sandbox_has(platform: SandboxPlatform, expected: str) -> None:
    assert platform.wheel_platform == expected


def test_a_glibc_older_than_the_python_build_needs_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="glibc 2.12"):
        SandboxPlatform("x86_64", "gnu", 12).wheel_platform


def test_pins_cover_packages_installed_from_an_index(monkeypatch) -> None:
    def distribution(name: str, version: str, direct_url: str | None = None) -> SimpleNamespace:
        return SimpleNamespace(metadata={"Name": name}, version=version, read_text=lambda _name: direct_url)

    monkeypatch.setattr(
        sandbox_python.importlib.metadata,
        "distributions",
        lambda: [
            distribution("Pillow", "12.3.0"),
            distribution("nemo-gym", "0.3.0", '{"url": "file:///gym", "dir_info": {"editable": true}}'),
            distribution("hermes-agent", "0.6.0", '{"url": "https://github.com/o/h.git", "vcs_info": {}}'),
            # A shadowed copy further down sys.path is not the one the server imports.
            distribution("pillow", "11.0.0"),
        ],
    )
    sandbox_python._agent_environment_pins.cache_clear()
    try:
        assert sandbox_python._agent_environment_pins() == ("Pillow==12.3.0",)
    finally:
        sandbox_python._agent_environment_pins.cache_clear()


def test_the_runtime_directory_is_keyed_by_its_inputs(monkeypatch) -> None:
    runtime = SandboxPythonRuntime(harness="hermes", requirements=("hermes-agent==1",), import_check="import x")

    assert runtime.directory.startswith("/tmp/nemo-gym-hermes-runtime-")
    assert runtime.python == f"{runtime.directory}/python/bin/python3"
    assert runtime.directory == SandboxPythonRuntime("hermes", ("hermes-agent==1",), "import x").directory
    assert runtime.directory != SandboxPythonRuntime("hermes", ("hermes-agent==2",), "import x").directory
    monkeypatch.setattr(sandbox_python, "_agent_environment_pins", lambda: ("openai==9.9.9",))
    assert runtime.directory != SandboxPythonRuntime("hermes", ("hermes-agent==1",), "import x").directory


@pytest.fixture
def host(monkeypatch, tmp_path):
    """A host whose uv records each install and writes a module into the target."""
    monkeypatch.setattr(sandbox_python, "_cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(sandbox_python.shutil, "which", lambda name: "/host/uv")
    monkeypatch.setattr(sandbox_python, "_agent_environment_pins", lambda: ("openai==1.2.3", "pillow==12.3.0"))
    host = SimpleNamespace(installs=[], fail=0)

    async def run(argv: list[str], *, cwd: Path) -> None:
        constraints = Path(argv[argv.index("--constraints") + 1]).read_text()
        overrides = Path(argv[argv.index("--overrides") + 1]).read_text() if "--overrides" in argv else None
        host.installs.append(SimpleNamespace(argv=argv, cwd=cwd, constraints=constraints, overrides=overrides))
        await asyncio.sleep(0.01)
        if host.fail:
            host.fail -= 1
            raise RuntimeError("uv exited with 2: no matching distribution")
        target = Path(argv[argv.index("--target") + 1])
        target.mkdir(parents=True)
        (target / "run_agent.py").write_text("")

    monkeypatch.setattr(sandbox_python, "_run", run)
    return host


async def test_the_host_installs_wheels_for_the_sandbox_platform(host, tmp_path) -> None:
    runtime = SandboxPythonRuntime(harness="hermes", requirements=(_HERMES,), import_check="import run_agent")

    archive = await runtime._site_packages_archive(SandboxPlatform("aarch64", "gnu", 31))

    (install,) = host.installs
    argv = install.argv
    assert argv[:3] == ["/host/uv", "pip", "install"]
    # Configuration files, including any in the directory uv runs from, never shape the install.
    assert "--no-config" in argv
    assert install.cwd.parent == tmp_path / "cache"
    assert argv[argv.index("--python-platform") + 1] == "aarch64-manylinux_2_31"
    assert argv[argv.index("--python-version") + 1] == sandbox_python.PYTHON_VERSION
    # Only the commit-pinned harness is built from source; every dependency must be a published wheel.
    assert argv[argv.index("--only-binary") + 1] == ":all:"
    # Files, not links into the host's uv cache, which the sandbox does not have.
    assert argv[argv.index("--link-mode") + 1] == "copy"
    assert [argv[i + 1] for i, arg in enumerate(argv) if arg == "--no-binary"] == ["hermes-agent"]
    assert argv[-1] == _HERMES
    # The agent environment's versions are constraints, so a dependency's own extras are still installed.
    assert install.constraints == "openai==1.2.3\npillow==12.3.0\n"
    assert install.overrides is None
    with tarfile.open(archive) as tar:
        assert f"{_SITE_PACKAGES}/run_agent.py" in tar.getnames()
    assert archive.parent == tmp_path / "cache" and [p.name for p in archive.parent.iterdir()] == [archive.name]

    assert await runtime._site_packages_archive(SandboxPlatform("aarch64", "gnu", 31)) == archive
    await runtime._site_packages_archive(SandboxPlatform("aarch64", "musl"))
    assert [install.argv[install.argv.index("--python-platform") + 1] for install in host.installs] == [
        "aarch64-manylinux_2_31",
        "aarch64-unknown-linux-musl",
    ]


async def test_concurrent_cold_seeds_share_one_build_and_a_failed_build_is_retried(host) -> None:
    runtime = SandboxPythonRuntime(harness="stirrup", requirements=("stirrup==0.1.12",), import_check="import x")
    platform = SandboxPlatform("x86_64", "gnu", 35)
    host.fail = 1

    with pytest.raises(RuntimeError, match="no matching distribution"):
        await runtime._site_packages_archive(platform)
    archives = await asyncio.gather(*(runtime._site_packages_archive(platform) for _ in range(3)))

    assert len(host.installs) == 2 and len(set(archives)) == 1
    assert "--no-binary" not in host.installs[-1].argv


async def test_the_adapter_overrides_reach_the_host_install(host) -> None:
    runtime = SandboxPythonRuntime(
        harness="stirrup", requirements=("stirrup==0.1.12",), import_check="import x", overrides=("pillow==12.3.0",)
    )

    await runtime._site_packages_archive(SandboxPlatform("x86_64", "gnu", 35))

    assert host.installs[0].overrides == "pillow==12.3.0\n"
    assert runtime.directory != SandboxPythonRuntime("stirrup", ("stirrup==0.1.12",), "import x").directory


async def test_the_host_build_requires_uv(host, monkeypatch) -> None:
    monkeypatch.setattr(sandbox_python.shutil, "which", lambda name: None)
    runtime = SandboxPythonRuntime(harness="stirrup", requirements=("stirrup==0.1.12",), import_check="import x")

    with pytest.raises(RuntimeError, match="requires uv on the Agent Server host"):
        await runtime._site_packages_archive(SandboxPlatform("x86_64", "gnu", 35))


async def test_concurrent_seeds_share_a_failed_build_and_a_cancelled_seed_leaves_the_build_running(host) -> None:
    runtime = SandboxPythonRuntime(harness="stirrup", requirements=("stirrup==0.1.12",), import_check="import x")
    platform = SandboxPlatform("x86_64", "gnu", 35)
    host.fail = 1

    # Seeds waiting on a build that fails get its error rather than each running it again in turn.
    failures = await asyncio.gather(
        *(runtime._site_packages_archive(platform) for _ in range(3)), return_exceptions=True
    )
    assert len(host.installs) == 1
    assert all(isinstance(failure, RuntimeError) for failure in failures)

    first = asyncio.create_task(runtime._site_packages_archive(platform))
    await asyncio.sleep(0)
    second = asyncio.create_task(runtime._site_packages_archive(platform))
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    assert (await second).is_file()
    assert len(host.installs) == 2


async def test_the_host_cache_is_rebuilt_when_the_runtime_inputs_change(host, monkeypatch) -> None:
    platform = SandboxPlatform("x86_64", "gnu", 35)

    def runtime(requirement: str = "stirrup==0.1.12", overrides: tuple[str, ...] = ()) -> SandboxPythonRuntime:
        return SandboxPythonRuntime("stirrup", (requirement,), "import x", overrides)

    first = await runtime()._site_packages_archive(platform)
    assert await runtime()._site_packages_archive(platform) == first
    upgraded = await runtime("stirrup==0.1.13")._site_packages_archive(platform)
    overridden = await runtime(overrides=("pillow==12.3.0",))._site_packages_archive(platform)
    monkeypatch.setattr(sandbox_python, "_agent_environment_pins", lambda: ("openai==9.9.9",))
    repinned = await runtime()._site_packages_archive(platform)

    assert len({first, upgraded, overridden, repinned}) == 4
    assert len(host.installs) == 4
    assert host.installs[-1].constraints == "openai==9.9.9\n"


def test_the_host_cache_lives_in_gyms_cache_dir(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(sandbox_python, "maybe_get_global_config_dict", lambda: {"cache_dir": str(tmp_path)})
    assert sandbox_python._cache_dir() == tmp_path / "sandbox_python"

    monkeypatch.setattr(sandbox_python, "maybe_get_global_config_dict", lambda: None)
    assert sandbox_python._cache_dir() == sandbox_python.CACHE_DIR / "sandbox_python"


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
def python_build(monkeypatch, tmp_path):
    """A download of ``body`` for the x86_64 glibc Python build, whose pinned checksum is ``pinned``."""
    build = SimpleNamespace(
        body=b"python build", status=200, pinned=hashlib.sha256(b"python build").hexdigest(), urls=[], requests=[]
    )
    monkeypatch.setattr(sandbox_python, "_cache_dir", lambda: tmp_path)
    monkeypatch.delenv("UV_PYTHON_INSTALL_MIRROR", raising=False)

    async def request(method: str, url: str, **kwargs) -> _Response:
        build.urls.append(url)
        build.requests.append(kwargs)
        return _Response(build.body, build.status)

    monkeypatch.setattr(sandbox_python, "request", request)
    monkeypatch.setattr(sandbox_python, "_PYTHON_SHA256", {"x86_64/gnu": build.pinned})
    return build


async def test_the_python_build_is_downloaded_once_and_checked_on_every_use(python_build) -> None:
    platform = SandboxPlatform("x86_64", "gnu", 35)

    path = await sandbox_python._python_archive(platform)

    assert path.read_bytes() == python_build.body
    assert await sandbox_python._python_archive(platform) == path
    assert python_build.urls == [
        "https://github.com/astral-sh/python-build-standalone/releases/download/20260805/"
        "cpython-3.13.14%2B20260805-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
    ]
    path.write_bytes(b"replaced")
    with pytest.raises(RuntimeError, match="does not match its pinned checksum"):
        await sandbox_python._python_archive(platform)


async def test_a_python_build_that_does_not_match_its_pin_is_not_cached(python_build, tmp_path) -> None:
    python_build.body = b"something else"

    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        await sandbox_python._python_archive(SandboxPlatform("x86_64", "gnu", 35))
    with pytest.raises(RuntimeError, match="No pinned Python build for x86_64/musl"):
        await sandbox_python._python_archive(SandboxPlatform("x86_64", "musl"))

    assert list(tmp_path.iterdir()) == []


async def test_a_failed_download_is_not_cached_and_the_next_seed_retries(python_build, tmp_path) -> None:
    platform = SandboxPlatform("x86_64", "gnu", 35)
    python_build.status = 404

    # An HTTP error carries no response_content, so it must not reach the server's middleware as one.
    with pytest.raises(RuntimeError, match="failed with HTTP 404"):
        await sandbox_python._python_archive(platform)
    assert list(tmp_path.iterdir()) == []

    python_build.status = 200
    assert (await sandbox_python._python_archive(platform)).read_bytes() == python_build.body
    # Connection failures end after a few attempts instead of retrying until the seed times out.
    assert [request["_max_num_tries"] for request in python_build.requests] == [3, 3]


async def test_the_python_build_comes_through_the_hosts_proxy_and_mirror(python_build, monkeypatch) -> None:
    for name in ("no_proxy", "NO_PROXY", "https_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.internal:3128")
    monkeypatch.setenv("UV_PYTHON_INSTALL_MIRROR", "https://mirror.internal/python-build-standalone/")

    await sandbox_python._python_archive(SandboxPlatform("x86_64", "gnu", 35))

    assert python_build.urls == [
        "https://mirror.internal/python-build-standalone/20260805/"
        "cpython-3.13.14%2B20260805-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
    ]
    assert python_build.requests[0]["proxy"] == "http://proxy.internal:3128"


def _add(tar: tarfile.TarFile, name: str, text: str = "", mode: int = 0o644) -> None:
    entry = tarfile.TarInfo(name)
    entry.size, entry.mode = len(text.encode()), mode
    tar.addfile(entry, io.BytesIO(text.encode()))


def _archives(directory: Path) -> tuple[Path, Path]:
    """Stand-ins for the host's two archives: this interpreter behind a ``python3`` wrapper, and one module."""
    directory.mkdir()
    # The wrapper puts the runtime's site-packages first, as the standalone interpreter's own site would.
    wrapper = (
        "#!/bin/sh\n"
        'here=$(cd "$(dirname "$0")/.." && pwd)\n'
        "shift 2\n"
        "code=$1\n"
        "shift\n"
        f"exec {shlex.quote(sys.executable)} -I -c \"import sys; sys.path.insert(0, '$here/lib/python3.13/"
        'site-packages\'); $code" "$@"\n'
    )
    with tarfile.open(directory / "python.tar.gz", "w:gz") as tar:
        _add(tar, "python/bin/python3", wrapper, mode=0o755)
    with tarfile.open(directory / "site-packages.tar.gz", "w:gz") as tar:
        _add(
            tar,
            f"{_SITE_PACKAGES}/fake_harness.py",
            "import pathlib\npathlib.Path.home().joinpath('.harness').mkdir(exist_ok=True)\n",
        )
    return directory / "python.tar.gz", directory / "site-packages.tar.gz"


@dataclass(frozen=True)
class _Runtime(SandboxPythonRuntime):
    root: str = ""

    @property
    def directory(self) -> str:
        return self.root


def _tree(root: Path) -> list[str]:
    return sorted(str(path.relative_to(root)) for path in root.rglob("*"))


@pytest.fixture
async def task(monkeypatch, tmp_path):
    """A local sandbox whose task brings a uv configuration and a ``uv`` on PATH that records being called."""
    workdir, home, bin_dir = tmp_path / "task", tmp_path / "home", tmp_path / "bin"
    for directory in (workdir, home, bin_dir):
        directory.mkdir()
    # Any uv run from the task directory would install from an index that does not exist.
    (workdir / "uv.toml").write_text('index-url = "http://127.0.0.1:9/simple"\n')
    (workdir / "pyproject.toml").write_text('[tool.uv]\nindex-url = "http://127.0.0.1:9/simple"\n')
    (bin_dir / "uv").write_text(f'#!/bin/sh\necho "$@" >> {shlex.quote(str(tmp_path / "uv-calls"))}\nexit 2\n')
    (bin_dir / "uv").chmod(0o755)
    python_archive, site_packages_archive = _archives(tmp_path / "archives")
    monkeypatch.setattr(
        sandbox_python, "probe_sandbox_platform", AsyncMock(return_value=SandboxPlatform("x86_64", "gnu"))
    )
    monkeypatch.setattr(sandbox_python, "_python_archive", AsyncMock(return_value=python_archive))
    monkeypatch.setattr(SandboxPythonRuntime, "_site_packages_archive", AsyncMock(return_value=site_packages_archive))
    sandbox = AsyncSandbox(LocalProvider())
    await sandbox.start(
        SandboxSpec(workdir=str(workdir), env={"HOME": str(home), "PATH": f"{bin_dir}:{os.environ['PATH']}"})
    )
    yield SimpleNamespace(sandbox=sandbox, root=tmp_path, workdir=workdir, home=home)
    await sandbox.stop()


def _runtime(task, import_check: str = "import fake_harness") -> _Runtime:
    return _Runtime(
        harness="fake", requirements=("fake-harness==1",), import_check=import_check, root=str(task.root / "rt")
    )


@pytest.mark.sandbox
async def test_seeding_ignores_the_task_uv_configuration_and_leaves_the_task_untouched(task) -> None:
    runtime = _runtime(task)
    workdir, home = _tree(task.workdir), _tree(task.home)

    assert not await runtime.is_installed(task.sandbox)
    await runtime.install(task.sandbox, timeout_s=60)

    assert await runtime.is_installed(task.sandbox)
    assert not (task.root / "uv-calls").exists()
    assert (_tree(task.workdir), _tree(task.home)) == (workdir, home)
    # The setup directory is gone; what the harness wrote at import stays in the runtime's own HOME.
    assert sorted(os.listdir(runtime.directory)) == ["home", "python"]
    assert (Path(runtime.directory) / "home/.harness").is_dir()


@pytest.mark.sandbox
async def test_concurrent_seeds_publish_one_runtime(task) -> None:
    runtime = _runtime(task)

    await asyncio.wait_for(asyncio.gather(*(runtime.install(task.sandbox, timeout_s=60) for _ in range(4))), 60)

    assert await runtime.is_installed(task.sandbox)
    assert sorted(os.listdir(runtime.directory)) == ["home", "python"]


@pytest.mark.sandbox
async def test_a_runtime_that_fails_its_check_is_replaced(task) -> None:
    runtime = _runtime(task)
    broken = Path(runtime.python)
    broken.parent.mkdir(parents=True)
    broken.write_text("#!/bin/sh\nexit 1\n")
    broken.chmod(0o755)

    assert not await runtime.is_installed(task.sandbox)
    await asyncio.wait_for(
        asyncio.gather(runtime.install(task.sandbox, timeout_s=60), runtime.install(task.sandbox, timeout_s=60)), 60
    )

    assert await runtime.is_installed(task.sandbox)
    assert sorted(os.listdir(runtime.directory)) == ["home", "python"]


@pytest.mark.sandbox
async def test_a_runtime_that_fails_its_check_is_never_published(task) -> None:
    runtime = _runtime(task, import_check="import fake_harness, missing_dependency")

    with pytest.raises(RuntimeError, match="No module named 'missing_dependency'"):
        await runtime.install(task.sandbox, timeout_s=60)

    assert os.listdir(runtime.directory) == ["home"]


async def _until(condition, timeout_s: float = 30) -> None:
    async with asyncio.timeout(timeout_s):
        while not condition():
            await asyncio.sleep(0.02)


@pytest.mark.sandbox
async def test_a_runtime_another_seed_accepted_is_never_set_aside(task) -> None:
    runtime = _runtime(task)
    gate, checking = task.root / "gate", task.root / "checking"
    # A broken runtime whose first check fails only once the test opens the gate; later checks fail at once.
    broken = Path(runtime.python)
    broken.parent.mkdir(parents=True)
    broken.write_text(
        "#!/bin/sh\n"
        f"if mkdir {shlex.quote(str(task.root / 'first-check'))} 2>/dev/null; then\n"
        f"    touch {shlex.quote(str(checking))}\n"
        f"    while [ ! -e {shlex.quote(str(gate))} ]; do sleep 0.02; done\n"
        "fi\n"
        "exit 1\n"
    )
    broken.chmod(0o755)

    first = asyncio.create_task(runtime.install(task.sandbox, timeout_s=60))
    await _until(checking.exists)
    # The second seed must not publish over the first seed's pending decision; if it does return, its runtime is
    # what its session runs, so it must survive the first seed's decision.
    second = asyncio.create_task(runtime.install(task.sandbox, timeout_s=60))
    done, _ = await asyncio.wait({second}, timeout=2)
    accepted = Path(runtime.directory) / "python/ACCEPTED"
    if done:
        await second
        accepted.touch()
    gate.touch()
    await asyncio.wait_for(asyncio.gather(first, second), 60)

    assert accepted.exists() or not done
    assert await runtime.is_installed(task.sandbox)
    assert sorted(os.listdir(runtime.directory)) == ["home", "python"]


@pytest.mark.sandbox
async def test_a_failed_upload_leaves_no_setup_directory_and_the_next_seed_installs(task, monkeypatch) -> None:
    runtime = _runtime(task)
    upload = task.sandbox.upload

    async def interrupted(source: Path, destination: str) -> None:
        if destination.endswith("/site-packages.tar.gz"):
            raise OSError("upload interrupted")
        await upload(source, destination)

    monkeypatch.setattr(task.sandbox, "upload", interrupted)
    with pytest.raises(OSError, match="upload interrupted"):
        await runtime.install(task.sandbox, timeout_s=60)
    # The Python archive had arrived; the attempt's setup directory is gone with it.
    assert os.listdir(runtime.directory) == []

    monkeypatch.setattr(task.sandbox, "upload", upload)
    await runtime.install(task.sandbox, timeout_s=60)
    assert await runtime.is_installed(task.sandbox)
    assert sorted(os.listdir(runtime.directory)) == ["home", "python"]


@pytest.mark.sandbox
async def test_a_seed_that_times_out_during_upload_removes_its_setup_directory(task, monkeypatch) -> None:
    runtime = _runtime(task)
    upload = task.sandbox.upload

    async def stalled(source: Path, destination: str) -> None:
        await upload(source, destination)
        if destination.endswith("/python.tar.gz"):
            await asyncio.Event().wait()

    monkeypatch.setattr(task.sandbox, "upload", stalled)
    with pytest.raises(TimeoutError, match="Installing the fake Python runtime took longer than 1 s"):
        await runtime.install(task.sandbox, timeout_s=1)

    assert os.listdir(runtime.directory) == []
    assert not await runtime.is_installed(task.sandbox)
