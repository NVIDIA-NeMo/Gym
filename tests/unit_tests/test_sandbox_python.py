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
    monkeypatch.setattr(sandbox_python, "CACHE_DIR", tmp_path / "cache")
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


class _Response:
    def __init__(self, body: bytes) -> None:
        self.body = body

    async def __aenter__(self) -> "_Response":
        return self

    async def __aexit__(self, *exc_info) -> None:
        return None

    def raise_for_status(self) -> None:
        pass

    async def read(self) -> bytes:
        return self.body


@pytest.fixture
def python_build(monkeypatch, tmp_path):
    """A download of ``body`` for the x86_64 glibc Python build, whose pinned checksum is ``pinned``."""
    build = SimpleNamespace(body=b"python build", pinned=hashlib.sha256(b"python build").hexdigest(), urls=[])
    monkeypatch.setattr(sandbox_python, "CACHE_DIR", tmp_path)

    async def request(method: str, url: str, **kwargs) -> _Response:
        build.urls.append(url)
        return _Response(build.body)

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
