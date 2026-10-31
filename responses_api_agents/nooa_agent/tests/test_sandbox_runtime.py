# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import io
import json
import os
import shlex
import shutil
import subprocess
import tarfile
import tomllib
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nemo_gym.sandbox import SandboxExecResult
from responses_api_agents.nooa_agent import sandbox_runtime


@pytest.fixture
def source_tree(tmp_path: Path) -> Path:
    for name in ("README.md", "LICENSE"):
        (tmp_path / name).write_text(name)
    (tmp_path / "pyproject.toml").write_text('[project]\ndependencies = ["openai==2.44.0", "ray[default]>=2.58.0"]')
    core = tmp_path / "nemo_gym"
    core.mkdir()
    (core / "module.py").write_text("current = True")
    (core / ".env").write_text("secret")
    (core / "secrets.json").write_text("secret")
    (core / "linked.py").symlink_to(core / ".env")
    agent = tmp_path / "responses_api_agents/nooa_agent"
    agent.mkdir(parents=True)
    (agent / "runner.py").write_text("current = True")
    (agent / "install_nooa_runtime.sh").write_text("#!/bin/bash\necho packaged-installer\n")
    (agent / "runtime").mkdir()
    (agent / "runtime/pyproject.toml").write_text(
        "[project]\ndependencies = []\n[tool.nemo-gym-runtime]\n"
        'core-dependencies = ["openai"]\nadditional-dependencies = ["requests"]\n'
    )
    (agent / "results.json").write_text("secret")
    pin = "a" * 40
    (agent / "requirements.txt").write_text(
        f"-e nemo-gym[dev] @ ../../\njsonschema>=4\nnooa @ git+https://github.com/NVIDIA-NeMo/labs-OO-Agents.git@{pin}\n"
    )
    return tmp_path


def test_archive_uses_current_source_without_private_files(source_tree: Path) -> None:
    blob, fingerprint = sandbox_runtime._runtime_archive(source_tree)
    with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
        assert set(archive.getnames()) == {
            "pyproject.toml",
            "README.md",
            "LICENSE",
            "nemo_gym/module.py",
            "responses_api_agents/nooa_agent/runner.py",
            "responses_api_agents/nooa_agent/install_nooa_runtime.sh",
            "runtime-requirements.txt",
        }
        assert (
            archive.extractfile(sandbox_runtime._INSTALLER_PATH).read()
            == (source_tree / sandbox_runtime._INSTALLER_PATH).read_bytes()
        )
        requirements = archive.extractfile("runtime-requirements.txt").read().decode()
        assert "[dev]" not in requirements
        assert f"https://github.com/NVIDIA-NeMo/labs-OO-Agents/archive/{'a' * 40}.tar.gz" in requirements
    assert sandbox_runtime._runtime_archive(source_tree) == (blob, fingerprint)
    # A zero gzip timestamp keeps uploaded bytes stable across wall-clock time.
    assert blob[4:8] == bytes(4)
    (source_tree / "nemo_gym/module.py").write_text("current = False")
    updated_blob, updated_fingerprint = sandbox_runtime._runtime_archive(source_tree)
    assert updated_blob != blob
    assert updated_fingerprint != fingerprint


def test_installer_changes_invalidate_runtime_fingerprint(source_tree: Path) -> None:
    before, fingerprint = sandbox_runtime._runtime_archive(source_tree)
    installer = source_tree / sandbox_runtime._INSTALLER_PATH
    installer.write_text("#!/bin/bash\necho revised-installer\n")
    after, updated_fingerprint = sandbox_runtime._runtime_archive(source_tree)
    assert after != before
    assert updated_fingerprint != fingerprint
    with tarfile.open(fileobj=io.BytesIO(after)) as archive:
        assert archive.extractfile(sandbox_runtime._INSTALLER_PATH).read() == installer.read_bytes()


def test_expected_python_checksum_invalidates_runtime_fingerprint(
    monkeypatch: pytest.MonkeyPatch, source_tree: Path
) -> None:
    before, fingerprint = sandbox_runtime._runtime_archive(source_tree)
    monkeypatch.setitem(sandbox_runtime._PYTHON_SHA256, "x86_64/gnu", "0" * 64)
    after, updated_fingerprint = sandbox_runtime._runtime_archive(source_tree)
    assert after == before
    assert updated_fingerprint != fingerprint


def test_archive_selects_thin_runtime_dependencies(source_tree: Path) -> None:
    blob, _ = sandbox_runtime._runtime_archive(source_tree)
    with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
        manifest = tomllib.loads(archive.extractfile("pyproject.toml").read().decode())
    assert manifest["project"]["dependencies"] == ["openai==2.44.0", "requests"]
    root = Path(sandbox_runtime.__file__).resolve().parents[2]
    actual_blob, _ = sandbox_runtime._runtime_archive(root)
    with tarfile.open(fileobj=io.BytesIO(actual_blob)) as archive:
        actual_manifest = tomllib.loads(archive.extractfile("pyproject.toml").read().decode())
    core_dependencies = tomllib.loads((root / "pyproject.toml").read_text())["project"]["dependencies"]
    # server_utils imports httptools even in the otherwise thin worker process.
    assert (
        next(value for value in core_dependencies if value.startswith("httptools"))
        in actual_manifest["project"]["dependencies"]
    )


@pytest.fixture
def portable_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AsyncMock:
    from responses_api_agents.miniswe_sandboxed_agent import bootstrap

    payload = b"verified portable Python archive"
    archive = tmp_path / "cached-python.tar.gz"
    archive.write_bytes(payload)
    for target in sandbox_runtime._PYTHON_SHA256:
        monkeypatch.setitem(sandbox_runtime._PYTHON_SHA256, target, hashlib.sha256(payload).hexdigest())
    download = AsyncMock(return_value=archive)
    monkeypatch.setattr(bootstrap, "cached_download", download)
    return download


@pytest.fixture
def runtime_package(
    monkeypatch: pytest.MonkeyPatch, source_tree: Path, portable_download: AsyncMock
) -> tuple[bytes, str]:
    package = sandbox_runtime._runtime_archive(source_tree)
    monkeypatch.setattr(sandbox_runtime, "_runtime_archive", lambda root, **kwargs: package)
    return package


@pytest.fixture
def sandbox() -> AsyncMock:
    sandbox = AsyncMock()
    sandbox.uploaded = {}

    async def upload(local: Path, remote: str) -> None:
        # Snapshot bytes while the controller's temporary directory still exists.
        assert remote not in sandbox.uploaded
        sandbox.uploaded[remote] = local.read_bytes()

    sandbox.upload.side_effect = upload
    return sandbox


def probe_environment(directory: Path, *, arch: str, libc: str) -> dict[str, str]:
    # Execute the actual probe with controlled target facts, without depending on
    # the controller's architecture, libc, or OS.
    commands = {
        "uname": f'case "$1" in -s) echo Linux ;; -m) echo {arch} ;; *) exit 1 ;; esac',
        "getconf": "echo glibc 2.31" if libc == "gnu" else "exit 1",
        "ldd": "echo musl libc >&2; exit 1" if libc == "musl" else "echo GNU libc",
    }
    directory.mkdir()
    for name, contents in commands.items():
        command = directory / name
        command.write_text(f"#!/bin/sh\n{contents}\n")
        command.chmod(0o755)
    return {**os.environ, "PATH": str(directory)}


@pytest.mark.skipif(shutil.which("bash") is None, reason="Runtime probe requires bash")
@pytest.mark.parametrize("arch", ["x86_64", "aarch64"])
def test_installer_probe_accepts_glibc(tmp_path: Path, arch: str) -> None:
    installer = Path(sandbox_runtime.__file__).with_name("install_nooa_runtime.sh")
    bash = shutil.which("bash")
    assert bash is not None
    result = subprocess.run(
        [bash, str(installer), "probe"],
        env=probe_environment(tmp_path / "bin", arch=arch, libc="gnu"),
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{arch}/gnu\n"


@pytest.mark.asyncio
@pytest.mark.skipif(shutil.which("bash") is None, reason="Runtime probe requires bash")
@pytest.mark.parametrize("arch", ["x86_64", "aarch64"])
async def test_musl_probe_without_a_musl_bundle_fails_closed_before_downloads_or_uploads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sandbox: AsyncMock,
    source_tree: Path,
    portable_download: AsyncMock,
    arch: str,
) -> None:
    """The probe reports a musl image; a glibc-only bundle configuration must not fall back to the network."""
    installer = Path(sandbox_runtime.__file__).with_name("install_nooa_runtime.sh")
    (source_tree / sandbox_runtime._INSTALLER_PATH).write_bytes(installer.read_bytes())
    package = sandbox_runtime._runtime_archive(source_tree)
    monkeypatch.setattr(sandbox_runtime, "_runtime_archive", lambda root, **kwargs: package)
    env = probe_environment(tmp_path / "bin", arch=arch, libc="musl")
    bash = shutil.which("bash")
    assert bash is not None

    async def execute(command: str, **kwargs: object) -> SandboxExecResult:
        arguments = shlex.split(command)
        assert arguments[0] == "bash" and arguments[2:] == ["probe"]
        uploaded_probe = tmp_path / "uploaded-probe.sh"
        uploaded_probe.write_bytes(sandbox.uploaded[arguments[1]])
        result = subprocess.run(
            [bash, str(uploaded_probe), "probe"], env=env, capture_output=True, text=True, timeout=5
        )
        assert result.stdout.strip() == f"{arch}/musl"
        return SandboxExecResult(result.stdout, result.stderr, result.returncode)

    sandbox.exec.side_effect = execute
    with pytest.raises(RuntimeError, match="runtime_wheelhouse_musl"):
        await sandbox_runtime.prepare_nooa_runtime(sandbox, wheelhouse_path=tmp_path / "glibc-bundle")
    portable_download.assert_not_awaited()
    sandbox.upload.assert_awaited_once()
    assert sandbox.upload.await_args.args[1].endswith(".sh")
    sandbox.exec.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["x86_64/gnu", "aarch64/gnu"])
async def test_cached_runtime_requires_successful_health_check(
    sandbox: AsyncMock, runtime_package: tuple[bytes, str], portable_download: AsyncMock, target: str
) -> None:
    source, fingerprint = runtime_package
    sandbox.exec.side_effect = [
        SandboxExecResult(target + "\n", "", 0),
        SandboxExecResult("", "", 0),
        SandboxExecResult("", "", 0),
    ]
    python = await sandbox_runtime.prepare_nooa_runtime(sandbox)
    runtime = f"/opt/nemo-gym-nooa/{fingerprint}-{target.replace('/', '-')}"
    assert python == f"{runtime}/python/bin/python3"
    sandbox.upload.assert_awaited_once()
    script = sandbox.upload.await_args.args[1]
    with tarfile.open(fileobj=io.BytesIO(source)) as archive:
        assert sandbox.uploaded[script] == archive.extractfile(sandbox_runtime._INSTALLER_PATH).read()
    commands = [shlex.split(call.args[0]) for call in sandbox.exec.await_args_list]
    assert commands[0] == ["bash", script, "probe"]
    assert commands[1] == ["bash", script, "check", runtime, fingerprint, sandbox_runtime._IMPORT_CHECK]
    assert commands[2] == ["rm", "-rf", "--", script.removesuffix(".sh"), script]
    portable_download.assert_not_awaited()


@pytest.mark.asyncio
async def test_unhealthy_runtime_stages_verified_archives_and_installs(
    sandbox: AsyncMock, runtime_package: tuple[bytes, str], portable_download: AsyncMock
) -> None:
    source, fingerprint = runtime_package
    sandbox.exec.side_effect = [
        SandboxExecResult("x86_64/gnu", "", 0),
        SandboxExecResult("", "bad import", 1),
        SandboxExecResult("ready", "", 0),
        SandboxExecResult("", "", 0),
    ]
    python = await sandbox_runtime.prepare_nooa_runtime(sandbox)
    script = sandbox.upload.await_args_list[0].args[1]
    scratch = script.removesuffix(".sh")
    expected = sandbox_runtime._PYTHON_SHA256["x86_64/gnu"]
    portable_download.assert_awaited_once_with(
        "https://github.com/astral-sh/python-build-standalone/releases/download/20260805/"
        "cpython-3.13.14%2B20260805-x86_64-unknown-linux-gnu-install_only.tar.gz",
        sha256=expected,
    )
    assert set(sandbox.uploaded) == {script, f"{scratch}/source.tar.gz", f"{scratch}/python.tar.gz"}
    assert sandbox.uploaded[f"{scratch}/source.tar.gz"] == source
    assert hashlib.sha256(sandbox.uploaded[f"{scratch}/python.tar.gz"]).hexdigest() == expected
    install = sandbox.exec.await_args_list[2]
    assert shlex.split(install.args[0]) == [
        "bash",
        script,
        "install",
        python.removesuffix("/python/bin/python3"),
        fingerprint,
        hashlib.sha256(source).hexdigest(),
        expected,
        sandbox_runtime._IMPORT_CHECK,
    ]
    assert install.kwargs == {"cwd": "/", "timeout_s": 1800}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("target", "exit_code"),
    [
        ("unexpected", 0),
        ("x86_64/gnu; rm -rf /", 0),
        ("", 0),
        ("x86_64/gnu", 7),
        ("riscv64/gnu", 0),
        ("x86_64/uclibc", 0),
    ],
)
async def test_probe_rejects_invalid_target_before_downloading(
    sandbox: AsyncMock, runtime_package: tuple[bytes, str], portable_download: AsyncMock, target: str, exit_code: int
) -> None:
    sandbox.exec.return_value = SandboxExecResult(target, "probe diagnostic", exit_code)
    with pytest.raises(RuntimeError, match="Unsupported NOOA sandbox target") as error:
        await sandbox_runtime.prepare_nooa_runtime(sandbox)
    assert "probe diagnostic" in str(error.value)
    assert f"exit={exit_code}" in str(error.value)
    portable_download.assert_not_awaited()
    sandbox.upload.assert_awaited_once()
    sandbox.exec.assert_awaited_once()


@pytest.mark.asyncio
async def test_changed_cached_archive_is_rejected_before_payload_uploads(
    sandbox: AsyncMock, runtime_package: tuple[bytes, str], portable_download: AsyncMock
) -> None:
    portable_download.return_value.write_bytes(b"cache replaced after initial verification")
    sandbox.exec.side_effect = [SandboxExecResult("x86_64/gnu", "", 0), SandboxExecResult("", "", 1)]
    with pytest.raises(RuntimeError, match="NOOA Python artifact checksum mismatch before sandbox upload"):
        await sandbox_runtime.prepare_nooa_runtime(sandbox)
    sandbox.upload.assert_awaited_once()
    assert sandbox.exec.await_count == 2


@pytest.mark.asyncio
async def test_concurrent_preparations_use_distinct_upload_paths(
    sandbox: AsyncMock, runtime_package: tuple[bytes, str], portable_download: AsyncMock
) -> None:
    both_checking = asyncio.Barrier(2)

    async def execute(command: str, **kwargs: object) -> SandboxExecResult:
        arguments = shlex.split(command)
        if arguments[:1] == ["rm"]:
            return SandboxExecResult("", "", 0)
        if arguments[2] == "probe":
            return SandboxExecResult("aarch64/gnu", "", 0)
        if arguments[2] == "check":
            await both_checking.wait()
            return SandboxExecResult("", "", 1)
        assert arguments[2] == "install"
        await asyncio.sleep(0)
        return SandboxExecResult("ready", "", 0)

    sandbox.exec.side_effect = execute
    first, second = await asyncio.gather(
        sandbox_runtime.prepare_nooa_runtime(sandbox), sandbox_runtime.prepare_nooa_runtime(sandbox)
    )
    assert first == second
    assert first.endswith("-aarch64-gnu/python/bin/python3")
    scripts = [path for path in sandbox.uploaded if path.endswith(".sh")]
    assert len(scripts) == 2
    assert len(sandbox.uploaded) == 6
    for script in scripts:
        scratch = script.removesuffix(".sh")
        assert sandbox.uploaded[f"{scratch}/source.tar.gz"] == runtime_package[0]
        assert f"{scratch}/python.tar.gz" in sandbox.uploaded
    installs = [shlex.split(call.args[0]) for call in sandbox.exec.await_args_list if " install " in call.args[0]]
    assert {arguments[1] for arguments in installs} == set(scripts)


@pytest.mark.asyncio
async def test_failed_install_preserves_diagnostics_and_retry_uses_fresh_paths(
    sandbox: AsyncMock, runtime_package: tuple[bytes, str], portable_download: AsyncMock
) -> None:
    sandbox.exec.side_effect = [
        SandboxExecResult("x86_64/gnu", "", 0),
        SandboxExecResult("", "", 1),
        SandboxExecResult("No matching distribution found", "exit status 17", 17),
        SandboxExecResult("x86_64/gnu", "", 0),
        SandboxExecResult("", "", 1),
        SandboxExecResult("ready", "", 0),
        SandboxExecResult("", "", 0),
    ]
    with pytest.raises(RuntimeError, match="NOOA runtime preparation failed") as error:
        await sandbox_runtime.prepare_nooa_runtime(sandbox)
    first_script = sandbox.upload.await_args_list[0].args[1]
    failure = str(error.value)
    assert "exit=17" in failure
    assert "No matching distribution found" in failure
    assert "exit status 17" in failure
    assert f"diagnostics={first_script.removesuffix('.sh')}" in failure
    assert sandbox.exec.await_count == 3
    await sandbox_runtime.prepare_nooa_runtime(sandbox)
    second_script = sandbox.upload.await_args_list[3].args[1]
    assert second_script != first_script
    cleanup = shlex.split(sandbox.exec.await_args.args[0])
    assert cleanup == ["rm", "-rf", "--", second_script.removesuffix(".sh"), second_script]
    assert len(sandbox.uploaded) == 6


@pytest.mark.asyncio
async def test_cancelled_install_retains_remote_diagnostics(
    sandbox: AsyncMock, runtime_package: tuple[bytes, str], portable_download: AsyncMock
) -> None:
    sandbox.exec.side_effect = [
        SandboxExecResult("x86_64/gnu", "", 0),
        SandboxExecResult("", "", 1),
        asyncio.CancelledError(),
    ]
    with pytest.raises(asyncio.CancelledError):
        await sandbox_runtime.prepare_nooa_runtime(sandbox)
    assert sandbox.exec.await_count == 3
    assert len(sandbox.uploaded) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup", [SandboxExecResult("", "permission denied", 1), OSError("transport unavailable")])
async def test_cleanup_failure_does_not_discard_healthy_runtime(
    sandbox: AsyncMock,
    runtime_package: tuple[bytes, str],
    portable_download: AsyncMock,
    cleanup: SandboxExecResult | OSError,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sandbox.exec.side_effect = [SandboxExecResult("x86_64/gnu", "", 0), SandboxExecResult("", "", 0), cleanup]
    python = await sandbox_runtime.prepare_nooa_runtime(sandbox)
    assert python == f"/opt/nemo-gym-nooa/{runtime_package[1]}-x86_64-gnu/python/bin/python3"
    assert "Could not remove NOOA setup files" in caplog.text
    assert sandbox.upload.await_args.args[1].removesuffix(".sh") in caplog.text
    portable_download.assert_not_awaited()


def test_runtime_entrypoint_import_does_not_load_controller_dependencies() -> None:
    import subprocess
    import sys

    # A fresh interpreter catches imports hidden by a test process's module cache.
    script = """
import importlib.abc
import sys
class RejectControllerDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'ray', 'datasets', 'pyarrow', 'pandas', 'mlflow'}:
            raise RuntimeError('task runtime imported controller dependency: ' + fullname)
sys.meta_path.insert(0, RejectControllerDependencies())
from responses_api_agents.nooa_agent.sandbox_entrypoint import SandboxInput
from responses_api_agents.nooa_agent.invocation import NOOAInvocationConfig, validate_invocation
from responses_api_agents.nooa_agent.task_agent import TaskAgent
config = NOOAInvocationConfig(agent_class='responses_api_agents.nooa_agent.task_agent:TaskAgent',
    invocation_adapter='responses_api_agents.nooa_agent.task_agent:invoke_solve_task')
assert validate_invocation(config)[0] is TaskAgent
"""
    subprocess.run([sys.executable, "-c", script], cwd=Path(sandbox_runtime.__file__).resolve().parents[2], check=True)


@pytest.fixture
def wheelhouse(tmp_path: Path, runtime_package: tuple[bytes, str], portable_download: AsyncMock) -> Path:
    path = tmp_path / "wheelhouse"
    path.mkdir()
    (path / "python.tar.gz").write_bytes(portable_download.return_value.read_bytes())
    with tarfile.open(path / "wheels.tar.gz", "w:gz") as archive:
        data = b"fixture wheel bytes"
        member = tarfile.TarInfo("fixture-1-py3-none-any.whl")
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
    manifest = {
        "version": 1,
        "target": "x86_64/gnu",
        "source_fingerprint": runtime_package[1],
        "python_version": sandbox_runtime._PYTHON_VERSION,
        "python_sha256": sandbox_runtime._PYTHON_SHA256["x86_64/gnu"],
        "wheels_sha256": hashlib.sha256((path / "wheels.tar.gz").read_bytes()).hexdigest(),
        "glibc": "2.28",
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    return path


@pytest.mark.asyncio
async def test_offline_runtime_stages_only_verified_local_artifacts(
    sandbox: AsyncMock, wheelhouse: Path, runtime_package: tuple[bytes, str], portable_download: AsyncMock
) -> None:
    sandbox.exec.side_effect = [
        SandboxExecResult("x86_64/gnu", "", 0),
        SandboxExecResult("glibc 2.31", "", 0),
        SandboxExecResult("", "", 1),
        SandboxExecResult("ready", "", 0),
        SandboxExecResult("", "", 0),
    ]
    python = await sandbox_runtime.prepare_nooa_runtime(sandbox, wheelhouse_path=wheelhouse)
    portable_download.assert_not_awaited()
    script = sandbox.upload.await_args_list[0].args[1]
    scratch = script.removesuffix(".sh")
    assert sandbox.uploaded[f"{scratch}/wheels.tar.gz"] == (wheelhouse / "wheels.tar.gz").read_bytes()
    assert sandbox.uploaded[f"{scratch}/python.tar.gz"] == (wheelhouse / "python.tar.gz").read_bytes()
    command = shlex.split(sandbox.exec.await_args_list[3].args[0])
    wheels_sha = json.loads((wheelhouse / "manifest.json").read_text())["wheels_sha256"]
    expected_fingerprint = hashlib.sha256(f"{runtime_package[1]}:{wheels_sha}".encode()).hexdigest()
    assert command[2] == "install"
    assert command[4] == expected_fingerprint
    assert command[-1] == wheels_sha
    assert python == f"/opt/nemo-gym-nooa/{expected_fingerprint}-x86_64-gnu/python/bin/python3"


def musl_wheelhouse(path: Path, gnu_wheelhouse: Path) -> Path:
    """A musl bundle shaped like the glibc fixture's, with the musl Python digest and a musl manifest."""
    path.mkdir()
    (path / "python.tar.gz").write_bytes((gnu_wheelhouse / "python.tar.gz").read_bytes())
    (path / "wheels.tar.gz").write_bytes((gnu_wheelhouse / "wheels.tar.gz").read_bytes())
    manifest = json.loads((gnu_wheelhouse / "manifest.json").read_text())
    manifest.pop("glibc")
    manifest.update(
        target="x86_64/musl",
        libc="musl",
        libc_version="1.2.3",
        python_sha256=sandbox_runtime._PYTHON_SHA256["x86_64/musl"],
    )
    (path / "manifest.json").write_text(json.dumps(manifest))
    return path


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["1.2.5", "1.2.3", "1.2.4_git20230717"])
async def test_offline_runtime_selects_the_musl_bundle_for_a_musl_image(
    version: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sandbox: AsyncMock,
    wheelhouse: Path,
    runtime_package: tuple[bytes, str],
    portable_download: AsyncMock,
) -> None:
    monkeypatch.setitem(
        sandbox_runtime._PYTHON_SHA256,
        "x86_64/musl",
        hashlib.sha256((wheelhouse / "python.tar.gz").read_bytes()).hexdigest(),
    )
    musl = musl_wheelhouse(tmp_path / "musl-wheelhouse", wheelhouse)
    sandbox.exec.side_effect = [
        SandboxExecResult("x86_64/musl", "", 0),
        SandboxExecResult("", f"musl libc (x86_64)\nVersion {version}\nDynamic Program Loader\n", 1),
        SandboxExecResult("", "", 1),
        SandboxExecResult("ready", "", 0),
        SandboxExecResult("", "", 0),
    ]
    python = await sandbox_runtime.prepare_nooa_runtime(sandbox, wheelhouse_path=wheelhouse, musl_wheelhouse_path=musl)
    portable_download.assert_not_awaited()
    assert sandbox.exec.await_args_list[1].args[0] == "ldd --version"
    wheels_sha = json.loads((musl / "manifest.json").read_text())["wheels_sha256"]
    expected_fingerprint = hashlib.sha256(f"{runtime_package[1]}:{wheels_sha}".encode()).hexdigest()
    assert python == f"/opt/nemo-gym-nooa/{expected_fingerprint}-x86_64-musl/python/bin/python3"
    script = sandbox.upload.await_args_list[0].args[1]
    scratch = script.removesuffix(".sh")
    assert sandbox.uploaded[f"{scratch}/wheels.tar.gz"] == (musl / "wheels.tar.gz").read_bytes()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ldd", ["musl libc (x86_64)\nVersion 1.1.24", "musl libc (x86_64)\nVersion 1.2.2_git20220101", "GNU libc 2.31", ""]
)
async def test_offline_runtime_rejects_an_old_or_unknown_musl_without_online_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sandbox: AsyncMock,
    wheelhouse: Path,
    portable_download: AsyncMock,
    ldd: str,
) -> None:
    monkeypatch.setitem(
        sandbox_runtime._PYTHON_SHA256,
        "x86_64/musl",
        hashlib.sha256((wheelhouse / "python.tar.gz").read_bytes()).hexdigest(),
    )
    musl = musl_wheelhouse(tmp_path / "musl-wheelhouse", wheelhouse)
    sandbox.exec.side_effect = [SandboxExecResult("x86_64/musl", "", 0), SandboxExecResult("", ldd, 1)]
    with pytest.raises(RuntimeError, match="requires musl >= 1.2.3"):
        await sandbox_runtime.prepare_nooa_runtime(sandbox, wheelhouse_path=wheelhouse, musl_wheelhouse_path=musl)
    portable_download.assert_not_awaited()
    sandbox.upload.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("libc", ["glibc 2.17", "unknown", "glibc 2.31 extra"])
async def test_offline_runtime_rejects_incompatible_libc_without_online_fallback(
    sandbox: AsyncMock, wheelhouse: Path, portable_download: AsyncMock, libc: str
) -> None:
    sandbox.exec.side_effect = [SandboxExecResult("x86_64/gnu", "", 0), SandboxExecResult(libc, "", 0)]
    with pytest.raises(RuntimeError, match="requires glibc >= 2.28"):
        await sandbox_runtime.prepare_nooa_runtime(sandbox, wheelhouse_path=wheelhouse)
    portable_download.assert_not_awaited()
    sandbox.upload.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ["source", "target", "python", "wheels", "missing"])
async def test_offline_runtime_rejects_incompatible_or_damaged_bundle_without_online_fallback(
    sandbox: AsyncMock, wheelhouse: Path, portable_download: AsyncMock, changed: str
) -> None:
    if changed in {"source", "target"}:
        manifest = json.loads((wheelhouse / "manifest.json").read_text())
        manifest["source_fingerprint" if changed == "source" else "target"] = "changed"
        (wheelhouse / "manifest.json").write_text(json.dumps(manifest))
    elif changed == "missing":
        (wheelhouse / "wheels.tar.gz").unlink()
    else:
        (wheelhouse / f"{changed}.tar.gz").write_bytes(b"replaced")
    sandbox.exec.return_value = SandboxExecResult("x86_64/gnu", "", 0)
    with pytest.raises(ValueError, match="NOOA wheelhouse"):
        await sandbox_runtime.prepare_nooa_runtime(sandbox, wheelhouse_path=wheelhouse)
    portable_download.assert_not_awaited()
    sandbox.upload.assert_awaited_once()
    sandbox.exec.assert_awaited_once()


@pytest.mark.asyncio
async def test_offline_runtime_revalidates_private_wheel_snapshot(
    monkeypatch: pytest.MonkeyPatch, sandbox: AsyncMock, wheelhouse: Path, portable_download: AsyncMock
) -> None:
    copyfile = sandbox_runtime.shutil.copyfile

    def replace_during_copy(source: Path, target: Path) -> Path:
        result = copyfile(source, target)
        if source.name == "wheels.tar.gz":
            target.write_bytes(b"replaced while copying")
        return result

    monkeypatch.setattr(sandbox_runtime.shutil, "copyfile", replace_during_copy)
    sandbox.exec.side_effect = [
        SandboxExecResult("x86_64/gnu", "", 0),
        SandboxExecResult("glibc 2.28", "", 0),
        SandboxExecResult("", "", 1),
    ]
    with pytest.raises(RuntimeError, match="wheelhouse changed before sandbox upload"):
        await sandbox_runtime.prepare_nooa_runtime(sandbox, wheelhouse_path=wheelhouse)
    portable_download.assert_not_awaited()
    sandbox.upload.assert_awaited_once()


@pytest.mark.asyncio
async def test_runtime_packaging_does_not_block_other_requests(
    monkeypatch: pytest.MonkeyPatch, sandbox: AsyncMock, runtime_package: tuple[bytes, str]
) -> None:
    import threading

    started = threading.Event()
    release = threading.Event()
    timed_out = threading.Event()

    def slow_archive(root: Path) -> tuple[bytes, str]:
        started.set()
        if not release.wait(timeout=2):
            timed_out.set()
        return runtime_package

    monkeypatch.setattr(sandbox_runtime, "_runtime_archive", slow_archive)
    sandbox.exec.side_effect = [
        SandboxExecResult("x86_64/gnu", "", 0),
        SandboxExecResult("", "", 0),
        SandboxExecResult("", "", 0),
    ]
    preparation = asyncio.create_task(sandbox_runtime.prepare_nooa_runtime(sandbox))
    try:
        assert await asyncio.to_thread(started.wait, 3)
        assert not preparation.done()
        assert not timed_out.is_set()
        # Other requests reach the loop while packaging is still blocked.
        await asyncio.sleep(0)
    finally:
        release.set()
        await preparation


@pytest.mark.parametrize("root_name,runtime_name", [("pydantic_core", "Pydantic-Core"), ("hydra-core", "hydra_core")])
def test_archive_normalizes_dependency_names(source_tree: Path, root_name: str, runtime_name: str) -> None:
    (source_tree / "pyproject.toml").write_text(f'[project]\ndependencies = ["{root_name}>=1"]')
    (source_tree / "responses_api_agents/nooa_agent/runtime/pyproject.toml").write_text(
        "[project]\ndependencies = []\n[tool.nemo-gym-runtime]\n"
        f'core-dependencies = ["{runtime_name}"]\nadditional-dependencies = []\n'
    )
    blob, _ = sandbox_runtime._runtime_archive(source_tree)
    with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
        contents = tomllib.loads(archive.extractfile("pyproject.toml").read().decode())
    assert contents["project"]["dependencies"] == [f"{root_name}>=1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["validate", "snapshot"])
async def test_bundle_work_keeps_loop_responsive_and_joins_cancelled_workers(
    monkeypatch: pytest.MonkeyPatch, sandbox: AsyncMock, wheelhouse: Path, stage: str
) -> None:
    import threading

    from responses_api_agents.nooa_agent import offline_runtime

    entered = threading.Event()
    release = threading.Event()
    completed = threading.Event()
    snapshot_paths = []
    module = offline_runtime if stage == "validate" else sandbox_runtime
    name = "load_runtime_wheelhouse" if stage == "validate" else "_stage_runtime_artifacts"
    original = getattr(module, name)

    def blocked(*args, **kwargs):
        if stage == "snapshot":
            snapshot_paths.append(args[0])
        entered.set()
        assert release.wait(5), "Event loop stalled during bundle work"
        result = original(*args, **kwargs)
        completed.set()
        return result

    monkeypatch.setattr(module, name, blocked)
    sandbox.exec.side_effect = [
        SandboxExecResult("x86_64/gnu", "", 0),
        SandboxExecResult("glibc 2.28", "", 0),
        SandboxExecResult("", "", 1),
    ]
    preparation = asyncio.create_task(sandbox_runtime.prepare_nooa_runtime(sandbox, wheelhouse_path=wheelhouse))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        preparation.cancel()
        await asyncio.sleep(0.01)
        assert not preparation.done()
        assert not completed.is_set()
        for path in snapshot_paths:
            assert path.exists()
        preparation.cancel()  # Repeated cancellation still cannot race the worker's files.
        await asyncio.sleep(0.01)
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await preparation
    assert completed.is_set()
    assert all(not path.exists() for path in snapshot_paths)
    assert sandbox.upload.await_count == 1


@pytest.mark.asyncio
async def test_runtime_io_bounds_concurrent_bundle_work() -> None:
    import threading

    lock = threading.Lock()
    release = threading.Event()
    two_started = threading.Event()
    active = 0
    peak = 0

    def work() -> None:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            if active == 2:
                two_started.set()
        assert release.wait(5)
        with lock:
            active -= 1

    tasks = [asyncio.create_task(sandbox_runtime._runtime_io(work)) for _ in range(6)]
    try:
        assert await asyncio.to_thread(two_started.wait, 3)
        await asyncio.sleep(0.01)
        assert peak == active == 2
    finally:
        release.set()
        await asyncio.gather(*tasks)
    assert peak == 2


@pytest.mark.asyncio
async def test_glibc_217_bundle_accepts_ubuntu_1804_task_image(
    sandbox: AsyncMock, wheelhouse: Path, portable_download: AsyncMock
) -> None:
    manifest_path = wheelhouse / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["glibc"] = "2.17"
    manifest_path.write_text(json.dumps(manifest))
    sandbox.exec.side_effect = [
        SandboxExecResult("x86_64/gnu", "", 0),
        SandboxExecResult("glibc 2.27", "", 0),
        SandboxExecResult("", "", 0),
        SandboxExecResult("", "", 0),
    ]
    assert (await sandbox_runtime.prepare_nooa_runtime(sandbox, wheelhouse_path=wheelhouse)).endswith("/bin/python3")
    portable_download.assert_not_awaited()
