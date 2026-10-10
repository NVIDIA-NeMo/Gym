# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import tarfile
import tomllib
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nemo_gym.sandbox import SandboxExecResult
from responses_api_agents.nooa_agent import sandbox_runtime


def test_archive_uses_current_source_without_private_files(tmp_path: Path) -> None:
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
    blob, first = sandbox_runtime._runtime_archive(tmp_path)
    with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
        assert set(archive.getnames()) == {
            "pyproject.toml",
            "README.md",
            "LICENSE",
            "nemo_gym/module.py",
            "responses_api_agents/nooa_agent/runner.py",
            "runtime-requirements.txt",
        }
        manifest = tomllib.loads(archive.extractfile("pyproject.toml").read().decode())
        assert manifest["project"]["dependencies"] == ["openai==2.44.0", "requests"]
        requirements = archive.extractfile("runtime-requirements.txt").read().decode()
        assert "[dev]" not in requirements
        assert f"https://github.com/NVIDIA-NeMo/labs-OO-Agents/archive/{pin}.tar.gz" in requirements
    assert first == sandbox_runtime._runtime_archive(tmp_path)[1]
    (core / "module.py").write_text("current = False")
    assert first != sandbox_runtime._runtime_archive(tmp_path)[1]


@pytest.mark.asyncio
async def test_cached_runtime_requires_successful_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox_runtime, "_runtime_archive", lambda root: (b"archive", "fingerprint"))
    sandbox = AsyncMock()
    sandbox.exec.return_value = SandboxExecResult("", "", 0)
    python = await sandbox_runtime.prepare_nooa_runtime(sandbox)
    assert python == "/opt/nemo-gym-nooa/fingerprint/python/bin/python3"
    command = sandbox.exec.call_args.args[0]
    assert "/ready" in command and "import nooa" in command
    sandbox.upload.assert_not_called()


@pytest.mark.asyncio
async def test_failed_import_rebuilds_target_native_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox_runtime, "_runtime_archive", lambda root: (b"archive", "fingerprint"))
    sandbox = AsyncMock()
    sandbox.exec.side_effect = [
        SandboxExecResult("", "bad import", 1),
        SandboxExecResult("", "", 0),
        SandboxExecResult(None, None, 0),
        SandboxExecResult("", "", 0),
    ]
    await sandbox_runtime.prepare_nooa_runtime(sandbox)
    sandbox.upload.assert_awaited_once()
    command = sandbox.exec.call_args.args[0]
    assert "urllib.request.urlretrieve" in command
    assert "uname -m" in command and "/lib/ld-musl-" in command
    assert "$arch-unknown-linux-$libc" in command
    assert "--force-reinstall" in command
    assert "unset PYTHONHOME PYTHONPATH VIRTUAL_ENV" in command
    assert 'touch "$root/ready"' in command
    assert sandbox.exec.call_args.kwargs["cwd"] == "/"


@pytest.mark.asyncio
async def test_install_failure_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox_runtime, "_runtime_archive", lambda root: (b"archive", "fingerprint"))
    sandbox = AsyncMock()
    sandbox.exec.side_effect = [
        SandboxExecResult("", "", 1),
        SandboxExecResult("", "", 0),
        SandboxExecResult("", "", 0),
        SandboxExecResult("No matching distribution found for ray", "exit status 1", 1),
    ]
    with pytest.raises(RuntimeError, match="No matching distribution found for ray.*exit status 1"):
        await sandbox_runtime.prepare_nooa_runtime(sandbox)


@pytest.mark.asyncio
async def test_bare_image_gets_controller_download_without_system_install(monkeypatch, tmp_path: Path) -> None:
    from unittest.mock import MagicMock

    sandbox = AsyncMock()
    sandbox.exec.return_value = SandboxExecResult("x86_64/gnu\n", "", 0)

    async def chunks(size):
        assert size == 1024 * 1024
        yield b"portable-python-archive"

    response = MagicMock()
    response.content.iter_chunked = chunks
    fetch = AsyncMock(return_value=response)
    monkeypatch.setattr(sandbox_runtime, "request", fetch)
    monkeypatch.setattr(sandbox_runtime, "raise_for_status", AsyncMock())
    await sandbox_runtime._stage_python_if_needed(sandbox, "/opt/nemo-gym-nooa/runtime", tmp_path)
    assert fetch.await_args.args == (
        "GET",
        "https://github.com/astral-sh/python-build-standalone/releases/download/20260805/"
        "cpython-3.13.14+20260805-x86_64-unknown-linux-gnu-install_only.tar.gz",
    )
    local, remote = sandbox.upload.await_args.args
    assert local.read_bytes() == b"portable-python-archive"
    assert remote == "/opt/nemo-gym-nooa/runtime/python.tar.gz"
    response.release.assert_called_once()
    command = sandbox.exec.await_args.args[0]
    assert "apt" not in command and "apk" not in command


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["unexpected", "x86_64/gnu; rm -rf /"])
async def test_controller_download_rejects_unknown_target(monkeypatch, tmp_path: Path, target: str) -> None:
    sandbox = AsyncMock()
    sandbox.exec.return_value = SandboxExecResult(target, "", 0)
    fetch = AsyncMock()
    monkeypatch.setattr(sandbox_runtime, "request", fetch)
    with pytest.raises(RuntimeError, match="Unsupported NOOA Python target"):
        await sandbox_runtime._stage_python_if_needed(sandbox, "/opt/runtime", tmp_path)
    fetch.assert_not_called()
    sandbox.upload.assert_not_called()


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
