# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import tarfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nemo_gym.sandbox import SandboxExecResult
from responses_api_agents.nooa_agent import sandbox_runtime


def test_archive_uses_current_source_without_private_files(tmp_path: Path) -> None:
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        (tmp_path / name).write_text(name)
    core = tmp_path / "nemo_gym"
    core.mkdir()
    (core / "module.py").write_text("current = True")
    (core / ".env").write_text("secret")
    (core / "secrets.json").write_text("secret")
    (core / "linked.py").symlink_to(core / ".env")
    agent = tmp_path / "responses_api_agents/nooa_agent"
    agent.mkdir(parents=True)
    (agent / "runner.py").write_text("current = True")
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
        SandboxExecResult("", "Unsupported target", 1),
    ]
    with pytest.raises(RuntimeError, match="Unsupported target"):
        await sandbox_runtime.prepare_nooa_runtime(sandbox)
