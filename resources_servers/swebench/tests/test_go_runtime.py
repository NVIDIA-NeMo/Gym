# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Exercise Go discovery in isolated command paths and the resources-server setup hook."""

import importlib
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from resources_servers.swebench import go_runtime


class LocalSandbox:
    def __init__(self, path: str) -> None:
        self.path = path

    async def exec(self, command: str, **kwargs: object) -> SimpleNamespace:
        result = subprocess.run(
            ["/bin/bash", "-c", command],
            env={**os.environ, "PATH": self.path},
            capture_output=True,
            text=True,
            timeout=10,
        )
        return SimpleNamespace(
            return_code=result.returncode, stdout=result.stdout, stderr=result.stderr, error_type=None
        )


def isolated_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    installed = tmp_path / "go" / "bin" / "go"
    installed.parent.mkdir(parents=True)
    commands = tmp_path / "commands"
    commands.mkdir()
    script = go_runtime._GO_SETUP
    for source in ("/usr/lib/go/bin/go", "/usr/lib/go-*/bin/go", "/opt/go/bin/go"):
        script = script.replace(source, str(tmp_path / "absent"))
    script = script.replace("/usr/local/go/bin/go", str(installed)).replace("/usr/local/bin", str(commands))
    monkeypatch.setattr(go_runtime, "_GO_SETUP", script)
    return installed, commands


@pytest.mark.asyncio
async def test_installed_compiler_becomes_visible_to_subsequent_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installed, commands = isolated_setup(tmp_path, monkeypatch)
    installed.write_text("#!/bin/sh\necho installed-go\n")
    installed.chmod(0o755)
    sandbox = LocalSandbox(f"{commands}:/usr/bin:/bin")
    assert (await sandbox.exec("go version")).return_code == 127
    await go_runtime.ensure_go_command(sandbox)
    await go_runtime.ensure_go_command(sandbox)
    assert (await sandbox.exec("go version")).stdout.strip() == "installed-go"
    assert (commands / "go").resolve() == installed


@pytest.mark.asyncio
async def test_already_selected_go_is_preserved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    installed, commands = isolated_setup(tmp_path, monkeypatch)
    selected = commands / "go"
    selected.write_text("#!/bin/sh\necho selected-go\n")
    selected.chmod(0o755)
    await go_runtime.ensure_go_command(LocalSandbox(f"{commands}:/usr/bin:/bin"))
    assert not selected.is_symlink()
    assert "selected-go" in selected.read_text()
    assert not installed.exists()


@pytest.mark.asyncio
async def test_missing_compiler_rejects_the_sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, commands = isolated_setup(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="Go runtime setup failed"):
        await go_runtime.ensure_go_command(LocalSandbox(f"{commands}:/usr/bin:/bin"))


@pytest.mark.asyncio
@pytest.mark.parametrize("route,prefix", [("swemer_v1", "SwemerV1"), ("swemer_v2", "SwemerV2")])
@pytest.mark.parametrize("framework", ["go", "pytest"])
async def test_after_start_prepares_only_go_tasks(
    route: str, prefix: str, framework: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module(f"resources_servers.{route}.app")
    setup = AsyncMock()
    monkeypatch.setattr(module, "ensure_go_command", setup)
    sandbox = object()
    await getattr(module, prefix + "ResourcesServer")._after_start(
        None, sandbox, SimpleNamespace(test_framework=framework)
    )
    if framework == "go":
        setup.assert_awaited_once_with(sandbox)
    else:
        setup.assert_not_awaited()
