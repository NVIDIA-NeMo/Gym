# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coverage for the agent_network_mode == "block_target" pip/git/curl/wget/uv/python guard.

Two layers are tested:
  - the Python side (`_install_pip_git_guard`): discovers real tool paths, backs each up, and
    uploads the right wrapper template for each tool group.
  - the bash wrapper templates themselves, executed for real (not mocked) with a fake "real"
    binary standing in for pip/git/python/uv, to prove the actual guard logic - blanket substring
    match vs. package-fetch-shape-aware match - behaves as documented rather than just asserting
    that some script got uploaded.
"""

import os
import stat
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.server_utils import ServerClient
from resources_servers.nl2repobench.app import (
    _GUARD_SCRIPT_TEMPLATE,
    _GUARD_SCRIPT_TEMPLATE_MODULE_AWARE,
    NL2RepoBenchResourcesServer,
    NL2RepoBenchResourcesServerConfig,
)
from resources_servers.nl2repobench.task_schema import Task


def _task(tmp_path: Path, name: str = "six") -> Task:
    task_dir = tmp_path / "tasks" / name
    task_dir.mkdir(parents=True)
    (task_dir / "start.md").write_text(f"# {name}\n\nImplement a tiny package called {name}.\n", encoding="utf-8")
    (task_dir / "test_case_count.txt").write_text("5", encoding="utf-8")
    (task_dir / "test_commands.json").write_text('["pip install -e .", "pytest tests"]', encoding="utf-8")
    (task_dir / "test_files.json").write_text('["tests"]', encoding="utf-8")
    return Task(task_dir)


def _server(tmp_path: Path) -> NL2RepoBenchResourcesServer:
    config = NL2RepoBenchResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="nl2repobench_resources_server",
        tasks_dir=tmp_path / "tasks",
        expected_task_count=1,
        sandbox_provider="test",
        sandbox_config={},
    )
    return NL2RepoBenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _fake_real_binary(tmp_path: Path, *, name: str) -> Path:
    """A stand-in for the real pip/git/python/uv binary: records invocation and exits 0."""

    path = tmp_path / f"real-{name}"
    path.write_text(f'#!/usr/bin/env bash\necho REAL_CALLED:{name}:"$@"\n')
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def _render(template: str, *, target: str, real_path: Path, tool_name: str | None = None) -> Path:
    if tool_name is None:
        tool_name = real_path.name.removeprefix("real-")
    script_path = real_path.parent / f"wrapper-{real_path.name}"
    script_path.write_text(template.format(target=target, real_path=str(real_path), tool_name=tool_name))
    script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)
    return script_path


def _run(script_path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(script_path), *args],
        capture_output=True,
        text=True,
        env={**os.environ},
    )


# --- blanket template (pip/pip3/git/curl/wget) ------------------------------------------------


def test_blanket_template_blocks_target_substring(tmp_path: Path) -> None:
    real = _fake_real_binary(tmp_path, name="pip")
    wrapper = _render(_GUARD_SCRIPT_TEMPLATE, target="six", real_path=real)

    result = _run(wrapper, "install", "six")

    assert result.returncode == 1
    assert "blocked" in result.stderr
    assert "REAL_CALLED" not in result.stdout


def test_blanket_template_passes_through_unrelated_calls(tmp_path: Path) -> None:
    real = _fake_real_binary(tmp_path, name="pip")
    wrapper = _render(_GUARD_SCRIPT_TEMPLATE, target="six", real_path=real)

    result = _run(wrapper, "install", "requests")

    assert result.returncode == 0
    assert "REAL_CALLED:pip:install requests" in result.stdout


def test_blanket_template_normalizes_underscores_and_case(tmp_path: Path) -> None:
    real = _fake_real_binary(tmp_path, name="git")
    wrapper = _render(_GUARD_SCRIPT_TEMPLATE, target="python-jose", real_path=real)

    result = _run(wrapper, "clone", "https://github.com/mpdavis/Python_Jose")

    assert result.returncode == 1
    assert "blocked" in result.stderr


# --- module-aware template (python/python3.x/uv) ----------------------------------------------


def test_module_aware_template_blocks_python_dash_m_pip(tmp_path: Path) -> None:
    real = _fake_real_binary(tmp_path, name="python3")
    wrapper = _render(_GUARD_SCRIPT_TEMPLATE_MODULE_AWARE, target="six", real_path=real)

    result = _run(wrapper, "-m", "pip", "install", "six")

    assert result.returncode == 1
    assert "package-fetch shape" in result.stderr


def test_module_aware_template_blocks_uv_pip(tmp_path: Path) -> None:
    real = _fake_real_binary(tmp_path, name="uv")
    wrapper = _render(_GUARD_SCRIPT_TEMPLATE_MODULE_AWARE, target="six", real_path=real)

    result = _run(wrapper, "pip", "install", "six")

    assert result.returncode == 1
    assert "package-fetch shape" in result.stderr


def test_module_aware_template_does_not_block_pytest_mentioning_target(tmp_path: Path) -> None:
    """The whole point of the module-aware template: `python -m pytest tests/test_six.py` for
    task "six" must NOT be blocked - only the package-fetch shape (`-m pip` / `uv pip`) is
    guarded, not every invocation that happens to mention the target name."""

    real = _fake_real_binary(tmp_path, name="python3")
    wrapper = _render(_GUARD_SCRIPT_TEMPLATE_MODULE_AWARE, target="six", real_path=real)

    result = _run(wrapper, "-m", "pytest", "tests/test_six.py")

    assert result.returncode == 0
    assert "REAL_CALLED:python3:-m pytest tests/test_six.py" in result.stdout


def test_module_aware_template_does_not_block_plain_script_run(tmp_path: Path) -> None:
    real = _fake_real_binary(tmp_path, name="python3")
    wrapper = _render(_GUARD_SCRIPT_TEMPLATE_MODULE_AWARE, target="six", real_path=real)

    result = _run(wrapper, "setup.py", "install")

    assert result.returncode == 0
    assert "REAL_CALLED:python3:setup.py install" in result.stdout


def test_module_aware_template_uv_run_is_not_pkg_fetch_shape(tmp_path: Path) -> None:
    real = _fake_real_binary(tmp_path, name="uv")
    wrapper = _render(_GUARD_SCRIPT_TEMPLATE_MODULE_AWARE, target="six", real_path=real)

    result = _run(wrapper, "run", "pytest")

    assert result.returncode == 0
    assert "REAL_CALLED:uv:run pytest" in result.stdout


# --- _install_pip_git_guard: discovery/backup/upload orchestration -----------------------------


async def test_install_pip_git_guard_wraps_only_discovered_tools(tmp_path: Path) -> None:
    task = _task(tmp_path)
    server = _server(tmp_path)

    # Only pip and python3 "exist" in this sandbox; every other candidate tool is absent (empty
    # `command -v` line), mirroring a minimal image.
    discover_outputs = {
        server._BLANKET_GUARD_TOOLS: "\n".join(
            "/usr/bin/pip" if name == "pip" else "" for name in server._BLANKET_GUARD_TOOLS
        ),
        server._MODULE_AWARE_GUARD_TOOLS: "\n".join(
            "/usr/bin/python3" if name == "python3" else "" for name in server._MODULE_AWARE_GUARD_TOOLS
        ),
    }

    sandbox = AsyncMock()

    async def fake_exec(*, command: str, cwd: str, timeout_s: float):
        for tool_names, stdout in discover_outputs.items():
            if all(f'command -v "{n}"' in command for n in tool_names):
                return MagicMock(return_code=0, stdout=stdout, stderr="")
        # mv / chmod calls
        return MagicMock(return_code=0, stdout="", stderr="")

    sandbox.exec = AsyncMock(side_effect=fake_exec)
    uploads: list[tuple[Path, str]] = []

    async def fake_upload(local_path: Path, remote_path: str):
        uploads.append((Path(local_path).read_text(), remote_path))

    sandbox.upload = AsyncMock(side_effect=fake_upload)

    await server._install_pip_git_guard(sandbox, task)

    assert len(uploads) == 2
    (pip_script, pip_dest), (python_script, python_dest) = uploads
    assert pip_dest == "/usr/bin/pip"
    assert python_dest == "/usr/bin/python3"
    assert 'TARGET="six"' in pip_script
    assert 'REAL="/usr/bin/pip.nl2repobench-real"' in pip_script
    assert "package-fetch shape" in python_script  # module-aware template, not blanket

    mv_calls = [c.kwargs["command"] for c in sandbox.exec.await_args_list if c.kwargs["command"].startswith("mv ")]
    assert "mv /usr/bin/pip /usr/bin/pip.nl2repobench-real" in mv_calls
    assert "mv /usr/bin/python3 /usr/bin/python3.nl2repobench-real" in mv_calls


async def test_install_pip_git_guard_raises_on_discovery_failure(tmp_path: Path) -> None:
    task = _task(tmp_path)
    server = _server(tmp_path)
    sandbox = AsyncMock()
    sandbox.exec = AsyncMock(return_value=MagicMock(return_code=1, stdout="", stderr="no such tool"))

    with pytest.raises(RuntimeError, match="Failed to discover"):
        await server._install_pip_git_guard(sandbox, task)
