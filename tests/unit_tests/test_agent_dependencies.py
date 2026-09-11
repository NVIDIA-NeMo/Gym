# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import tarfile
from unittest.mock import AsyncMock

import pytest

from nemo_gym.sandbox import SandboxExecResult
from nemo_gym.sandbox.agent_dependencies import build_source_archive, install_agent_dependencies
from nemo_gym.sandbox.agent_runtime_config import AgentRuntimeConfig


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    files = {
        "pyproject.toml": "[project]\nname = 'nemo-gym'\nversion = '1.0'\n",
        "README.md": "Gym",
        "LICENSE": "Apache-2.0",
        "nemo_gym/__init__.py": "",
        "nemo_gym/new_runtime.py": "# uncommitted runtime changes\n",
        "responses_api_agents/__init__.py": "",
        "responses_api_agents/example/app.py": "# selected harness\n",
        "responses_api_agents/example/requirements.txt": "-e nemo-gym @ ../../\n",
        "responses_api_agents/example/setup.sh": "echo setup\n",
        "responses_api_agents/example/.venv/secret.txt": "excluded",
        "responses_api_agents/example/data/rollouts.json": "excluded",
        "resources_servers/verifier/answers.json": "excluded",
        "env.yaml": "secret: excluded",
    }
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return root


def config(source):
    return AgentRuntimeConfig(dependencies={"source_root": str(source)})


def test_source_bundle_contains_current_harness_and_core_only(source, tmp_path):
    (source / "nemo_gym/escape.py").symlink_to(source / "env.yaml")
    archive = tmp_path / "source.tar.gz"
    relative = build_source_archive(config(source), "responses_api_agents.example.app:Agent", archive)
    assert relative == "responses_api_agents/example"
    with tarfile.open(archive) as bundle:
        names = set(bundle.getnames())
        assert "nemo_gym/new_runtime.py" in names
        assert "responses_api_agents/example/setup.sh" in names
        assert "responses_api_agents/example/requirements.txt" in names
        assert "responses_api_agents/__init__.py" in names
        assert "env.yaml" not in names
        assert "nemo_gym/escape.py" not in names
        assert not any("secret" in name or "rollouts" in name or "answers" in name for name in names)


@pytest.mark.parametrize("harness_path", ["../outside", "/tmp/outside"])
def test_rejects_source_escape(source, tmp_path, harness_path):
    runtime = AgentRuntimeConfig(dependencies={"source_root": str(source), "harness_path": harness_path})
    with pytest.raises(ValueError, match="relative directory"):
        build_source_archive(runtime, "custom:Agent", tmp_path / "source.tar.gz")


def test_rejects_ambiguous_dependencies(source, tmp_path):
    (source / "responses_api_agents/example/pyproject.toml").write_text("[project]\n")
    with pytest.raises(ValueError, match="exactly one"):
        build_source_archive(config(source), "responses_api_agents.example.app:Agent", tmp_path / "source.tar.gz")


async def test_provisions_isolated_python_and_installs_harness_manifest(source, tmp_path):
    sandbox = AsyncMock()
    sandbox.exec.return_value = SandboxExecResult(stdout="installed", stderr="", return_code=0)
    runtime = config(source)
    runtime.env = {"CUSTOM_INDEX_SETTING": "preserved"}
    python, env = await install_agent_dependencies(
        sandbox, runtime, "responses_api_agents.example.app:Agent", "/tmp/task-worker", tmp_path
    )
    assert python == "/tmp/task-worker/venv/bin/python"
    assert env["CUSTOM_INDEX_SETTING"] == "preserved"
    assert env["UV_PYTHON_INSTALL_DIR"] == "/tmp/task-worker/python"
    command = sandbox.exec.await_args.args[0]
    assert "venv --seed --python 3.13.14 /tmp/task-worker/venv" in command
    assert "cd /tmp/task-worker/source/responses_api_agents/example" in command
    assert "-r requirements.txt" in command
    assert "--override overrides.txt" in command
    assert "-e ." in command
    assert "--system" not in command
    assert sandbox.upload.await_args.args[1] == "/tmp/task-worker/source.tar.gz"


async def test_install_failure_reports_stderr(source, tmp_path):
    sandbox = AsyncMock()
    sandbox.exec.return_value = SandboxExecResult(stdout="", stderr="cannot download Python", return_code=1)
    with pytest.raises(RuntimeError, match="cannot download Python"):
        await install_agent_dependencies(
            sandbox, config(source), "responses_api_agents.example.app:Agent", "/tmp/task-worker", tmp_path
        )
