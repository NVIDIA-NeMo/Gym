# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replacement prompts remain literal through OpenCode's configuration substitutions."""

import json
import os
import shutil
import subprocess
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from responses_api_agents.opencode_agent.app import _serialize_replacement_config
from responses_api_agents.opencode_agent.tests.test_app import _make_agent
from responses_api_agents.opencode_agent.tests.test_sandbox_sessions import seed, setup  # noqa: F401


@pytest.mark.parametrize(
    "prompt",
    [
        "{env:G10_TEMPLATE_VALUE}",
        "{file:existing-prompt.txt}",
        "{file:missing-prompt.txt}",
        '  λ\n"quoted" \\path\\end\n  ',
        r"literal \u007b and {env:G10_TEMPLATE_VALUE}",
        "{} {{env:G10_TEMPLATE_VALUE}} {ordinary braces}",
    ],
)
def test_only_the_replacement_prompt_is_protected_without_mutation(prompt):
    config = {
        "agent": {"build": {"prompt": prompt, "description": prompt}, "plan": {"prompt": prompt}},
        "username": prompt,
        "tools": {"bash": False},
    }
    original = deepcopy(config)
    serialized = _serialize_replacement_config(config)
    assert json.loads(serialized) == original
    assert config == original
    # Unrelated copies of identical text retain normal JSON serialization and native expansion.
    ordinary = json.dumps(prompt)
    assert ('"description":' + ordinary) in serialized
    assert ('"plan":' + json.dumps({"prompt": prompt})) in serialized
    assert ('"username":' + ordinary) in serialized


@pytest.fixture
def pinned_opencode():
    configured = os.environ.get("GYM_TEST_OPENCODE_BINARY")
    binary = configured or shutil.which("opencode")
    if binary is None:
        pytest.skip("Set GYM_TEST_OPENCODE_BINARY to OpenCode 1.17.11")
    version = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=30)
    if configured is None and (version.returncode != 0 or version.stdout.strip() != "1.17.11"):
        pytest.skip("This integration targets the pinned OpenCode 1.17.11")
    assert version.returncode == 0 and version.stdout.strip() == "1.17.11"
    return binary


@pytest.mark.parametrize("path", ["local", "sandbox"])
async def test_pinned_cli_keeps_the_generated_replacement_literal(path, setup, tmp_path, pinned_opencode):
    prompt = (
        '  λ\n"quoted" \\path\\end\n'
        "{env:G10_TEMPLATE_VALUE}\n{file:existing-prompt.txt}\n{file:missing-prompt.txt}\n"
        r"literal \u007b"
        "\n  "
    )
    project = tmp_path / "project"
    project.mkdir()
    (project / "existing-prompt.txt").write_text("included fixture")
    assert not (project / "missing-prompt.txt").exists()
    ordinary_prompt = "{env:G10_TEMPLATE_VALUE}::{file:existing-prompt.txt}"
    if path == "local":
        agent = _make_agent(
            repo_dir=str(project),
            workspace_root=str(tmp_path / "work"),
            system_prompt_mode="replace",
            env={
                "OPENCODE_CONFIG_CONTENT": json.dumps(
                    {"agent": {"plan": {"prompt": ordinary_prompt}}, "username": "{env:G10_TEMPLATE_VALUE}"}
                )
            },
        )
        process = MagicMock(returncode=0, communicate=AsyncMock(return_value=(b"", b"")))
        with (
            patch.object(agent, "_ensure_local_runtime", AsyncMock()),
            patch(
                "responses_api_agents.opencode_agent.app.asyncio.create_subprocess_exec", return_value=process
            ) as run,
            patch("responses_api_agents.opencode_agent.app.parse_opencode_session", return_value=([], {})),
        ):
            await agent._run_opencode("task", prompt, collect_observations=False)
        serialized = run.call_args.kwargs["env"]["OPENCODE_CONFIG_CONTENT"]
    else:
        agent, sandbox = setup
        agent.config.system_prompt_mode = "replace"
        agent.config.system_prompt = prompt
        with TestClient(agent.setup_webserver()) as client:
            client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).raise_for_status()
            client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"}).raise_for_status()
            serialized = sandbox.params()["env"]["OPENCODE_CONFIG_CONTENT"]
    assert json.loads(serialized)["agent"]["build"]["prompt"] == prompt
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path / "home"),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
        "OPENCODE_CONFIG_CONTENT": serialized,
        "OPENCODE_DISABLE_AUTOUPDATE": "true",
        "OPENCODE_DISABLE_MODELS_FETCH": "true",
        "G10_TEMPLATE_VALUE": "expanded fixture",
    }
    result = subprocess.run(
        [pinned_opencode, "--pure", "debug", "agent", "build"],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["prompt"] == prompt
    if path == "local":
        other = subprocess.run(
            [pinned_opencode, "--pure", "debug", "agent", "plan"],
            cwd=project,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert other.returncode == 0, other.stderr
        assert json.loads(other.stdout)["prompt"] == "expanded fixture::included fixture"
