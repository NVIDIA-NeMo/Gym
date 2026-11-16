# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The pinned OpenCode CLI merges invocation prompt overrides with project config."""

import json
import os
import shutil
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemo_gym.config import ModelServerRef
from responses_api_agents.opencode_agent.tests.test_app import _make_agent
from responses_api_agents.pi_agent.tests.test_sandbox import _finishing_model, _http_server


async def test_pinned_opencode_merges_replacement_without_persisting_it(tmp_path):
    configured_binary = os.environ.get("GYM_TEST_OPENCODE_BINARY")
    binary = configured_binary or shutil.which("opencode")
    if binary is None:
        pytest.skip("Set GYM_TEST_OPENCODE_BINARY to the pinned OpenCode 1.17.11 executable")
    version = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=30)
    if configured_binary is None and version.stdout.strip() != "1.17.11":
        pytest.skip("PATH OpenCode is not 1.17.11")
    assert version.returncode == 0 and version.stdout.strip() == "1.17.11"
    repo = tmp_path / "repo"
    configured = {
        "default_agent": "plan",
        "agent": {"build": {"prompt": "configured prompt", "temperature": 0.25}},
        "tools": {"bash": False},
    }
    agent = _make_agent(
        repo_dir=str(repo),
        workspace_root=str(tmp_path / "work"),
        system_prompt_mode="replace",
        opencode_config=configured,
        env={
            "HOME": str(tmp_path / "home"),
            "XDG_CONFIG_HOME": str(tmp_path / "config"),
            "XDG_CACHE_HOME": str(tmp_path / "cache"),
            "XDG_STATE_HOME": str(tmp_path / "state"),
            "OPENCODE_DISABLE_AUTOUPDATE": "true",
            "OPENCODE_DISABLE_MODELS_FETCH": "true",
            "OPENCODE_CONFIG_CONTENT": json.dumps({"agent": {"build": {"top_p": 0.8}}}),
        },
    )
    process = MagicMock(returncode=0, communicate=AsyncMock(return_value=(b"", b"")))
    with (
        patch.object(agent, "_ensure_local_runtime", AsyncMock()),
        patch(
            "responses_api_agents.opencode_agent.app.asyncio.create_subprocess_exec", return_value=process
        ) as launch,
        patch("responses_api_agents.opencode_agent.app.parse_opencode_session", return_value=([], {})),
    ):
        await agent._run_opencode("user task", "G10 replacement marker", collect_observations=False)
    assert launch.call_args.args[launch.call_args.args.index("--agent") + 1] == "build"
    result = subprocess.run(
        [binary, "debug", "agent", "build"],
        cwd=repo,
        env=launch.call_args.kwargs["env"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    build = json.loads(result.stdout)
    assert build["prompt"] == "G10 replacement marker"
    assert build["temperature"] == 0.25 and build["topP"] == 0.8
    assert build["tools"]["bash"] is False
    persisted = json.loads((repo / "opencode.json").read_text())
    persisted.pop("$schema", None)  # The CLI adds its schema URL when loading the project.
    assert persisted == configured


async def test_pinned_opencode_replacement_selects_build_over_a_plan_default(tmp_path, monkeypatch):
    binary = os.environ.get("GYM_TEST_OPENCODE_BINARY") or shutil.which("opencode")
    if binary is None:
        pytest.skip("The pinned OpenCode 1.17.11 executable is required")
    version = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=30)
    if version.returncode != 0 or version.stdout.strip() != "1.17.11":
        pytest.skip("This integration targets OpenCode 1.17.11")
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.delenv(key, raising=False)
    requests = []
    with _http_server(_finishing_model(requests, [])) as model_url:
        agent = _make_agent(
            command=binary,
            repo_dir=str(tmp_path / "repo"),
            workspace_root=str(tmp_path / "work"),
            model="synthetic",
            model_server=ModelServerRef(type="responses_api_models", name="policy"),
            system_prompt_mode="replace",
            opencode_config={
                "default_agent": "plan",
                "agent": {"plan": {"prompt": "G10 plan marker"}},
                "tools": {"*": False},
                "small_model": "nemo/synthetic",
                "enabled_providers": ["nemo"],
            },
            env={
                "HOME": str(tmp_path / "home"),
                "XDG_CONFIG_HOME": str(tmp_path / "config"),
                "XDG_CACHE_HOME": str(tmp_path / "cache"),
                "XDG_STATE_HOME": str(tmp_path / "state"),
                "OPENCODE_DISABLE_AUTOUPDATE": "true",
                "OPENCODE_DISABLE_MODELS_FETCH": "true",
            },
        )
        with (
            patch.object(agent, "_ensure_local_runtime", AsyncMock()),
            patch.object(agent, "_resolve_model_base_url", return_value=model_url),
        ):
            await agent._run_opencode(
                "Synthetic user task", "G10 replacement marker", collect_observations=False, timeout=30
            )
    assert requests
    task_requests = [
        request
        for request in requests
        if any(
            message["role"] == "user" and "Synthetic user task" in json.dumps(message["content"])
            for message in request["messages"]
        )
    ]
    assert task_requests
    systems = [
        message["content"]
        for request in task_requests
        for message in request["messages"]
        if message["role"] == "system"
    ]
    assert any("G10 replacement marker" in system for system in systems)
    assert all("G10 plan marker" not in system for system in systems)
