# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A ``/run`` body without ``model`` (as ``prepare.py`` writes rows) must still produce a response."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from responses_api_agents.anyswe_agent.app import AnySweAgent, AnySweInstanceConfig
from responses_api_agents.anyswe_agent.prepare import _to_gym_row
from responses_api_agents.anyswe_agent.tests.test_app import _config


class _FakeSandbox:
    """A sandbox in which the agent runs and leaves no patch (so no grading sandbox is started)."""

    envs: list[dict] = []

    def __init__(self, provider, spec) -> None:
        pass

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def upload(self, local: Path, remote: str) -> None:
        pass

    async def download(self, remote: str, local: Path) -> None:
        raise AssertionError("nothing to download")

    async def exec(self, command: str, **kwargs):
        if kwargs.get("env"):
            _FakeSandbox.envs.append(kwargs["env"])
        return SimpleNamespace(
            return_code=1 if command.startswith("test -f") else 0, stdout="", stderr="", error_type=None
        )


def _params(tmp_path: Path, body: dict) -> AnySweInstanceConfig:
    (tmp_path / "instruction.txt").write_text("Fix it")
    (tmp_path / "agent_runner.py").write_text("")
    metrics = tmp_path / "metrics.json"
    metrics.write_text("{}")
    return AnySweInstanceConfig(
        **_config().model_dump(),
        run_session_id="s",
        base_results_dir=tmp_path,
        model_server_url="http://policy:8000",
        resolved_sandbox_provider={"opensandbox": {}},
        sandbox_default_metadata={},
        problem_info={"instance_id": "repo__repo-1", "instance_dict": "{}"},
        body=body,
        persistent_dir=tmp_path,
        metrics_fpath=metrics,
        container="registry.example.com/swebench:repo__repo-1",
    )


@pytest.mark.parametrize(("model", "expected"), [(None, "model"), ("policy", "policy")])
def test_response_reports_the_model_name_the_harness_used(tmp_path: Path, model, expected: str) -> None:
    body = _to_gym_row({"instance_id": "repo__repo-1", "problem_statement": "Fix it"}, "test")[
        "responses_create_params"
    ]
    if model is not None:
        body["model"] = model
    _FakeSandbox.envs = []

    with patch("responses_api_agents.anyswe_agent.app.AsyncSandbox", _FakeSandbox):
        response = asyncio.run(AnySweAgent.__new__(AnySweAgent)._run_agent_in_sandbox(_params(tmp_path, body)))

    assert response.model == expected
    assert [env["NGSWE_MODEL_NAME"] for env in _FakeSandbox.envs] == [expected]
    assert json.loads(response.metadata["metrics"])["patch_exists"] is False
