# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
import os
import runpy
import shutil
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock

import pytest
from fastapi import Request, Response
from pydantic import BaseModel

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient


def test_runner_preserves_image_python_for_terminal_tools(tmp_path, monkeypatch):
    image_bin = tmp_path / "bin"
    image_bin.mkdir()
    image_python = image_bin / "python"
    image_python.write_text("#!/bin/sh\nexit 0\n")
    image_python.chmod(0o755)
    monkeypatch.setenv("PATH", str(image_bin))
    monkeypatch.setattr(sys, "path", sys.path.copy())

    runpy.run_path(str(Path(__file__).parents[1] / "agent_runner.py"))

    assert shutil.which("python") == str(image_python)
    assert "/opt/agent/bin" in os.environ["PATH"].split(os.pathsep)


@pytest.fixture
def runner_env(tmp_path: Path, monkeypatch):
    # Importing the container entrypoint adjusts PATH and sys.path.
    monkeypatch.setenv("PATH", os.environ["PATH"])
    monkeypatch.setattr(sys, "path", sys.path.copy())
    runner = importlib.import_module("responses_api_agents.kernel_gym.agent_runner")
    solution = tmp_path / "solution.py"
    solution.write_text("original solution\n")
    response_path = tmp_path / "response.json"
    paths = {"/workspace/solution.py": solution, "/trajectories_mount/response.json": response_path}
    monkeypatch.setattr(runner, "Path", paths.__getitem__)

    class HarnessConfig(BaseModel):
        host: str
        port: int
        name: str
        entrypoint: str
        max_turns: int
        model_server: ModelServerRef
        resources_server: ResourcesServerRef

    class Harness:
        def __init__(self, config, server_client):
            self.config = config
            self.server_client = server_client

    module = ModuleType("kernel_gym_test_harness")
    module.HarnessConfig = HarnessConfig
    module.Harness = Harness
    monkeypatch.setitem(sys.modules, module.__name__, module)
    for key, value in {
        "KB_AGENT_MODULE": module.__name__,
        "KB_AGENT_CLASS": "Harness",
        "KB_AGENT_CONFIG_CLASS": "HarnessConfig",
        "KB_AGENT_KWARGS": json.dumps({"max_turns": 3}),
        "KB_MODEL_URL": "http://model.invalid/ng-rollout/test-rollout",
        "KB_MODEL_NAME": "fallback-model",
        "KB_BODY": NeMoGymResponseCreateParamsNonStreaming(input="optimize").model_dump_json(),
    }.items():
        monkeypatch.setenv(key, value)
    transport = AsyncMock()
    monkeypatch.setattr(ServerClient, "request", transport)
    response = NeMoGymResponse(
        id="response-1",
        created_at=0,
        model="served-model",
        object="response",
        status="completed",
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        output=[{"type": "message", "id": "message-1", "content": [{"type": "output_text", "text": "done"}]}],
        usage={
            "input_tokens": 11,
            "output_tokens": 7,
            "total_tokens": 18,
            "input_tokens_details": {"cached_tokens": 3},
            "output_tokens_details": {"reasoning_tokens": 2},
        },
    )
    return runner, module, solution, response_path, response, transport


def test_preserves_solution_written_by_harness_and_full_response(runner_env):
    runner, module, solution, response_path, response, _ = runner_env
    submitted = "def kernel(x):\n    return x + 1\n"

    class EditingHarness(module.Harness):
        async def responses(self, request, body):
            solution.write_text(submitted)
            return response

    module.Harness = EditingHarness
    runner.main()

    assert solution.read_text() == submitted
    assert NeMoGymResponse.model_validate_json(response_path.read_text()) == response


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "Here is the kernel:\n```python\ndef kernel(x):\n    return x + 1\n```\nDone.",
            "def kernel(x):\n    return x + 1\n",
        ),
        ("```cpp\nint kernel() { return 1; }\n```", "int kernel() { return 1; }\n"),
        ("def kernel(x):\n    return x + 1\n", "def kernel(x):\n    return x + 1\n"),
    ],
)
def test_saves_one_shot_code_when_harness_does_not_edit_solution(runner_env, text, expected):
    runner, module, solution, response_path, result, _ = runner_env
    result.output[0].content[0].text = text

    class OneShotHarness(module.Harness):
        async def responses(self, request, body, response):
            assert isinstance(response, Response)
            return result

    module.Harness = OneShotHarness
    runner.main()

    assert solution.read_text() == expected
    assert NeMoGymResponse.model_validate_json(response_path.read_text()) == result


@pytest.mark.parametrize("accepts_response", [False, True])
@pytest.mark.parametrize("requested_model", [None, "requested-model"])
@pytest.mark.parametrize("api_key", [None, "test-model-key"])
def test_preserves_request_model_endpoint_and_auth(
    runner_env, monkeypatch, accepts_response, requested_model, api_key
):
    runner, module, _, _, result, transport = runner_env
    body = NeMoGymResponseCreateParamsNonStreaming(
        model=requested_model,
        input=[{"role": "user", "content": "optimize this kernel"}],
        temperature=0.7,
        max_output_tokens=400,
        parallel_tool_calls=False,
        metadata={"task_name": "kernel-1"},
        tools=[{"type": "function", "name": "bash", "parameters": {"type": "object"}, "strict": False}],
    )
    monkeypatch.setenv("KB_BODY", body.model_dump_json())
    monkeypatch.setenv("KB_AGENT_KWARGS", json.dumps({"max_turns": 3, "model_api_key": api_key}))
    seen = []

    class Harness(module.Harness):
        async def responses(self, request, body):
            seen.append((self, request, body))
            await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="/v1/responses",
                json=body.model_dump(),
                cookies={"session_id": "kernel-session"},
            )
            return result

    class HarnessWithResponse(Harness):
        async def responses(self, request, body, response):
            assert isinstance(response, Response)
            return await super().responses(request, body)

    module.Harness = HarnessWithResponse if accepts_response else Harness
    runner.main()

    assert len(seen) == 1
    agent, request, received = seen[0]
    assert isinstance(request, Request) and request.url.path == "/v1/responses"
    assert received.model_dump() == body.model_copy(update={"model": requested_model or "fallback-model"}).model_dump()
    assert agent.config.max_turns == 3
    assert agent.config.model_server == ModelServerRef(name="policy_model", type="responses_api_models")
    assert agent.config.resources_server == ResourcesServerRef(name="unused", type="resources_servers")
    assert "model_api_key" not in agent.config.model_dump()
    assert agent.server_client._resolve_base_url("policy_model") == "http://model.invalid/ng-rollout/test-rollout"
    expected = {
        "server_name": "policy_model",
        "url_path": "/v1/responses",
        "method": "POST",
        "json": received.model_dump(),
        "cookies": {"session_id": "kernel-session"},
    }
    if api_key:
        expected["headers"] = {"Authorization": f"Bearer {api_key}"}
    transport.assert_awaited_once_with(**expected)
