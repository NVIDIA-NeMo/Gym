# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import SandboxExecResult
from nemo_gym.server_utils import ServerClient
from responses_api_agents.hermes_sandboxed_agent.app import (
    HermesSandboxedAgent,
    HermesSandboxedAgentConfig,
    HermesSandboxedRunRequest,
    trajectory_response,
)
from responses_api_agents.hermes_sandboxed_agent.runner import split_input


@pytest.fixture
def agent(tmp_path):
    return HermesSandboxedAgent(
        config=HermesSandboxedAgentConfig(
            name="hermes",
            host="127.0.0.1",
            port=8000,
            entrypoint="app.py",
            model="real-model",
            resources_server={"type": "resources_servers", "name": "benchmark"},
            model_server={"type": "responses_api_models", "name": "policy"},
            results_dir=str(tmp_path),
        ),
        server_client=MagicMock(spec=ServerClient),
    )


def test_input_preserves_system_and_history():
    assert split_input(
        [
            {"role": "system", "content": "system"},
            {"role": "developer", "content": "developer"},
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": [{"type": "input_text", "text": "new"}]},
        ]
    ) == (
        "new",
        [{"role": "user", "content": "old"}, {"role": "assistant", "content": "answer"}],
        "system\n\ndeveloper",
    )


@pytest.mark.parametrize(
    "items",
    [
        [],
        [{"role": "assistant", "content": "x"}],
        [{"role": "user", "content": [{"type": "input_image", "image_url": "x"}]}],
        [{"type": "function_call_output", "call_id": "x", "output": "x"}],
    ],
)
def test_input_rejects_unsupported_items(items):
    with pytest.raises(ValueError):
        split_input(items)


def test_trajectory_contains_actual_reasoning_tools_and_usage():
    result = {
        "completed": True,
        "n_input": 1,
        "messages": [
            {"role": "user", "content": "problem"},
            {
                "role": "assistant",
                "reasoning": "inspect",
                "tool_calls": [{"id": "tool-1", "function": {"name": "terminal", "arguments": '{"command":"pwd"}'}}],
            },
            {"role": "tool", "tool_call_id": "tool-1", "content": "/app"},
            {"role": "assistant", "content": "fixed"},
        ],
        "usage": {"input_tokens": 100, "output_tokens": 20, "cached_tokens": 10, "reasoning_tokens": 5},
    }
    response = trajectory_response(result, NeMoGymResponseCreateParamsNonStreaming(input="problem"), "real-model")
    assert [i.type for i in response.output] == ["reasoning", "function_call", "function_call_output", "message"]
    assert response.output[1].call_id == response.output[2].call_id == "tool-1"
    assert response.output[2].output == "/app"
    assert response.usage.total_tokens == 120
    assert response.usage.input_tokens_details.cached_tokens == 10
    assert response.status == "completed"


@pytest.mark.parametrize(
    ("result", "error", "status"),
    [
        ({"completed": False}, None, "incomplete"),
        ({"completed": True, "interrupted": True}, None, "incomplete"),
        ({"completed": True}, "timeout", "failed"),
        ({"completed": True, "failed": True}, None, "failed"),
    ],
)
def test_failed_or_aborted_never_marked_completed(result, error, status):
    response = trajectory_response(result, NeMoGymResponseCreateParamsNonStreaming(input="problem"), "model", error)
    assert response.status == status
    assert response.output == []  # Never fabricate training tokens or a successful answer.


@pytest.mark.asyncio
async def test_runner_request_has_no_gold_and_runs_outside_repo(agent, monkeypatch):
    sandbox = SimpleNamespace(
        exec=AsyncMock(
            side_effect=[
                SandboxExecResult(return_code=0, stdout="/app\n", stderr=""),
                SandboxExecResult(return_code=0, stdout="ran", stderr=""),
            ]
        ),
        upload=AsyncMock(),
    )

    async def download(remote, local):
        local.write_text(json.dumps({"completed": True, "api_calls": 2, "messages": []}))

    sandbox.download = download
    monkeypatch.setattr(HermesSandboxedAgent, "resolve_model_base_url", lambda *args: "http://proxy/ng-rollout/id/v1")
    response, metrics = await agent._run_in_sandbox(
        sandbox, NeMoGymResponseCreateParamsNonStreaming(input="fix it"), "id"
    )
    uploaded = sandbox.upload.call_args_list[0].args[0]
    params = json.loads(uploaded.read_text())
    assert params["input"] == "fix it"
    assert params["workdir"] == "/app"
    assert params["base_url"] == "http://proxy/ng-rollout/id/v1"
    assert "patch" not in params and "test_patch" not in params and "api_key" not in params
    command = sandbox.exec.call_args.args[0]
    assert " -I " in command
    assert sandbox.exec.call_args.kwargs["cwd"].startswith("/tmp/nemo-hermes-")
    assert metrics["hermes_finished"] and response.status == "completed"


@pytest.mark.asyncio
async def test_missing_result_preserves_process_failure(agent, monkeypatch):
    sandbox = SimpleNamespace(
        exec=AsyncMock(
            side_effect=[
                SandboxExecResult(return_code=0, stdout="/app\n", stderr=""),
                SandboxExecResult(return_code=125, stdout="", stderr="deadline expired", error_type="timeout"),
            ]
        ),
        upload=AsyncMock(),
        download=AsyncMock(side_effect=FileNotFoundError("result.json")),
    )
    monkeypatch.setattr(HermesSandboxedAgent, "resolve_model_base_url", lambda *args: "http://proxy/v1")
    response, metrics = await agent._run_in_sandbox(
        sandbox, NeMoGymResponseCreateParamsNonStreaming(input="fix"), None
    )
    assert response.status == "failed"
    assert metrics["hermes_return_code"] == 125
    assert metrics["hermes_error_type"] == "timeout"
    from pathlib import Path

    persisted = json.loads(Path(metrics["hermes_result_path"]).read_text())
    assert persisted["stderr"] == "deadline expired"


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_endpoint", [None, "/verify", "/close_session"])
@pytest.mark.parametrize(
    ("finished", "evaluation_completed", "failure"),
    [
        (False, True, "Hermes response status: incomplete"),
        (True, False, "Verification did not complete"),
        (True, True, None),
    ],
)
async def test_run_cookies_descriptor_reward_and_cleanup(
    agent, monkeypatch, failed_endpoint, finished, evaluation_completed, failure
):
    import responses_api_agents.hermes_sandboxed_agent.app as module

    body = HermesSandboxedRunRequest.model_validate({"responses_create_params": {"input": "fix"}, "patch": "gold"})
    response = trajectory_response({"completed": finished}, body.responses_create_params, "real-model")
    seeded = SimpleNamespace(cookies={"session": "seeded"}, data={"sandbox_descriptor": {"sandbox_id": "box"}})
    verified = SimpleNamespace(
        data=body.model_dump()
        | {"response": response.model_dump(), "reward": 1, "evaluation_completed": evaluation_completed}
    )

    async def post(*, url_path, **kwargs):
        if url_path == failed_endpoint:
            raise RuntimeError(f"{url_path} unavailable")
        return {"/seed_session": seeded, "/verify": verified, "/close_session": SimpleNamespace()}[url_path]

    agent.server_client.post = AsyncMock(side_effect=post)
    sandbox = SimpleNamespace(stop=AsyncMock())
    connected = AsyncMock(return_value=sandbox)
    monkeypatch.setattr(module.AsyncSandbox, "connect", connected)
    monkeypatch.setattr(module, "get_global_config_dict", lambda: {"sandbox": {"fake": {}}})
    monkeypatch.setattr(module, "create_provider", lambda config: "provider")
    monkeypatch.setattr(module, "raise_for_status", AsyncMock())
    monkeypatch.setattr(module, "get_response_json", AsyncMock(side_effect=lambda r: r.data))
    monkeypatch.setattr(
        HermesSandboxedAgent,
        "_run_in_sandbox",
        AsyncMock(
            return_value=(
                response,
                {
                    "hermes_result_path": "artifact",
                    "hermes_return_code": 0,
                    "hermes_error_type": None,
                    "hermes_finished": finished,
                    "turns_used": 1,
                },
            )
        ),
    )
    if failed_endpoint == "/verify":
        with pytest.raises(RuntimeError, match="verify unavailable"):
            await agent.run(SimpleNamespace(cookies={"original": "cookie"}), body)
    else:
        result = await agent.run(SimpleNamespace(cookies={"original": "cookie"}), body)
        assert result.verifier_reward == 1
        wire = result.model_dump(mode="json")
        if failure:
            assert result.reward is None
            assert wire["_ng_failure_class"] == "agent_run_error"
            assert wire["_ng_failure_message"] == failure
            assert "reward" not in wire and "response" not in wire
        else:
            assert wire["reward"] == 1
            assert wire["response"]["status"] == "completed"
            assert "_ng_failure_class" not in wire
    assert agent.server_client.post.call_args_list[0].kwargs["json"]["create_pty"] is False
    assert agent.server_client.post.call_args.kwargs["url_path"] == "/close_session"
    assert agent.server_client.post.call_args.kwargs["cookies"] == {"original": "cookie", "session": "seeded"}
    connected.assert_awaited_once_with({"sandbox_id": "box"}, provider="provider")
    assert sandbox.stop.await_count == (1 if failed_endpoint == "/close_session" else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("bare_handle", [False, True])
async def test_failed_connect_uses_benchmark_cleanup_with_seed_cookie(agent, monkeypatch, bare_handle):
    import responses_api_agents.hermes_sandboxed_agent.app as module

    seed = SimpleNamespace(
        cookies={"session": "seeded"},
        data={"sandbox_descriptor": {"sandbox_id": "box"}},
    )
    if bare_handle:
        seed.data = {"sandbox_handle": "box"}
    agent.server_client.post = AsyncMock(side_effect=[seed, SimpleNamespace()])
    monkeypatch.setattr(module, "get_global_config_dict", lambda: {"sandbox": {"fake": {}}})
    monkeypatch.setattr(module, "create_provider", lambda config: "provider")
    monkeypatch.setattr(module, "raise_for_status", AsyncMock())
    monkeypatch.setattr(module, "get_response_json", AsyncMock(return_value=seed.data))
    monkeypatch.setattr(module.AsyncSandbox, "connect", AsyncMock(side_effect=RuntimeError("cannot attach")))

    error, message = (ValueError, "must return sandbox_descriptor") if bare_handle else (RuntimeError, "cannot attach")
    with pytest.raises(error, match=message):
        await agent.run(
            SimpleNamespace(cookies={}),
            HermesSandboxedRunRequest.model_validate({"responses_create_params": {"input": "fix"}}),
        )

    cleanup = agent.server_client.post.call_args.kwargs
    assert cleanup["url_path"] == "/close_session"
    assert cleanup["cookies"] == {"session": "seeded"}
