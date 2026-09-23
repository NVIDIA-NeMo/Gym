# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
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
from responses_api_agents.hermes_sandboxed_agent.runner import progress_result, split_input


@pytest.fixture
def agent(tmp_path):
    return HermesSandboxedAgent(
        config=HermesSandboxedAgentConfig(
            name="hermes",
            host="127.0.0.1",
            port=8000,
            entrypoint="app.py",
            model="real-model",
            context_length=262144,
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


def test_runner_emits_info_logs_to_stderr(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    runner = Path(__file__).parents[1] / "runner.py"
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"run_dir": str(tmp_path)}))
    # A fresh interpreter is essential: pytest has already configured logging.
    probe = """
import logging, runpy, sys
runner = runpy.run_path(sys.argv[1])
def run(params):
    logging.getLogger("run_agent").info("Starting turn")
    logging.getLogger("tools.lazy_deps").info("Dependencies ready")
    print("Hermes stdout")
    return {"completed": True}
runner["_run_worker"].__globals__["run"] = run
sys.argv = sys.argv[1:]
sys.exit(runner["_run_worker"]())
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(runner), str(request)],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert result.stdout == "Hermes stdout\n"
    assert "INFO:run_agent:Starting turn" in result.stderr
    assert "INFO:tools.lazy_deps:Dependencies ready" in result.stderr


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
    ("result", "error", "status", "budget_stop"),
    [
        ({"completed": False}, None, "incomplete", "false"),
        ({"completed": True, "interrupted": True}, None, "incomplete", "false"),
        ({"completed": True}, "timeout", "failed", "false"),
        ({"completed": True, "failed": True}, None, "failed", "false"),
        ({"completed": False, "budget_exhausted": True}, None, "incomplete", "true"),
        ({"completed": False, "budget_exhausted": True, "failed": True}, None, "failed", "false"),
        ({"completed": False, "budget_exhausted": True, "interrupted": True}, None, "incomplete", "true"),
    ],
)
def test_failed_or_aborted_never_marked_completed(result, error, status, budget_stop):
    response = trajectory_response(result, NeMoGymResponseCreateParamsNonStreaming(input="problem"), "model", error)
    assert response.status == status
    assert response.output == []  # Never fabricate training tokens or a successful answer.
    assert response.metadata["budget_exhausted"] == budget_stop


@pytest.mark.asyncio
async def test_runner_request_has_no_gold_and_runs_outside_repo(agent, monkeypatch):
    sandbox = SimpleNamespace(
        exec=AsyncMock(
            side_effect=[
                SandboxExecResult(return_code=0, stdout="/app\n", stderr=""),
                SandboxExecResult(return_code=0, stdout="ran", stderr="INFO:run_agent:Starting turn\n"),
            ]
        ),
        upload=AsyncMock(),
    )

    async def download(remote, local):
        if remote.endswith("/cleanup.json"):
            local.write_text(json.dumps({"cleanup_confirmed": True, "return_code": 0}))
            return
        local.write_text(json.dumps({"completed": True, "api_calls": 2, "messages": []}))

    sandbox.download = download
    monkeypatch.setattr(HermesSandboxedAgent, "resolve_model_base_url", lambda *args: "http://proxy/ng-rollout/id/v1")
    response, metrics = await agent._run_in_sandbox(
        sandbox,
        NeMoGymResponseCreateParamsNonStreaming(
            input="fix it", instructions="Keep the public API", temperature=0, max_output_tokens=128
        ),
        "id",
    )
    uploaded = sandbox.upload.call_args_list[0].args[0]
    params = json.loads(uploaded.read_text())
    assert params["input"] == "fix it"
    assert params["workdir"] == "/app"
    assert params["base_url"] == "http://proxy/ng-rollout/id/v1"
    assert params["context_length"] == 262144
    assert params["temperature"] == 0
    assert params["max_tokens"] == 128
    assert params["instructions"] == "Keep the public API"
    assert "patch" not in params and "test_patch" not in params and "api_key" not in params
    command = sandbox.exec.call_args.args[0]
    assert " -I " in command
    assert sandbox.exec.call_args.kwargs["cwd"].startswith("/tmp/nemo-hermes-")
    assert metrics["hermes_finished"] and response.status == "completed"
    persisted = json.loads(uploaded.with_name("agent_result.json").read_text())
    assert persisted["stdout"] == "ran"
    assert persisted["stderr"] == "INFO:run_agent:Starting turn\n"


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
    with pytest.raises(RuntimeError, match="cleanup was not confirmed"):
        await agent._run_in_sandbox(sandbox, NeMoGymResponseCreateParamsNonStreaming(input="fix"), None)
    from pathlib import Path

    persisted = json.loads(next(Path(agent.config.results_dir).glob("*/agent_result.json")).read_text())
    assert persisted["stderr"] == "deadline expired"
    assert persisted["return_code"] == 125
    assert persisted["error_type"] == "timeout"


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_endpoint", [None, "/verify", "/close_session", "/cleanup_timeout"])
@pytest.mark.parametrize(
    ("finished", "budget_exhausted", "evaluation_completed", "failure"),
    [
        (False, False, True, "Hermes response status: incomplete"),
        (True, False, False, "Verification did not complete"),
        (True, False, True, None),
        (False, True, True, None),
        (False, True, False, "Verification did not complete"),
    ],
)
@pytest.mark.parametrize("verifier_reward", [0, 1])
async def test_run_cookies_descriptor_reward_and_cleanup(
    agent, monkeypatch, failed_endpoint, finished, budget_exhausted, evaluation_completed, failure, verifier_reward
):
    import responses_api_agents.hermes_sandboxed_agent.app as module

    body = HermesSandboxedRunRequest.model_validate({"responses_create_params": {"input": "fix"}, "patch": "gold"})
    response = trajectory_response(
        {"completed": finished, "budget_exhausted": budget_exhausted}, body.responses_create_params, "real-model"
    )
    seeded = SimpleNamespace(cookies={"session": "seeded"}, data={"sandbox_descriptor": {"sandbox_id": "box"}})
    verified = SimpleNamespace(
        data=body.model_dump()
        | {"response": response.model_dump(), "reward": verifier_reward, "evaluation_completed": evaluation_completed}
    )

    async def post(*, url_path, **kwargs):
        if failed_endpoint == "/cleanup_timeout" and url_path == "/close_session":
            await asyncio.Event().wait()
        if url_path == failed_endpoint:
            raise RuntimeError(f"{url_path} unavailable")
        return {"/seed_session": seeded, "/verify": verified, "/close_session": SimpleNamespace()}[url_path]

    agent.server_client.post = AsyncMock(side_effect=post)
    sandbox = SimpleNamespace(stop=AsyncMock())
    if failed_endpoint == "/cleanup_timeout":
        agent.config.cleanup_timeout = 0.01

        async def hung_stop():
            await asyncio.Event().wait()

        sandbox.stop.side_effect = hung_stop
    connected = AsyncMock(return_value=sandbox)
    monkeypatch.setattr(module.AsyncSandbox, "connect", connected)
    monkeypatch.setattr(module, "get_global_config_dict", lambda: {"sandbox": {"fake": {}}})
    provider = SimpleNamespace(aclose=AsyncMock())
    monkeypatch.setattr(module, "create_provider", lambda config: provider)
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
        result = await agent.run(SimpleNamespace(cookies={"original": "cookie"}), body)
        wire = result.model_dump(mode="json")
        assert wire["_ng_failure_class"] == "agent_run_error"
        assert "verify unavailable" in wire["_ng_failure_message"]
        assert "reward" not in wire and "response" not in wire
        assert result.hermes_result_path == "artifact"
    else:
        result = await agent.run(SimpleNamespace(cookies={"original": "cookie"}), body)
        assert result.verifier_reward == verifier_reward
        wire = result.model_dump(mode="json")
        if failure:
            assert result.reward is None
            assert wire["_ng_failure_class"] == "agent_run_error"
            assert wire["_ng_failure_message"] == failure
            assert "reward" not in wire and "response" not in wire
        else:
            assert wire["reward"] == verifier_reward
            assert wire["response"]["status"] == ("completed" if finished else "incomplete")
            assert "_ng_failure_class" not in wire
    assert agent.server_client.post.call_args.kwargs["url_path"] == "/close_session"
    assert agent.server_client.post.call_args.kwargs["cookies"] == {"original": "cookie", "session": "seeded"}
    connected.assert_awaited_once_with({"sandbox_id": "box"}, provider=provider)
    provider.aclose.assert_awaited_once()
    assert sandbox.stop.await_count == (1 if failed_endpoint in ("/close_session", "/cleanup_timeout") else 0)


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
    provider = SimpleNamespace(aclose=AsyncMock())
    monkeypatch.setattr(module, "create_provider", lambda config: provider)
    monkeypatch.setattr(module, "raise_for_status", AsyncMock())
    monkeypatch.setattr(module, "get_response_json", AsyncMock(return_value=seed.data))
    monkeypatch.setattr(module.AsyncSandbox, "connect", AsyncMock(side_effect=RuntimeError("cannot attach")))

    result = await agent.run(
        SimpleNamespace(cookies={}),
        HermesSandboxedRunRequest.model_validate({"responses_create_params": {"input": "fix"}}),
    )
    wire = result.model_dump(mode="json")
    assert wire["_ng_failure_class"] == "agent_run_error"
    assert ("must return sandbox_descriptor" if bare_handle else "cannot attach") in wire["_ng_failure_message"]
    assert "reward" not in wire and "response" not in wire

    cleanup = agent.server_client.post.call_args.kwargs
    assert cleanup["url_path"] == "/close_session"
    assert cleanup["cookies"] == {"session": "seeded"}
    assert provider.aclose.await_count == (0 if bare_handle else 1)


@pytest.mark.asyncio
async def test_timeout_recovers_checkpoint_and_keeps_verifier_eligible_trajectory(agent, monkeypatch):
    from responses_api_agents.hermes_sandboxed_agent.runner import classify_stop

    progress = {
        "api_calls": 8,
        "n_input": 1,
        "messages": [{"role": "user", "content": "fix"}, {"role": "assistant", "content": "working"}],
    }
    sandbox = SimpleNamespace(
        exec=AsyncMock(
            side_effect=[
                SandboxExecResult(return_code=0, stdout="/app\n", stderr=""),
                SandboxExecResult(return_code=125, stdout="", stderr="deadline expired", error_type="timeout"),
            ]
        ),
        upload=AsyncMock(),
    )

    async def download(remote, local):
        if remote.endswith("/cleanup.json"):
            local.write_text(json.dumps({"cleanup_confirmed": True, "return_code": 0}))
            return
        if remote.endswith("/result.json"):
            raise FileNotFoundError(remote)
        local.write_text(json.dumps(progress))

    sandbox.download = download
    monkeypatch.setattr(HermesSandboxedAgent, "resolve_model_base_url", lambda *args: "http://proxy/v1")
    response, metrics = await agent._run_in_sandbox(
        sandbox, NeMoGymResponseCreateParamsNonStreaming(input="fix"), None
    )
    assert response.status == "incomplete" and response.metadata["budget_exhausted"] == "true"
    assert response.output[0].content[0].text == "working"
    assert metrics["turns_used"] == 8 and metrics["agent_timed_out"]
    assert metrics["hermes_error_type"] == "timeout"  # Keep the process evidence.
    truncated = classify_stop(
        progress
        | {
            "partial": True,
            "error": "Model used all output tokens on reasoning with none left for the response. Try lowering reasoning effort or increasing max_tokens.",
        }
    )
    assert truncated["stop_reason"] == "output_tokens" and "error" not in truncated
    crash = classify_stop(progress | {"failed": True, "error": "connection refused"})
    assert crash["failed"] and not crash.get("budget_exhausted")


def test_timeout_before_any_model_reply_is_not_a_scored_budget_stop():
    from responses_api_agents.hermes_sandboxed_agent.runner import classify_stop

    result = classify_stop({"api_calls": 1, "messages": [{"role": "user", "content": "fix"}]}, timed_out=True)
    assert not result.get("budget_exhausted")
    response = trajectory_response(result, NeMoGymResponseCreateParamsNonStreaming(input="fix"), "model", "timeout")
    assert response.status == "failed"


def test_timeout_preserves_tool_call_while_tool_is_blocked():
    user = {"role": "user", "content": "fix"}
    assistant = {"role": "assistant", "tool_calls": [{"id": "in-flight"}]}
    agent = SimpleNamespace(
        _session_messages=[user],
        _db_flush_scan_prefix=[user, assistant],
        _api_call_count=1,
        session_input_tokens=20,
        session_output_tokens=10,
        session_cache_read_tokens=0,
        session_reasoning_tokens=0,
    )
    assert progress_result(agent, 1)["messages"] == [user, assistant]


@pytest.mark.parametrize(
    "overrides",
    [
        {"input": []},
        {"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "image", "detail": "auto"}]}]},
        {"top_p": 0.9},
        {"previous_response_id": "old-response"},
        {"tool_choice": "none"},
        {"tools": [{"type": "function", "name": "custom", "parameters": {}, "strict": False}]},
        {"reasoning": {"effort": "high"}},
    ],
)
async def test_unsupported_request_does_not_create_sandbox(agent, overrides):
    from fastapi import HTTPException

    body = HermesSandboxedRunRequest.model_validate({"responses_create_params": {"input": "fix"} | overrides})
    with pytest.raises(HTTPException) as exc:
        await agent.run(SimpleNamespace(cookies={}), body)
    assert exc.value.status_code == 422
    agent.server_client.post.assert_not_called()


@pytest.mark.parametrize(
    "receipt", [None, {}, {"cleanup_confirmed": "true"}, {"cleanup_confirmed": True, "error": "cleanup failed"}]
)
async def test_missing_or_uncertain_cleanup_never_returns_a_verifier_eligible_response(agent, monkeypatch, receipt):
    sandbox = SimpleNamespace(
        exec=AsyncMock(
            side_effect=[
                SandboxExecResult(return_code=0, stdout="/app\n", stderr=""),
                SandboxExecResult(return_code=0, stdout="", stderr=""),
            ]
        ),
        upload=AsyncMock(),
    )

    async def download(remote, local):
        if remote.endswith("/cleanup.json"):
            if receipt is None:
                raise FileNotFoundError(remote)
            local.write_text(json.dumps(receipt))
        else:
            local.write_text(json.dumps({"completed": True}))

    sandbox.download = download
    monkeypatch.setattr(HermesSandboxedAgent, "resolve_model_base_url", lambda *args: "http://proxy/v1")
    with pytest.raises(RuntimeError, match="cleanup was not confirmed"):
        await agent._run_in_sandbox(sandbox, NeMoGymResponseCreateParamsNonStreaming(input="fix"), None)
