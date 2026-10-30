# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import shlex
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from nemo_gym.global_config import OBSERVABILITY_ENABLED_KEY_NAME
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.deepseek_harness_agent import app as module
from responses_api_agents.deepseek_harness_agent.app import (
    DeepSeekHarnessAgent,
    DeepSeekHarnessAgentConfig,
    DeepSeekHarnessRunRequest,
    task_text,
)
from responses_api_agents.deepseek_harness_agent.trajectory import convert_events


def event(kind, data, seq=1, session="root"):
    return {
        "method": "session.event",
        "payload": {"sessionId": session, "event": {"type": kind, "data": data, "seq": seq}},
    }


def transcript(session="root"):
    return [
        event(
            "assistant/message",
            {
                "message": {
                    "content": [
                        {"type": "reasoning", "text": "Inspect the file."},
                        {"type": "tool-call", "id": "call_1", "name": "bash", "arguments": '{"command":"cat answer"}'},
                    ]
                },
                "usage": {"inputTokens": 20, "cacheReadTokens": 10, "outputTokens": 5},
            },
            session=session,
        ),
        event("tool/call", {"callId": "call_1", "name": "bash", "arguments": "{}"}, seq=2, session=session),
        event(
            "tool/result",
            {
                "message": {
                    "content": [
                        {"type": "tool-result", "toolCallId": "call_1", "content": [{"type": "text", "text": "42"}]}
                    ]
                }
            },
            seq=3,
            session=session,
        ),
        event(
            "assistant/message",
            {
                "message": {"content": [{"type": "text", "text": "42"}]},
                "usage": {"inputTokens": 30, "outputTokens": 2},
            },
            seq=4,
            session=session,
        ),
    ]


def test_transcript_preserves_call_order_reasoning_and_cached_usage():
    notifications = transcript() + [
        event("assistant/message", {"message": {"content": [{"type": "text", "text": "child"}]}}, session="child")
    ]
    output, usage = convert_events(notifications, "root")
    assert [item.type for item in output] == ["reasoning", "function_call", "function_call_output", "message"]
    assert output[1].call_id == output[2].call_id == "call_1"
    assert output[2].output == "42"
    assert output[0].summary[0].text == "Inspect the file."
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (60, 7, 67)
    assert usage.input_tokens_details.cached_tokens is None
    assert usage.output_tokens_details.reasoning_tokens is None


@pytest.mark.parametrize(
    "block", [{"type": "image"}, {"type": "tool-result", "toolCallId": "x", "content": [{"type": "image"}]}]
)
def test_unsupported_output_is_not_silently_discarded(block):
    with pytest.raises(ValueError):
        convert_events([event("assistant/message", {"message": {"content": [block]}})], "root")


@pytest.mark.parametrize(
    "input_value",
    ["hi", [{"role": "user", "content": "hi"}], [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}]],
)
def test_text_inputs(input_value):
    assert task_text(NeMoGymResponseCreateParamsNonStreaming(input=input_value)) == "hi"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input": " "},
        {"input": []},
        {"input": [{"role": "system", "content": "hi"}]},
        {"input": "hi", "instructions": "instruction"},
        {"input": "hi", "previous_response_id": "old"},
        {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": "https://example.com/image.png", "detail": "auto"}
                    ],
                }
            ]
        },
    ],
)
def test_unsupported_inputs_fail_before_starting_a_sandbox(kwargs):
    with pytest.raises(HTTPException, match="DSH"):
        task_text(NeMoGymResponseCreateParamsNonStreaming(**kwargs))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    config = DeepSeekHarnessAgentConfig(
        host="127.0.0.1",
        port=8080,
        entrypoint="app.py",
        name="dsh",
        resources_server={"type": "resources_servers", "name": "tasks"},
        model_server={"type": "responses_api_models", "name": "policy_model"},
        model="test-model",
        sandbox_provider="sandbox",
        results_dir=tmp_path,
    )
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {OBSERVABILITY_ENABLED_KEY_NAME: True}
    agent = DeepSeekHarnessAgent(config=config, server_client=client)
    monkeypatch.setattr(
        DeepSeekHarnessAgent, "resolve_model_base_url", lambda self, name, rollout: f"http://model/{rollout}/v1"
    )
    monkeypatch.setattr(module, "resolve_provider_config", lambda *args: {})
    provider = AsyncMock()
    monkeypatch.setattr(module, "create_provider", lambda *args: provider)
    monkeypatch.setattr(module, "raise_for_status", AsyncMock())
    monkeypatch.setattr(module, "get_response_json", lambda response: response.json())
    files = {}
    order = []
    sandbox = AsyncMock()
    state = SimpleNamespace(
        agent=agent,
        sandbox=sandbox,
        provider=provider,
        files=files,
        order=order,
        mode="completed",
        seed={"sandbox_handle": "opaque-id"},
    )

    async def upload(local, remote):
        files[remote] = Path(local).read_text()

    async def download(remote, local):
        Path(local).write_text(files[remote])
        order.append("download")

    async def execute(command, **kwargs):
        if command.startswith("mkdir"):
            return SimpleNamespace(return_code=0)
        assert "user task" not in command
        input_path = shlex.split(command)[-1]
        data = json.loads(files[input_path])
        assert data["harness"]["base_url"] == "http://model/rollout-1/v1"
        assert data["harness"]["env"]["DSH_PERMISSION_MODE"] == "danger-full-access"
        root = str(Path(input_path).parent)
        files[root + "/events.jsonl"] = "\n".join(json.dumps(x) for x in transcript(data["session_id"]))
        if state.mode == "cancel":
            raise asyncio.CancelledError()
        if state.mode == "timeout":
            raise TimeoutError("runner deadline")
        files[root + "/result.json"] = json.dumps({"finish_reason": state.mode, "error": None})
        return SimpleNamespace(return_code=0 if state.mode == "completed" else 1, stdout="", stderr="")

    async def post(server, path, **kwargs):
        order.append(path)
        if path == "/seed_session":
            assert kwargs["json"]["task_metadata"] == {"opaque": "kept"}
            assert kwargs["cookies"] == {"original": "cookie"}
            return SimpleNamespace(cookies={"resource": "cookie"}, json=AsyncMock(return_value=state.seed))
        assert kwargs["cookies"] == {"original": "cookie", "resource": "cookie"}
        assert kwargs["json"]["task_metadata"] == {"opaque": "kept"}
        if state.mode == "verifier-error":
            raise RuntimeError("verifier unavailable")
        return SimpleNamespace(json=AsyncMock(return_value=kwargs["json"] | {"reward": 1.0}))

    sandbox.upload.side_effect = upload
    sandbox.download.side_effect = download
    sandbox.exec.side_effect = execute
    sandbox.stop.side_effect = lambda: order.append("stop")
    client.post = AsyncMock(side_effect=post)
    monkeypatch.setattr(module.AsyncSandbox, "connect", AsyncMock(return_value=sandbox))
    state.body = DeepSeekHarnessRunRequest(
        responses_create_params={"input": "user task ' $(false)"},
        task_metadata={"opaque": "kept"},
        _ng_rollout_id="rollout-1",
    )
    state.request = lambda: Request({"type": "http", "headers": [(b"cookie", b"original=cookie")]})
    return state


@pytest.mark.parametrize("descriptor", [False, True])
async def test_run_keeps_resources_opaque_and_verifies_before_cleanup(setup, descriptor):
    if descriptor:
        setup.seed = {"sandbox_descriptor": {"provider_token": "opaque", "workdir": "/workspace"}}
    result = await setup.agent.run(setup.request(), setup.body)
    assert result.reward == 1
    assert result.dsh_finish_reason == "completed"
    assert result.dsh_error is None
    assert Path(result.dsh_artifacts, "events.jsonl").exists()
    assert setup.order[-2:] == ["/verify", "stop"]
    expected = setup.seed.get("sandbox_descriptor", {"sandbox_id": "opaque-id"})
    assert module.AsyncSandbox.connect.await_args.args[0] == expected


@pytest.mark.parametrize("mode", ["timeout", "max-tokens"])
async def test_partial_runs_keep_their_trajectory_and_finish_reason(setup, mode):
    setup.mode = mode
    result = await setup.agent.run(setup.request(), setup.body)
    assert result.response.status == "incomplete"
    assert result.response.output[-1].content[0].text == "42"
    assert result.dsh_finish_reason == ("error" if mode == "timeout" else "max-tokens")
    assert bool(result.dsh_error) == (mode == "timeout")
    setup.sandbox.stop.assert_awaited_once()


@pytest.mark.parametrize("mode,error", [("verifier-error", RuntimeError), ("cancel", asyncio.CancelledError)])
async def test_failures_release_sandbox_and_capture_partial_events(setup, mode, error):
    setup.mode = mode
    with pytest.raises(error):
        await setup.agent.run(setup.request(), setup.body)
    setup.sandbox.stop.assert_awaited_once()
    assert "download" in setup.order


async def test_independent_runs_have_independent_homes_and_artifacts(setup):
    a, b = await asyncio.gather(
        setup.agent.run(setup.request(), setup.body), setup.agent.run(setup.request(), setup.body)
    )
    assert a.dsh_artifacts != b.dsh_artifacts
    inputs = [json.loads(value) for name, value in setup.files.items() if name.endswith("input.json")]
    assert len({item["session_id"] for item in inputs}) == 2


async def test_resources_can_retain_sandbox_ownership_and_skip_verification(setup):
    setup.agent.config.stop_sandbox = False
    setup.agent.config.skip_verification = True
    result = await setup.agent.run(setup.request(), setup.body)
    assert result.reward == 0
    assert "/verify" not in setup.order
    setup.sandbox.stop.assert_not_awaited()
    setup.provider.aclose.assert_awaited_once()


async def test_cleanup_failure_preserves_verifier_result(setup, caplog):
    setup.sandbox.stop.side_effect = RuntimeError("sandbox already stopped")
    result = await setup.agent.run(setup.request(), setup.body)
    assert result.reward == 1
    assert result.dsh_finish_reason == "completed"
    assert "Could not clean up DSH sandbox connection" in caplog.text


async def test_connection_failure_releases_provider(setup):
    module.AsyncSandbox.connect.side_effect = TimeoutError("connect failed")
    with pytest.raises(TimeoutError, match="connect failed"):
        await setup.agent.run(setup.request(), setup.body)
    setup.provider.aclose.assert_awaited_once()
    setup.sandbox.stop.assert_not_awaited()


async def test_responses_requires_prepared_run_context(setup):
    with pytest.raises(HTTPException, match="/run"):
        await setup.agent.responses(setup.request(), setup.body.responses_create_params)


async def test_preparation_failure_closes_sandbox_without_calling_verifier(setup):
    setup.sandbox.exec.side_effect = None
    setup.sandbox.exec.return_value = SimpleNamespace(return_code=1)
    with pytest.raises(RuntimeError, match="run directory"):
        await setup.agent.run(setup.request(), setup.body)
    setup.sandbox.stop.assert_awaited_once()
    assert "/verify" not in setup.order


@pytest.mark.parametrize("fault", ["missing-events", "partial-event", "partial-result", "runtime-error"])
async def test_artifact_failures_cannot_report_success(setup, fault):
    original = setup.sandbox.download.side_effect

    async def download(remote, local):
        if remote.endswith("events.jsonl"):
            if fault == "missing-events":
                raise FileNotFoundError(remote)
            if fault == "partial-event":
                setup.files[remote] += '\n{"payload":'
        if remote.endswith("result.json") and fault == "runtime-error":
            setup.files[remote] = json.dumps({"finish_reason": "completed", "error": "runtime shutdown failed"})
        if remote.endswith("result.json") and fault == "partial-result":
            setup.files[remote] = '{"finish_reason":'
        await original(remote, local)

    setup.sandbox.download.side_effect = download
    result = await setup.agent.run(setup.request(), setup.body)
    assert result.dsh_error
    assert result.dsh_finish_reason == "error"
    setup.sandbox.stop.assert_awaited_once()
