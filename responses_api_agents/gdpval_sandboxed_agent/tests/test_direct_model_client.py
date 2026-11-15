# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import sys
from http.cookies import SimpleCookie
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp
import pytest

from responses_api_agents.gdpval_sandboxed_agent import direct_model_client as transport


class Response:
    status = 200
    cookies = SimpleCookie("model=reply")

    def __init__(self):
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True

    async def read(self):
        return b'{"id":"served","usage":{"completion_tokens":3},"token_ids":[7,8,9]}'


async def test_transport_keeps_rollout_prefix_tokens_and_cookies(monkeypatch, tmp_path):
    response = Response()
    request = AsyncMock(return_value=response)
    monkeypatch.setattr(transport, "request", request)
    monkeypatch.setattr(transport, "raise_for_status", AsyncMock())
    client = transport.DirectModelTransport(
        "http://model/ng-rollout/episode/training-token-capture/v1", timing_path=tmp_path / "timing.jsonl"
    )
    message = {"role": "tool", "name": "execute_bash", "tool_call_id": "call1", "content": "ok"}
    payload = {"model": "policy", "aws_region_name": None, "messages": [message], "prompt_token_ids": [1, 2]}
    out = await client.post(
        server_name="ignored", url_path="/v1/chat/completions", json=payload, cookies={"session": "s"}
    )
    assert request.call_args.args == (
        "POST",
        "http://model/ng-rollout/episode/training-token-capture/v1/chat/completions",
    )
    sent = request.call_args.kwargs
    assert sent["cookies"] == {"session": "s"}
    assert sent["_max_num_tries"] == 1
    assert sent["json"]["prompt_token_ids"] == [1, 2]
    assert sent["json"]["messages"] == [{"role": "tool", "tool_call_id": "call1", "content": "ok"}]
    assert "aws_region_name" not in sent["json"] and message["name"] == "execute_bash"
    assert json.loads(await out.read())["token_ids"] == [7, 8, 9]
    assert out.cookies["model"].value == "reply" and response.closed
    timing = json.loads((tmp_path / "timing.jsonl").read_text())
    assert timing["response_id"] == "served" and timing["usage"] == {"completion_tokens": 3}
    assert "session" not in timing and "messages" not in timing


async def test_retry_count_is_owned_by_adapter(monkeypatch):
    request = AsyncMock(side_effect=[aiohttp.ClientPayloadError("cut"), Response()])
    monkeypatch.setattr(transport, "request", request)
    monkeypatch.setattr(transport, "raise_for_status", AsyncMock())
    monkeypatch.setattr(transport.asyncio, "sleep", AsyncMock())
    client = transport.DirectModelTransport("http://model/v1", read_retries=1)
    await client.post(server_name="model", url_path="/v1/chat/completions", json={"messages": []}, cookies={})
    assert request.await_count == 2
    assert all(call.kwargs["_max_num_tries"] == 1 for call in request.call_args_list)


@pytest.mark.parametrize("url", ["model/v1", "https://secret@model/v1", "http://model/v1?secret=x"])
def test_endpoint_rejects_ambiguous_or_secret_bearing_urls(url):
    with pytest.raises(ValueError):
        transport.DirectModelTransport(url)


@pytest.fixture
def pinned_client_module(monkeypatch):
    """Exercise the timing hook without installing the controller on the agent host."""
    module = ModuleType("openhands.agenthub.nemo_gym_client")

    class OriginalClient:
        @staticmethod
        def _update_model_call_time(start_time):
            raise AssertionError("The adapter must replace the shared-file timing writer")

    module.NemoGymClient = OriginalClient
    agenthub = ModuleType("openhands.agenthub")
    agenthub.nemo_gym_client = module
    monkeypatch.setitem(sys.modules, "openhands", ModuleType("openhands"))
    monkeypatch.setitem(sys.modules, "openhands.agenthub", agenthub)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


@pytest.mark.parametrize("runtime_content", ["", '{"total_runtime_call_time":'])
async def test_model_timing_ignores_partial_runtime_metrics(
    monkeypatch, tmp_path, pinned_client_module, runtime_content
):
    runtime = tmp_path / "metrics.json"
    runtime.write_text(runtime_content)
    monkeypatch.setenv("NEMO_GYM_METRICS_FPATH", str(runtime))
    monkeypatch.setattr(transport.time, "time", lambda: 15.0)
    original = pinned_client_module.NemoGymClient
    llm = SimpleNamespace(config=SimpleNamespace(base_url="http://model/v1"))
    async with transport.use_direct_model_client():
        policy = pinned_client_module.NemoGymClient(llm)
        condenser = pinned_client_module.NemoGymClient(llm)
        policy._update_model_call_time(13.0)
        condenser._update_model_call_time(12.0)
        assert json.loads((tmp_path / "model-metrics.json").read_text()) == {"total_model_call_time": 5.0}
        assert runtime.read_text() == runtime_content
        runtime.write_text('{"total_runtime_call_time": 7.0}')
        policy._update_model_call_time(14.0)
        assert json.loads(runtime.read_text()) == {"total_runtime_call_time": 7.0}
        assert json.loads((tmp_path / "model-metrics.json").read_text()) == {"total_model_call_time": 6.0}
    assert pinned_client_module.NemoGymClient is original
    assert not list(tmp_path.glob("*.tmp"))


async def test_model_timing_publish_failure_does_not_interrupt_policy(monkeypatch, tmp_path, pinned_client_module):
    runtime = tmp_path / "metrics.json"
    runtime.write_text('{"runtime": 3}')
    monkeypatch.setenv("NEMO_GYM_METRICS_FPATH", str(runtime))
    monkeypatch.setattr(transport.time, "time", lambda: 15.0)
    replace = transport.os.replace
    attempted = []

    def fail_first_publish(source, destination):
        attempted.append(json.loads(source.read_text()))
        if len(attempted) == 1:
            raise OSError("diagnostic storage unavailable")
        # A reader never sees the partially written temporary file at the final path.
        assert not destination.exists()
        replace(source, destination)

    monkeypatch.setattr(transport.os, "replace", fail_first_publish)
    async with transport.use_direct_model_client():
        client = pinned_client_module.NemoGymClient(
            SimpleNamespace(config=SimpleNamespace(base_url="http://model/v1"))
        )
        client._update_model_call_time(13.0)
        client._update_model_call_time(12.0)
    assert attempted == [{"total_model_call_time": 2.0}, {"total_model_call_time": 5.0}]
    assert json.loads((tmp_path / "model-metrics.json").read_text()) == {"total_model_call_time": 5.0}
    assert json.loads(runtime.read_text()) == {"runtime": 3}
    assert not list(tmp_path.glob("*.tmp"))
