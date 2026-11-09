# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.client import ServerClient
from nemo_gym.core import FailureBoundary, NgFailureEvent
from nemo_gym.verification import JudgeError, is_judge_failure
from resources_servers.opencode_multimodal_benchmarks.app import VisualBenchmarksConfig, VisualBenchmarksServer
from resources_servers.opencode_multimodal_benchmarks.tests.test_app import request


@pytest.mark.asyncio
async def test_native_judge_uses_captured_model_route_without_local_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = MagicMock(spec=ServerClient)
    http = MagicMock()
    http.status = 200
    http.read = AsyncMock(
        return_value=json.dumps(
            {
                "id": "judge-call-123",
                "object": "chat.completion",
                "created": 1,
                "model": "judge",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "A",
                        },
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        ).encode()
    )
    client.post = AsyncMock(return_value=http)
    cfg = VisualBenchmarksConfig
    cls = VisualBenchmarksServer
    server = cls(
        config=cfg(
            host="localhost",
            port=1,
            name="visual",
            entrypoint="app.py",
            judge_server="visual_judge",
            judge_attempts=1,
            judge_model="test-judge",
        ),
        server_client=client,
    )
    body = request("VStarBench", "the fuzzy feline")
    result = await server.verify(body)
    assert result.reward == (1)
    assert client.post.call_args.kwargs["server_name"] == "visual_judge"
    assert client.post.call_args.kwargs["url_path"] == "/v1/chat/completions"
    assert "headers" not in client.post.call_args.kwargs
    client.post.side_effect = RuntimeError("judge transport unavailable")
    with pytest.raises(NgFailureEvent, match="judge transport unavailable") as caught:
        await server.verify(body)
    assert is_judge_failure(caught.value)
    assert caught.value.operation == "judge.request"

    # A transient typed judge failure remains eligible for the configured retry.
    server.config.judge_attempts = 2
    client.post.side_effect = [RuntimeError("temporary judge failure"), http]
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    recovered = await server.verify(body)
    assert recovered.reward == (1)


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["model", "route"])
async def test_missing_judge_configuration_fails_before_network(monkeypatch, missing):
    from resources_servers.opencode_multimodal_benchmarks import app as app_module

    transport = AsyncMock(side_effect=AssertionError("Must not make a network call"))
    monkeypatch.setattr(app_module, "request", transport)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-test-key")
    config_type = VisualBenchmarksConfig
    server_type = VisualBenchmarksServer
    config = config_type(host="localhost", port=1, name="visual", entrypoint="app.py")
    assert config.judge_endpoint is None and config.judge_model is None
    if missing == "route":
        config.judge_model = "test-judge"
    server = server_type(config=config, server_client=MagicMock(spec=ServerClient))
    with pytest.raises(JudgeError, match="Configure judge_"):
        await server._judge("prompt")
    transport.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("recover", [False, True])
async def test_native_judge_deadline_cancels_attempt_and_preserves_retry_policy(monkeypatch, recover):
    from resources_servers.opencode_multimodal_benchmarks import app as app_module

    client = MagicMock(spec=ServerClient)
    config_type = VisualBenchmarksConfig
    server_type = VisualBenchmarksServer
    server = server_type(
        config=config_type(
            host="localhost",
            port=1,
            name="visual",
            entrypoint="app.py",
            judge_server="visual_judge",
            judge_model="test-judge",
            judge_timeout=0.01,
            judge_concurrency=1,
            judge_attempts=2 if recover else 1,
        ),
        server_client=client,
    )
    attempts = 0
    cancelled = []

    async def post(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.append(attempts)
                raise
        response = MagicMock(status=200)
        response.read = AsyncMock(
            return_value=json.dumps(
                {
                    "id": "judge",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "judge",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": "A",
                            },
                        }
                    ],
                }
            ).encode()
        )
        return response

    client.post = AsyncMock(side_effect=post)
    module = app_module
    monkeypatch.setattr(module.asyncio, "sleep", AsyncMock())
    body = request("VStarBench", "the fuzzy feline")
    if recover:
        result = await asyncio.wait_for(server.verify(body), timeout=1)
        assert result.reward == 1
    else:
        with pytest.raises(JudgeError, match="TimeoutError"):
            await asyncio.wait_for(server.verify(body), timeout=1)
    assert attempts == (2 if recover else 1)
    assert cancelled == [1]
    assert not server._judge_sem.locked()


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"judge_model": "judge"},
        {"judge_model": "judge", "judge_server": "policy_model"},
        {"judge_model": "judge", "judge_server": "judge", "judge_endpoint": "https://judge.example/chat"},
    ],
)
def test_required_judge_configuration_fails_at_startup(settings):
    with pytest.raises(JudgeError):
        VisualBenchmarksServer(
            config=VisualBenchmarksConfig(
                host="localhost", port=1, name="visual", entrypoint="app.py", judge_required=True, **settings
            ),
            server_client=MagicMock(spec=ServerClient),
        )


@pytest.mark.asyncio
async def test_direct_judge_owns_retries_and_releases_slot_during_backoff(monkeypatch):
    from resources_servers.opencode_multimodal_benchmarks import app as module

    monkeypatch.setenv("OPENAI_API_KEY", "fake-test-key")
    server = VisualBenchmarksServer(
        config=VisualBenchmarksConfig(
            host="localhost",
            port=1,
            name="visual",
            entrypoint="app.py",
            judge_model="judge",
            judge_endpoint="https://judge.example/chat",
            judge_attempts=2,
            judge_concurrency=1,
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    failed = MagicMock(status=503)
    failed.__aenter__ = AsyncMock(return_value=failed)
    failed.__aexit__ = AsyncMock(return_value=None)
    success = MagicMock(status=200)
    success.__aenter__ = AsyncMock(return_value=success)
    success.__aexit__ = AsyncMock(return_value=None)
    success.json = AsyncMock(return_value={"choices": [{"message": {"content": "A"}}]})
    transport = AsyncMock(side_effect=[failed, success])
    monkeypatch.setattr(module, "request", transport)

    async def backoff(seconds):
        assert not server._judge_sem.locked()

    monkeypatch.setattr(module.asyncio, "sleep", backoff)
    assert await server._judge("prompt") == "A"
    assert transport.await_count == 2
    assert all(call.kwargs["_max_num_tries"] == 1 for call in transport.call_args_list)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 403, 422])
async def test_direct_judge_does_not_retry_permanent_http_errors(monkeypatch, status):
    from resources_servers.opencode_multimodal_benchmarks import app as module

    monkeypatch.setenv("OPENAI_API_KEY", "fake-test-key")
    response = MagicMock(status=status)
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=None)
    transport = AsyncMock(return_value=response)
    monkeypatch.setattr(module, "request", transport)
    server = VisualBenchmarksServer(
        config=VisualBenchmarksConfig(
            host="localhost",
            port=1,
            name="visual",
            entrypoint="app.py",
            judge_model="judge",
            judge_endpoint="https://judge.example/chat",
            judge_attempts=3,
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    with pytest.raises(JudgeError):
        await server._judge("prompt")
    transport.assert_awaited_once()


def test_typo_in_judge_configuration_is_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="judge_modle"):
        VisualBenchmarksConfig(host="localhost", port=1234, name="visual", entrypoint="app.py", judge_modle="typo")


def test_policy_judge_alias_is_rejected_at_startup():
    from unittest.mock import MagicMock

    from nemo_gym.client import ServerClient
    from nemo_gym.verification import JudgeError

    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {
        "agent": {
            "responses_api_agents": {
                "simple_agent": {"model_server": {"type": "responses_api_models", "name": "custom_policy"}}
            }
        }
    }
    with pytest.raises(JudgeError, match="policy model"):
        VisualBenchmarksServer(
            config=VisualBenchmarksConfig(
                host="localhost",
                port=1234,
                name="visual",
                entrypoint="app.py",
                judge_required=True,
                judge_model="policy",
                judge_server="custom_policy",
            ),
            server_client=client,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("status,attempts", [(400, 1), (401, 1), (422, 1), (429, 3), (503, 3)])
async def test_native_retry_policy_follows_remote_cause_not_proxy_carrier(monkeypatch, status, attempts):
    from aiohttp import ClientResponseError

    from resources_servers.opencode_multimodal_benchmarks import app

    cause = NgFailureEvent(
        boundary=FailureBoundary.RESOURCES_SERVER,
        failure_kind="judge_failed",
        failure_reason="provider",
        http_status=status,
    )
    failure = NgFailureEvent(
        boundary=FailureBoundary.RESOURCES_SERVER, failure_kind="judge_failed", failure_reason="remote judge"
    )
    failure.__cause__ = cause
    failure.http_error = ClientResponseError(MagicMock(), (), status=500)
    call = AsyncMock(side_effect=failure)
    monkeypatch.setattr(app, "call_judge", call)
    monkeypatch.setattr(app.asyncio, "sleep", AsyncMock())
    server = VisualBenchmarksServer(
        config=VisualBenchmarksConfig(
            host="localhost",
            port=1,
            name="visual",
            entrypoint="app.py",
            judge_server="judge",
            judge_model="judge",
            judge_attempts=3,
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    with pytest.raises(NgFailureEvent):
        await server._judge("extract the option")
    assert call.await_count == attempts
