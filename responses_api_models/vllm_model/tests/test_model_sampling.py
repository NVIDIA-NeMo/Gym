# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from nemo_gym.base_responses_api_model import CaptureStore
from nemo_gym.config_types import TOKEN_CAPTURE_PATH_SEGMENT
from nemo_gym.rollout_correlation import ModelCallSampling, model_sampling_base_url
from nemo_gym.server_utils import ServerClient
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


def _model_server(tmp_path, operator_overrides=None):
    return VLLMModel(
        config=VLLMModelConfig(
            host="127.0.0.1",
            port=0,
            name="policy",
            entrypoint="app.py",
            base_url="http://unused/v1",
            api_key="gym",  # pragma: allowlist secret
            model="served-policy",
            return_token_id_information=False,
            uses_reasoning_parser=False,
            sampling_overrides=operator_overrides,
        ),
        server_client=MagicMock(
            spec=ServerClient,
            global_config_dict={"observability_enabled": True, "model_call_capture_dir": str(tmp_path)},
        ),
    )


def _model_provider():
    provider = MagicMock()
    provider.create_chat_completion = AsyncMock(
        return_value={
            "id": "chatcmpl-sampling",
            "object": "chat.completion",
            "created": 0,
            "model": "served-policy",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "done"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )
    return provider


@pytest.mark.parametrize("dialect", ["chat/completions", "responses"])
@pytest.mark.parametrize("checkpoint_policy", [False, True])
@pytest.mark.parametrize(
    "operator_overrides, expected_conflicts",
    [
        (None, {}),
        ({"temperature": 0.0, "top_p": 0.8}, {}),
        (
            {"temperature": 0.4, "top_p": 0.6},
            {"temperature": {"requested": 0.0, "effective": 0.4}, "top_p": {"requested": 0.8, "effective": 0.6}},
        ),
        ({"temperature": 0.0, "top_p": 0.6}, {"top_p": {"requested": 0.8, "effective": 0.6}}),
    ],
)
def test_model_server_applies_controls_before_capture_and_preserves_operator_overrides(
    tmp_path, dialect, checkpoint_policy, operator_overrides, expected_conflicts, caplog
):
    server = _model_server(tmp_path, operator_overrides)
    if checkpoint_policy:
        server.config.checkpoint_policy = True
        server.server_client.global_config_dict["checkpoint"] = {"enabled": True, "control_auth_token": "test"}
    provider = _model_provider()
    server._clients = [provider]
    app = server.setup_webserver()
    if checkpoint_policy:
        gate = app.state.nemo_gym_policy_gate
        gate.admit = AsyncMock(wraps=gate.admit)
    capture_path = f"/{TOKEN_CAPTURE_PATH_SEGMENT}" if checkpoint_policy else ""
    base_url = model_sampling_base_url(
        f"http://testserver/ng-rollout/integration-a2{capture_path}/v1",
        ModelCallSampling(temperature=0.0, top_p=0.8, max_output_tokens=73),
    )
    body = (
        {"messages": [{"role": "user", "content": "task"}], "max_tokens": 1024}
        if dialect == "chat/completions"
        else {"input": [{"role": "user", "content": "task"}], "max_output_tokens": 1024}
    )
    response = TestClient(app).post(f"{base_url}/{dialect}", json=body | {"temperature": 1.0, "top_p": 1.0})

    assert response.status_code == 200, response.text
    if checkpoint_policy:
        gate.admit.assert_awaited_once_with("integration-a2")
    sent = provider.create_chat_completion.await_args.kwargs
    assert sent["model"] == "served-policy"
    assert {name: sent[name] for name in ("temperature", "top_p")} == (
        operator_overrides or {"temperature": 0.0, "top_p": 0.8}
    )
    assert sent.get("max_completion_tokens", sent.get("max_tokens")) == 73
    warnings = [record for record in caplog.records if "sampling_overrides replaces" in record.getMessage()]
    assert [record.args[1] for record in warnings] == ([expected_conflicts] if expected_conflicts else [])
    [captured] = CaptureStore(tmp_path).read("integration-a2")
    assert captured["request"]["temperature"] == 0.0
    assert captured["request"]["top_p"] == 0.8
    assert captured["request"]["max_completion_tokens" if dialect == "chat/completions" else "max_output_tokens"] == 73


def test_model_server_closed_checkpoint_gate_holds_sampling_requests_before_capture(tmp_path):
    server = _model_server(tmp_path)
    server.config.checkpoint_policy = True
    server.server_client.global_config_dict["checkpoint"] = {"enabled": True, "control_auth_token": "test"}
    provider = _model_provider()
    server._clients = [provider]
    app = server.setup_webserver()
    gate = app.state.nemo_gym_policy_gate
    reached_admission = Event()
    admit = gate.admit

    async def observed_admit(capture_key):
        reached_admission.set()
        await admit(capture_key)

    gate.admit = AsyncMock(side_effect=observed_admit)
    base_url = model_sampling_base_url(
        f"http://testserver/ng-rollout/training-a2/{TOKEN_CAPTURE_PATH_SEGMENT}/v1",
        ModelCallSampling(temperature=0.0, max_output_tokens=31),
    )

    control = {"checkpoint_id": "sampling-test", "deadline_ts": time.time() + 10}
    control_headers = {"authorization": "Bearer test"}
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=1) as pool:
        prepared = client.post("/ng-control/v1/checkpoint/prepare", json=control, headers=control_headers)
        assert prepared.status_code == 200, prepared.text
        pending = pool.submit(client.post, f"{base_url}/responses", json={"input": "task"})
        try:
            assert reached_admission.wait(timeout=2)
            gate.admit.assert_awaited_once_with("training-a2")
            assert client.portal.call(lambda: {waiter.capture_key for waiter in gate.waiters}) == {"training-a2"}
            assert not pending.done()
            provider.create_chat_completion.assert_not_awaited()
            assert CaptureStore(tmp_path).read("training-a2") == []
        finally:
            resumed = client.post("/ng-control/v1/checkpoint/resume", json=control, headers=control_headers)
            client.portal.call(gate.open)  # Release the request even if a control-route assertion fails.
        assert resumed.status_code == 200, resumed.text
        response = pending.result(timeout=5)

    assert response.status_code == 200, response.text
    assert provider.create_chat_completion.await_args.kwargs["temperature"] == 0.0
    [captured] = CaptureStore(tmp_path).read("training-a2")
    assert captured["request"]["temperature"] == 0.0
    assert captured["request"]["max_output_tokens"] == 31


def test_rollout_named_ng_sampling_is_not_mistaken_for_controls(tmp_path):
    server = _model_server(tmp_path)
    provider = _model_provider()
    server._clients = [provider]

    response = TestClient(server.setup_webserver()).post(
        "/ng-rollout/ng-sampling/v1/responses", json={"input": "task", "temperature": 0.9}
    )

    assert response.status_code == 200, response.text
    assert provider.create_chat_completion.await_args.kwargs["temperature"] == 0.9
    [captured] = CaptureStore(tmp_path).read("ng-sampling")
    assert captured["request"]["temperature"] == 0.9


def test_misplaced_sampling_prefix_does_not_silently_drop_controls(tmp_path):
    server = _model_server(tmp_path)
    provider = _model_provider()
    server._clients = [provider]
    sampling_prefix = model_sampling_base_url("http://testserver/v1", ModelCallSampling(temperature=0.0)).removesuffix(
        "/v1"
    )

    response = TestClient(server.setup_webserver()).post(
        f"{sampling_prefix}/ng-rollout/misplaced/v1/responses", json={"input": "task", "temperature": 0.9}
    )

    assert response.status_code == 422, response.text
    provider.create_chat_completion.assert_not_awaited()
    assert CaptureStore(tmp_path).read("misplaced") == []


def test_anthropic_messages_rejects_sampling_but_preserves_unprefixed_requests(tmp_path):
    server = _model_server(tmp_path)
    provider = _model_provider()
    server._clients = [provider]
    client = TestClient(server.setup_webserver())
    body = {"messages": [{"role": "user", "content": "task"}], "max_tokens": 128, "temperature": 0.9}

    response = client.post("/v1/messages", json=body)

    assert response.status_code == 200, response.text
    assert provider.create_chat_completion.await_args.kwargs["temperature"] == 0.9
    provider.create_chat_completion.reset_mock()
    base_url = model_sampling_base_url(
        "http://testserver/ng-rollout/unsupported-a2/v1", ModelCallSampling(temperature=0.0)
    )

    response = client.post(f"{base_url}/messages", json=body)

    assert response.status_code == 422, response.text
    assert "not supported for /v1/messages" in response.json()["detail"]
    provider.create_chat_completion.assert_not_awaited()
    assert CaptureStore(tmp_path).read("unsupported-a2") == []


def test_model_server_metadata_route_is_not_treated_as_generation(tmp_path):
    server = _model_server(tmp_path)
    app = server.setup_webserver()

    @app.get("/v1/models")
    async def models(request: Request):
        return {"object": "list", "body": (await request.body()).decode(), "query": request.url.query}

    base_url = model_sampling_base_url(
        "http://testserver/ng-rollout/metadata-a2/v1",
        ModelCallSampling(temperature=0.0, max_output_tokens=23),
    )

    response = TestClient(app).request("GET", f"{base_url}/models?check=ready", content=b"not-generation-json")

    assert response.status_code == 200, response.text
    assert response.json() == {"object": "list", "body": "not-generation-json", "query": "check=ready"}
    assert CaptureStore(tmp_path).read("metadata-a2") == []
