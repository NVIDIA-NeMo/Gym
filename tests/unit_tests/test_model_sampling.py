# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Event
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemo_gym.base_responses_api_model import CaptureStore, ModelCallCaptureConfig, install_model_call_capture
from nemo_gym.config_types import TOKEN_CAPTURE_PATH_SEGMENT
from nemo_gym.model_sampling import ModelCallSampling, ModelSamplingMiddleware, model_sampling_base_url
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


def test_request_controls_preserve_zero_and_leave_omitted_controls_unset():
    request = NeMoGymResponseCreateParamsNonStreaming(input="task", temperature=0.0, top_p=1.0)
    original = request.model_dump()

    sampling = ModelCallSampling.from_request(request)

    assert sampling.model_dump(exclude_none=True) == {"temperature": 0.0, "top_p": 1.0}
    assert request.model_dump() == original
    empty = ModelCallSampling.from_request(NeMoGymResponseCreateParamsNonStreaming(input="task"))
    assert model_sampling_base_url("http://model/ng-rollout/task-a2/v1", empty) == (
        "http://model/ng-rollout/task-a2/v1"
    )


@pytest.mark.parametrize(
    "controls",
    [
        {"temperature": True},
        {"temperature": "0.7"},
        {"temperature": -0.1},
        {"temperature": 2.1},
        {"temperature": float("nan")},
        {"top_p": float("inf")},
        {"top_p": -0.1},
        {"top_p": 1.1},
        {"max_output_tokens": True},
        {"max_output_tokens": 1.5},
        {"max_output_tokens": "128"},
        {"max_output_tokens": 0},
        {"max_output_tokens": -1},
        {"max_output_tokens": 2**53},
    ],
)
def test_invalid_generation_controls_are_rejected(controls):
    with pytest.raises(ValidationError):
        ModelCallSampling(**controls)
    with pytest.raises(HTTPException) as error:
        ModelCallSampling.from_request(
            SimpleNamespace(**({"temperature": None, "top_p": None, "max_output_tokens": None} | controls))
        )
    assert error.value.status_code == 422


def test_token_cap_supports_the_largest_exact_javascript_integer():
    assert ModelCallSampling(max_output_tokens=2**53 - 1).max_output_tokens == 2**53 - 1


@pytest.mark.parametrize("dialect", ["chat/completions", "responses"])
@pytest.mark.parametrize("stream", [False, True])
def test_concurrent_rollouts_capture_their_own_effective_model_requests(tmp_path, dialect, stream):
    app = FastAPI()
    entered = []
    both_entered = asyncio.Event()
    stream_body = b'data: {"id":"completion","choices":[]}\n\ndata: [DONE]\n\n'

    @app.post(f"/v1/{dialect}")
    async def generate(request: Request):
        body = await request.json()
        entered.append((body, request.headers.get("x-session-id"), request.url.query))
        if len(entered) == 2:
            both_entered.set()
        await asyncio.wait_for(both_entered.wait(), timeout=5)
        if stream:
            return StreamingResponse(iter([stream_body]), media_type="text/event-stream", headers={"x-test": "kept"})
        return JSONResponse({"id": "completion", "received": body}, headers={"x-test": "kept"})

    install_model_call_capture(
        app,
        ModelCallCaptureConfig(observability_enabled=True, model_call_capture_dir=tmp_path),
        model_server_name="policy",
    )
    app.add_middleware(ModelSamplingMiddleware)
    original = {
        "model": "policy",
        "stream": stream,
        "temperature": 0.9,
        "top_p": 0.9,
        "metadata": {"trace": "untouched"},
    }
    if dialect == "chat/completions":
        original |= {
            "messages": [{"role": "user", "content": "task"}],
            "max_tokens": 2048,
            "max_completion_tokens": 4096,
        }
    else:
        original |= {"input": "task", "max_output_tokens": 2048}
    before = deepcopy(original)
    controls = [
        ModelCallSampling(temperature=0.0, top_p=0.7, max_output_tokens=17),
        ModelCallSampling(temperature=1.2, top_p=1.0, max_output_tokens=31),
    ]

    with TestClient(app) as client, ThreadPoolExecutor(max_workers=2) as pool:

        def post(index):
            base_url = model_sampling_base_url(f"http://testserver/ng-rollout/task-{index}-a2/v1", controls[index])
            return client.post(
                f"{base_url}/{dialect}?trace=keep",
                json=original,
                headers={"x-session-id": f"session-{index}"},
            )

        results = list(pool.map(post, range(2)))

    assert original == before
    assert len(entered) == 2
    for index, result in enumerate(results):
        assert result.status_code == 200, result.text
        assert result.headers["x-test"] == "kept"
        expected = original | {"temperature": controls[index].temperature, "top_p": controls[index].top_p}
        if dialect == "chat/completions":
            expected.pop("max_tokens")
            expected["max_completion_tokens"] = controls[index].max_output_tokens
        else:
            expected["max_output_tokens"] = controls[index].max_output_tokens
        if stream:
            assert result.content == stream_body
            assert result.headers["content-type"].startswith("text/event-stream")
        else:
            assert result.json() == {"id": "completion", "received": expected}
        assert (expected, f"session-{index}", "trace=keep") in entered
        [captured] = CaptureStore(tmp_path).read(f"task-{index}-a2")
        assert captured["request"] == expected


@pytest.mark.parametrize("dialect", ["chat/completions", "responses"])
def test_omitted_controls_preserve_harness_defaults(dialect):
    app = FastAPI()

    @app.post(f"/v1/{dialect}")
    async def echo(request: Request):
        return await request.json()

    app.add_middleware(ModelSamplingMiddleware)
    request_body = {"temperature": 0.3, "top_p": 0.8, "max_tokens": 43, "metadata": {"kept": "yes"}}
    base_url = model_sampling_base_url("http://testserver/v1", ModelCallSampling(temperature=0.0))

    response = TestClient(app).post(f"{base_url}/{dialect}", json=request_body)

    assert response.status_code == 200
    assert response.json() == request_body | {"temperature": 0.0}


@pytest.mark.parametrize(
    "dialect,harness_limits,expected_limit",
    [
        ("chat/completions", {}, 73),
        ("chat/completions", {"max_tokens": 100}, 73),
        ("chat/completions", {"max_tokens": 11}, 11),
        ("chat/completions", {"max_completion_tokens": 100}, 73),
        ("chat/completions", {"max_completion_tokens": 11}, 11),
        ("chat/completions", {"max_tokens": 11, "max_completion_tokens": 19}, 11),
        ("chat/completions", {"max_tokens": 19, "max_completion_tokens": 11}, 11),
        ("chat/completions", {"max_tokens": True, "max_completion_tokens": 19}, 19),
        ("responses", {}, 73),
        ("responses", {"max_output_tokens": 100}, 73),
        ("responses", {"max_output_tokens": 11}, 11),
    ],
)
def test_request_token_cap_never_expands_a_smaller_harness_budget(dialect, harness_limits, expected_limit):
    app = FastAPI()

    @app.post(f"/v1/{dialect}")
    async def model(request: Request):
        return await request.json()

    app.add_middleware(ModelSamplingMiddleware)
    base_url = model_sampling_base_url(
        "http://testserver/v1", ModelCallSampling(temperature=0.0, top_p=0.8, max_output_tokens=73)
    )

    response = TestClient(app).post(f"{base_url}/{dialect}", json={"temperature": 1.0, "top_p": 1.0} | harness_limits)

    assert response.status_code == 200, response.text
    limit_field = "max_completion_tokens" if dialect == "chat/completions" else "max_output_tokens"
    assert response.json() == {"temperature": 0.0, "top_p": 0.8, limit_field: expected_limit}


@pytest.mark.parametrize("method,path", [("GET", "/v1/models"), ("POST", "/v1/control"), ("GET", "/health")])
def test_non_generating_routes_keep_their_request_body(method, path):
    app = FastAPI()

    async def endpoint(request: Request):
        return {"path": request.url.path, "body": (await request.body()).decode(), "query": request.url.query}

    app.add_api_route(path, endpoint, methods=[method])
    app.add_middleware(ModelSamplingMiddleware)
    # Administrative endpoints do not need a generation payload, even when a client reuses its model base URL.
    base_url = model_sampling_base_url("http://testserver/v1", ModelCallSampling(temperature=0.0))
    prefix = base_url.removesuffix("/v1")
    response = TestClient(app).request(method, f"{prefix}{path}?key=value", content=b"not-json")

    assert response.status_code == 200
    assert response.json() == {"path": path, "body": "not-json", "query": "key=value"}


@pytest.mark.parametrize("payload", [{"temperature": True}, {"unknown": 1}, [], {"max_output_tokens": -1}])
def test_invalid_url_controls_never_reach_generation(payload):
    app = FastAPI()

    @app.post("/v1/responses")
    async def generate():
        pytest.fail("Invalid controls reached the model")

    app.add_middleware(ModelSamplingMiddleware)
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    response = TestClient(app).post(f"/ng-sampling/{encoded}/v1/responses", json={"input": "task"})

    assert response.status_code == 422


@pytest.mark.parametrize("body", [b"invalid-json", b"[]", b"null"])
def test_invalid_generation_body_never_reaches_model(body):
    app = FastAPI()

    @app.post("/v1/responses")
    async def generate():
        pytest.fail("Invalid body reached the model")

    app.add_middleware(ModelSamplingMiddleware)
    base_url = model_sampling_base_url("http://testserver/v1", ModelCallSampling(temperature=0.0))
    response = TestClient(app).post(f"{base_url}/responses", content=body)

    assert response.status_code == 422


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


@pytest.mark.parametrize("rollout_path", ["", f"/ng-rollout/mounted-a2/{TOKEN_CAPTURE_PATH_SEGMENT}"])
@pytest.mark.parametrize("mount", ["/relay", "/ng-sampling/relay"])
@pytest.mark.parametrize("sampled", [False, True])
def test_sampling_url_preserves_an_external_relay_mount(rollout_path, mount, sampled):
    model_app = FastAPI()

    @model_app.post(f"{rollout_path}/v1/responses")
    async def generate(request: Request):
        return {"path": request.url.path, "body": await request.json(), "query": request.url.query}

    model_app.add_middleware(ModelSamplingMiddleware)
    relay = FastAPI()
    relay.mount(mount, model_app)
    controls = ModelCallSampling(temperature=0.0, max_output_tokens=31) if sampled else None
    base_url = model_sampling_base_url(f"http://testserver{mount}{rollout_path}/v1", controls)

    response = TestClient(relay).post(f"{base_url}/responses?trace=kept", json={"input": "task"})

    assert response.status_code == 200, response.text
    assert response.json() == {
        "path": f"{mount}{rollout_path}/v1/responses",
        "body": {"input": "task"} | ({"temperature": 0.0, "max_output_tokens": 31} if sampled else {}),
        "query": "trace=kept",
    }


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


@pytest.mark.parametrize(
    "encoded",
    ["not+urlsafe", "a" * 1025]
    + [
        base64.urlsafe_b64encode(raw).decode().rstrip("=")
        for raw in [b'{"top_p":0.1,"top_p":0.9}', b'{"temperature":NaN}', b'{"top_p":Infinity}']
    ],
)
def test_malformed_or_ambiguous_sampling_url_is_rejected(encoded):
    app = FastAPI()

    @app.post("/v1/responses")
    async def generate():
        pytest.fail("Malformed controls reached the model")

    app.add_middleware(ModelSamplingMiddleware)

    response = TestClient(app).post(f"/ng-sampling/{encoded}/v1/responses", json={"input": "task"})

    assert response.status_code == 422


@pytest.mark.parametrize("disconnect_early", [False, True])
async def test_chunked_requests_are_reassembled_or_abandoned_on_disconnect(disconnect_early):
    original = '{"input":"caf\u00e9","temperature":1.0}'.encode()
    split = original.index(b"\xc3") + 1  # A multibyte code point can span two HTTP body chunks.
    incoming = [
        {"type": "http.request", "body": original[:split], "more_body": True},
        *([] if disconnect_early else [{"type": "http.request", "body": original[split:], "more_body": False}]),
        {"type": "http.disconnect"},
    ]
    forwarded = []
    sent = []

    async def receive():
        return incoming.pop(0)

    async def send(message):
        sent.append(message)

    async def downstream(scope, receive, send):
        body = await receive()
        forwarded.append((scope, body))
        assert await receive() == {"type": "http.disconnect"}
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"done"})

    path = model_sampling_base_url("/v1", ModelCallSampling(temperature=0.0)) + "/responses"
    await asyncio.wait_for(
        ModelSamplingMiddleware(downstream)(
            {
                "type": "http",
                "method": "POST",
                "path": path,
                "raw_path": path.encode(),
                "headers": [(b"content-length", str(len(original)).encode()), (b"x-trace", b"kept")],
            },
            receive,
            send,
        ),
        timeout=2,
    )

    assert not incoming
    if disconnect_early:
        assert forwarded == []
        assert sent == []
    else:
        [(scope, body)] = forwarded
        assert scope["path"] == "/v1/responses"
        assert body["type"] == "http.request" and body["more_body"] is False
        assert json.loads(body["body"]) == {"input": "caf\u00e9", "temperature": 0.0}
        assert dict(scope["headers"])[b"content-length"] == str(len(body["body"])).encode()
        assert dict(scope["headers"])[b"x-trace"] == b"kept"
        assert sent[-1]["body"] == b"done"


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
