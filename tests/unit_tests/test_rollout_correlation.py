# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import base64
import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from types import SimpleNamespace
from urllib.parse import urlsplit

import orjson
import pytest
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from omegaconf import OmegaConf
from pydantic import ConfigDict, ValidationError
from starlette.testclient import TestClient

import nemo_gym.server_utils
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModelConfig,
    CaptureStore,
    ModelCallCaptureConfig,
    SimpleResponsesAPIModel,
    _CaptureMiddleware,
    install_model_call_capture,
)
from nemo_gym.config_types import TOKEN_CAPTURE_PATH_SEGMENT, BaseServerConfig
from nemo_gym.observability.model_calls import merge_model_call_capture_into_record
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_correlation import (
    ModelCallSampling,
    ModelSamplingMiddleware,
    current_rollout_id,
    maybe_rollout_id_from_run_body,
    model_sampling_base_url,
)
from nemo_gym.server_utils import ServerClient, get_response_json


def _model_response(model: str, text: str = "") -> dict:
    return {
        "id": f"resp-{model}",
        "created_at": 0.0,
        "model": model,
        "object": "response",
        "output": [
            {
                "id": f"msg-{model}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def _model_app(capture_dir, name: str) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/responses")
    async def responses(body: dict = Body()) -> JSONResponse:
        return JSONResponse(_model_response(name, "[[A=B]]" if name == "judge" else "answer"))

    install_model_call_capture(
        app,
        ModelCallCaptureConfig(observability_enabled=True, model_call_capture_dir=capture_dir),
        model_server_name=name,
    )
    return app


class _JudgeResourcesServer(SimpleResourcesServer):
    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/tool")(self.tool)
        return app

    async def tool(self, body: dict = Body()) -> dict:
        tool_model = await self.server_client.post(
            server_name="tool_model",
            url_path="/v1/responses",
            json={"input": body["input"]},
        )
        return await get_response_json(tool_model)

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        judge = await self.server_client.post(
            server_name="judge",
            url_path="/v1/responses",
            json={"input": "grade"},
        )
        await get_response_json(judge)
        return BaseVerifyResponse(**body.model_dump(), reward=1.0)


class _AgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class _Agent(SimpleResponsesAPIAgent):
    async def responses(self, body):
        raise NotImplementedError

    async def run(self, body: _AgentRunRequest) -> BaseVerifyResponse:
        policy = await self.server_client.post(
            server_name="policy",
            url_path="/v1/responses",
            json=body.responses_create_params,
        )
        response = orjson.loads(await policy.read())
        tool = await self.server_client.post(
            server_name="resources",
            url_path="/tool",
            json={"input": "lookup"},
        )
        await get_response_json(tool)
        verify = await self.server_client.post(
            server_name="resources",
            url_path="/verify",
            json={
                "responses_create_params": body.responses_create_params.model_dump(),
                "response": response,
            },
        )
        return BaseVerifyResponse.model_validate(orjson.loads(await verify.read()))


class _Response:
    def __init__(self, response) -> None:
        self.status = response.status_code
        self.ok = response.is_success
        self.cookies = response.cookies
        self._content = response.content

    async def read(self) -> bytes:
        return self._content


@pytest.mark.asyncio
async def test_verify_correlates_policy_and_judge_calls_and_preserves_raw_capture(tmp_path, monkeypatch) -> None:
    capture_dir = tmp_path / "captures"
    config = OmegaConf.create(
        {
            "observability_enabled": True,
            "policy": {"responses_api_models": {"model": {"host": "policy.test", "port": 80}}},
            "tool_model": {"responses_api_models": {"model": {"host": "tool-model.test", "port": 80}}},
            "judge": {"responses_api_models": {"model": {"host": "judge.test", "port": 80}}},
            "resources": {"resources_servers": {"judge": {"host": "resources.test", "port": 80}}},
            "agent": {"responses_api_agents": {"agent": {"host": "agent.test", "port": 80}}},
        }
    )
    server_client = ServerClient(
        head_server_config=BaseServerConfig(host="head.test", port=80),
        global_config_dict=config,
    )
    resources = _JudgeResourcesServer(
        config=BaseResourcesServerConfig(
            host="resources.test",
            port=80,
            entrypoint="app.py",
            name="resources",
        ),
        server_client=server_client,
    )
    agent = _Agent(
        config=BaseResponsesAPIAgentConfig(
            host="agent.test",
            port=80,
            entrypoint="app.py",
            name="agent",
        ),
        server_client=server_client,
    )
    clients = {
        "policy.test": TestClient(_model_app(capture_dir, "policy")),
        "tool-model.test": TestClient(_model_app(capture_dir, "tool_model")),
        "judge.test": TestClient(_model_app(capture_dir, "judge")),
        "resources.test": TestClient(resources.setup_webserver()),
        "agent.test": TestClient(agent.setup_webserver()),
    }

    async def dispatch(method: str, url: str, **kwargs):
        parsed = urlsplit(url)
        response = clients[parsed.hostname].request(method, parsed.path, json=kwargs.get("json"))
        return _Response(response)

    monkeypatch.setattr(nemo_gym.server_utils, "request", dispatch)

    verify = await server_client.post(
        server_name="agent",
        url_path="/run",
        json={
            "_ng_task_index": 4,
            "_ng_rollout_index": 2,
            "responses_create_params": {"input": "solve"},
        },
    )
    assert orjson.loads(await verify.read())["reward"] == 1.0

    store = CaptureStore(capture_dir)
    capture_path = store.path_for("4-2")
    assert capture_path.is_file()
    exchanges = store.read("4-2")
    assert [exchange["model_ref"]["name"] for exchange in exchanges] == ["policy", "tool_model", "judge"]
    assert all(exchange.get("request") is not None or exchange.get("request_raw") for exchange in exchanges)
    assert all(exchange.get("response") is not None or exchange.get("response_raw") for exchange in exchanges)

    rollout = {"_ng_task_index": 4, "_ng_rollout_index": 2}
    merge_model_call_capture_into_record(rollout, [capture_dir])
    assert capture_path.is_file()
    assert len(capture_path.read_bytes()) > 0


def test_rollout_id_does_not_serialize_run_body() -> None:
    class UndumpableRunRequest(_AgentRunRequest):
        def model_dump(self, *args, **kwargs):
            raise AssertionError("run body must not be serialized")

    body = UndumpableRunRequest.model_validate(
        {
            "_ng_task_index": 4,
            "_ng_rollout_index": 2,
            "responses_create_params": {"input": "solve"},
        }
    )

    assert maybe_rollout_id_from_run_body(body) == "4-2"


def test_explicit_rollout_alias_stays_request_scoped() -> None:
    body = BaseRunRequest.model_validate(
        {
            "_ng_rollout_id": "rollout-explicit",
            "responses_create_params": {"input": "solve"},
        }
    )
    assert maybe_rollout_id_from_run_body(body) == "rollout-explicit"
    assert "_ng_rollout_id" not in body.model_dump(by_alias=True)


@pytest.mark.asyncio
async def test_capture_middleware_exposes_current_rollout_id_on_plain_forward() -> None:
    """``_CaptureMiddleware`` already parses and strips the ``/ng-rollout/<id>`` prefix on
    every request, capture enabled or not. It should publish that id through
    ``current_rollout_id()`` around each place it dispatches to the wrapped app, rather than
    requiring a second middleware to re-parse the same prefix.

    This exercises the plain-forward path (no store, capture not requested) -- the common
    case for a model server with observability disabled.
    """
    observed: dict[str, object] = {}

    async def inner_app(scope, receive, send) -> None:
        observed["path"] = scope["path"]
        observed["rollout_id_inside"] = current_rollout_id()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    middleware = _CaptureMiddleware(inner_app, store=None, model_server_name="policy")
    scope = {
        "type": "http",
        "path": "/ng-rollout/rollout-42/v1/chat/completions",
        "raw_path": b"/ng-rollout/rollout-42/v1/chat/completions",
        "headers": [],
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    sent: list[dict] = []

    async def send(message):
        sent.append(message)

    await middleware(scope, receive, send)

    assert observed["path"] == "/v1/chat/completions"  # the prefix is still stripped
    assert observed["rollout_id_inside"] == "rollout-42"
    assert current_rollout_id() is None  # does not leak past the request


def test_capture_middleware_exposes_current_rollout_id_when_capture_is_enabled(tmp_path) -> None:
    """Same guarantee on the buffering/full-capture dispatch path (observability
    enabled), not just the plain-forward path above -- this is the branch with
    streaming/exception handling around the downstream call.
    """
    app = FastAPI()

    @app.post("/v1/responses")
    async def responses() -> dict:
        return {"rollout_id": current_rollout_id()}

    install_model_call_capture(
        app,
        ModelCallCaptureConfig(observability_enabled=True, model_call_capture_dir=tmp_path),
        model_server_name="policy",
    )
    client = TestClient(app)

    response = client.post("/ng-rollout/rollout-9/v1/responses", json={})
    assert response.status_code == 200
    assert response.json() == {"rollout_id": "rollout-9"}


class _EchoRolloutIdConfig(BaseResponsesAPIModelConfig):
    pass


class _EchoRolloutIdModel(SimpleResponsesAPIModel):
    config: _EchoRolloutIdConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        @app.get("/observed-rollout-id")
        async def observed_rollout_id() -> dict:
            return {"rollout_id": current_rollout_id()}

        return app

    async def chat_completions(self, body: dict = Body()) -> dict:
        raise NotImplementedError

    async def responses(self, body: dict = Body()) -> dict:
        raise NotImplementedError


def test_model_server_exposes_current_rollout_id_to_its_own_handler() -> None:
    """Regression test for the gap this change fixes.

    Before this change, ``current_rollout_id()`` was populated for resources and
    agent servers but always ``None`` inside a model server's own handler, even
    though every correlated call already carries the id in its URL prefix.
    """
    server_client = ServerClient(
        head_server_config=BaseServerConfig(host="head.test", port=80),
        global_config_dict=OmegaConf.create({}),
    )
    model = _EchoRolloutIdModel(
        config=_EchoRolloutIdConfig(host="policy.test", port=80, entrypoint="app.py", name="policy"),
        server_client=server_client,
    )
    client = TestClient(model.setup_webserver())

    correlated = client.get("/ng-rollout/rollout-7/observed-rollout-id")
    assert correlated.status_code == 200
    assert correlated.json() == {"rollout_id": "rollout-7"}

    uncorrelated = client.get("/observed-rollout-id")
    assert uncorrelated.status_code == 200
    assert uncorrelated.json() == {"rollout_id": None}


def test_an_explicit_rollout_id_with_an_attempt_suffix_cannot_also_take_an_attempt_index() -> None:
    # A controller may name a replacement attempt itself, as "<rollout>-a<N>".
    assert maybe_rollout_id_from_run_body({"_ng_rollout_id": "job-a2"}) == "job-a2"
    assert maybe_rollout_id_from_run_body({"_ng_rollout_id": "job", "_ng_attempt_index": 2}) == "job-a2"

    # Both at once would build "job-a2-a1", which decodes to no rollout and attempt.
    with pytest.raises(ValueError, match="already ends with the attempt suffix"):
        maybe_rollout_id_from_run_body({"_ng_rollout_id": "job-a2", "_ng_attempt_index": 1})


@pytest.mark.parametrize("top_p", [1e-12, 1.0])
def test_request_controls_preserve_zero_temperature_and_leave_omitted_controls_unset(top_p):
    request = NeMoGymResponseCreateParamsNonStreaming(input="task", temperature=0.0, top_p=top_p)
    original = request.model_dump()

    sampling = ModelCallSampling.from_request(request)

    assert sampling.model_dump(exclude_none=True) == {"temperature": 0.0, "top_p": top_p}
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
        {"top_p": 0.0},
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


@pytest.mark.parametrize(
    "payload", [{"temperature": True}, {"top_p": 0.0}, {"unknown": 1}, [], {"max_output_tokens": -1}]
)
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
