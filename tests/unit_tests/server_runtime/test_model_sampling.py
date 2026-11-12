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
from urllib.parse import urlsplit

import orjson
import pytest
from fastapi import Body, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from omegaconf import OmegaConf
from pydantic import ConfigDict
from starlette.testclient import TestClient

from nemo_gym.agents.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.client.http_client import get_response_json
from nemo_gym.client.server_client import ServerClient
from nemo_gym.config.config_types import TOKEN_CAPTURE_PATH_SEGMENT, BaseServerConfig
from nemo_gym.core.model_call_sampling import (
    MODEL_SAMPLING_NULL,
    ModelCallSampling,
    current_model_sampling_segment,
    encode_model_sampling,
    model_sampling_path_segment,
)
from nemo_gym.models.model_call_capture import CaptureStore, ModelCallCaptureConfig, install_model_call_capture
from nemo_gym.server_runtime.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.server_runtime.request_handling.model_sampling import (
    CallerSamplingMiddleware,
    ModelSamplingContextMiddleware,
    ModelSamplingMiddleware,
)


def _prefix(sampling: ModelCallSampling | None) -> str:
    return model_sampling_path_segment(encode_model_sampling(sampling))


def _echo_app(*paths: str, **middleware_options) -> FastAPI:
    app = FastAPI()

    async def echo(request: Request) -> dict:
        return {
            "path": request.url.path,
            "body": await request.json(),
            "segment": current_model_sampling_segment(),
        }

    for path in paths:
        app.add_api_route(path, echo, methods=["POST"])
    app.add_middleware(ModelSamplingMiddleware, **middleware_options)
    return app


def test_a_resources_server_exposing_tools_over_mcp_accepts_the_null_marker():
    """In a training run, a resources server with expose_tools_over_mcp installs the null marker and still starts."""
    from nemo_gym.server_runtime.mcp_auto_exposure import harvest_tools

    class Server:
        def mcp_tools(self, harvested, catchall):
            return harvested

    app = FastAPI()

    @app.post("/lookup")
    async def lookup() -> dict:
        return {}

    app.add_middleware(CallerSamplingMiddleware)
    assert CallerSamplingMiddleware.applies_to_mcp_requests
    assert "lookup" in harvest_tools(app, Server())


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
    original = {"model": "policy", "stream": stream, "temperature": 0.9, "top_p": 0.9, "metadata": {"trace": "kept"}}
    if dialect == "chat/completions":
        original |= {
            "messages": [{"role": "user", "content": "task"}],
            "max_tokens": 2048,
            "max_completion_tokens": 4096,
        }
    else:
        original |= {"input": "task", "max_output_tokens": 2048}
    before = deepcopy(original)
    sampling = [
        ModelCallSampling(temperature=0.0, top_p=0.7, max_output_tokens=17),
        ModelCallSampling(temperature=1.2, top_p=1.0, max_output_tokens=31),
    ]

    with TestClient(app) as client, ThreadPoolExecutor(max_workers=2) as pool:

        def post(index):
            return client.post(
                f"/ng-rollout/task-{index}-a2{_prefix(sampling[index])}/v1/{dialect}?trace=keep",
                json=original,
                headers={"x-session-id": f"session-{index}"},
            )

        results = list(pool.map(post, range(2)))

    assert original == before
    assert len(entered) == 2
    for index, result in enumerate(results):
        assert result.status_code == 200, result.text
        assert result.headers["x-test"] == "kept"
        expected = original | {"temperature": sampling[index].temperature, "top_p": sampling[index].top_p}
        if dialect == "chat/completions":
            expected.pop("max_tokens")
            expected["max_completion_tokens"] = sampling[index].max_output_tokens
        else:
            expected["max_output_tokens"] = sampling[index].max_output_tokens
        if stream:
            assert result.content == stream_body
            assert result.headers["content-type"].startswith("text/event-stream")
        else:
            assert result.json() == {"id": "completion", "received": expected}
        assert (expected, f"session-{index}", "trace=keep") in entered
        # Capture records what the provider received, not what the harness sent.
        [captured] = CaptureStore(tmp_path).read(f"task-{index}-a2")
        assert captured["request"] == expected


@pytest.mark.parametrize("dialect", ["chat/completions", "responses", "messages"])
def test_unset_values_keep_the_callers_own(dialect):
    app = _echo_app(f"/v1/{dialect}")
    request_body = {"temperature": 0.3, "top_p": 0.8, "max_tokens": 43, "metadata": {"kept": "yes"}}

    response = TestClient(app).post(f"{_prefix(ModelCallSampling(temperature=0.0))}/v1/{dialect}", json=request_body)

    assert response.status_code == 200
    assert response.json()["body"] == request_body | {"temperature": 0.0}


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
        ("responses", {"max_output_tokens": 11.0}, 11),
        # Not a usable limit, so the cap stands.
        ("responses", {"max_output_tokens": 0}, 73),
        ("responses", {"max_output_tokens": -1}, 73),
        ("responses", {"max_output_tokens": 0.5}, 73),
        # Claude Code asks for 32,000 tokens on every call.
        ("messages", {"max_tokens": 32000}, 73),
        ("messages", {"max_tokens": 11}, 11),
    ],
)
def test_the_cap_never_raises_a_smaller_limit_the_harness_set(dialect, harness_limits, expected_limit):
    app = _echo_app(f"/v1/{dialect}")
    sampling = ModelCallSampling(temperature=0.0, top_p=0.8, max_output_tokens=73)

    response = TestClient(app).post(
        f"{_prefix(sampling)}/v1/{dialect}", json={"temperature": 1.0, "top_p": 1.0} | harness_limits
    )

    assert response.status_code == 200, response.text
    limit_field = {"chat/completions": "max_completion_tokens", "responses": "max_output_tokens"}.get(
        dialect, "max_tokens"
    )
    assert response.json()["body"] == {"temperature": 0.0, "top_p": 0.8, limit_field: expected_limit}


@pytest.mark.parametrize("limit", [b"Infinity", b"-Infinity", b"NaN"])
def test_a_non_finite_limit_leaves_the_cap(limit):
    app = _echo_app("/v1/responses")
    sampling = ModelCallSampling(temperature=0.0, top_p=0.8, max_output_tokens=73)

    response = TestClient(app).post(
        f"{_prefix(sampling)}/v1/responses",
        content=b'{"max_output_tokens": ' + limit + b"}",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["body"] == {"temperature": 0.0, "top_p": 0.8, "max_output_tokens": 73}


def test_the_body_reaches_the_route_as_fastapi_parses_it():
    """A lone surrogate escape and an integer wider than 64 bits pass through unchanged."""
    app = FastAPI()

    async def inspect(request: Request) -> dict:
        body = await request.json()
        return {"content": ascii(body["messages"][0]["content"]), "seed": body["seed"], "top_p": body["top_p"]}

    app.add_api_route("/v1/chat/completions", inspect, methods=["POST"])
    app.add_middleware(ModelSamplingMiddleware)
    sampling = ModelCallSampling(temperature=0.0, top_p=0.8, max_output_tokens=73)

    response = TestClient(app).post(
        f"{_prefix(sampling)}/v1/chat/completions",
        content=b'{"messages": [{"role": "user", "content": "\\ud83d"}], "seed": 1180591620717411303424}',
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 200, response.text
    assert response.json() == {"content": "'\\ud83d'", "seed": 2**70, "top_p": 0.8}


def test_the_segment_is_read_after_an_agent_instance_name():
    app = _echo_app("/ng-rollout/r-a1/ng-agent-instance/judge/v1/responses", required=True)
    sampling = ModelCallSampling(temperature=0.0, top_p=0.8, max_output_tokens=73)

    response = TestClient(app).post(
        f"/ng-rollout/r-a1/ng-agent-instance/judge{_prefix(sampling)}/v1/responses", json={"temperature": 1.0}
    )

    assert response.status_code == 200, response.text
    assert response.json()["path"] == "/ng-rollout/r-a1/ng-agent-instance/judge/v1/responses"
    assert response.json()["body"] == {"temperature": 0.0, "top_p": 0.8, "max_output_tokens": 73}


@pytest.mark.parametrize("dialect", ["chat/completions", "responses", "messages"])
def test_null_leaves_the_request_as_the_caller_sent_it(dialect):
    app = _echo_app(f"/ng-rollout/judge-a1/v1/{dialect}", required=True)
    body = {"temperature": 0.0, "max_tokens": 5, "metadata": {"kept": "yes"}}

    response = TestClient(app).post(f"/ng-rollout/judge-a1{_prefix(None)}/v1/{dialect}", json=body)

    assert response.status_code == 200, response.text
    # The segment is stripped and stays current, so the Model Server can tell its generation server.
    assert response.json() == {
        "path": f"/ng-rollout/judge-a1/v1/{dialect}",
        "body": body,
        "segment": MODEL_SAMPLING_NULL,
    }


_REQUIRED_PATHS = (
    "/v1/responses",
    "/ng-rollout/4-2/v1/chat/completions",
    "/ng-rollout/4-2/ng-agent-instance/judge/v1/messages",
)


@pytest.mark.parametrize("path", _REQUIRED_PATHS)
def test_a_required_server_refuses_a_generation_call_without_a_segment(path):
    method = "POST"
    app = _echo_app(*_REQUIRED_PATHS, required=True)

    response = TestClient(app).request(method, path, json={"input": "task"})

    assert response.status_code == 422
    assert "/ng-sampling/" in response.json()["detail"]
    # Without the requirement, the call passes unchanged: Gym-only evaluation adds no segment.
    unrequired = TestClient(_echo_app(*_REQUIRED_PATHS))
    assert unrequired.request(method, path, json={"input": "task"}).json()["body"] == {"input": "task"}


@pytest.mark.parametrize("method,path", [("GET", "/v1/models"), ("POST", "/v1/embeddings"), ("GET", "/health")])
def test_a_required_server_serves_other_routes_without_a_segment(method, path):
    app = FastAPI()

    async def endpoint(request: Request):
        return {"body": (await request.body()).decode()}

    app.add_api_route(path, endpoint, methods=[method])
    app.add_middleware(ModelSamplingMiddleware, required=True)

    response = TestClient(app).request(method, path, content=b"not-json")

    assert response.status_code == 200
    assert response.json() == {"body": "not-json"}


@pytest.mark.parametrize("method,path", [("GET", "/v1/models"), ("POST", "/v1/control"), ("GET", "/health")])
def test_non_generation_routes_under_a_segment_keep_their_body(method, path):
    app = FastAPI()

    async def endpoint(request: Request):
        return {"path": request.url.path, "body": (await request.body()).decode(), "query": request.url.query}

    app.add_api_route(path, endpoint, methods=[method])
    app.add_middleware(ModelSamplingMiddleware)
    # A client that reuses its model base URL for metadata sends no generation payload.
    response = TestClient(app).request(
        method, f"{_prefix(ModelCallSampling(temperature=0.0))}{path}?key=value", content=b"not-json"
    )

    assert response.status_code == 200
    assert response.json() == {"path": path, "body": "not-json", "query": "key=value"}


@pytest.mark.parametrize(
    "encoded",
    [
        "not+urlsafe",
        "a" * 1025,
        *(
            base64.urlsafe_b64encode(raw).decode().rstrip("=")
            for raw in [
                b'{"top_p":0.1,"top_p":0.9}',
                b'{"temperature":NaN}',
                b'{"top_p":0.0}',
                b'{"unknown":1}',
                b"[]",
                b'{"max_output_tokens":-1}',
            ]
        ),
    ],
)
def test_invalid_values_never_reach_generation(encoded):
    app = FastAPI()

    @app.post("/v1/responses")
    async def generate():
        pytest.fail("Invalid values reached the model")

    app.add_middleware(ModelSamplingMiddleware)

    response = TestClient(app).post(f"/ng-sampling/{encoded}/v1/responses", json={"input": "task"})

    assert response.status_code == 422


@pytest.mark.parametrize("body", [b"invalid-json", b"[]", b"null"])
def test_an_invalid_generation_body_never_reaches_the_model(body):
    app = FastAPI()

    @app.post("/v1/responses")
    async def generate():
        pytest.fail("Invalid body reached the model")

    app.add_middleware(ModelSamplingMiddleware)

    response = TestClient(app).post(f"{_prefix(ModelCallSampling(temperature=0.0))}/v1/responses", content=body)

    assert response.status_code == 422


def test_a_segment_before_the_rollout_prefix_is_refused_rather_than_dropped():
    app = FastAPI()

    @app.post("/ng-rollout/misplaced/v1/responses")
    async def generate():
        pytest.fail("A misplaced segment reached the model")

    app.add_middleware(ModelSamplingMiddleware)

    response = TestClient(app).post(
        f"{_prefix(ModelCallSampling(temperature=0.0))}/ng-rollout/misplaced/v1/responses", json={"input": "task"}
    )

    assert response.status_code == 422


def test_a_rollout_named_like_the_segment_is_not_taken_for_one():
    response = TestClient(_echo_app("/ng-rollout/ng-sampling/v1/responses", required=True)).post(
        "/ng-rollout/ng-sampling/v1/responses", json={"input": "task", "temperature": 0.9}
    )

    # A rollout prefix is not a segment, so a required server refuses the call rather than misreading it.
    assert response.status_code == 422


@pytest.mark.parametrize("rollout_path", ["", f"/ng-rollout/mounted-a2/{TOKEN_CAPTURE_PATH_SEGMENT}"])
@pytest.mark.parametrize("mount", ["/relay", "/ng-sampling/relay"])
@pytest.mark.parametrize("sampled", [False, True])
def test_a_relay_mount_is_kept(rollout_path, mount, sampled):
    model_app = FastAPI()

    @model_app.post(f"{rollout_path}/v1/responses")
    async def generate(request: Request):
        return {"path": request.url.path, "body": await request.json(), "query": request.url.query}

    model_app.add_middleware(ModelSamplingMiddleware)
    relay = FastAPI()
    relay.mount(mount, model_app)
    sampling = ModelCallSampling(temperature=0.0, max_output_tokens=31) if sampled else None
    segment = _prefix(sampling) if sampled else ""

    response = TestClient(relay).post(
        f"{mount}{rollout_path}{segment}/v1/responses?trace=kept", json={"input": "task"}
    )

    assert response.status_code == 200, response.text
    assert response.json() == {
        "path": f"{mount}{rollout_path}/v1/responses",
        "body": {"input": "task"} | ({"temperature": 0.0, "max_output_tokens": 31} if sampled else {}),
        "query": "trace=kept",
    }


@pytest.mark.parametrize("disconnect_early", [False, True])
async def test_chunked_requests_are_reassembled_or_abandoned_on_disconnect(disconnect_early):
    original = '{"input":"café","temperature":1.0}'.encode()
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

    path = f"{_prefix(ModelCallSampling(temperature=0.0))}/v1/responses"
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
        assert json.loads(body["body"]) == {"input": "café", "temperature": 0.0}
        assert dict(scope["headers"])[b"content-length"] == str(len(body["body"])).encode()
        assert dict(scope["headers"])[b"x-trace"] == b"kept"
        assert sent[-1]["body"] == b"done"


def test_an_agent_strips_an_inbound_segment_and_keeps_it_for_its_model_calls():
    app = FastAPI()

    @app.post("/ng-rollout/{rollout_id}/v1/responses")
    async def responses(rollout_id: str):
        return {"rollout_id": rollout_id, "segment": current_model_sampling_segment()}

    app.add_middleware(ModelSamplingContextMiddleware)
    client = TestClient(app)
    segment = encode_model_sampling(ModelCallSampling(temperature=1.0))

    response = client.post(f"/ng-rollout/4-2/ng-sampling/{segment}/v1/responses", json={})

    assert response.json() == {"rollout_id": "4-2", "segment": segment}
    assert client.post("/ng-rollout/4-2/v1/responses", json={}).json() == {"rollout_id": "4-2", "segment": None}
    assert client.post("/ng-rollout/4-2/ng-sampling/not+valid/v1/responses", json={}).status_code == 422


def test_a_resources_server_marks_its_model_calls_as_keeping_their_own_sampling():
    app = FastAPI()

    @app.post("/verify")
    async def verify():
        return {"segment": current_model_sampling_segment()}

    app.add_middleware(CallerSamplingMiddleware)

    assert TestClient(app).post("/verify").json() == {"segment": MODEL_SAMPLING_NULL}


class _RecordingModelApp:
    """A Model Server app that records each call's path, body and current segment."""

    def __init__(self, name: str, *, required: bool) -> None:
        self.calls: list[dict] = []
        self.app = FastAPI()

        @self.app.post("/v1/responses")
        async def responses(request: Request) -> dict:
            self.calls.append(
                {"path": request.url.path, "body": await request.json(), "segment": current_model_sampling_segment()}
            )
            return {
                "id": f"resp-{name}",
                "created_at": 0.0,
                "model": name,
                "object": "response",
                "output": [],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            }

        self.app.add_middleware(ModelSamplingMiddleware, required=required)


class _JudgeResourcesServer(SimpleResourcesServer):
    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/tool")(self.tool)
        return app

    async def tool(self, body: dict = Body()) -> dict:
        response = await self.server_client.post(server_name="policy", url_path="/v1/responses", json=body)
        return await get_response_json(response)

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        judge = await self.server_client.post(
            server_name="policy", url_path="/v1/responses", json={"input": "grade", "temperature": 0.0}
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
        tool = await self.server_client.post(server_name="resources", url_path="/tool", json={"input": "lookup"})
        await get_response_json(tool)
        verify = await self.server_client.post(
            server_name="resources",
            url_path="/verify",
            json={"responses_create_params": body.responses_create_params.model_dump(), "response": response},
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
@pytest.mark.parametrize("training", [True, False])
async def test_a_rollouts_values_reach_its_agents_calls_and_untrained_callers_keep_their_own(
    monkeypatch, training
) -> None:
    """In training, the agent's call carries the rollout's values; the judge's and a tool's carry ``null``.

    Without the global block or the row key, as in Gym-only evaluation, no call carries a segment.
    """
    config = {
        "policy": {"responses_api_models": {"model": {"host": "policy.test", "port": 80}}},
        "resources": {"resources_servers": {"judge": {"host": "resources.test", "port": 80}}},
        "agent": {"responses_api_agents": {"agent": {"host": "agent.test", "port": 80}}},
    }
    if training:
        config["model_call_sampling"] = {"required_model_servers": ["policy"]}
    server_client = ServerClient(
        head_server_config=BaseServerConfig(host="head.test", port=80),
        global_config_dict=OmegaConf.create(config),
    )
    resources = _JudgeResourcesServer(
        config=BaseResourcesServerConfig(host="resources.test", port=80, entrypoint="app.py", name="resources"),
        server_client=server_client,
    )
    agent = _Agent(
        config=BaseResponsesAPIAgentConfig(host="agent.test", port=80, entrypoint="app.py", name="agent"),
        server_client=server_client,
    )
    policy = _RecordingModelApp("policy", required=training)
    clients = {
        "policy.test": TestClient(policy.app),
        "resources.test": TestClient(resources.setup_webserver()),
        "agent.test": TestClient(agent.setup_webserver()),
    }

    async def dispatch(method: str, url: str, **kwargs):
        parsed = urlsplit(url)
        path = f"{parsed.path}?{parsed.query}" if parsed.query else parsed.path
        return _Response(clients[parsed.hostname].request(method, path, json=kwargs.get("json")))

    monkeypatch.setattr("nemo_gym.client.server_client.request", dispatch)
    sampling = {"temperature": 0.6, "top_p": 0.95, "max_output_tokens": 64}
    row = {"responses_create_params": {"input": "solve", "temperature": 0.2, "max_output_tokens": 4096}}
    if training:
        row["_ng_model_sampling"] = sampling

    verify = await server_client.post(server_name="agent", url_path="/run", json=row)

    assert orjson.loads(await verify.read())["reward"] == 1.0
    agent_call, tool_call, judge_call = policy.calls
    assert agent_call["path"] == tool_call["path"] == judge_call["path"] == "/v1/responses"
    if training:
        assert agent_call["segment"] == encode_model_sampling(ModelCallSampling(**sampling))
        assert agent_call["body"] == {"input": "solve", **sampling}
        assert tool_call["segment"] == judge_call["segment"] == MODEL_SAMPLING_NULL
    else:
        assert agent_call["segment"] is tool_call["segment"] is judge_call["segment"] is None
        assert agent_call["body"] == row["responses_create_params"]
    assert judge_call["body"] == {"input": "grade", "temperature": 0.0}
    assert tool_call["body"] == {"input": "lookup"}
