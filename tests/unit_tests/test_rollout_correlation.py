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
import functools
from urllib.parse import urlsplit

import httpx
import orjson
import pytest
import requests
from fastapi import Body, FastAPI
from fastapi.responses import JSONResponse
from omegaconf import OmegaConf
from pydantic import ConfigDict
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
from nemo_gym.config_types import BaseServerConfig
from nemo_gym.observability.model_calls import merge_model_call_capture_into_record
from nemo_gym.rollout_correlation import (
    ROLLOUT_ID_HEADER,
    RolloutContextMiddleware,
    add_rollout_id_header,
    add_rollout_id_header_async,
    current_rollout_id,
    maybe_rollout_id_from_run_body,
    rollout_context,
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


def _rollout_echo_app() -> FastAPI:
    app = FastAPI()

    @app.get("/whoami")
    async def _whoami() -> dict:
        return {"rollout_id": current_rollout_id()}

    app.add_middleware(RolloutContextMiddleware)
    return app


def test_rollout_id_header_names_the_rollout_like_the_path_prefix() -> None:
    client = TestClient(_rollout_echo_app())

    assert client.get("/whoami", headers={ROLLOUT_ID_HEADER: "3-0"}).json() == {"rollout_id": "3-0"}
    assert client.get("/ng-rollout/3-0/whoami", headers={ROLLOUT_ID_HEADER: "3-0"}).json() == {"rollout_id": "3-0"}
    assert client.get("/ng-rollout/3-0/whoami").json() == {"rollout_id": "3-0"}
    assert client.get("/whoami").json() == {"rollout_id": None}


@pytest.mark.parametrize(
    ("path", "headers", "message"),
    [
        ("/ng-rollout/3-0/whoami", [(ROLLOUT_ID_HEADER, "4-0")], "names rollout '3-0'"),
        ("/whoami", [(ROLLOUT_ID_HEADER, ".hidden")], "must be letters"),
        ("/whoami", [(ROLLOUT_ID_HEADER, "3-0"), (ROLLOUT_ID_HEADER, "4-0")], "repeated with different values"),
    ],
)
def test_contradictory_or_malformed_rollout_id_headers_are_refused(path, headers, message) -> None:
    response = TestClient(_rollout_echo_app()).get(path, headers=headers)

    assert response.status_code == 400
    assert message in response.json()["detail"]


async def test_a_shared_client_tags_each_concurrent_request_with_its_own_rollout() -> None:
    base_url = "http://gym-model"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_rollout_echo_app()),
        base_url=base_url,
        event_hooks={"request": [functools.partial(add_rollout_id_header_async, base_urls=[base_url])]},
    ) as client:

        async def call(rollout_id: str) -> str:
            with rollout_context(rollout_id):
                await asyncio.sleep(0)
                return (await client.get("/whoami")).json()["rollout_id"]

        rollout_ids = [f"{task}-0" for task in range(20)]
        assert await asyncio.gather(*(call(rollout_id) for rollout_id in rollout_ids)) == rollout_ids


def test_add_rollout_id_header_tags_only_listed_gym_servers_and_only_once() -> None:
    base_urls = ["http://gym-model:8000/"]
    request = httpx.Request("POST", "http://gym-model:8000/v1/responses")
    with rollout_context("3-0"):
        add_rollout_id_header(request, base_urls=base_urls)
        add_rollout_id_header(request, base_urls=base_urls)
        unrelated = httpx.Request("POST", "http://other-host/v1/responses")
        add_rollout_id_header(unrelated, base_urls=base_urls)
        lookalike = httpx.Request("POST", "http://gym-model:80001/v1/responses")
        add_rollout_id_header(lookalike, base_urls=base_urls)
        prepared = requests.Request("POST", "http://gym-model:8000/v1/chat/completions").prepare()
        add_rollout_id_header(prepared, base_urls=base_urls)
    outside_rollout = httpx.Request("POST", "http://gym-model:8000/v1/responses")
    add_rollout_id_header(outside_rollout, base_urls=base_urls)

    assert request.headers.get_list(ROLLOUT_ID_HEADER) == ["3-0"]
    assert ROLLOUT_ID_HEADER not in unrelated.headers
    assert ROLLOUT_ID_HEADER not in lookalike.headers
    assert prepared.headers[ROLLOUT_ID_HEADER] == "3-0"
    assert ROLLOUT_ID_HEADER not in outside_rollout.headers
