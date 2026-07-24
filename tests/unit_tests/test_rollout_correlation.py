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
from urllib.parse import urlsplit

import orjson
import pytest
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
    CaptureStore,
    ModelCallCaptureConfig,
    install_model_call_capture,
    merge_model_call_capture_into_record,
)
from nemo_gym.config_types import BaseServerConfig
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
    assert [exchange["model_ref"]["name"] for exchange in exchanges] == ["policy", "judge"]
    assert all(exchange.get("request") is not None or exchange.get("request_raw") for exchange in exchanges)
    assert all(exchange.get("response") is not None or exchange.get("response_raw") for exchange in exchanges)

    rollout = {"_ng_task_index": 4, "_ng_rollout_index": 2}
    merge_model_call_capture_into_record(rollout, [capture_dir])
    assert capture_path.is_file()
    assert len(capture_path.read_bytes()) > 0
