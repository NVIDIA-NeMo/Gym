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
"""Integration test for the adapters lift on Agent + Resources bases.

Confirms that the same ``adapters`` config field + ``install_middleware``
call shipped on ``BaseResponsesAPIModelConfig`` is now also live on
``BaseResponsesAPIAgentConfig`` and ``BaseResourcesServerConfig``, so any
server inheriting from ``SimpleResponsesAPIAgent`` or ``SimpleResourcesServer``
picks up the middleware automatically.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

from fastapi import Body
from fastapi.testclient import TestClient

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import ServerClient


class _StubAgent(SimpleResponsesAPIAgent):
    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:  # type: ignore[override]
        raise NotImplementedError

    async def run(self, body: BaseRunRequest = Body()) -> BaseVerifyResponse:  # type: ignore[override]
        return BaseVerifyResponse(
            responses_create_params=body.responses_create_params,
            response={"id": "stub", "model": "stub", "usage": {"total_tokens": 7}},
            reward=1.0,
        )


class _StubResources(SimpleResourcesServer):
    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:  # type: ignore[override]
        return BaseVerifyResponse(
            responses_create_params=body.responses_create_params,
            response=body.response,
            reward=0.5,
        )


def _agent_config(adapters: list[dict] | None) -> BaseResponsesAPIAgentConfig:
    return BaseResponsesAPIAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="stub_agent",
        adapters=adapters,
    )


def _resources_config(adapters: list[dict] | None) -> BaseResourcesServerConfig:
    return BaseResourcesServerConfig(
        host="0.0.0.0",
        port=8081,
        entrypoint="",
        name="stub_resources",
        adapters=adapters,
    )


def test_agent_base_installs_adapter_chain(caplog) -> None:
    agent = _StubAgent(
        config=_agent_config(adapters=[{"name": "logging", "config": {}}]),
        server_client=MagicMock(spec=ServerClient),
    )
    app = agent.setup_webserver()

    # POST to a path the agent server doesn't route — middleware fires
    # before routing, so the logging interceptor records the request
    # regardless of the 404 that follows.
    with caplog.at_level(logging.INFO, logger="nemo_gym.adapters.interceptors.request_logging"):
        TestClient(app).post("/_probe", json={"x": 1})

    assert any("request POST /_probe" in rec.message for rec in caplog.records), [
        rec.message for rec in caplog.records
    ]


def test_agent_base_skips_middleware_when_adapters_none(caplog) -> None:
    agent = _StubAgent(
        config=_agent_config(adapters=None),
        server_client=MagicMock(spec=ServerClient),
    )
    app = agent.setup_webserver()

    with caplog.at_level(logging.INFO, logger="nemo_gym.adapters.interceptors.request_logging"):
        TestClient(app).post("/_probe", json={"x": 1})

    assert not any("request POST /_probe" in rec.message for rec in caplog.records)


def test_resources_base_installs_adapter_chain(caplog) -> None:
    resources = _StubResources(
        config=_resources_config(adapters=[{"name": "logging", "config": {}}]),
        server_client=MagicMock(spec=ServerClient),
    )
    app = resources.setup_webserver()

    with caplog.at_level(logging.INFO, logger="nemo_gym.adapters.interceptors.request_logging"):
        TestClient(app).post("/_probe", json={"x": 1})

    assert any("request POST /_probe" in rec.message for rec in caplog.records), [
        rec.message for rec in caplog.records
    ]


class _StubAgentInlineMiddleware(SimpleResponsesAPIAgent):
    """Mirrors the harbor_agent / mini_swe_agent override pattern: rebuilds
    the FastAPI app from scratch but calls install_middleware inline so the
    `adapters` field still applies.
    """

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:  # type: ignore[override]
        raise NotImplementedError

    async def run(self, body: BaseRunRequest = Body()) -> BaseVerifyResponse:  # type: ignore[override]
        raise NotImplementedError

    def setup_webserver(self):
        from fastapi import FastAPI

        from nemo_gym.adapters import install_middleware

        app = FastAPI()
        app.post("/v1/responses")(self.responses)
        app.post("/run")(self.run)
        install_middleware(app, self.config.adapters)
        return app


def test_agent_override_with_inline_install_middleware(caplog) -> None:
    """Regression test for the harbor/mini_swe override fix.

    The override doesn't call super().setup_webserver(), so the base class's
    install_middleware call doesn't fire. Per the fix, each override adds its
    own install_middleware(app, self.config.adapters) call so the YAML
    `adapters:` block still takes effect.
    """
    agent = _StubAgentInlineMiddleware(
        config=_agent_config(adapters=[{"name": "logging", "config": {}}]),
        server_client=MagicMock(spec=ServerClient),
    )
    app = agent.setup_webserver()

    with caplog.at_level(logging.INFO, logger="nemo_gym.adapters.interceptors.request_logging"):
        TestClient(app).post("/_probe", json={"x": 1})

    assert any("request POST /_probe" in rec.message for rec in caplog.records), [
        rec.message for rec in caplog.records
    ]


def test_resources_base_skips_middleware_when_adapters_none(caplog) -> None:
    resources = _StubResources(
        config=_resources_config(adapters=None),
        server_client=MagicMock(spec=ServerClient),
    )
    app = resources.setup_webserver()

    with caplog.at_level(logging.INFO, logger="nemo_gym.adapters.interceptors.request_logging"):
        TestClient(app).post("/_probe", json={"x": 1})

    assert not any("request POST /_probe" in rec.message for rec in caplog.records)
