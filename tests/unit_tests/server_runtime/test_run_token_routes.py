# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A server that requires the run's token answers its orchestration routes only for callers that send it."""

from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest
from fastapi import Body, FastAPI, Request
from fastapi.testclient import TestClient
from omegaconf import DictConfig
from pydantic import BaseModel
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount

from nemo_gym.agents.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.client.server_client import ServerClient
from nemo_gym.config.config_types import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.core.episode_types import BaseEpisodeRequest, BaseEpisodeResponse
from nemo_gym.core.run_token import RUN_TOKEN_ENV, RUN_TOKEN_HEADER
from nemo_gym.server_runtime.base_environment_server import (
    BaseEnvironmentServer,
    BaseEnvironmentServerConfig,
    CleanupContext,
)
from nemo_gym.server_runtime.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.server_runtime.mcp_auto_exposure import TOKEN_HEADER, maybe_auto_expose
from nemo_gym.server_runtime.request_handling.run_token import require_run_token_on_routes


pytest.importorskip("mcp")

TOKEN = "the-run-token"
REQUIRED = DictConfig({"require_run_token": True})
NOT_REQUIRED = DictConfig({})
REFUSAL = {"detail": f"This route needs the run's token ({RUN_TOKEN_HEADER})"}
RPC_HEADERS = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}


@pytest.fixture(autouse=True)
def _a_run_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RUN_TOKEN_ENV, TOKEN)


class NoteRequest(BaseModel):
    topic: str = "today"


class Notes(SimpleResourcesServer):
    """A resources server with one tool."""

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        return BaseVerifyResponse(**body.model_dump(), reward=1.0)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        @app.post("/note")
        async def note(body: NoteRequest, request: Request) -> dict:
            """Read the note on a topic."""
            return {"note": "hi"}

        return app


def _started(server: Any, global_config: DictConfig = REQUIRED) -> tuple[FastAPI, list[str]]:
    """Build the server's app as run_webserver does, up to the token check."""
    app = server.setup_webserver()
    maybe_auto_expose(server, app)
    return app, server.setup_run_token_check(app, global_config)


def _notes(server_type: type[SimpleResourcesServer] = Notes, **options: Any) -> SimpleResourcesServer:
    config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="notes", **options)
    return server_type(config=config, server_client=MagicMock(spec=ServerClient))


def test_a_resources_server_answers_its_episode_and_run_routes_only_with_the_token() -> None:
    app, guarded = _started(_notes(expose_tools_over_mcp=True))

    assert sorted(guarded) == ["/aggregate_metrics", "/close_session", "/reverify_mode", "/seed_session", "/verify"]
    with TestClient(app) as client:
        for headers in ({}, {RUN_TOKEN_HEADER: "another-run"}, {RUN_TOKEN_HEADER: ""}):
            refused = client.post("/seed_session", json={}, headers=headers)
            assert (refused.status_code, refused.json()) == (401, REFUSAL)
        assert client.post("/seed_session", json={}, headers={RUN_TOKEN_HEADER: TOKEN}).status_code == 200
        # The rollout prefix comes off before routing, so the prefixed route is the same route.
        assert client.post("/ng-rollout/r-1/seed_session", json={}).status_code == 401
        # The refusal comes before the route reads the body: without the token an invalid body is never validated.
        assert client.post("/verify", json={}).status_code == 401
        assert client.post("/verify", json={}, headers={RUN_TOKEN_HEADER: TOKEN}).status_code == 422
        assert client.post("/close_session", json={}).status_code == 401
        assert client.post("/aggregate_metrics", json={}).status_code == 401
        assert client.get("/reverify_mode").status_code == 401
        assert client.get("/reverify_mode", headers={RUN_TOKEN_HEADER: TOKEN}).status_code == 200
        # A tool route and the MCP endpoint answer whoever calls them, with the credentials they check themselves.
        assert client.post("/note", json={}).json() == {"note": "hi"}
        listed = client.post(
            "/mcp", headers=RPC_HEADERS, json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        )
        assert [tool["name"] for tool in listed.json()["result"]["tools"]] == ["note"]


def test_a_run_that_does_not_require_the_token_checks_nothing() -> None:
    app, guarded = _started(_notes(), NOT_REQUIRED)

    assert guarded == []
    with TestClient(app) as client:
        assert client.post("/seed_session", json={}).status_code == 200
        assert client.get("/reverify_mode").status_code == 200


class Lending(Notes):
    """A server whose routes code in a sandbox reaches: it requires the token in every run, guards its tool too, and
    answers a call without the token as for a route that does not exist."""

    requires_run_token: ClassVar[bool] = True

    def run_token_routes(self) -> frozenset[str]:
        return super().run_token_routes() | {"/note"}

    def run_token_refusal(self) -> Response:
        return JSONResponse(status_code=404, content={"detail": "Not Found"})


def test_a_server_can_require_the_token_in_every_run_on_routes_of_its_own() -> None:
    app, guarded = _started(_notes(Lending, expose_tools_over_mcp=True), NOT_REQUIRED)

    assert "/note" in guarded and "/seed_session" in guarded
    with TestClient(app) as client:
        assert client.post("/note", json={}).json() == {"detail": "Not Found"}
        assert client.post("/note", json={}, headers={RUN_TOKEN_HEADER: TOKEN}).json() == {"note": "hi"}
        seeded = client.post("/seed_session", json={}, headers={RUN_TOKEN_HEADER: TOKEN})
        session_token = seeded.json()["mcp"]["headers"][TOKEN_HEADER]
        # Over MCP the tool's handler runs without its route, under the session token the MCP endpoint checks.
        called = client.post(
            "/mcp",
            headers={**RPC_HEADERS, TOKEN_HEADER: session_token},
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "note", "arguments": {}}},
        )
        assert called.json()["result"]["structuredContent"] == {"note": "hi"}


def test_a_required_token_that_the_process_lacks_stops_the_server_at_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(RUN_TOKEN_ENV)
    server = _notes()

    with pytest.raises(
        RuntimeError, match=rf"notes requires the run's token, but its process has none.*{RUN_TOKEN_ENV}"
    ):
        server.setup_run_token_check(server.setup_webserver(), REQUIRED)
    # Not required, no token needed.
    assert server.setup_run_token_check(server.setup_webserver(), NOT_REQUIRED) == []


class Agent(SimpleResponsesAPIAgent):
    """An agent with no /run of its own: its /run answers 422, which shows the call got past the token check."""

    async def responses(self, body: Any = Body()) -> dict:
        return {"answered": True}


def test_an_agent_answers_its_run_session_and_stop_routes_only_with_the_token() -> None:
    config = BaseResponsesAPIAgentConfig(host="", port=0, entrypoint="", name="agent")
    app, guarded = _started(Agent(config=config, server_client=MagicMock(spec=ServerClient)))

    assert sorted(guarded) == [
        "/aggregate_metrics",
        "/rollouts/{rollout_id}/stop",
        "/run",
        "/v1/agent_sessions",
        "/v1/agent_sessions/capabilities",
        "/v1/agent_sessions/close",
    ]
    run = {"responses_create_params": {"input": "hi"}}
    with TestClient(app) as client:
        assert (client.post("/run", json=run).status_code, client.post("/run", json=run).json()) == (401, REFUSAL)
        assert "does not implement /run" in client.post("/run", json=run, headers={RUN_TOKEN_HEADER: TOKEN}).text
        assert client.post("/rollouts/r-1/stop", json={}).status_code == 401
        assert client.post("/v1/agent_sessions", json={}).status_code == 401
        assert client.post("/v1/agent_sessions/close", json={}).status_code == 401
        assert client.get("/v1/agent_sessions/capabilities").status_code == 401
        assert client.get("/v1/agent_sessions/capabilities", headers={RUN_TOKEN_HEADER: TOKEN}).status_code == 200
        # A harness in a sandbox may call the agent's own model route.
        answered = client.post("/v1/responses", json={"input": "hi"})
        assert answered.status_code != 401 and answered.json() != REFUSAL


class Environment(BaseEnvironmentServer[BaseEpisodeRequest, BaseEpisodeResponse]):
    request_model = BaseEpisodeRequest
    response_model = BaseEpisodeResponse

    async def run(self, request: BaseEpisodeRequest, cleanup: CleanupContext) -> BaseEpisodeResponse:
        raise NotImplementedError

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        return AggregateMetrics()


def test_an_environment_server_answers_its_run_routes_only_with_the_token() -> None:
    config = BaseEnvironmentServerConfig(
        host="",
        port=0,
        entrypoint="",
        name="environment",
        default_episode_timeout_seconds=1,
        cleanup_timeout_seconds=1,
    )
    app, guarded = _started(Environment(config=config, server_client=MagicMock(spec=ServerClient)))

    assert sorted(guarded) == ["/aggregate_metrics", "/run"]
    with TestClient(app) as client:
        assert client.post("/run", json={}).status_code == 401
        assert client.post("/run", json={}, headers={RUN_TOKEN_HEADER: TOKEN}).status_code == 422
        assert client.post("/aggregate_metrics", json={"verify_responses": []}).status_code == 401
        assert (
            client.post(
                "/aggregate_metrics", json={"verify_responses": []}, headers={RUN_TOKEN_HEADER: TOKEN}
            ).status_code
            == 200
        )


def test_only_the_named_routes_are_guarded_and_a_mount_never_is() -> None:
    app = FastAPI()
    app.get("/kept")(lambda: {"kept": True})
    app.post("/guarded")(lambda: {"guarded": True})
    app.router.routes.append(Mount("/mounted", app=FastAPI()))

    guarded = require_run_token_on_routes(
        app, ["/guarded", "/mounted", "/absent"], lambda: JSONResponse(status_code=401, content={})
    )

    assert guarded == ["/guarded"]
    with TestClient(app) as client:
        assert client.get("/kept").json() == {"kept": True}
        assert client.post("/guarded").status_code == 401
        assert client.post("/guarded", headers={RUN_TOKEN_HEADER: TOKEN}).json() == {"guarded": True}
