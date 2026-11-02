# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model -> agent (sessions) -> single_agent_turn environment server over real local sockets."""

from unittest.mock import MagicMock

import pytest
from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer
from fastapi import Body, FastAPI, Response
from omegaconf import OmegaConf

import nemo_gym.client.http_client
import nemo_gym.openai_utils as openai_utils_module
import nemo_gym.server_utils
from environment_servers.single_agent_turn.app import (
    SingleAgentTurnEnvironmentServer,
    SingleAgentTurnEnvironmentServerConfig,
)
from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionResponse,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.client.server_client import ServerClient
from nemo_gym.config.config_types import AgentServerRef, BaseServerConfig, ResourcesServerRef
from nemo_gym.core.episode import EpisodeId, TaskId
from nemo_gym.core.responses_api_types import NeMoGymResponse
from nemo_gym.failure_transport import FailureHeader, RequestFailureLog
from nemo_gym.openai_utils import NeMoGymAsyncOpenAI
from nemo_gym.server_runtime.episode_types import FailureReport, MaterializedTask
from nemo_gym.server_runtime.single_agent_turn_types import SingleAgentTurnRequest, SingleAgentTurnTaskInput
from tests.unit_tests.test_agent_environment_collection_failures import _serve


def _agent_response() -> NeMoGymResponse:
    return NeMoGymResponse(
        id="response",
        created_at=0,
        model="model",
        object="response",
        output=[
            {
                "type": "message",
                "role": "assistant",
                "id": "message",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Patch applied", "annotations": []}],
                "prompt_token_ids": [1, 2],
                "generation_token_ids": [3, 4],
                "generation_log_probs": [-0.1, -0.2],
            }
        ],
        tool_choice="auto",
        parallel_tool_calls=True,
        tools=[],
    )


def _request() -> SingleAgentTurnRequest:
    return SingleAgentTurnRequest(
        episode_id=EpisodeId(rollout_id="rollout", attempt=2),
        task=MaterializedTask(
            task_id=TaskId(taskset="test:source", task_id="task"),
            task_input=SingleAgentTurnTaskInput(
                responses_create_params={"input": "task"},
                task_data={"instance_id": "task"},
            ),
        ),
    )


class _SessionModelAgent(SimpleResponsesAPIAgent):
    model_url: str = ""

    async def responses(self, body: dict = Body(default_factory=dict)) -> dict:
        client = NeMoGymAsyncOpenAI(api_key="k", base_url=self.model_url, max_http_attempts=2)
        return await client.create_response(model="m", input="task")

    async def run(self, body: dict = Body(default_factory=dict)) -> dict:
        raise NotImplementedError

    async def _seed_agent_session_state(self, body) -> AgentSessionState:
        return AgentSessionState(request=body)

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        return AgentCloseSessionResponse(agent_session_id=state.request.agent_session_id)


def _resources_app(verified: list) -> FastAPI:
    app = FastAPI()

    @app.post("/seed_session")
    async def seed(response: Response, body: dict = Body(...)) -> dict:
        response.set_cookie("session", "resources-cookie")  # The environment requires a resources session cookie.
        return {"resources_session_id": body.get("resources_session_id", "resources-session")}

    @app.post("/verify")
    async def verify(body: dict = Body(...)) -> dict:
        verified.append(body)
        return {"responses_create_params": {"input": "task"}, "response": body["response"], "reward": 1.0}

    @app.post("/close_session")
    async def close(body: dict = Body(...)) -> dict:
        return {"resources_session_id": body.get("resources_session_id", "resources-session")}

    return app


async def _episode(monkeypatch, statuses: list[int], *, via_http: bool = False):
    model_hits, observed, verified = [], [], []
    reply = _agent_response().model_dump(mode="json")

    async def model(request):
        model_hits.append(request.path)
        status = statuses[min(len(model_hits) - 1, len(statuses) - 1)]
        return web.json_response(reply if status == 200 else {"error": "busy"}, status=status)

    real_observe = openai_utils_module.observe_failure
    monkeypatch.setattr(openai_utils_module, "observe_failure", lambda e: (observed.append(e), real_observe(e)))
    monkeypatch.setattr(openai_utils_module, "_retry_wait_seconds", lambda *args: 0)
    app = web.Application()
    app.router.add_route("POST", "/{tail:.*}", model)
    servers = []
    async with TestServer(app) as model_server, ClientSession() as session:
        monkeypatch.setattr(nemo_gym.client.http_client, "_GLOBAL_AIOHTTP_CLIENT", session)
        try:
            agent = _SessionModelAgent(
                config=BaseResponsesAPIAgentConfig(host="127.0.0.1", port=0, entrypoint="", name="agent"),
                server_client=MagicMock(spec=ServerClient),
                model_url=str(model_server.make_url("/v1")),
            )
            agent_app = agent.setup_webserver()
            agent.setup_exception_middleware(agent_app)
            servers.append(await _serve(agent_app))
            servers.append(await _serve(_resources_app(verified)))
            config = {
                "resources": {"resources_servers": {"test": {"host": "127.0.0.1", "port": servers[1][2]}}},
                "agent": {"responses_api_agents": {"test": {"host": "127.0.0.1", "port": servers[0][2]}}},
            }
            client = ServerClient(
                head_server_config=BaseServerConfig(host="127.0.0.1", port=1),
                global_config_dict=OmegaConf.create(config),
            )
            environment = SingleAgentTurnEnvironmentServer(
                config=SingleAgentTurnEnvironmentServerConfig(
                    name="environment",
                    host="127.0.0.1",
                    port=0,
                    entrypoint="app.py",
                    resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
                    agent_server=AgentServerRef(type="responses_api_agents", name="agent"),
                    cleanup_timeout_seconds=10,
                ),
                server_client=client,
            )
            # The environment's middleware serves each request inside RequestFailureLog.capture; do the same here.
            relayed: dict = {}
            if via_http:
                env_app = environment.setup_webserver()
                environment.setup_exception_middleware(env_app)
                servers.append(await _serve(env_app))
                reply = await nemo_gym.client.http_client.request(
                    method="POST",
                    url=f"http://127.0.0.1:{servers[-1][2]}/run",
                    json=_request().model_dump(mode="json"),
                    _max_num_tries=1,
                )
                relayed = FailureHeader.decode(reply.headers).events
                result = type(_episode_response()).model_validate_json(await reply.read())
                print("TYPED_ENV_HTTP_HEADERS", reply.status, dict(reply.headers))
            else:
                with RequestFailureLog.capture() as observed_here:
                    result = await environment.run_request(_request())
                relayed = dict(observed_here.report.events)
        finally:
            for server, task, _ in servers:
                server.should_exit = True
                await task
    print("TYPED_ENV", statuses, result.model_dump(mode="json"))
    return result, model_hits, observed, verified, dict(relayed)


def _set_tracing(monkeypatch, traced: bool) -> None:
    """Select the traced or the untraced request path; diagnostics must not depend on it."""
    monkeypatch.setattr(nemo_gym.server_utils, "is_trace_exporter_active", lambda: traced)
    monkeypatch.setattr(nemo_gym.server_utils, "is_span_group_enabled", lambda group: traced)


def _episode_response():
    from environment_servers.single_agent_turn.app import SingleAgentTurnResponse

    return SingleAgentTurnResponse.model_construct()


@pytest.mark.parametrize("traced", [False, True])
async def test_over_http_a_recovered_model_call_reaches_the_environment_reply(monkeypatch, traced: bool) -> None:
    _set_tracing(monkeypatch, traced)
    result, model_hits, observed, verified, relayed = await _episode(monkeypatch, [503, 200], via_http=True)

    assert result.failure is None and result.result.reward == 1.0 and len(model_hits) == 2
    assert list(relayed) == [observed[0].event.event_id]


@pytest.mark.parametrize("traced", [False, True])
async def test_over_http_an_exhausted_model_call_is_the_environments_agent_stage_failure(
    monkeypatch, traced: bool
) -> None:
    _set_tracing(monkeypatch, traced)
    result, model_hits, observed, verified, relayed = await _episode(monkeypatch, [502], via_http=True)

    failure = result.failure
    report = result.failure_report
    report = result.failure_report
    assert failure is not None and failure.stage == "agent" and len(model_hits) == 2
    events = report.events
    assert events[report.failure_id].boundary == "environment"
    chain, current = [], report.failure_id
    while current is not None:
        chain.append(current)
        current = events[current].cause_id
    assert [events[e].boundary for e in chain].count("agent") == 1
    assert chain[-1] == observed[-1].event.event_id
    assert observed[0].event.event_id not in chain and (
        observed[0].event.event_id in relayed or observed[0].event.event_id in events
    )


async def test_a_recovered_model_call_verifies_and_reports_the_observation(monkeypatch) -> None:
    result, model_hits, observed, verified, relayed = await _episode(monkeypatch, [503, 200])

    assert result.failure is None and result.result.reward == 1.0 and len(verified) == 1
    assert len(model_hits) == 2 and len(observed) == 1
    assert list(relayed) == [
        observed[0].event.event_id
    ]  # The recovered attempt reaches the environment's reply, same id.


async def test_an_exhausted_model_call_is_the_environments_own_agent_stage_failure(monkeypatch) -> None:
    result, model_hits, observed, verified, relayed = await _episode(monkeypatch, [502])

    failure = result.failure
    report = result.failure_report
    report = result.failure_report
    assert failure is not None and failure.stage == "agent" and not verified
    assert len(model_hits) == 2
    events = report.events
    own = events[report.failure_id]
    assert own.boundary == "environment"
    chain, current = [], report.failure_id
    while current is not None:
        chain.append(current)
        current = events[current].cause_id
    types = [events[event_id].boundary for event_id in chain]
    assert "agent" in types and types.count("agent") == 1
    assert chain[-1] == observed[-1].event.event_id  # The explicit chain ends at the final attempt's own ModelFailure.
    # The earlier attempt is a sibling observation: it travels in the reply's observations, not in the chain.
    assert observed[0].event.event_id not in chain and observed[0].event.event_id in relayed


@pytest.mark.parametrize("typed_cause", [False, True])
def test_an_unhandled_environment_error_owns_its_report_without_borrowing_the_causes_kind(typed_cause: bool) -> None:
    from fastapi.testclient import TestClient

    from nemo_gym.core.failure_kinds import MODEL_REQUEST_ERROR
    from nemo_gym.server_runtime.episode_types import EpisodeFailure, FailureBoundary
    from nemo_gym.typed_exceptions import NgFailureEvent
    from tests.unit_tests.server_runtime.test_environment_server import (
        _environment_server,
        _EnvironmentServer,
        _request,
    )

    provider = RuntimeError("unexpected dependency failure")
    model = NgFailureEvent(
        boundary=FailureBoundary.MODEL,
        failure_kind=MODEL_REQUEST_ERROR,
        failure_reason="model request failed",
        http_status=503,
    )

    class FailedEnvironment(_EnvironmentServer):
        async def run(self, request, cleanup):
            if typed_cause:
                raise model from provider
            raise provider

    template = _environment_server()
    server = FailedEnvironment(config=template.config, server_client=template.server_client)
    with TestClient(server.setup_webserver()) as client:
        reply = client.post("/run", json=_request().model_dump(mode="json"))

    assert reply.status_code == 200 and reply.json()["result"] is None
    failure = EpisodeFailure.model_validate(reply.json()["failure"])
    report = FailureReport.model_validate(reply.json()["failure_report"])
    report = FailureReport.model_validate(reply.json()["failure_report"])
    assert failure.terminal is True and failure.failure_kind is None
    own = report.events[report.failure_id]
    assert own.boundary is FailureBoundary.ENVIRONMENT and own.failure_kind == "environment_episode_failed"
    assert own.failure_reason == failure.failure_reason
    cause = report.events[own.cause_id] if typed_cause else own
    if typed_cause:
        assert (cause.event_id, cause.boundary, cause.failure_kind, cause.http_status) == (
            model.event.event_id,
            FailureBoundary.MODEL,
            MODEL_REQUEST_ERROR,
            503,
        )
    assert cause.cause_id is None
    assert cause.foreign_error.exception_type == "RuntimeError" and cause.foreign_error.message == ""
