# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent sessions for the Agentic-MME agent, as the single-agent-turn Environment Server drives them."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from aiohttp import ClientConnectionError
from fastapi import HTTPException, Request, Response
from omegaconf import OmegaConf

from nemo_gym.agents import (
    AGENT_OUTCOME_METADATA_KEY,
    AgentCloseSessionRequest,
    AgentOutcomePolicyConfig,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    OutcomeAction,
    agent_outcome_policy,
)
from nemo_gym.core import NeMoGymResponse
from resources_servers.agentic_mme.tests.test_app import response
from responses_api_agents.agentic_mme_agent.tests.test_app import (
    FakeResponse,
    UnreadableResponse,
    body,
    call,
    make_agent,
)


ROLLOUT = "amme-rollout"


def session_request(rollout_id: str = ROLLOUT, session: dict | None = None) -> Request:
    return Request(
        {
            "type": "http",
            "headers": [],
            "method": "POST",
            "scheme": "http",
            "server": ("localhost", 80),
            "path": f"/ng-rollout/{rollout_id}/v1/responses",
            "query_string": b"",
            "session": session if session is not None else {},
            "path_params": {"rollout_id": rollout_id},
        }
    )


def seed_body(**kwargs) -> AgentSeedSessionRequest:
    return AgentSeedSessionRequest(
        agent_session_id="amme-session",
        episode_id={"rollout_id": ROLLOUT},
        task_id={"taskset": "agentic_mme", "task_id": "example-0"},
        **kwargs,
    )


def close_body(seed: AgentSeedSessionRequest) -> AgentCloseSessionRequest:
    return AgentCloseSessionRequest(agent_session_id=seed.agent_session_id, episode_id=seed.episode_id)


@pytest.mark.asyncio
async def test_session_runs_one_episode_and_closes() -> None:
    agent, seen = make_agent([response(output=[call()]), response("<answer>red</answer>")])
    request = session_request()
    seed = seed_body()
    await agent.seed_agent_session(request, seed)
    assert await agent.seed_agent_session(request, seed) is not None

    outgoing = Response()
    result = await agent.responses(request, outgoing, body().responses_create_params)

    # The output also holds the tool image as a user message, so read the final answer from the last item.
    assert result.output[-1].content[0].text == "<answer>red</answer>"
    assert len(seen) == 2
    # Every model call goes through the rollout's route, so token capture can join the episode's calls.
    model_paths = [c.kwargs["url_path"] for c in agent.server_client.post.call_args_list]
    assert model_paths and all(path.startswith(f"/ng-rollout/{ROLLOUT}/") for path in model_paths)
    assert "model-session=test" in outgoing.headers["set-cookie"]

    closed = await agent.close_agent_session(request, close_body(seed))
    assert closed.agent_session_id == seed.agent_session_id
    assert await agent.close_agent_session(request, close_body(seed)) == closed
    with pytest.raises(HTTPException):
        await agent.responses(request, Response(), body().responses_create_params)


@pytest.mark.asyncio
async def test_identical_retry_joins_and_a_different_request_is_refused() -> None:
    agent, _ = make_agent([])
    release = asyncio.Event()
    model_calls = 0

    async def post(**kwargs):
        nonlocal model_calls
        model_calls += 1
        await release.wait()
        return FakeResponse(response("<answer>red</answer>").model_dump())

    agent.server_client.post.side_effect = post
    request = session_request()
    await agent.seed_agent_session(request, seed_body())
    params = body().responses_create_params

    first = asyncio.create_task(agent.responses(request, Response(), params))
    retry = asyncio.create_task(agent.responses(request, Response(), params.model_copy(deep=True)))
    await asyncio.sleep(0)
    with pytest.raises(HTTPException) as refused:
        await agent.responses(request, Response(), params.model_copy(update={"temperature": 0.5}))
    assert refused.value.status_code == 409

    release.set()
    first_result, retry_result = await asyncio.gather(first, retry)
    assert first_result.output[-1].content[0].text == "<answer>red</answer>"
    assert retry_result.model_dump() == first_result.model_dump()
    assert model_calls == 1


@pytest.mark.asyncio
async def test_close_cancels_a_running_episode() -> None:
    agent, _ = make_agent([])
    started = asyncio.Event()

    async def post(**kwargs):
        started.set()
        await asyncio.Event().wait()

    agent.server_client.post.side_effect = post
    request = session_request()
    seed = seed_body()
    await agent.seed_agent_session(request, seed)
    running = asyncio.create_task(agent.responses(request, Response(), body().responses_create_params))
    await started.wait()

    await agent.close_agent_session(request, close_body(seed))

    with pytest.raises(HTTPException) as closing:
        await running
    assert closing.value.status_code == 409


@pytest.mark.asyncio
async def test_a_request_under_another_rollout_is_refused() -> None:
    agent, _ = make_agent([response("<answer>red</answer>")])
    session: dict = {}
    await agent.seed_agent_session(session_request(session=session), seed_body())
    with pytest.raises(HTTPException) as refused:
        await agent.responses(
            session_request(rollout_id="another-rollout", session=session), Response(), body().responses_create_params
        )
    assert refused.value.status_code == 409


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config, seed_kwargs, message",
    [
        (
            {},
            {
                "sandbox_access": {
                    "connection": {"provider_config_ref": "sandbox", "descriptor": {"id": "lent"}},
                    "workdir": "/workspace",
                }
            },
            "sandbox",
        ),
        ({"retrieval": {"mode": "replay"}}, {}, "retrieval_replay"),
    ],
)
async def test_seed_refuses_what_the_agent_cannot_serve(config, seed_kwargs, message) -> None:
    agent, _ = make_agent([], **config)
    seed = seed_body(**seed_kwargs)
    with pytest.raises(AgentSeedRejectedError, match=message):
        await agent.seed_agent_session(session_request(), seed)


async def run_session_with_failing_call(failure, failing_call: int):
    """Run one session episode whose model call number ``failing_call`` (from 0) fails with ``failure``."""
    agent, _ = make_agent([response(output=[call()]), response("<answer>red</answer>")])
    post = agent.server_client.post.side_effect
    model_calls = 0

    async def post_or_fail(**kwargs):
        nonlocal model_calls
        if kwargs["url_path"].endswith("/v1/responses"):
            model_calls += 1
            if model_calls == failing_call + 1:
                if isinstance(failure, Exception):
                    raise failure
                return failure
        return await post(**kwargs)

    agent.server_client.post.side_effect = post_or_fail
    request = session_request()
    await agent.seed_agent_session(request, seed_body())
    return await agent.responses(request, Response(), body().responses_create_params)


CONTEXT_LIMIT = b'{"error": {"code": "request_too_large"}}'
IMAGE_LIMIT = b'{"error": {"message": "At most 1 image(s) may be provided in one prompt."}}'
UNSUPPORTED_PARAMETER = b'{"error": {"message": "Unsupported parameter: top_k"}}'
# The Environment Server's outcome policy that every Agentic-MME config uses.
BENCHMARK_POLICY = AgentOutcomePolicyConfig.model_validate(
    OmegaConf.to_container(
        OmegaConf.load(
            Path(__file__).resolve().parents[3] / "resources_servers/agentic_mme/configs/agentic_mme.yaml"
        ).agentic_mme_environment_server.environment_servers.single_agent_turn_legacy.agent_outcome_policy
    )
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "error_body", "outcome", "action"),
    [
        (400, CONTEXT_LIMIT, {"failure_kind": "model_context_limit"}, OutcomeAction.VERIFY),
        (413, CONTEXT_LIMIT, {"failure_kind": "model_context_limit"}, OutcomeAction.VERIFY),
        (400, IMAGE_LIMIT, {"failure_kind": "model_request_error", "status_code": 400}, OutcomeAction.FAIL),
        (400, UNSUPPORTED_PARAMETER, {"failure_kind": "model_request_error", "status_code": 400}, OutcomeAction.FAIL),
        (503, IMAGE_LIMIT, {"failure_kind": "model_request_error", "status_code": 503}, OutcomeAction.RETRY),
        (401, IMAGE_LIMIT, {"failure_kind": "model_request_error", "status_code": 401}, OutcomeAction.FAIL),
    ],
    ids=["context_400", "context_413", "image_limit", "unsupported_parameter", "unavailable", "bad_key"],
)
async def test_refused_model_call_is_graded_only_when_the_context_refused_it(
    status, error_body, outcome, action
) -> None:
    # As on /run: a context or request-size refusal is the model's own result and scores 0. A serving limit or an
    # unsupported parameter is not, so under the benchmark's outcome policy the rollout fails ungraded.
    refused = FakeResponse(None)
    refused.ok, refused.status = False, status
    refused.read = AsyncMock(return_value=error_body)
    result = await run_session_with_failing_call(refused, failing_call=1)

    assert result.status == "failed" and result.incomplete_details is None
    assert error_body.decode() in result.error.message
    assert json.loads(result.metadata[AGENT_OUTCOME_METADATA_KEY]) == outcome
    # The work before the refused call stays in the response: the tool call, its output and the tool image.
    assert [item.type for item in result.output][:2] == ["function_call", "function_call_output"]
    assert agent_outcome_policy(result, config=BENCHMARK_POLICY) == action


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "failure_kind"),
    [
        (TimeoutError(), "model_transport_error"),
        (ClientConnectionError("connection reset"), "model_transport_error"),
        (UnreadableResponse(body=b'{"not": "a response"}'), "unknown"),
    ],
    ids=["timeout", "connection", "not_a_response"],
)
async def test_model_call_that_fails_without_a_status_is_retried(failure, failure_kind) -> None:
    result = await run_session_with_failing_call(failure, failing_call=0)

    assert result.status == "failed" and result.output == []
    assert json.loads(result.metadata[AGENT_OUTCOME_METADATA_KEY]) == {"failure_kind": failure_kind}
    assert agent_outcome_policy(result) == OutcomeAction.RETRY


@pytest.mark.asyncio
async def test_unanswered_episode_without_a_failed_call_is_not_marked_failed() -> None:
    # Only a failed model call becomes a failed response; a model that stops on its own keeps its status.
    cut = NeMoGymResponse.model_validate(
        response(output=[call()]).model_dump()
        | {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}
    )
    agent, _ = make_agent([cut])
    request = session_request()
    await agent.seed_agent_session(request, seed_body())
    result = await agent.responses(request, Response(), body().responses_create_params)

    assert result.status == "incomplete" and AGENT_OUTCOME_METADATA_KEY not in (result.metadata or {})
