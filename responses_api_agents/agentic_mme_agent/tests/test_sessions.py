# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent sessions for the Agentic-MME agent, as the single-agent-turn Environment Server drives them."""

import asyncio

import pytest
from fastapi import HTTPException, Request, Response

from nemo_gym.agents import AgentCloseSessionRequest, AgentSeedRejectedError, AgentSeedSessionRequest
from resources_servers.agentic_mme.tests.test_app import response
from responses_api_agents.agentic_mme_agent.tests.test_app import FakeResponse, body, call, make_agent


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
