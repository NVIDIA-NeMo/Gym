# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent sessions for the Agentic-MME agent, as the single-agent-turn Environment Server drives them."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiohttp import ClientConnectionError
from fastapi import HTTPException, Request, Response
from omegaconf import OmegaConf

from environment_servers.single_agent_turn_legacy.app import (
    SingleAgentTurnLegacyEnvironmentServer,
    SingleAgentTurnLegacyEnvironmentServerConfig,
)
from nemo_gym.agents import (
    AGENT_OUTCOME_METADATA_KEY,
    AgentCloseSessionRequest,
    AgentOutcomePolicyConfig,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    OutcomeAction,
    agent_outcome_policy,
)
from nemo_gym.client import ServerClient
from nemo_gym.config import BaseServerConfig
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollouts import RolloutCollectionConfig, RolloutCollectionHelper
from resources_servers.agentic_mme.app import AgenticMMEConfig, AgenticMMEServer, AgenticMMEVerifyRequest
from resources_servers.agentic_mme.tests.test_app import no_session, response
from resources_servers.agentic_mme.tests.test_opencode_recipes import load
from responses_api_agents.agentic_mme_agent.app import AgenticMMEAgent, AgenticMMEAgentConfig
from responses_api_agents.agentic_mme_agent.retrieval import Retrieval
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


class Reply(FakeResponse):
    """A successful HTTP reply, as the collector and the Environment Server read it."""

    status = 200
    headers: dict = {}


@pytest.mark.asyncio
async def test_a_collected_session_rollout_keeps_the_tool_trace_and_its_counts(monkeypatch) -> None:
    """The default recipe's rollout row keeps what /run returned: the trace, its retrieval metadata and the counts.

    Collection, the single-agent-turn Environment Server, the agent's session routes and the grader run in this
    process; only the policy model and the live search provider are scripted.
    """
    # A live search result: the provider's answer, the leak guard's count and the cache outcome stay out of the
    # model's tool output.
    searched = {
        "ok": True,
        "tool": "google_search",
        "context": "1. Red",
        "raw": {"organic": [{"title": "provider-only answer"}]},
        "contaminated": 1,
        "cache": "miss",
    }
    monkeypatch.setattr(Retrieval, "call", AsyncMock(return_value=searched))
    replies = [
        response(output=[call("google_search", {"query": "image color"}, call_id="search")]),
        response(output=[call(call_id="crop")]),
        response(output=[call("zoom", {}, call_id="unknown")]),
        response("<answer>red</answer>"),
    ]
    config = load("config")
    config.policy_model = {"responses_api_models": {"openai_model": {"host": "model", "port": 9000}}}
    transport = ServerClient(head_server_config=BaseServerConfig(host="head", port=1), global_config_dict=config)
    agent_config = OmegaConf.to_container(
        config.agentic_mme_agent.responses_api_agents.agentic_mme_agent, resolve=True
    )
    agent_config["retrieval"] = {
        "mode": "live",
        "serper_api_key": "k",
        "serpapi_api_key": "k",
        "allow_image_upload": True,
    }
    agent = AgenticMMEAgent(
        config=AgenticMMEAgentConfig(name="agentic_mme_agent", host="agent", port=8001, **agent_config),
        server_client=transport,
    )
    grader = AgenticMMEServer(
        config=AgenticMMEConfig(name="agentic_mme", host="resources", port=8000, entrypoint=""),
        server_client=transport,
    )
    pairing = config.agentic_mme_environment_server.environment_servers.single_agent_turn_legacy
    environment = SingleAgentTurnLegacyEnvironmentServer(
        config=SingleAgentTurnLegacyEnvironmentServerConfig(
            name="agentic_mme_environment_server",
            host="environment",
            port=8002,
            **OmegaConf.to_container(pairing, resolve=True),
        ),
        server_client=transport,
    )
    calls, model_inputs, agent_session = [], [], {}

    async def post(self, server_name, url_path, **kwargs):
        body = kwargs["json"]
        calls.append((server_name, url_path))
        if server_name == "agentic_mme_environment_server":
            return Reply(await environment.run_legacy(body))
        if server_name == "policy_model":
            model_inputs.append(body.model_dump(mode="json")["input"])
            return Reply(replies.pop(0).model_dump(mode="json"))
        body = body.model_dump(mode="json") if hasattr(body, "model_dump") else body
        if server_name == "agentic_mme":
            if url_path == "/verify":
                verified = await grader.verify(no_session(), AgenticMMEVerifyRequest.model_validate(body))
                return Reply(verified.model_dump(mode="json"))
            # Seed and close. The Environment Server needs the seed's session cookie.
            cookie = {"session": SimpleNamespace(value="resources-cookie")}
            return Reply({"resources_session_id": body["resources_session_id"]}, cookie)
        path = url_path.split("?")[0]
        request = Request(
            {
                "type": "http",
                "headers": [],
                "method": "POST",
                "scheme": "http",
                "server": ("agent", 8001),
                "path": path,
                "query_string": b"",
                "session": agent_session,
            }
        )
        if url_path == "/v1/agent_sessions":
            result = await agent.seed_agent_session(request, AgentSeedSessionRequest.model_validate(body))
        elif url_path == "/v1/agent_sessions/close":
            result = await agent.close_agent_session(request, AgentCloseSessionRequest.model_validate(body))
        else:
            request.scope["path_params"] = {"rollout_id": path.split("/")[2]}
            params = NeMoGymResponseCreateParamsNonStreaming.model_validate(body)
            result = await agent.responses(request, Response(), params)
        return Reply(result.model_dump(mode="json"))

    monkeypatch.setattr(ServerClient, "post", post)
    monkeypatch.setattr(ServerClient, "_resolve_base_url", lambda self, name: f"http://{name}:8000")
    monkeypatch.setattr(RolloutCollectionHelper, "setup_server_client", lambda self, head=None: transport)
    task = body().model_dump(mode="json", include={"responses_create_params", "verifier_metadata"})
    rows = RolloutCollectionHelper._preprocess_raw_rows(
        [(0, json.dumps(task), task)],
        RolloutCollectionConfig(
            input_jsonl_fpath="input.jsonl",
            output_jsonl_fpath="output.jsonl",
            agent_name="agentic_mme_agent",
            num_repeats=1,
        ),
    )
    _, row = await next(RolloutCollectionHelper().run_examples(rows))

    # The episode ran under an agent session, not through the agent's /run.
    assert ("agentic_mme_agent", "/v1/agent_sessions/close") in calls
    assert ("agentic_mme_agent", "/run") not in calls
    assert row["reward"] == 1.0
    assert [event["tool_name"] for event in row["tool_trace"]] == ["google_search", "crop", "zoom"]
    search = row["tool_trace"][0]["output"]
    assert search["raw"] == searched["raw"] and search["contaminated"] == 1 and search["cache"] == "miss"
    assert row["tool_trace"][2]["output"] == {"ok": False, "error": "unknown tool"}
    assert (row["tool_call_count"], row["successful_tool_calls"], row["tool_error_count"]) == (3, 2, 1)
    # Two successful calls against a reference of none.
    assert row["overthink"] == 2.0 and row["process_scores_available"] is False
    # The model saw only the search context, and the row stores the trace once.
    assert "provider-only answer" not in json.dumps(model_inputs)
    assert json.loads(model_inputs[1][-1]["output"]) == {"ok": True, "tool": "google_search", "context": "1. Red"}
    assert row["response"].get("metrics") is None
