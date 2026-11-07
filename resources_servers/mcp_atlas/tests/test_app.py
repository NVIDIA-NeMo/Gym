# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest

from nemo_gym.base_resources_server import ResourcesCloseSessionRequest
from nemo_gym.config_types import ModelServerRef
from nemo_gym.episode_types import EpisodeId
from nemo_gym.judge import JudgeError
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.mcp_atlas.app import (
    MCPAtlasResourcesServer,
    MCPAtlasResourcesServerConfig,
    MCPAtlasSeedSessionRequest,
    MCPAtlasVerifyRequest,
    tool_output_text,
)
from resources_servers.mcp_atlas.sandbox import SandboxError
from resources_servers.mcp_atlas.tests.fake_agent_environment import FakeAgentEnvironment
from resources_servers.mcp_atlas.tests.test_sandbox import pool
from resources_servers.mcp_atlas.tests.test_verifier import message, response


TOOLS = ["calculator_calculate", "github_get_repository", "met-museum_get-museum-object"]
META = {
    "task_id": "t1",
    "enabled_tools": TOOLS,
    "servers": ["calculator", "github", "met-museum"],
    "claims": ["The repo was created in 2013.", "The domain was registered in 2006."],
}


def judge_client(*outcomes: str) -> MagicMock:
    """A ServerClient whose judge answers the given outcomes in order; any other string is sent as is."""
    client = MagicMock(spec=ServerClient)
    replies = []
    for outcome in outcomes:
        if outcome in ("fulfilled", "partially_fulfilled", "not_fulfilled"):
            outcome = json.dumps({"coverage_outcome": outcome, "justification": "j"})
        completion = {
            "id": "c",
            "object": "chat.completion",
            "created": 0,
            "model": "judge",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": outcome},
                }
            ],
        }
        reply = AsyncMock()
        reply.ok = True
        reply.read = AsyncMock(return_value=orjson.dumps(completion))
        replies.append(reply)
    client.post = AsyncMock(side_effect=replies)
    return client


def make_server(base_url: str, client: MagicMock, **overrides) -> MCPAtlasResourcesServer:
    config = MCPAtlasResourcesServerConfig(
        host="0.0.0.0",
        port=0,
        entrypoint="app.py",
        name="mcp_atlas",
        sandbox_base_url=base_url,
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        **overrides,
    )
    return MCPAtlasResourcesServer(config=config, server_client=client)


def http_request(session_id: str = "s1") -> SimpleNamespace:
    return SimpleNamespace(session={SESSION_ID_KEY: session_id}, cookies={})


def verify_body(text: str) -> MCPAtlasVerifyRequest:
    return MCPAtlasVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=response(message(text)),
        verifier_metadata=META,
    )


@pytest.fixture
async def environment():
    env = FakeAgentEnvironment(
        TOOLS,
        {
            "github_get_repository": (500, {"detail": "Failed to call tool: Not Found: repository does not exist"}),
            "calculator_calculate": (200, [{"type": "text", "text": "14"}]),
        },
    )
    url = await env.start()
    yield env, url
    await env.stop()


async def seed(server: MCPAtlasResourcesServer, session_id: str = "s1") -> None:
    body = MCPAtlasSeedSessionRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]), verifier_metadata=META
    )
    await server.seed_session(http_request(session_id), body)


def test_exactly_one_sandbox_source_is_required() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        make_server("", judge_client())


def test_tool_output_text() -> None:
    assert tool_output_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "a\nb"
    assert tool_output_text([]) == "success"
    assert tool_output_text([{"type": "text", "text": "  "}]) == "success"


async def test_tool_calls_are_proxied_and_bounded(environment) -> None:
    env, url = environment
    server = make_server(url, judge_client(), max_tool_calls=3)
    await seed(server)
    request = http_request()
    assert await server.call_tool(request, "calculator_calculate", {"expression": "2+3*4"}) == "14"
    # A model error (unknown repository) is a tool output, not an environment failure.
    assert "does not exist" in await server.call_tool(request, "github_get_repository", {"owner": "x"})
    await server.call_tool(request, "met-museum_get-museum-object", {"objectId": 1})
    assert env.calls[-1]["tool_args"] == {"objectId": 1, "returnImage": False}
    assert "not available" in await server.call_tool(request, "fetch_fetch", {"url": "x"})
    assert "limit of 3 tool calls" in await server.call_tool(request, "calculator_calculate", {"expression": "1"})
    assert server._rollouts["s1"].failures == []


async def test_verify_passes_at_threshold_and_reports_coverage(environment) -> None:
    _, url = environment
    server = make_server(url, judge_client("fulfilled", "partially_fulfilled"))
    await seed(server)
    result = await server.verify(http_request(), verify_body("Created 2013; domain from about 2006."))
    assert result.coverage == 0.75
    assert result.reward == 1.0
    assert result.passed_at_0_50 is True
    assert result.n_claims == 2
    assert result.mask_sample is False and result.failure_kind is None
    assert "s1" not in server._rollouts


async def test_verify_below_threshold_scores_zero(environment) -> None:
    _, url = environment
    server = make_server(url, judge_client("fulfilled", "not_fulfilled"))
    await seed(server)
    result = await server.verify(http_request(), verify_body("Created 2013."))
    assert result.coverage == 0.5
    assert result.reward == 0.0
    assert result.passed_at_0_50 is True


async def test_empty_answer_scores_zero_without_the_judge(environment) -> None:
    _, url = environment
    client = judge_client()
    server = make_server(url, client)
    await seed(server)
    result = await server.verify(http_request(), verify_body("   "))
    assert result.reward == 0.0 and result.coverage == 0.0
    client.post.assert_not_called()


async def test_service_error_masks_the_rollout(environment) -> None:
    env, url = environment
    env.results["github_get_repository"] = (500, {"detail": "Failed to call tool: 429 Too Many Requests"})
    server = make_server(url, judge_client("fulfilled", "fulfilled"))
    await seed(server)
    await server.call_tool(http_request(), "github_get_repository", {"owner": "x"})
    result = await server.verify(http_request(), verify_body("Created 2013, registered 2006."))
    assert result.mask_sample is True
    assert result.failure_kind == "mcp_atlas:service_error"
    assert "429" in result.failure_reason
    # The measurement is still recorded for the audit.
    assert result.coverage == 1.0


async def test_service_errors_are_kept_when_masking_is_off(environment) -> None:
    env, url = environment
    env.results["github_get_repository"] = (500, {"detail": "Failed to call tool: rate limit exceeded"})
    server = make_server(url, judge_client("fulfilled", "fulfilled"), mask_service_errors=False)
    await seed(server)
    await server.call_tool(http_request(), "github_get_repository", {"owner": "x"})
    result = await server.verify(http_request(), verify_body("ok"))
    assert result.mask_sample is False
    assert result.failure_kind == "mcp_atlas:service_error"


async def test_unreachable_sandbox_masks_the_rollout() -> None:
    server = make_server("http://127.0.0.1:9", judge_client("fulfilled", "fulfilled"))
    await seed(server)
    output = await server.call_tool(http_request(), "calculator_calculate", {"expression": "1"})
    assert output.startswith("Error: tool call failed")
    result = await server.verify(http_request(), verify_body("Created 2013, registered 2006."))
    assert result.mask_sample is True
    assert result.failure_kind == "mcp_atlas:sandbox_unreachable"


async def test_calls_after_the_sandbox_stopped_are_refused_and_masked(environment) -> None:
    env, url = environment
    server = make_server(url, judge_client("fulfilled", "fulfilled"))
    await seed(server)
    server._rollouts["s1"].box.stopped = True  # as the reaper leaves it
    output = await server.call_tool(http_request(), "calculator_calculate", {"expression": "1"})
    assert output == "Error: tool call failed: SandboxError"
    assert env.calls == []
    result = await server.verify(http_request(), verify_body("Created 2013, registered 2006."))
    assert result.mask_sample is True
    assert result.failure_kind == "mcp_atlas:sandbox_unreachable"


def close_body(resources_session_id: str = "run:r1") -> ResourcesCloseSessionRequest:
    return ResourcesCloseSessionRequest(
        resources_session_id=resources_session_id, episode_id=EpisodeId(rollout_id="r1")
    )


async def test_close_stops_the_sandbox_and_refuses_a_later_seed() -> None:
    server = make_server("http://unused", judge_client())
    server._pool = pool(TOOLS)
    await seed(server)
    box = server._rollouts["s1"].box
    # A /run close names its run in the body; the cookie names the session this server seeded.
    await server.close_resources_session(http_request(), close_body())
    assert box.stopped
    assert "s1" not in server._rollouts and server._pool.get("s1") is None
    await server.close_resources_session(http_request(), close_body())
    await server.close_resources_session(http_request("never-seeded"), close_body("run:never-seeded"))
    with pytest.raises(ValueError, match="already closed"):
        await seed(server)


async def test_a_close_whose_sandbox_does_not_stop_can_be_retried() -> None:
    server = make_server("http://unused", judge_client())
    server._pool = pool(TOOLS, stop_errors=1)
    await seed(server)
    with pytest.raises(SandboxError, match="sandbox stop failed"):
        await server.close_resources_session(http_request(), close_body())
    assert "s1" in server._rollouts and "s1" not in server._closed_sessions
    await server.close_resources_session(http_request(), close_body())
    assert server._pool.get("s1") is None and "s1" in server._closed_sessions


async def test_reverify_judges_a_saved_row_again(environment) -> None:
    env, url = environment
    env.results["github_get_repository"] = (500, {"detail": "Failed to call tool: 429 Too Many Requests"})
    server = make_server(url, judge_client("fulfilled", "partially_fulfilled", "fulfilled", "fulfilled"))
    await seed(server)
    await server.call_tool(http_request(), "github_get_repository", {"owner": "x"})
    first = await server.verify(http_request(), verify_body("Created 2013; domain from about 2006."))
    assert first.coverage == 0.75
    # gym eval reverify posts the saved row, with its computed fields, and no session.
    saved = MCPAtlasVerifyRequest.model_validate(first.model_dump(mode="json"))
    again = await server.verify(http_request("reverify"), saved)
    assert again.coverage == 1.0
    assert again.tool_calls == 1
    assert again.service_errors == first.service_errors
    assert again.mask_sample is True and again.failure_kind == "mcp_atlas:service_error"


async def test_verify_without_a_session_is_masked(environment) -> None:
    _, url = environment
    server = make_server(url, judge_client())
    result = await server.verify(http_request("never-seeded"), verify_body("text"))
    assert result.mask_sample is True
    assert result.failure_kind == "mcp_atlas:session_lost"
    assert result.reward == 0.0


async def test_unreadable_verdicts_are_retried(environment) -> None:
    _, url = environment
    client = judge_client("not json", "fulfilled", "partially_fulfilled")
    server = make_server(url, client, judge_max_attempts=2)
    await seed(server)
    result = await server.verify(http_request(), verify_body("Created 2013; domain from about 2006."))
    assert result.coverage == 0.75
    assert client.post.await_count == 3


async def test_judge_error_after_the_last_unreadable_verdict(environment) -> None:
    _, url = environment
    client = judge_client("not json", '{"coverage_outcome": "maybe"}', "", "[]")
    server = make_server(url, client, judge_max_attempts=2)
    await seed(server)
    # The measurement fails instead of scoring the claims 0.
    with pytest.raises(JudgeError, match="no readable judge verdict after 2 attempts"):
        await server.verify(http_request(), verify_body("answer"))


async def test_failed_judge_call_raises_judge_error(environment) -> None:
    _, url = environment
    client = MagicMock(spec=ServerClient)
    failed = AsyncMock()
    failed.ok = False
    failed.raise_for_status = MagicMock(side_effect=RuntimeError("HTTP 503"))
    failed.content.read = AsyncMock(return_value=b"unavailable")
    client.post = AsyncMock(return_value=failed)
    server = make_server(url, client)
    await seed(server)
    # judge_failsafe, applied to /verify by the base class, turns this into a masked judge_failed row.
    with pytest.raises(JudgeError):
        await server.verify(http_request(), verify_body("answer"))
