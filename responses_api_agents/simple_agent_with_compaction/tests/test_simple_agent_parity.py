# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare the actual agent loops; only the external model/resources are scripted."""

from copy import deepcopy
from http.cookies import SimpleCookie
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Response

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig, SimpleAgentRunRequest
from responses_api_agents.simple_agent_with_compaction.app import (
    SimpleAgentWithCompaction,
    SimpleAgentWithCompactionConfig,
    SimpleAgentWithCompactionRunRequest,
)
from responses_api_agents.simple_agent_with_compaction.tests.test_app import tool_call
from responses_api_agents.simple_agent_with_compaction.tests.test_client import answer, http_response, reasoning


async def exercise(
    compaction_agent, responses, *, max_steps=5, skip_verification=False, tool_status=200, context_history=None
):
    calls = []
    queue = [response.model_copy(deep=True) for response in responses]
    server = MagicMock(spec=ServerClient, global_config_dict={"token_id_capture": {"enabled": True}})
    config_type = SimpleAgentWithCompactionConfig if compaction_agent else SimpleAgentConfig
    agent_type = SimpleAgentWithCompaction if compaction_agent else SimpleAgent
    agent = agent_type(
        config=config_type(
            host="127.0.0.1",
            port=8080,
            entrypoint="",
            name="agent",
            model_server={"type": "responses_api_models", "name": "model"},
            resources_server={"type": "resources_servers", "name": "resources"},
            max_steps=max_steps,
            skip_verification=skip_verification,
            skip_verification_reward=0.3,
            **({"context_history": context_history} if context_history is not None else {}),
        ),
        server_client=server,
    )

    async def post(**kwargs):
        if kwargs["server_name"] == "agent":
            # Execute simple_agent's real self-HTTP handler, including cookie propagation.
            response = Response()
            model_response = await agent.responses(
                MagicMock(cookies=kwargs["cookies"], path_params={}), response, kwargs["json"]
            )
            cookies = SimpleCookie()
            for header in response.headers.getlist("set-cookie"):
                cookies.load(header)
            return http_response(
                model_response.model_dump(mode="json"), cookies={k: v.value for k, v in cookies.items()}
            )
        payload = kwargs["json"]
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump(mode="json")
        # Capture scope, selected-parent header and retry policy deliberately differ.
        path = "/v1/responses" if kwargs["server_name"] == "model" else kwargs["url_path"]
        calls.append((path, deepcopy(payload), dict(kwargs.get("cookies") or {})))
        if path == "/seed_session":
            return http_response({}, cookies={"resource_session": "seeded"})
        if path == "/v1/responses":
            return http_response(queue.pop(0).model_dump(mode="json"), cookies={"model_session": "model"})
        if path == "/step":
            return http_response({"result": "observed"}, status=tool_status, cookies={"resource_session": "advanced"})
        if path == "/verify":
            # Verifiers may annotate the response as well as returning top-level metrics.
            return http_response(
                payload
                | {
                    "response": payload["response"] | {"verifier_annotation": "checked"},
                    "reward": 0.75,
                    "resolved": True,
                    "custom_metric": 7,
                }
            )
        raise AssertionError(path)

    server.post = AsyncMock(side_effect=post)
    body_type = SimpleAgentWithCompactionRunRequest if compaction_agent else SimpleAgentRunRequest
    body = body_type.model_validate(
        {
            "responses_create_params": {"input": "task", "max_output_tokens": 8},
            "_ng_rollout_id": "dispatch_g0",
            "task_id": "task-1",
            "verifier_metadata": {"expected": "answer"},
        }
    )
    original = body.model_dump()
    result = await agent.run(MagicMock(cookies={}), body)
    assert body.model_dump() == original
    return result, calls


@pytest.mark.parametrize(
    "case",
    [
        "answer",
        "tools",
        "multiple_tools",
        "malformed",
        "http_error",
        "reasoning",
        "non_assistant",
        "max_steps",
        "max_output_tokens",
        "content_filter",
        "skip_verification",
        "usage",
    ],
)
async def test_identity_policy_matches_simple_agent_task_tool_and_verifier_behavior(case):
    responses = [answer(1, output=[tool_call()]), answer(2)]
    options = {}
    if case == "answer":
        responses = [answer(1)]
    elif case == "multiple_tools":
        responses[0] = answer(1, output=[tool_call(), tool_call() | {"id": "function-2", "call_id": "tool-2"}])
    elif case == "malformed":
        responses[0] = answer(1, output=[tool_call("bad-json")])
    elif case == "http_error":
        options["tool_status"] = 500
    elif case == "reasoning":
        responses.insert(0, answer(0, output=[reasoning(0)]))
    elif case == "non_assistant":
        responses.insert(0, answer(0, output=[{"type": "message", "role": "user", "content": "continue"}]))
    elif case == "max_steps":
        options["max_steps"] = 1
    elif case in {"max_output_tokens", "content_filter"}:
        responses[0] = answer(1, output=[tool_call()], incomplete={"reason": case})
    elif case == "skip_verification":
        options["skip_verification"] = True
    elif case == "usage":
        responses.insert(0, answer(0, output=[reasoning(0)]))
        for response in (responses[0], responses[-1]):
            response.usage = NeMoGymResponse.model_validate(
                response.model_dump()
                | {
                    "usage": {
                        "input_tokens": 10,
                        "input_tokens_details": {"cached_tokens": 2},
                        "output_tokens": 3,
                        "output_tokens_details": {"reasoning_tokens": 1},
                        "total_tokens": 13,
                    }
                }
            ).usage

    ordinary, ordinary_calls = await exercise(False, responses, **options)
    managed, managed_calls = await exercise(True, responses, **options)
    assert managed_calls == ordinary_calls
    assert managed.model_dump(exclude={"context_compaction_result"}) == ordinary.model_dump()
    assert managed.context_compaction_result.segments[-1].selected_actions[-1].response_id == managed.response.id
    if case == "usage":
        assert managed.response.usage.total_tokens == 26


async def test_compaction_changes_model_view_not_verifier_history():
    responses = [answer(1, output=[reasoning(1), tool_call()]), answer(2)]
    ordinary, ordinary_calls = await exercise(False, responses)
    managed, managed_calls = await exercise(
        True,
        responses,
        context_history={
            "enabled": True,
            "policy": {"type": "recency", "config": {"reasoning": {"enabled": True, "keep_last_blocks": 0}}},
        },
    )
    assert [path for path, _, _ in managed_calls] == [path for path, _, _ in ordinary_calls]
    assert len(managed.context_compaction_result.segments) == 2
    original_input = ordinary_calls[3][1]["input"]
    compacted_input = managed_calls[3][1]["input"]
    assert any(item.get("type") == "reasoning" for item in original_input)
    assert not any(item.get("type") == "reasoning" for item in compacted_input)
    assert managed_calls[-1] == ordinary_calls[-1]
    assert managed.model_dump(exclude={"context_compaction_result"}) == ordinary.model_dump()
