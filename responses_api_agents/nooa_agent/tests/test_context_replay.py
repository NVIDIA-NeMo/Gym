# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preserve only the live suffix belonging to the same unedited policy response."""

from copy import deepcopy

import pytest
from nooa.unifiedllm import CacheBoundary, LLMResponse

from nemo_gym.core import NeMoGymResponseOutputMessage
from responses_api_agents.nooa_agent.tests.test_gym_llm import make_llm, model_response


async def test_normal_continuations_extend_the_exact_prior_request_and_reply():
    llm, client, state = make_llm(
        model_response(
            NeMoGymResponseOutputMessage(
                id="answer", content=[{"type": "output_text", "text": "reply", "annotations": []}]
            )
        ),
        max_policy_calls=3,
    )
    prefix = [{"role": "system", "content": "harness"}, {"role": "user", "content": "task"}]
    live = {"role": "user", "content": "live context\n exact whitespace  "}
    first = await llm.acall([*prefix, CacheBoundary(), live])
    original = state.calls[0].request.model_dump(mode="json")["input"]
    output = state.calls[0].response.model_dump(mode="json")["output"]
    second = await llm.acall([*prefix, first, CacheBoundary(), {"role": "user", "content": "fresh live context"}])
    second_input = state.calls[1].request.model_dump(mode="json")["input"]
    assert second_input[: len(original) + len(output)] == original + output
    assert second_input[-1]["content"] == "fresh live context"
    await llm.acall([*prefix, first, second, CacheBoundary(), {"role": "user", "content": "third context"}])
    assert (
        state.calls[2].request.model_dump(mode="json")["input"][: len(second_input) + len(output)]
        == second_input + output
    )
    assert client.post.await_count == 3


@pytest.mark.parametrize("case", ["edited", "foreign", "missing_response", "already_present"])
async def test_context_is_not_invented_for_edited_foreign_or_removed_history(case):
    llm, _, state = make_llm(
        model_response(
            NeMoGymResponseOutputMessage(
                id="answer", content=[{"type": "output_text", "text": "reply", "annotations": []}]
            )
        )
    )
    live = {"role": "user", "content": "old context"}
    response = await llm.acall([{"role": "user", "content": "task"}, CacheBoundary(), live])
    if case == "edited":
        response = LLMResponse(content="edited", id=response.id, replay_scope=response.replay_scope)
    elif case == "foreign":
        response = LLMResponse(parts=response.parts, id=response.id, replay_scope="foreign")
    history = [] if case == "missing_response" else [response]
    if case == "already_present":
        history.insert(0, deepcopy(live))
    await llm.acall([*history, CacheBoundary(), {"role": "user", "content": "new context"}])
    items = state.calls[-1].request.model_dump(mode="json")["input"]
    assert sum(item.get("content") == "old context" for item in items) == (1 if case == "already_present" else 0)
