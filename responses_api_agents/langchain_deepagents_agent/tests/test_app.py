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

from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.langchain_deepagents_agent.app import DeepAgentsAgent
from responses_api_agents.langchain_deepagents_agent.reasoning_search_agent import (
    ReasoningSearchDeepAgent,
    ReasoningSearchDeepAgentConfig,
)
from responses_api_agents.langchain_deepagents_agent.responses_langchain_bridge import (
    GymResponsesChatModel,
    to_responses,
)


def _config(**kwargs) -> ReasoningSearchDeepAgentConfig:
    kwargs.setdefault("resources_server", ResourcesServerRef(type="resources_servers", name="reasoning_gym"))
    kwargs.setdefault("model_server", ModelServerRef(type="responses_api_models", name="policy_model"))
    kwargs.setdefault("tavily_api_key", "test-tavily-key")
    kwargs.setdefault("max_input_tokens", 202800)
    return ReasoningSearchDeepAgentConfig(host="0.0.0.0", port=8080, entrypoint="", name="", **kwargs)


def _make_agent(**kwargs) -> ReasoningSearchDeepAgent:
    return ReasoningSearchDeepAgent(config=_config(**kwargs), server_client=MagicMock(spec=ServerClient))


# --- build_agent abstractness / concreteness --------------------------------------------------------


def test_base_class_build_agent_is_abstract():
    with pytest.raises(NotImplementedError):
        DeepAgentsAgent.build_agent(MagicMock(), MagicMock())


def test_concrete_agent_builds_successfully():
    agent = _make_agent()
    assert agent.agent is not None


def test_build_agent_sets_model_profile_from_max_input_tokens():
    """deepagents' SummarizationMiddleware reads model.profile["max_input_tokens"] to compute its
    trigger/keep thresholds; without it, it falls back to a hardcoded 170k-token/6-message default
    unrelated to whatever model is actually configured. Calls build_agent() directly with a fresh model
    (rather than inspecting the compiled create_deep_agent() graph) since the graph doesn't expose the
    model object it wraps."""
    agent = _make_agent(max_input_tokens=202800)
    model = GymResponsesChatModel(agent=agent)

    agent.build_agent(model)

    assert model.profile == {"max_input_tokens": 202800}


def test_reasoning_search_deep_agent_config_requires_max_input_tokens():
    with pytest.raises(ValidationError, match="max_input_tokens"):
        ReasoningSearchDeepAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            resources_server=ResourcesServerRef(type="resources_servers", name="reasoning_gym"),
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
            tavily_api_key="test-tavily-key",
            # max_input_tokens deliberately omitted
        )


def test_config_rejects_non_none_max_steps():
    """max_steps (inherited from SimpleAgentConfig) has no effect here — deepagents runs its own internal
    tool loop, with no per-model-call step counter for it to bound. Silently accepting and ignoring it
    would let a caller believe it's enforced when it isn't, so config construction must fail loudly."""
    with pytest.raises(ValidationError, match="max_steps"):
        _config(max_steps=3)


# --- cookie propagation ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_responses_propagates_inbound_cookies_to_outgoing_response():
    from fastapi import Response

    agent = _make_agent()
    agent.agent = MagicMock()
    agent.agent.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="done")]})

    request = MagicMock()
    request.cookies = {"session": "abc123"}
    request.path_params = {}
    response = Response()

    body = MagicMock()
    body.input = "hello"

    await agent.responses(request, response, body)
    assert response.raw_headers  # a Set-Cookie header was added
    assert any(b"session=abc123" in value for _, value in response.raw_headers)


# --- RunnableConfig construction ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_responses_seeds_model_cookies_empty_not_from_resources_server_cookies():
    """`request.cookies` here are resources-server session cookies chained from /seed_session (see
    SimpleAgent.run()'s self-POST) — they must never become the jar `_agenerate()` sends to the model
    server. The model-server cookie holder starts `None`, matching SimpleAgent._create_episode's
    `model_server_cookies = None`."""
    from fastapi import Response

    agent = _make_agent()
    agent.agent = MagicMock()
    agent.agent.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="done")]})

    request = MagicMock()
    request.cookies = {"session": "resources-server-cookie"}
    request.path_params = {}
    request.url.path = "/v1/responses"

    await agent.responses(request, Response(), MagicMock(input="hello"))

    run_config = agent.agent.ainvoke.call_args.kwargs["config"]
    assert run_config["configurable"]["model_cookies"] == {"cookies": None}


@pytest.mark.asyncio
async def test_responses_forwards_capture_mode_path_into_model_url_path():
    """A request arriving at .../training-token-capture/v1/responses must produce a model_url_path that
    still carries that segment — dropping it silently stops training token IDs from being captured."""
    from fastapi import Response

    agent = _make_agent()
    agent.agent = MagicMock()
    agent.agent.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="done")]})

    request = MagicMock()
    request.cookies = {}
    request.path_params = {"rollout_id": "abc123"}
    request.url.path = "/ng-rollout/abc123/training-token-capture/v1/responses"

    await agent.responses(request, Response(), MagicMock(input="hello"))

    run_config = agent.agent.ainvoke.call_args.kwargs["config"]
    assert "training-token-capture" in run_config["configurable"]["model_url_path"]


@pytest.mark.asyncio
async def test_responses_forwards_body_reasoning_as_plain_dict():
    """`body.reasoning` must reach `configurable` as a plain dict, not a Pydantic object: its field type
    (Reasoning) is a TypedDict, not a BaseModel, so a real request's `body.reasoning` is already a plain
    dict at runtime — calling .model_dump() on it would raise AttributeError. Uses a real
    NeMoGymResponseCreateParamsNonStreaming (not a MagicMock body) so this is checked against the actual
    runtime type instead of a mock that would hide the bug."""
    from fastapi import Response

    agent = _make_agent()
    agent.agent = MagicMock()
    agent.agent.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="done")]})

    request = MagicMock()
    request.cookies = {}
    request.path_params = {"rollout_id": "abc123"}
    request.url.path = "/ng-rollout/abc123/v1/responses"

    body = NeMoGymResponseCreateParamsNonStreaming.model_validate(
        {
            "input": [{"type": "message", "role": "user", "content": "hi"}],
            "reasoning": {"summary": "auto"},
        }
    )

    await agent.responses(request, Response(), body)

    run_config = agent.agent.ainvoke.call_args.kwargs["config"]
    model_reasoning = run_config["configurable"]["model_reasoning"]
    assert model_reasoning == {"summary": "auto"}
    assert isinstance(model_reasoning, dict)


# --- deepagents summarization middleware --------------------------------------------------------------


def _model_response(text: str, input_tokens: int = 0, output_tokens: int = 0):
    """A model-server response carrying usage, so summarization's own extra model call is observable in
    the accumulated total rather than having to be inferred from call counts alone."""
    body = to_responses([AIMessage(content=text)], "policy_model")
    body["usage"] = {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": input_tokens + output_tokens,
    }
    response = MagicMock()
    response.ok = True
    response.cookies = {}
    response.read = AsyncMock(return_value=orjson.dumps(body))
    return response


def _run_config(usage_holder: dict) -> RunnableConfig:
    return {
        "configurable": {
            "model_url_path": "/v1/responses",
            "model_cookies": {"cookies": None},
            "model_usage": usage_holder,
            "model_reasoning": None,
        }
    }


def _long_history(turns: int = 12) -> list:
    """Enough distinct messages for summarization to have something to partition — a single huge message
    is not enough, since deepagents bails out when its computed cutoff index is <= 0."""
    messages: list = []
    for i in range(turns):
        messages.append(HumanMessage(content=f"user turn {i}: " + "lorem ipsum dolor sit amet " * 12))
        messages.append(AIMessage(content=f"assistant turn {i}: " + "consectetur adipiscing elit " * 12))
    return messages


async def _invoke_with_limit(max_input_tokens: int, usage_holder: dict | None = None) -> int:
    """Drive a real create_deep_agent() graph over a fixed history; return how many model calls it made."""
    agent = _make_agent(max_input_tokens=max_input_tokens)
    post = AsyncMock(side_effect=[_model_response("ok", input_tokens=100, output_tokens=10) for _ in range(30)])
    agent.server_client.post = post

    await agent.agent.ainvoke({"messages": _long_history()}, config=_run_config(usage_holder or {"usage": None}))
    return post.call_count


@pytest.mark.asyncio
async def test_max_input_tokens_controls_deepagents_summarization():
    """`create_deep_agent()` always installs a SummarizationMiddleware, and there's no kwarg to configure
    it — it reads the context window off `model.profile["max_input_tokens"]` and triggers at a fraction of
    it. With no profile it would instead fall back to a fixed 170k-token threshold with no relationship to
    the configured model, which is the failure this config field exists to prevent.

    Same history both times, so the only variable is the configured limit: under a tiny limit deepagents
    summarizes (costing an extra model call through the same GymResponsesChatModel), under a large one it
    doesn't."""
    calls_small_limit = await _invoke_with_limit(200)
    calls_large_limit = await _invoke_with_limit(1_000_000)

    assert calls_large_limit == 1, "history fits well within the limit, so nothing should be summarized"
    assert calls_small_limit > calls_large_limit, (
        "a small max_input_tokens must make deepagents summarize, which costs an extra model call; "
        "if this fails, the profile is no longer reaching SummarizationMiddleware"
    )


@pytest.mark.asyncio
async def test_summarization_extra_model_call_is_included_in_usage_accounting():
    """Summarization issues its own model call through the same GymResponsesChatModel. That call's tokens
    must land in the rollout's accumulated usage — otherwise a rollout under-reports what it actually
    spent, by an amount that grows with how often compaction fires."""
    usage_holder: dict = {"usage": None}
    call_count = await _invoke_with_limit(200, usage_holder)

    assert call_count > 1, "expected summarization to add a model call on top of the agent's own turn"
    usage = usage_holder["usage"]
    assert usage is not None
    # every mocked call reports the same 100 in / 10 out, so a correct total is exactly per-call * calls
    assert usage.input_tokens == 100 * call_count
    assert usage.output_tokens == 10 * call_count
