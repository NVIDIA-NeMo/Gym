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

import asyncio
import re
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
from fastapi import HTTPException, Response
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from nemo_gym.agents import AgentCloseSessionRequest, AgentSeedSessionRequest, check_agent_seed_rejected
from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef, ResourcesServerRef
from nemo_gym.core import EpisodeId, NeMoGymResponseCreateParamsNonStreaming, TaskId
from nemo_gym.sandbox import (
    AsyncSandbox,
    DirectSandboxConnection,
    SandboxAccess,
    SandboxSpec,
    get_provider_class,
    register_provider,
)
from nemo_gym.server_runtime import DirectHTTPToolAccess, MCPStreamableHTTPConnection, MCPToolAccess
from responses_api_agents.langchain_deepagents_agent import app as app_module
from responses_api_agents.langchain_deepagents_agent.app import DeepAgentsAgent
from responses_api_agents.langchain_deepagents_agent.reasoning_search_agent import (
    ReasoningSearchDeepAgent,
    ReasoningSearchDeepAgentConfig,
)
from responses_api_agents.langchain_deepagents_agent.responses_langchain_bridge import (
    GymResponsesChatModel,
    to_responses,
)
from responses_api_agents.langchain_deepagents_agent.sandbox_backend import TaskSandboxError


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


def _model_response(text_or_message: "str | AIMessage", input_tokens: int = 0, output_tokens: int = 0):
    """A model-server response carrying usage, so summarization's own extra model call is observable in
    the accumulated total rather than having to be inferred from call counts alone. Accepts a whole
    AIMessage (not just text) so a mocked turn can carry tool calls and keep the agent loop going."""
    message = AIMessage(content=text_or_message) if isinstance(text_or_message, str) else text_or_message
    body = to_responses([message], "policy_model")
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


# --- verify-body regression (tool-using rollout) -------------------------------------------------------


@pytest.mark.asyncio
async def test_responses_result_is_what_verify_receives_for_a_tool_using_rollout():
    """SimpleAgent.run() (inherited, unmodified) forwards this method's return value verbatim as
    `body.response` on the `/verify` request (see simple_agent/app.py's `model_response_json`). So the
    NeMoGymResponse asserted on here is exactly what a resources server's verify() sees for a tool-using
    rollout: one function_call, one function_call_output, and the final assistant text — not just the
    final text alone. Regression test for cwing-nvidia's PR #2929 review comment (now fixed): `to_responses()`
    used to discard the whole tool-call/tool-result trace.

    Deliberately does not assert on resources-server session state/metrics: this agent's tools (e.g.
    TavilySearch in reasoning_search_agent.py; `ls` here, deepagents' built-in filesystem tool, so this
    test needs no network) execute inside the agent harness and never touch a resources server's own
    session — there is nothing on that side for a test at this level to observe. See README.md's "Tool
    ownership" section."""
    from fastapi import Response

    agent = _make_agent()
    call_id = "call_1"
    agent.server_client.post = AsyncMock(
        side_effect=[
            _model_response(AIMessage(content="", tool_calls=[{"name": "ls", "args": {}, "id": call_id}])),
            _model_response("final answer"),
        ]
    )

    request = MagicMock()
    request.cookies = {}
    request.path_params = {}
    body = MagicMock()
    body.input = "list files"

    result = await agent.responses(request, Response(), body)

    function_calls = [item for item in result.output if item.type == "function_call"]
    function_call_outputs = [item for item in result.output if item.type == "function_call_output"]
    messages = [item for item in result.output if item.type == "message"]

    assert len(function_calls) == 1
    assert function_calls[0].name == "ls"
    assert len(function_call_outputs) == 1
    assert function_call_outputs[0].call_id == call_id
    assert len(messages) == 1
    assert messages[0].content[0].text == "final answer"


# --- offloaded-history correlation (thread_id) ---------------------------------------------------------

_HISTORY_PATH_RE = re.compile(r"/conversation_history/[\w\-]+\.md")


def _file_text(entry) -> str:
    """StateBackend stores each virtual file as either a plain string or a small object with `.content`,
    depending on its configured file_format."""
    return str(entry if isinstance(entry, str) else getattr(entry, "content", entry))


def _looping_post(counters: dict, pointed_at: list[set[str]], tool_turns: int):
    """Mocked model server for a rollout that summarizes repeatedly: summarization calls (recognised by
    deepagents' own "Context Extraction Assistant" prompt) get summary text, agent turns get an `ls` tool
    call so the loop keeps running and history keeps growing back past the tiny max_input_tokens. `ls` is a
    deepagents built-in over the in-state virtual filesystem, so nothing here touches the network.

    Records, per agent turn, which offloaded-history paths that turn's request pointed the model at."""

    async def fake_post(**kwargs):
        blob = orjson.dumps(kwargs["json"]).decode()
        if "Context Extraction Assistant" in blob:
            return _model_response("summary")
        counters["agent_turns"] += 1
        pointed_at.append(set(_HISTORY_PATH_RE.findall(blob)))
        if counters["agent_turns"] <= tool_turns:
            call_id = f"call_{counters['agent_turns']}"
            return _model_response(
                AIMessage(content="listing", tool_calls=[{"name": "ls", "args": {}, "id": call_id}])
            )
        return _model_response("final answer")

    return fake_post


async def _rollout_with_repeated_summarization(thread_id: str | None, tool_turns: int = 5):
    """Drive a real create_deep_agent() graph through several summarizations in one rollout, invoking the
    graph directly so the thread_id can be varied (responses() always sets one).

    Returns the set of history paths the model was pointed at on each agent turn, plus the final virtual
    filesystem."""
    agent = _make_agent(max_input_tokens=200)
    counters = {"agent_turns": 0}
    pointed_at: list[set[str]] = []
    agent.server_client.post = AsyncMock(side_effect=_looping_post(counters, pointed_at, tool_turns))
    configurable = dict(_run_config({"usage": None})["configurable"])
    if thread_id is not None:
        configurable["thread_id"] = thread_id

    final_state = await agent.agent.ainvoke({"messages": _long_history()}, config={"configurable": configurable})
    return pointed_at, (final_state.get("files") or {})


@pytest.mark.asyncio
async def test_responses_propagates_rollout_id_as_langgraph_thread_id():
    """deepagents' SummarizationMiddleware names its offloaded-history file after `configurable.thread_id`,
    so the rollout id has to reach it for offloaded history to correlate with the rollout."""
    from fastapi import Response

    agent = _make_agent()
    agent.agent = MagicMock()
    agent.agent.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="done")]})

    request = MagicMock()
    request.cookies = {}
    request.path_params = {"rollout_id": "abc123"}
    request.url.path = "/ng-rollout/abc123/v1/responses"

    await agent.responses(request, Response(), MagicMock(input="hello"))

    run_config = agent.agent.ainvoke.call_args.kwargs["config"]
    assert run_config["configurable"]["thread_id"] == "abc123"


@pytest.mark.asyncio
async def test_responses_thread_id_falls_back_to_unscoped_without_a_rollout_id():
    """Matches SimpleAgent.responses()'s own `rollout_id or "unscoped"` fallback. Safe to share one constant
    across rollouts because the graph is compiled without a checkpointer and the default StateBackend keeps
    the virtual filesystem in per-ainvoke graph state."""
    from fastapi import Response

    agent = _make_agent()
    agent.agent = MagicMock()
    agent.agent.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="done")]})

    request = MagicMock()
    request.cookies = {}
    request.path_params = {}
    request.url.path = "/v1/responses"

    await agent.responses(request, Response(), MagicMock(input="hello"))

    run_config = agent.agent.ainvoke.call_args.kwargs["config"]
    assert run_config["configurable"]["thread_id"] == "unscoped"


@pytest.mark.asyncio
async def test_rollout_id_reaches_the_offloaded_history_path_end_to_end():
    """End-to-end over the whole chain the propagation exists for: a request carrying a rollout_id must
    make every summarization in that rollout offload to one path named for the rollout. Asserted on the
    paths the model is actually handed in its requests, which is the only place the pointer appears —
    responses() returns messages, not the virtual filesystem."""
    from fastapi import Response

    agent = _make_agent(max_input_tokens=200)
    counters = {"agent_turns": 0}
    pointed_at: list[set[str]] = []
    agent.server_client.post = AsyncMock(side_effect=_looping_post(counters, pointed_at, tool_turns=5))

    request = MagicMock()
    request.cookies = {}
    request.path_params = {"rollout_id": "abc123"}
    request.url.path = "/ng-rollout/abc123/v1/responses"
    body = NeMoGymResponseCreateParamsNonStreaming.model_validate(
        {"input": [{"type": "message", "role": "user", "content": m.content} for m in _long_history()]}
    )

    await agent.responses(request, Response(), body)

    assert len(pointed_at) > 2, "expected several agent turns, each preceded by a summarization"
    assert set().union(*pointed_at) == {"/conversation_history/abc123.md"}


@pytest.mark.asyncio
async def test_stable_thread_id_keeps_offloaded_history_in_one_appended_log():
    """The point of propagating thread_id, asserted on behaviour rather than on the config key.

    SummarizationMiddleware re-reads `/conversation_history/{thread_id}.md` and appends to it on every
    summarization, so one rollout should accumulate one running log. Without a thread_id it instead
    generates a fresh `session_<uuid>` *per summarization event*: the read-back finds nothing to append to,
    each eviction lands in its own file, and the path the model is handed — under a prompt claiming the
    full history was saved there — only ever covers the most recent eviction, with everything older left
    in files the model is never told about."""
    with_id_pointers, with_id_files = await _rollout_with_repeated_summarization("rollout-abc123")
    without_id_pointers, without_id_files = await _rollout_with_repeated_summarization(None)

    # Guard the setup itself: this only tests anything if summarization actually fired repeatedly.
    assert len(with_id_pointers) > 2, "expected several agent turns, each preceded by a summarization"

    with_id_paths = set().union(*with_id_pointers)
    without_id_paths = set().union(*without_id_pointers)

    assert len(with_id_paths) == 1, f"one rollout should mean one history log, got {sorted(with_id_paths)}"
    assert with_id_paths == {"/conversation_history/rollout-abc123.md"}, "log should be named for the rollout"
    assert len(with_id_files) == 1

    assert len(without_id_paths) > 1, (
        "characterises the behaviour this fix exists to prevent: without a thread_id every summarization "
        "points the model at a different file. If this ever fails, deepagents changed and the propagation "
        "in responses() may no longer be load-bearing"
    )

    # The decisive difference: the first summarization is what evicts the original task context, so the
    # accumulated log must still contain it at the end of the rollout.
    log_text = _file_text(next(iter(with_id_files.values())))
    assert "user turn 0" in log_text and "user turn 11" in log_text

    scattered_latest = _file_text(without_id_files[sorted(without_id_pointers[-1])[0]])
    assert "user turn 0" not in scattered_latest, (
        "without a stable thread_id the file the model is finally pointed at has lost the original context"
    )


# --- agent sessions (single_agent_turn / single_agent_turn_legacy) ------------------------------------


def _session_agent(global_config: dict | None = None) -> ReasoningSearchDeepAgent:
    agent = _make_agent()
    agent.server_client.global_config_dict = global_config or {}
    agent.agent = MagicMock()

    async def ainvoke(state: dict, config: RunnableConfig) -> dict:
        # Like the real graph, return the input history followed by the new messages.
        return {"messages": [*state["messages"], AIMessage(content="<answer>42</answer>")]}

    agent.agent.ainvoke = AsyncMock(side_effect=ainvoke)
    return agent


def _seed_body(**kwargs) -> AgentSeedSessionRequest:
    return AgentSeedSessionRequest(
        agent_session_id="agent-session",
        episode_id=EpisodeId(rollout_id="rollout", attempt=0),
        task_id=TaskId(taskset="reasoning_gym", task_id="0"),
        **kwargs,
    )


def test_config_rejects_execute_tools_false():
    with pytest.raises(ValidationError, match="execute_tools=false is not supported"):
        _config(execute_tools=False)


@pytest.mark.parametrize(
    ("unsupported", "reason"),
    [
        (
            {"tool_accesses": [DirectHTTPToolAccess(name="resources", required=True, base_url="http://r:8000")]},
            "DeepAgents agent does not support required tool access: resources",
        ),
        (
            {
                "tool_accesses": [
                    MCPToolAccess(
                        name="resources",
                        required=True,
                        connection=MCPStreamableHTTPConnection(url="http://r:8000/mcp"),
                    )
                ]
            },
            "DeepAgents agent does not support required tool access: resources",
        ),
        (
            {
                "sandbox_access": SandboxAccess(
                    connection=DirectSandboxConnection(provider_config_ref="runtime", descriptor={}),
                    workdir="/work",
                    user="agent",
                )
            },
            "DeepAgents agent does not support a sandbox user override",
        ),
        (
            {
                "sandbox_access": SandboxAccess(
                    connection=DirectSandboxConnection(provider_config_ref="runtime", descriptor={}),
                    workdir="work",
                )
            },
            "DeepAgents agent requires an absolute sandbox workdir",
        ),
    ],
)
def test_unsupported_seed_is_refused_with_a_final_422(unsupported: dict, reason: str):
    agent = _session_agent()
    assert check_agent_seed_rejected(agent.setup_webserver(), _seed_body(**unsupported)) == reason


def test_session_skips_an_optional_grant_and_runs_the_graph_without_mirroring_cookies():
    agent = _session_agent()
    client = TestClient(agent.setup_webserver())
    seed = client.post(
        "/v1/agent_sessions",
        json=_seed_body(
            tool_accesses=[
                DirectHTTPToolAccess(
                    name="resources", required=False, base_url="http://r:8000", cookies={"resources_cookie": "grant"}
                )
            ]
        ).model_dump(mode="json"),
    )
    assert seed.status_code == 200

    result = client.post(
        "/ng-rollout/rollout/v1/responses",
        json={"input": [{"role": "user", "content": "6 * 7?"}]},
        cookies={"resources_cookie": "from-seed"},
    )

    assert result.status_code == 200
    assert result.json()["output"][0]["content"][0]["text"] == "<answer>42</answer>"
    assert agent.agent.ainvoke.await_count == 1
    # The Environment Server owns the Resources cookies on the session path; the agent sends none back.
    assert "resources_cookie" not in result.cookies

    close = client.post(
        "/v1/agent_sessions/close",
        json=AgentCloseSessionRequest(
            agent_session_id="agent-session", episode_id=EpisodeId(rollout_id="rollout", attempt=0)
        ).model_dump(mode="json"),
    )
    # The skipped grant's cookies never become the jar, which would replace the seed cookies /verify uses.
    assert close.status_code == 200
    assert close.json()["resources_cookies"] is None


def test_session_close_reports_observation_gaps_and_no_resources_cookie_jar():
    agent = _session_agent()
    client = TestClient(agent.setup_webserver())
    episode_id = EpisodeId(rollout_id="rollout", attempt=0)
    assert client.post("/v1/agent_sessions", json=_seed_body().model_dump(mode="json")).status_code == 200
    assert client.post("/ng-rollout/rollout/v1/responses", json={"input": "6 * 7?"}).status_code == 200

    close = client.post(
        "/v1/agent_sessions/close",
        json=AgentCloseSessionRequest(agent_session_id="agent-session", episode_id=episode_id).model_dump(mode="json"),
    )

    assert close.status_code == 200
    body = close.json()
    # No jar: the seed's Resources cookie, which the Environment Server sends to /verify, stays unchanged.
    assert body["resources_cookies"] is None
    observations = body["agent_observations"]
    assert observations["source"] == "langchain_deepagents_agent"
    assert observations["records"] == []
    assert [gap["code"] for gap in observations["gaps"]] == [
        "agent_transcript_unavailable",
        "model_call_ownership_unavailable",
    ]


def test_activation_without_a_seeded_session_is_refused():
    agent = _session_agent()
    client = TestClient(agent.setup_webserver())
    assert client.post("/v1/agent_sessions", json=_seed_body().model_dump(mode="json")).status_code == 200
    episode_id = EpisodeId(rollout_id="rollout", attempt=0)
    close_body = AgentCloseSessionRequest(agent_session_id="agent-session", episode_id=episode_id)
    assert client.post("/v1/agent_sessions/close", json=close_body.model_dump(mode="json")).status_code == 200

    # The client still carries the closed session's cookie.
    result = client.post("/ng-rollout/rollout/v1/responses", json={"input": "6 * 7?"})

    assert result.status_code == 409
    agent.agent.ainvoke.assert_not_awaited()


def test_checkpointing_restarts_deepagents_rollouts_instead_of_exporting_them():
    """The graph's state lives inside one ainvoke(), which a checkpoint cannot export, so the agent takes part
    as restart-only rather than claiming SimpleAgent's resumable sessions."""
    agent = _session_agent({"checkpoint": {"enabled": True, "control_auth_token": "t"}})

    agent.setup_webserver()

    assert agent.checkpoint_participant is None
    assert agent._restart_only is not None


# --- borrowed task sandbox -------------------------------------------------------------------------------


class _LendableLocalProvider(get_provider_class("local")):
    """The local provider, made connectable by sandbox id so a test can lend its sandbox to the agent, as a
    Resources Server lends a Docker or OpenSandbox sandbox. Records each provider close of a sandbox."""

    name = "lendable_local"
    handles: ClassVar[dict] = {}
    closed: ClassVar[list] = []

    async def serialize_handle(self, handle, *, scope=None) -> dict:
        self.handles[handle.sandbox_id] = handle
        return {"sandbox_id": handle.sandbox_id}

    async def connect(self, descriptor):
        return self.handles[descriptor["sandbox_id"]]

    async def close(self, handle) -> None:
        self.closed.append(handle.sandbox_id)
        await super().close(handle)


@pytest.fixture
async def lent_sandbox(tmp_path):
    """A running task sandbox owned by the test (the Resources Server's role), with a task file in its workdir."""
    register_provider("lendable_local", _LendableLocalProvider, override=True)
    workdir = tmp_path / "task"
    workdir.mkdir()
    (workdir / "input.txt").write_text("from the task image\n")
    owner = AsyncSandbox(_LendableLocalProvider())
    await owner.start(SandboxSpec(workdir=str(workdir)))
    access = SandboxAccess(
        connection=DirectSandboxConnection(provider_config_ref="lender", descriptor=await owner.serialize()),
        workdir=str(workdir),
    )
    yield owner, access
    await owner.stop()


def _sandbox_agent() -> ReasoningSearchDeepAgent:
    agent = _make_agent()
    agent.server_client.global_config_dict = {"lender": {"lendable_local": {}}}
    return agent


def _request(rollout_id: str = "rollout") -> SimpleNamespace:
    return SimpleNamespace(session={}, cookies={}, path_params={"rollout_id": rollout_id})


def _close_body() -> AgentCloseSessionRequest:
    return AgentCloseSessionRequest(agent_session_id="agent-session", episode_id=EpisodeId(rollout_id="rollout"))


def _tool_names(post: AsyncMock, call: int) -> set[str]:
    return {tool["name"] for tool in post.call_args_list[call].kwargs["json"]["tools"]}


@pytest.mark.asyncio
async def test_borrowed_sandbox_runs_the_graph_tools_there_and_close_leaves_it_running(lent_sandbox):
    owner, access = lent_sandbox
    workdir = access.workdir
    agent = _sandbox_agent()
    post = AsyncMock(
        side_effect=[
            _model_response(
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "execute", "args": {"command": "pwd && cp input.txt copied.txt"}, "id": "call_exec"},
                        {
                            "name": "write_file",
                            "args": {"file_path": f"{workdir}/note.txt", "content": "written by the graph"},
                            "id": "call_write",
                        },
                        {"name": "read_file", "args": {"file_path": f"{workdir}/input.txt"}, "id": "call_read"},
                    ],
                )
            ),
            _model_response("done"),
        ]
    )
    agent.server_client.post = post
    request = _request()
    await agent.seed_agent_session(request, _seed_body(sandbox_access=access))

    result = await agent.responses(request, Response(), NeMoGymResponseCreateParamsNonStreaming(input="go"))

    outputs = {item.call_id: item.output for item in result.output if item.type == "function_call_output"}
    # The shell runs in the lent sandbox's workdir; the file tools read and write its files.
    assert outputs["call_exec"].startswith(f"{workdir}\n") and "exit code 0" in outputs["call_exec"]
    assert "from the task image" in outputs["call_read"]
    assert Path(workdir, "copied.txt").read_text() == "from the task image\n"
    assert Path(workdir, "note.txt").read_text() == "written by the graph"
    # The sandbox graph offers deepagents' execute tool, and every model call takes the rollout-prefixed route.
    assert "execute" in _tool_names(post, 0)
    assert {call.kwargs["url_path"] for call in post.call_args_list} == {"/ng-rollout/rollout/v1/responses"}

    close = await agent.close_agent_session(request, _close_body())

    [sandbox] = [record for record in close.agent_observations.records if record.kind == "sandbox"]
    assert (sandbox.role, sandbox.provider, sandbox.outcome) == ("agent", "lendable_local", "completed")
    assert sandbox.sandbox_id == owner.sandbox_id
    # Close disconnected without stopping: the owner still runs commands in it and stops it itself.
    assert owner.sandbox_id not in _LendableLocalProvider.closed
    assert (await owner.exec("cat copied.txt")).stdout == "from the task image\n"


@pytest.mark.asyncio
async def test_in_process_session_keeps_files_in_graph_state_without_an_execute_tool(tmp_path):
    agent = _make_agent()
    path = str(tmp_path / "in_process.txt")
    post = AsyncMock(
        side_effect=[
            _model_response(
                AIMessage(
                    content="",
                    tool_calls=[{"name": "write_file", "args": {"file_path": path, "content": "kept"}, "id": "w"}],
                )
            ),
            _model_response(
                AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"file_path": path}, "id": "r"}])
            ),
            _model_response("done"),
        ]
    )
    agent.server_client.post = post
    request = _request()
    await agent.seed_agent_session(request, _seed_body())

    result = await agent.responses(request, Response(), NeMoGymResponseCreateParamsNonStreaming(input="go"))

    outputs = {item.call_id: item.output for item in result.output if item.type == "function_call_output"}
    assert "kept" in outputs["r"]
    # deepagents' StateBackend keeps the file in graph state, never on the Agent Server's disk.
    assert not (tmp_path / "in_process.txt").exists()
    assert "execute" not in _tool_names(post, 0)
    close = await agent.close_agent_session(request, _close_body())
    assert close.agent_observations.records == []


def _hanging_graph(started: asyncio.Event, order: list[str]) -> MagicMock:
    async def ainvoke(state: dict, config: RunnableConfig) -> dict:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            order.append("cancelled")
            raise

    graph = MagicMock()
    graph.ainvoke = AsyncMock(side_effect=ainvoke)
    return graph


async def _seed_with_borrowed_stub(monkeypatch, agent: ReasoningSearchDeepAgent, graph: MagicMock, order: list[str]):
    async def disconnect() -> None:
        order.append("disconnect")

    sandbox = SimpleNamespace(disconnect=disconnect, sandbox_id="sb-1", provider_name="docker")
    monkeypatch.setattr(app_module, "create_provider", lambda config: MagicMock())
    monkeypatch.setattr(app_module.AsyncSandbox, "connect", AsyncMock(return_value=sandbox))
    # After the agent's in-process graph is built: the seed builds the sandbox session's graph.
    monkeypatch.setattr(type(agent), "build_agent", lambda self, model, backend=None: graph)
    access = SandboxAccess(
        connection=DirectSandboxConnection(provider_config_ref="lender", descriptor={"sandbox_id": "sb-1"}),
        workdir="/app",
    )
    request = _request()
    await agent.seed_agent_session(request, _seed_body(sandbox_access=access))
    return request


@pytest.mark.asyncio
async def test_close_cancels_the_running_graph_before_releasing_the_borrowed_sandbox(monkeypatch):
    agent = _sandbox_agent()
    started, order = asyncio.Event(), []
    request = await _seed_with_borrowed_stub(monkeypatch, agent, _hanging_graph(started, order), order)
    activation = asyncio.create_task(
        agent.responses(request, Response(), NeMoGymResponseCreateParamsNonStreaming(input="x"))
    )
    await asyncio.wait_for(started.wait(), 5)

    close = await asyncio.wait_for(agent.close_agent_session(request, _close_body()), 5)

    assert order == ["cancelled", "disconnect"]
    [sandbox] = [record for record in close.agent_observations.records if record.kind == "sandbox"]
    assert (sandbox.sandbox_id, sandbox.outcome) == ("sb-1", "cancelled")
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(activation, 5)


@pytest.mark.asyncio
async def test_activation_retries_join_one_graph_run_and_conflicting_requests_are_refused(monkeypatch):
    agent = _sandbox_agent()
    release, runs = asyncio.Event(), []

    async def ainvoke(state: dict, config: RunnableConfig) -> dict:
        runs.append(state["messages"][0].content)
        await release.wait()
        return {"messages": [*state["messages"], AIMessage(content="ran once")]}

    graph = MagicMock()
    graph.ainvoke = AsyncMock(side_effect=ainvoke)
    request = await _seed_with_borrowed_stub(monkeypatch, agent, graph, [])
    body = NeMoGymResponseCreateParamsNonStreaming(input="do it")
    first = asyncio.create_task(agent.responses(request, Response(), body))
    await asyncio.sleep(0)
    # A client resending the same POST after a lost reply joins the run in flight.
    second = asyncio.create_task(agent.responses(request, Response(), body.model_copy(deep=True)))
    with pytest.raises(HTTPException) as different:
        await asyncio.wait_for(
            agent.responses(request, Response(), NeMoGymResponseCreateParamsNonStreaming(input="other")), 5
        )
    with pytest.raises(HTTPException) as other_rollout:
        other_route = SimpleNamespace(session=request.session, cookies={}, path_params={"rollout_id": "other"})
        await asyncio.wait_for(agent.responses(other_route, Response(), body), 5)
    release.set()

    replies = await asyncio.wait_for(asyncio.gather(first, second), 5)

    assert [reply.output[0].content[0].text for reply in replies] == ["ran once", "ran once"]
    assert runs == ["do it"]
    assert (different.value.status_code, other_rollout.value.status_code) == (409, 409)


@pytest.mark.asyncio
async def test_a_sandbox_runtime_failure_fails_the_activation_and_close_reports_it(monkeypatch):
    agent = _sandbox_agent()

    async def ainvoke(state: dict, config: RunnableConfig) -> dict:
        raise TaskSandboxError("sandbox sb-1 could not run a command (sandbox): gone")

    graph = MagicMock()
    graph.ainvoke = AsyncMock(side_effect=ainvoke)
    request = await _seed_with_borrowed_stub(monkeypatch, agent, graph, [])

    with pytest.raises(TaskSandboxError):
        await agent.responses(request, Response(), NeMoGymResponseCreateParamsNonStreaming(input="x"))
    close = await agent.close_agent_session(request, _close_body())

    [sandbox] = [record for record in close.agent_observations.records if record.kind == "sandbox"]
    assert (sandbox.outcome, sandbox.error_type) == ("failed", "TaskSandboxError")


@pytest.mark.asyncio
async def test_a_failed_connect_releases_the_provider_client(monkeypatch):
    agent = _sandbox_agent()
    provider = MagicMock(aclose=AsyncMock())
    monkeypatch.setattr(app_module, "create_provider", lambda config: provider)
    monkeypatch.setattr(app_module.AsyncSandbox, "connect", AsyncMock(side_effect=RuntimeError("unreachable")))
    access = SandboxAccess(
        connection=DirectSandboxConnection(provider_config_ref="lender", descriptor={"sandbox_id": "sb-1"}),
        workdir="/app",
    )

    with pytest.raises(RuntimeError, match="unreachable"):
        await agent.seed_agent_session(_request(), _seed_body(sandbox_access=access))

    provider.aclose.assert_awaited_once()
