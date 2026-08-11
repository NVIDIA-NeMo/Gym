# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for Finance Agent v2 (responses_api_agents/finance_agent_v2)."""

import ast
import asyncio
import inspect
import json
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import finance_agent.get_agent as upstream_get_agent
import pytest
from fastapi.testclient import TestClient

# Imported by name: `finance_agent.get_agent` is a function as well as a module, and
# the package binds the function, so attribute access on the import above misses this.
from finance_agent.get_agent import Parameters as UpstreamParameters

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymEasyInputMessage
from nemo_gym.server_utils import ServerClient
from responses_api_agents.finance_agent_v2.app import (
    UPSTREAM_ABORT_TOOL_ERRORS,
    UPSTREAM_DONE_TOOL,
    UPSTREAM_MAX_TIME_SECONDS,
    UPSTREAM_NO_TOOL_CALL_NUDGE,
    UPSTREAM_VALID_TOOLS,
    FinanceAgentV2,
    FinanceAgentV2Config,
    FinanceAgentV2RunRequest,
)


_MODEL_SERVER = "model_server"
_RS_SERVER = "resources_server"
_INPUT = {"input": [{"role": "user", "content": "What was revenue?"}]}

_REPO_ROOT = Path(__file__).resolve().parents[3]

# v1's nudge. Kept only so the tests can assert we did not silently inherit it —
# responses_api_agents/finance_agent still uses it and must keep doing so.
_V1_NUDGE = "Continue."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> FinanceAgentV2Config:
    defaults = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="finance_agent_v2",
        resources_server=ResourcesServerRef(type="resources_servers", name=_RS_SERVER),
        model_server=ModelServerRef(type="responses_api_models", name=_MODEL_SERVER),
    )
    defaults.update(overrides)
    return FinanceAgentV2Config(**defaults)


def _make_agent_and_client(config: FinanceAgentV2Config | None = None):
    """Create agent + TestClient pair (the canonical test pattern for responses_api_agents)."""
    config = config or _make_config()
    agent = FinanceAgentV2(config=config, server_client=MagicMock(spec=ServerClient))
    app = agent.setup_webserver()
    client = TestClient(app)
    return agent, client


def _text_response(text: str, resp_id: str = "resp_1") -> dict:
    return {
        "id": resp_id,
        "created_at": 0.0,
        "model": "test",
        "object": "response",
        "output": [
            {
                "id": "msg_1",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def _tool_call_response(tool_name: str, arguments: str, call_id: str = "call_1", resp_id: str = "resp_1") -> dict:
    return _multi_tool_call_response([(tool_name, arguments, call_id)], resp_id=resp_id)


def _multi_tool_call_response(calls: list[tuple[str, str, str]], resp_id: str = "resp_1") -> dict:
    return {
        "id": resp_id,
        "created_at": 0.0,
        "model": "test",
        "object": "response",
        "output": [
            {
                "id": f"fc_{i}",
                "call_id": call_id,
                "name": name,
                "arguments": arguments,
                "type": "function_call",
                "status": "completed",
            }
            for i, (name, arguments, call_id) in enumerate(calls)
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def _reasoning_response(text: str = "thinking", resp_id: str = "resp_1") -> dict:
    return {
        "id": resp_id,
        "created_at": 0.0,
        "model": "test",
        "object": "response",
        "output": [
            {
                "id": "r1",
                "summary": [{"text": text, "type": "summary_text"}],
                "status": "completed",
                "type": "reasoning",
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def _dotjson_mock(*json_responses: dict) -> AsyncMock:
    """Mock for server_client.post return value, matching the pattern used by simple_agent tests.

    Sets up both paths used by the finance agent:
    - .read() → JSON string  (used by get_response_json for model calls)
    - .content.read() → bytes (used by tool-call output path)
    """
    mock = AsyncMock()
    mock.ok = True
    mock.content = MagicMock()
    if len(json_responses) == 1:
        data = json.dumps(json_responses[0])
        mock.read.return_value = data
        mock.content.read = AsyncMock(return_value=data.encode())
    else:
        strings = [json.dumps(r) for r in json_responses]
        mock.read.side_effect = strings
        mock.content.read = AsyncMock(side_effect=[s.encode() for s in strings])
    mock.cookies = MagicMock()
    return mock


def _route(model_mock, rs_mock):
    def route_post(**kwargs):
        return model_mock if kwargs["server_name"] == _MODEL_SERVER else rs_mock

    return route_post


# ---------------------------------------------------------------------------
# Tests: Upstream parity
#
# These are the tripwires for the failure mode this component exists to avoid:
# Vals edits vals-ai/finance-agent-v2, we bump the pin, and a behavioral
# difference lands silently. Everything upstream exposes as a module-level value
# is imported by app.py, so it cannot drift. The nudge text is the one value
# that must be copied (it is an inline literal inside a closure), so it is
# checked against upstream's source here.
# ---------------------------------------------------------------------------


class TestUpstreamParity:
    def test_done_tool_name_comes_from_upstream(self) -> None:
        assert UPSTREAM_DONE_TOOL == "submit_final_result"

    def test_time_budget_comes_from_upstream(self) -> None:
        """Upstream v2 bounds the run at one hour; v1 used max_turns=50 instead."""
        assert UPSTREAM_MAX_TIME_SECONDS == 60 * 60

    def test_valid_tools_include_v2_additions(self) -> None:
        """calculator and price_history are v2-only; their absence means the pin
        moved back to a v1-era tree."""
        assert "calculator" in UPSTREAM_VALID_TOOLS
        assert "price_history" in UPSTREAM_VALID_TOOLS

    def test_abort_error_type_comes_from_upstream(self) -> None:
        assert UPSTREAM_ABORT_TOOL_ERRORS == ("RetryExhaustedError",)

    def test_upstream_agent_parameters_gained_no_new_knobs(self) -> None:
        """Upstream's ``Parameters`` is the whole surface of per-run policy Vals can
        configure, and this loop mirrors every field of it. A new one is a policy
        knob that FABv2 would silently not honor — for instance a per-turn tool-call
        cap, which their engine supports but v2 has never set.

        On failure, read the upstream diff and decide whether to mirror the field.
        """
        assert set(UpstreamParameters.model_fields) == {
            "model_name",
            "max_time_seconds",
            "max_turns",
            "tools",
            "llm_config",
        }

    def test_nudge_matches_upstream_source(self) -> None:
        """The nudge is an inline literal in upstream's ``get_agent._before_query``
        and cannot be imported, so assert the copy still appears verbatim there.

        Compared against the parsed string constants rather than the raw text:
        upstream writes the nudge as adjacent literals, which the parser folds
        into one constant, so this is an exact match and not a substring search.

        If this fails, Vals reworded the nudge. Read the upstream diff, decide
        whether to follow it, and re-baseline if you do.
        """
        literals = {
            node.value
            for node in ast.walk(ast.parse(inspect.getsource(upstream_get_agent)))
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        assert UPSTREAM_NO_TOOL_CALL_NUDGE in literals, (
            "UPSTREAM_NO_TOOL_CALL_NUDGE no longer appears in upstream get_agent; Vals reworded the no-tool-call nudge"
        )

    def test_nudge_is_not_the_v1_text(self) -> None:
        """Guards against reverting to responses_api_agents/finance_agent's nudge."""
        assert UPSTREAM_NO_TOOL_CALL_NUDGE != _V1_NUDGE
        assert "submit_final_result" in UPSTREAM_NO_TOOL_CALL_NUDGE

    def test_upstream_pins_match_the_resource_server(self) -> None:
        """The agent and the tools it drives must come from one upstream commit."""

        def _pins(path: Path) -> set[str]:
            return {
                line.strip()
                for line in path.read_text().splitlines()
                if line.strip().startswith(("finance-agent", "model-library"))
            }

        agent_pins = _pins(_REPO_ROOT / "responses_api_agents" / "finance_agent_v2" / "requirements.txt")
        server_pins = _pins(_REPO_ROOT / "resources_servers" / "finance_agent_v2" / "requirements.txt")
        assert agent_pins, "no upstream pins found in the agent requirements"
        assert len(agent_pins) == 2, f"expected finance-agent and model-library pins, got {agent_pins}"
        assert agent_pins == server_pins

    def test_openai_override_matches_the_nemo_gym_pin(self) -> None:
        """``overrides.txt`` exists only to drop model-library's openai floor.

        uv applies an override to every declared constraint on the package, so a
        looser bound here would replace nemo-gym's cap too and silently install a
        newer client than the one Gym is tested against. Bump both together.
        """
        gym_pin = re.search(r'"openai(<=|==)([\d.]+)"', (_REPO_ROOT / "pyproject.toml").read_text())
        assert gym_pin, "nemo-gym no longer pins openai; revisit this override"

        for component in ("responses_api_agents", "resources_servers"):
            overrides = (_REPO_ROOT / component / "finance_agent_v2" / "overrides.txt").read_text()
            assert f"openai[aiohttp]<={gym_pin.group(2)}" in overrides, (
                f"{component}/finance_agent_v2/overrides.txt is out of sync with nemo-gym's openai pin"
            )


# ---------------------------------------------------------------------------
# Tests: Config and Construction
# ---------------------------------------------------------------------------


class TestFinanceAgentV2Config:
    def test_default_config_matches_upstream_v2_policy(self) -> None:
        config = _make_config()
        assert config.max_steps is None, "upstream v2 sets no turn cap"
        assert config.max_time_seconds == float(UPSTREAM_MAX_TIME_SECONDS)
        assert config.done_tools == [UPSTREAM_DONE_TOOL]
        assert config.no_tool_call_nudge == UPSTREAM_NO_TOOL_CALL_NUDGE
        assert config.abort_on_tool_error_types == list(UPSTREAM_ABORT_TOOL_ERRORS)
        assert config.model_call_timeout is None
        assert config.tool_call_timeout is None
        assert config.truncate_on_overflow is False

    def test_custom_config(self) -> None:
        config = _make_config(
            max_steps=10,
            max_time_seconds=60.0,
            done_tools=["submit_final_result", "abort"],
            no_tool_call_nudge="Keep going.",
            abort_on_tool_error_types=[],
            model_call_timeout=30.0,
            tool_call_timeout=60.0,
            truncate_on_overflow=True,
        )
        assert config.max_steps == 10
        assert config.max_time_seconds == 60.0
        assert config.done_tools == ["submit_final_result", "abort"]
        assert config.no_tool_call_nudge == "Keep going."
        assert config.abort_on_tool_error_types == []

    def test_sanity_construction(self) -> None:
        agent, _ = _make_agent_and_client()
        assert agent is not None


# ---------------------------------------------------------------------------
# Tests: _aborting_error_type
# ---------------------------------------------------------------------------


class TestAbortingErrorType:
    def test_matches_abort_error(self) -> None:
        agent, _ = _make_agent_and_client()
        payload = json.dumps({"error": "RetryExhaustedError: gave up after 5 attempts"})
        assert agent._aborting_error_type(payload) == "RetryExhaustedError"

    def test_ignores_other_errors(self) -> None:
        agent, _ = _make_agent_and_client()
        payload = json.dumps({"error": "ConnectionError: server unavailable"})
        assert agent._aborting_error_type(payload) is None

    def test_ignores_successful_payload_mentioning_the_type(self) -> None:
        """Only the error field aborts, so a tool echoing the name in its
        results cannot kill the rollout."""
        agent, _ = _make_agent_and_client()
        payload = json.dumps({"results": "the filing mentions RetryExhaustedError: nope"})
        assert agent._aborting_error_type(payload) is None

    def test_ignores_non_json_output(self) -> None:
        agent, _ = _make_agent_and_client()
        assert agent._aborting_error_type("RetryExhaustedError: plain text") is None

    def test_disabled_when_config_is_empty(self) -> None:
        agent, _ = _make_agent_and_client(_make_config(abort_on_tool_error_types=[]))
        payload = json.dumps({"error": "RetryExhaustedError: gave up"})
        assert agent._aborting_error_type(payload) is None


# ---------------------------------------------------------------------------
# Tests: responses() via TestClient
# ---------------------------------------------------------------------------


class TestResponses:
    def test_text_only_response_injects_v2_nudge_and_keeps_looping(self) -> None:
        """Upstream pins should_stop to False, so prose between tool calls must
        not end the run. v2 reworded the nudge to name submit_final_result; the
        v1 loop injects a bare "Continue.", which is the drift this component
        exists to fix.
        """
        agent, client = _make_agent_and_client(_make_config(max_steps=3))
        agent.server_client.post.return_value = _dotjson_mock(_text_response("Hello!"))

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        output = res.json()["output"]

        assert agent.server_client.post.call_count == 3, (
            "loop should have run max_steps=3 model calls instead of breaking after first text response"
        )

        nudges = [o for o in output if o["type"] == "message" and o["role"] == "user"]
        assert nudges, "no nudge injected after a text-only response"
        assert all(o["content"] == UPSTREAM_NO_TOOL_CALL_NUDGE for o in nudges)
        assert all(o["content"] != _V1_NUDGE for o in nudges), "inherited the v1 nudge"

    def test_nudge_is_configurable(self) -> None:
        agent, client = _make_agent_and_client(_make_config(max_steps=1, no_tool_call_nudge="Keep going."))
        agent.server_client.post.return_value = _dotjson_mock(_text_response("Hello!"))

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        nudges = [o for o in res.json()["output"] if o["type"] == "message" and o["role"] == "user"]
        assert [o["content"] for o in nudges] == ["Keep going."]

    def test_done_tool_terminates_loop(self) -> None:
        agent, client = _make_agent_and_client()

        model_mock = _dotjson_mock(_tool_call_response("submit_final_result", json.dumps({"final_result": "$100B"})))
        agent.server_client.post = AsyncMock(side_effect=_route(model_mock, _dotjson_mock({"status": "ok"})))

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        fn_names = [o.get("name") for o in res.json()["output"] if o["type"] == "function_call"]
        assert fn_names == ["submit_final_result"]

    def test_nudge_loop_still_stops_at_done_tool(self) -> None:
        agent, client = _make_agent_and_client(_make_config(max_steps=5))

        model_mock = _dotjson_mock(
            _text_response("Thinking out loud..."),
            _tool_call_response("submit_final_result", json.dumps({"final_result": "42"})),
        )
        agent.server_client.post = AsyncMock(side_effect=_route(model_mock, _dotjson_mock({"status": "ok"})))

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        fn_names = [o.get("name") for o in res.json()["output"] if o["type"] == "function_call"]
        assert "submit_final_result" in fn_names

    def test_max_steps_terminates_loop(self) -> None:
        agent, client = _make_agent_and_client(_make_config(max_steps=2))

        tool_call = _tool_call_response("edgar_search", json.dumps({"search_query": "AAPL"}))
        model_mock = _dotjson_mock(tool_call, tool_call)
        agent.server_client.post = AsyncMock(side_effect=_route(model_mock, _dotjson_mock({"results": "data"})))

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        assert len([o for o in res.json()["output"] if o["type"] == "function_call"]) == 2

    def test_exhausted_time_budget_prevents_any_model_call(self) -> None:
        """The budget is checked before the turn's query, so an already-expired
        budget must not pay for one more call."""
        agent, client = _make_agent_and_client(_make_config(max_time_seconds=0))
        agent.server_client.post = AsyncMock()

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        assert res.json()["id"] == "error"
        assert agent.server_client.post.call_count == 0

    def test_time_budget_stops_loop_mid_run(self) -> None:
        agent, client = _make_agent_and_client(_make_config(max_steps=5, max_time_seconds=0.05))

        tool_call = _tool_call_response("edgar_search", json.dumps({"search_query": "AAPL"}))
        rs_mock = _dotjson_mock({"results": "data"})

        async def route_post(**kwargs):
            if kwargs["server_name"] == _MODEL_SERVER:
                await asyncio.sleep(0.06)
                return _dotjson_mock(tool_call)
            return rs_mock

        agent.server_client.post = AsyncMock(side_effect=route_post)

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        model_calls = [c for c in agent.server_client.post.call_args_list if c.kwargs["server_name"] == _MODEL_SERVER]
        assert len(model_calls) == 1, "budget should have expired before the second turn"

    def test_stop_reason_recorded_for_a_clean_submission(self) -> None:
        agent, client = _make_agent_and_client()

        model_mock = _dotjson_mock(_tool_call_response("submit_final_result", json.dumps({"final_result": "$100B"})))
        agent.server_client.post = AsyncMock(side_effect=_route(model_mock, _dotjson_mock({"status": "ok"})))

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        assert res.json()["metadata"]["stop_reason"] == "done_tool"
        assert res.json()["metadata"]["steps"] == "1"

    @pytest.mark.parametrize(
        "config, expected",
        [
            (dict(max_steps=2), "max_turns"),
            (dict(max_steps=5, max_time_seconds=0.05), "max_time"),
        ],
    )
    def test_stop_reason_distinguishes_truncated_rollouts(self, config, expected) -> None:
        """A trajectory cut short scores like a confident miss under dealbreaker
        gating, so the results file has to say which one it was."""
        agent, client = _make_agent_and_client(_make_config(**config))
        tool_call = _tool_call_response("edgar_search", json.dumps({"search_query": "AAPL"}))
        rs_mock = _dotjson_mock({"results": "data"})

        async def route_post(**kwargs):
            if kwargs["server_name"] == _MODEL_SERVER:
                if config.get("max_time_seconds"):
                    await asyncio.sleep(0.06)
                return _dotjson_mock(tool_call)
            return rs_mock

        agent.server_client.post = AsyncMock(side_effect=route_post)

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        assert res.json()["metadata"]["stop_reason"] == expected

    def test_abort_error_terminates_rollout(self) -> None:
        """Upstream's on_tool_result hook re-raises RetryExhaustedError, ending
        the run rather than letting the model continue without the data."""
        agent, client = _make_agent_and_client(_make_config(max_steps=5))

        tool_call = _tool_call_response("price_history", json.dumps({"ticker": "AAPL"}))
        model_mock = _dotjson_mock(tool_call, tool_call)
        rs_mock = _dotjson_mock({"error": "RetryExhaustedError: gave up after 5 attempts"})
        agent.server_client.post = AsyncMock(side_effect=_route(model_mock, rs_mock))

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        model_calls = [c for c in agent.server_client.post.call_args_list if c.kwargs["server_name"] == _MODEL_SERVER]
        assert len(model_calls) == 1, "rollout should abort instead of starting a second turn"

    def test_ordinary_tool_error_does_not_abort(self) -> None:
        agent, client = _make_agent_and_client(_make_config(max_steps=2))

        tool_call = _tool_call_response("price_history", json.dumps({"ticker": "AAPL"}))
        model_mock = _dotjson_mock(tool_call, tool_call)
        rs_mock = _dotjson_mock({"error": "ConnectionError: server unavailable"})
        agent.server_client.post = AsyncMock(side_effect=_route(model_mock, rs_mock))

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        model_calls = [c for c in agent.server_client.post.call_args_list if c.kwargs["server_name"] == _MODEL_SERVER]
        assert len(model_calls) == 2, "a plain tool failure is fed back to the model"

    def test_model_call_timeout(self) -> None:
        agent, client = _make_agent_and_client(_make_config(model_call_timeout=0.01))

        async def slow_post(**kwargs):
            await asyncio.sleep(10)

        agent.server_client.post = AsyncMock(side_effect=slow_post)

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        assert res.json()["id"] == "error"

    def test_tool_call_timeout_returns_error(self) -> None:
        agent, client = _make_agent_and_client(_make_config(tool_call_timeout=0.01, max_steps=2))

        model_mock = _dotjson_mock(
            _tool_call_response("edgar_search", json.dumps({"search_query": "AAPL"})),
            _text_response("I'll try something else."),
        )

        async def route_post(**kwargs):
            if kwargs["server_name"] == _MODEL_SERVER:
                return model_mock
            await asyncio.sleep(10)

        agent.server_client.post = AsyncMock(side_effect=route_post)

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        tool_outputs = [o for o in res.json()["output"] if o["type"] == "function_call_output"]
        assert tool_outputs
        assert "timed out" in json.loads(tool_outputs[0]["output"])["error"]

    def test_tool_call_exception_returns_error(self) -> None:
        agent, client = _make_agent_and_client(_make_config(max_steps=2))

        model_mock = _dotjson_mock(
            _tool_call_response("edgar_search", json.dumps({"search_query": "AAPL"})),
            _text_response("Something went wrong."),
        )

        def route_post(**kwargs):
            if kwargs["server_name"] == _MODEL_SERVER:
                return model_mock
            raise ConnectionError("server unavailable")

        agent.server_client.post = AsyncMock(side_effect=route_post)

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        tool_outputs = [o for o in res.json()["output"] if o["type"] == "function_call_output"]
        assert tool_outputs
        assert "ConnectionError" in json.loads(tool_outputs[0]["output"])["error"]

    def test_model_error_terminates_loop(self) -> None:
        agent, client = _make_agent_and_client()
        agent.server_client.post = AsyncMock(side_effect=RuntimeError("internal server error"))

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        assert res.json()["id"] == "error"

    def test_context_overflow_with_truncation(self) -> None:
        """Overflow with truncate_on_overflow=True retries after dropping the
        oldest exchange. Needs >=2 tool-call rounds so there is something to drop.
        """
        agent, client = _make_agent_and_client(_make_config(truncate_on_overflow=True, max_steps=10))

        tc1 = _tool_call_response("edgar_search", json.dumps({"search_query": "AAPL"}), call_id="c1")
        tc2 = _tool_call_response("edgar_search", json.dumps({"search_query": "MSFT"}), call_id="c2")
        submit = _tool_call_response("submit_final_result", json.dumps({"final_result": "Got it."}), call_id="c3")

        model_calls = {"n": 0}

        def route_post(**kwargs):
            if kwargs["server_name"] == _MODEL_SERVER:
                model_calls["n"] += 1
                if model_calls["n"] <= 2:
                    return _dotjson_mock(tc1 if model_calls["n"] == 1 else tc2)
                if model_calls["n"] == 3:
                    raise Exception("maximum context length is 8192 tokens")
                return _dotjson_mock(submit)
            return _dotjson_mock({"results": "filing data"})

        agent.server_client.post = AsyncMock(side_effect=route_post)

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        fn_names = [o.get("name") for o in res.json()["output"] if o["type"] == "function_call"]
        assert "submit_final_result" in fn_names
        assert model_calls["n"] == 4

    def test_context_overflow_without_truncation_breaks(self) -> None:
        agent, client = _make_agent_and_client(_make_config(truncate_on_overflow=False))
        agent.server_client.post = AsyncMock(side_effect=Exception("maximum context length is 8192 tokens"))

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        assert res.json()["id"] == "error"

    def test_usage_accumulation(self) -> None:
        agent, client = _make_agent_and_client(_make_config(max_steps=3))

        usage1 = {
            "input_tokens": 10,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 20,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 30,
        }
        usage2 = {
            "input_tokens": 100,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 200,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 300,
        }

        agent.server_client.post.return_value = _dotjson_mock(
            _reasoning_response() | {"usage": usage1},
            _text_response("Done.") | {"usage": usage2},
        )

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        u = res.json()["usage"]
        assert (u["input_tokens"], u["output_tokens"], u["total_tokens"]) == (110, 220, 330)

    def test_string_input_converted_to_message(self) -> None:
        agent, client = _make_agent_and_client(_make_config(max_steps=1))
        agent.server_client.post.return_value = _dotjson_mock(_text_response("Hi!"))

        res = client.post("/v1/responses", json={"input": "hello"})
        assert res.status_code == 200

        body = agent.server_client.post.call_args_list[0].kwargs["json"]
        assert isinstance(body.input[0], NeMoGymEasyInputMessage)
        assert body.input[0].content == "hello"

    def test_incomplete_details_max_tokens_breaks(self) -> None:
        agent, client = _make_agent_and_client()
        agent.server_client.post.return_value = _dotjson_mock(
            _text_response("partial") | {"incomplete_details": {"reason": "max_output_tokens"}}
        )

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        assert len(res.json()["output"]) == 1


# ---------------------------------------------------------------------------
# Tests: run() — top-level error handling
# ---------------------------------------------------------------------------


class TestRun:
    @pytest.mark.asyncio
    async def test_run_catches_exceptions(self) -> None:
        """run() wraps exceptions and returns reward=0."""
        agent, _ = _make_agent_and_client()
        agent.server_client.post = AsyncMock(side_effect=RuntimeError("catastrophic failure"))

        body = FinanceAgentV2RunRequest.model_validate(
            {"responses_create_params": {"input": [{"role": "user", "content": "test"}]}}
        )
        req = MagicMock()
        req.cookies = {}
        result = await agent.run(req, body)
        assert result.reward == 0.0
        assert result.response.id == "error"
