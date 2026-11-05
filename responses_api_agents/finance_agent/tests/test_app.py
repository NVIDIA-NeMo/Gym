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
"""Tests for Finance Agent (responses_api_agents/finance_agent).

The loop is exercised under both shipped profiles, read from the configs that
ship with the two resource servers rather than restated here, so a profile that
stops parsing or stops setting a required field fails these tests.
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientConnectionError, ClientResponseError, RequestInfo
from fastapi.testclient import TestClient
from multidict import CIMultiDict, CIMultiDictProxy
from omegaconf import OmegaConf
from pydantic import ValidationError
from yarl import URL

from nemo_gym.agents import AGENT_OUTCOME_METADATA_KEY, AgentSeedSessionRequest, OutcomeAction, agent_outcome_policy
from nemo_gym.client import ServerClient
from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig, ModelServerRef, ResourcesServerRef
from nemo_gym.core import (
    EpisodeId,
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    TaskId,
)
from nemo_gym.sandbox import DirectSandboxConnection, SandboxAccess
from nemo_gym.server_runtime import DirectHTTPToolAccess, MCPStreamableHTTPConnection, MCPToolAccess
from nemo_gym.testing.agent_session_conformance import check_agent_seed_rejected
from responses_api_agents.finance_agent.app import (
    FinanceAgent,
    FinanceAgentConfig,
    FinanceAgentRunRequest,
)


_MODEL_SERVER = "model_server"
_RS_SERVER = "resources_server"
_INPUT = {"input": [{"role": "user", "content": "What was revenue?"}]}

_REPO_ROOT = Path(__file__).resolve().parents[3]

#: Every server instance of this loop, and the config file declaring it.
_SHIPPED_CONFIGS = {
    "finance_agent": _REPO_ROOT / "resources_servers/finance_sec_search/configs/finance_sec_search.yaml",
    "finance_agent_v2": _REPO_ROOT / "resources_servers/finance_agent_v2/configs/finance_agent_v2.yaml",
}

#: Each config that runs the loop behind single_agent_turn_legacy, and the Environment Server it declares.
_SINGLE_AGENT_TURN_CONFIGS = {
    "resources_servers/finance_sec_search/configs/finance_sec_search_single_agent_turn.yaml": (
        "finance_agent_environment_server"
    ),
    "resources_servers/finance_agent_v2/configs/finance_agent_v2_single_agent_turn.yaml": (
        "finance_agent_v2_environment_server"
    ),
    "benchmarks/finance_sec_search/config_no_web_search_single_agent_turn.yaml": (
        "finance_sec_search_benchmark_environment_server"
    ),
    "benchmarks/finance_sec_search/config_web_search_single_agent_turn.yaml": (
        "finance_sec_search_web_search_benchmark_environment_server"
    ),
    "benchmarks/finance_agent_v2/config_single_agent_turn.yaml": "finance_agent_v2_benchmark_environment_server",
}

#: Fields with no default, so each profile has to state its own policy.
_POLICY_FIELDS = (
    "no_tool_call_nudge",
    "max_time_seconds",
    "abort_on_tool_error_types",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolved_config(config_path: Path) -> dict:
    initial = OmegaConf.merge(
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        {"config_paths": [str(config_path)]},
    )
    resolved = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=initial,
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    return OmegaConf.to_container(resolved, resolve=True)


def _shipped_block(instance: str) -> dict:
    """The agent block as the server starts with it. Resolved rather than read straight from the
    file, because a profile may pull shared fields in through `config_paths`."""
    return _resolved_config(_SHIPPED_CONFIGS[instance])[instance]["responses_api_agents"]["finance_agent"]


def _shipped_policy(instance: str) -> dict:
    block = _shipped_block(instance)
    return {field: block[field] for field in _POLICY_FIELDS if field in block}


_V1_POLICY = _shipped_policy("finance_agent")
_V2_POLICY = _shipped_policy("finance_agent_v2")


def _make_config(policy: dict | None = None, **overrides) -> FinanceAgentConfig:
    defaults = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="finance_agent",
        resources_server=ResourcesServerRef(type="resources_servers", name=_RS_SERVER),
        model_server=ModelServerRef(type="responses_api_models", name=_MODEL_SERVER),
    )
    defaults.update(_V1_POLICY if policy is None else policy)
    defaults.update(overrides)
    return FinanceAgentConfig(**defaults)


def _v2_config(**overrides) -> FinanceAgentConfig:
    return _make_config(policy=_V2_POLICY, **overrides)


def _make_agent_and_client(config: FinanceAgentConfig | None = None):
    """Create agent + TestClient pair (the canonical test pattern for responses_api_agents)."""
    config = config or _make_config()
    agent = FinanceAgent(config=config, server_client=MagicMock(spec=ServerClient))
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
    return {
        "id": resp_id,
        "created_at": 0.0,
        "model": "test",
        "object": "response",
        "output": [
            {
                "id": "fc_1",
                "call_id": call_id,
                "name": tool_name,
                "arguments": arguments,
                "type": "function_call",
                "status": "completed",
            }
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
# Tests: Config and Construction
# ---------------------------------------------------------------------------


class TestFinanceAgentConfig:
    @pytest.mark.parametrize("instance", sorted(_SHIPPED_CONFIGS))
    def test_shipped_profile_states_every_required_field(self, instance: str) -> None:
        """A required field with no default is only useful if the shipped configs
        set it; otherwise the server fails to start."""
        block = _shipped_block(instance)
        assert set(_POLICY_FIELDS) <= set(block), f"{instance} is missing {set(_POLICY_FIELDS) - set(block)}"

    @pytest.mark.parametrize("instance", sorted(_SHIPPED_CONFIGS))
    def test_shipped_profile_builds_a_valid_config(self, instance: str) -> None:
        config = _make_config(policy=_shipped_policy(instance))
        assert config.no_tool_call_nudge
        assert config.continue_if_not_tool_call is True

    @pytest.mark.parametrize("field", _POLICY_FIELDS)
    def test_omitting_a_policy_field_is_rejected(self, field: str) -> None:
        policy = {k: v for k, v in _V2_POLICY.items() if k != field}
        with pytest.raises(ValidationError):
            _make_config(policy=policy)

    def test_the_two_profiles_differ_where_the_loop_branches(self) -> None:
        """The loop has no benchmark-specific branches, so everything that makes
        the two harnesses different has to show up here."""
        assert _V1_POLICY["max_time_seconds"] is None
        assert _V2_POLICY["max_time_seconds"] > 0
        assert _V1_POLICY["abort_on_tool_error_types"] == []
        assert _V2_POLICY["abort_on_tool_error_types"]
        assert _V1_POLICY["no_tool_call_nudge"] != _V2_POLICY["no_tool_call_nudge"]

    @pytest.mark.parametrize(("config_path", "environment_server"), sorted(_SINGLE_AGENT_TURN_CONFIGS.items()))
    def test_single_agent_turn_config_grants_the_tools(self, config_path: str, environment_server: str) -> None:
        """The agent refuses a session without a direct HTTP grant, so each such config must ask for one,
        and keep the relay config's instances so rows still route to the one server in front of the agent."""
        resolved = _resolved_config(_REPO_ROOT / config_path)
        environment_servers = [
            name for name, block in resolved.items() if isinstance(block, dict) and "environment_servers" in block
        ]
        assert environment_servers == [environment_server]
        server = resolved[environment_server]["environment_servers"]
        assert list(server) == ["single_agent_turn_legacy"]
        assert server["single_agent_turn_legacy"]["resources_tool_transports"] == ["direct_http"]
        agent = server["single_agent_turn_legacy"]["agent_server"]["name"]
        assert list(resolved[agent]["responses_api_agents"]) == ["finance_agent"]

    def test_default_config(self) -> None:
        config = _make_config()
        assert config.max_steps is None
        assert config.done_tools == ["submit_final_result"]
        assert config.model_call_timeout is None
        assert config.tool_call_timeout is None
        assert config.truncate_on_overflow is False
        assert config.continue_if_not_tool_call is True

    def test_custom_config(self) -> None:
        config = _make_config(
            max_steps=10,
            max_time_seconds=60.0,
            done_tools=["submit_final_result", "abort"],
            no_tool_call_nudge="Keep going.",
            continue_if_not_tool_call=False,
            abort_on_tool_error_types=["RetryExhaustedError"],
            model_call_timeout=30.0,
            tool_call_timeout=60.0,
            truncate_on_overflow=True,
        )
        assert config.max_steps == 10
        assert config.max_time_seconds == 60.0
        assert config.done_tools == ["submit_final_result", "abort"]
        assert config.no_tool_call_nudge == "Keep going."
        assert config.continue_if_not_tool_call is False
        assert config.abort_on_tool_error_types == ["RetryExhaustedError"]
        assert config.model_call_timeout == 30.0
        assert config.tool_call_timeout == 60.0
        assert config.truncate_on_overflow is True

    def test_sanity_construction(self) -> None:
        agent, _ = _make_agent_and_client()
        assert agent is not None


# ---------------------------------------------------------------------------
# Tests: Context Overflow Detection
# ---------------------------------------------------------------------------


class TestContextOverflowDetection:
    @pytest.mark.parametrize(
        "msg",
        [
            "maximum context length is 8192 tokens",
            "context length is only 4096 tokens",
            "maximum input length of 32768 tokens exceeded",
            "Please reduce the length of the input",
            "prompt is too long",
            "input exceeded the context window",
            "too large for model with 8192 maximum context length",
            "longer than the model's context length",
            "payload too large",
        ],
    )
    def test_detects_overflow(self, msg: str) -> None:
        assert FinanceAgent._is_context_overflow_error(Exception(msg))

    @pytest.mark.parametrize(
        "msg",
        [
            "connection refused",
            "timeout error",
            "Internal server error",
            "rate limit exceeded",
        ],
    )
    def test_ignores_non_overflow(self, msg: str) -> None:
        assert not FinanceAgent._is_context_overflow_error(Exception(msg))


# ---------------------------------------------------------------------------
# Tests: _is_model_output
# ---------------------------------------------------------------------------


class TestIsModelOutput:
    def test_function_call_is_model_output(self) -> None:
        item = MagicMock(type="function_call")
        assert FinanceAgent._is_model_output(item)

    def test_reasoning_is_model_output(self) -> None:
        item = MagicMock(type="reasoning")
        assert FinanceAgent._is_model_output(item)

    def test_assistant_message_is_model_output(self) -> None:
        item = MagicMock(type="message", role="assistant")
        assert FinanceAgent._is_model_output(item)

    def test_user_message_is_not_model_output(self) -> None:
        item = MagicMock(type="message", role="user")
        assert not FinanceAgent._is_model_output(item)

    def test_function_call_output_is_not_model_output(self) -> None:
        item = MagicMock(type="function_call_output")
        assert not FinanceAgent._is_model_output(item)


# ---------------------------------------------------------------------------
# Tests: _truncate_oldest_exchange
# ---------------------------------------------------------------------------


class TestTruncateOldestExchange:
    @staticmethod
    def _item(type_: str, role: str | None = None) -> MagicMock:
        return MagicMock(type=type_, role=role)

    def test_empty_list(self) -> None:
        assert FinanceAgent._truncate_oldest_exchange([]) == []

    def test_single_item(self) -> None:
        items = [self._item("message", "assistant")]
        assert len(FinanceAgent._truncate_oldest_exchange(items)) == 1

    def test_removes_first_exchange(self) -> None:
        items = [
            self._item("reasoning"),
            self._item("function_call"),
            self._item("function_call_output"),
            self._item("function_call_output"),
            self._item("message", "assistant"),
            self._item("function_call"),
            self._item("function_call_output"),
        ]
        result = FinanceAgent._truncate_oldest_exchange(items)
        assert result[0] is items[4]
        assert len(result) == 3

    def test_no_truncation_when_single_exchange(self) -> None:
        items = [self._item("reasoning"), self._item("function_call")]
        result = FinanceAgent._truncate_oldest_exchange(items)
        assert result == items


# ---------------------------------------------------------------------------
# Tests: _aborting_error_type
# ---------------------------------------------------------------------------


class TestAbortingErrorType:
    def test_matches_abort_error_in_the_servers_envelope(self) -> None:
        """The resource server double-encodes: the error object is a JSON string
        inside ``results``, which is the only shape a tool failure arrives in."""
        agent, _ = _make_agent_and_client(_v2_config())
        payload = json.dumps({"results": json.dumps({"error": "RetryExhaustedError: gave up after 5 attempts"})})
        assert agent._aborting_error_type(payload) == "RetryExhaustedError"

    def test_matches_abort_error_without_an_envelope(self) -> None:
        """A tool call that never reached the server is rendered locally flat."""
        agent, _ = _make_agent_and_client(_v2_config())
        payload = json.dumps({"error": "RetryExhaustedError: gave up after 5 attempts"})
        assert agent._aborting_error_type(payload) == "RetryExhaustedError"

    def test_ignores_other_errors(self) -> None:
        agent, _ = _make_agent_and_client(_v2_config())
        payload = json.dumps({"results": json.dumps({"error": "ConnectionError: server unavailable"})})
        assert agent._aborting_error_type(payload) is None

    def test_ignores_successful_payload_mentioning_the_type(self) -> None:
        """Only the error field aborts, so a tool echoing the name in its
        results cannot kill the rollout."""
        agent, _ = _make_agent_and_client(_v2_config())
        payload = json.dumps({"results": "the filing mentions RetryExhaustedError: nope"})
        assert agent._aborting_error_type(payload) is None

    def test_ignores_non_json_output(self) -> None:
        agent, _ = _make_agent_and_client(_v2_config())
        assert agent._aborting_error_type("RetryExhaustedError: plain text") is None

    def test_ignores_a_successful_reply_that_happens_to_be_json(self) -> None:
        """A tool whose own output is a JSON object with no error field is a
        success, however it is worded."""
        agent, _ = _make_agent_and_client(_v2_config())
        payload = json.dumps({"results": json.dumps({"summary": "RetryExhaustedError: appears in this filing"})})
        assert agent._aborting_error_type(payload) is None

    def test_disabled_under_the_v1_profile(self) -> None:
        agent, _ = _make_agent_and_client()
        payload = json.dumps({"results": json.dumps({"error": "RetryExhaustedError: gave up"})})
        assert agent._aborting_error_type(payload) is None


# ---------------------------------------------------------------------------
# Tests: responses() via TestClient
# ---------------------------------------------------------------------------


class TestResponses:
    def test_text_only_turn_ends_episode_when_continue_if_not_tool_call_is_false(self) -> None:
        """Training overlay: a text-only assistant turn ends the
        episode after a single model call.

        Gym shipped profiles leave ``continue_if_not_tool_call`` true.
        GRPO sets it to false because NeMo-RL requires each turn's prompt tokens
        to extend the previous turn's, and re-serializing a reasoning turn
        back into history strips its thinking block, breaking that prefix.
        """
        agent, client = _make_agent_and_client(_make_config(max_steps=3, continue_if_not_tool_call=False))
        agent.server_client.post.return_value = _dotjson_mock(_text_response("Hello!"))

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200

        assert agent.server_client.post.call_count == 1, (
            "text-only turn should end the episode instead of looping to max_steps"
        )
        output = res.json()["output"]
        assert not [o for o in output if o["type"] == "message" and o["role"] == "user"], (
            "no 'Continue.' should be injected when continue_if_not_tool_call is false"
        )
        assert res.json()["metadata"]["stop_reason"] == "text_only"

    def test_text_only_response_injects_continue_until_max_steps(self) -> None:
        """Text-only assistant responses must inject the nudge and keep looping
        (mirrors vals-ai/finance-agent ``_before_query`` + ``_should_stop=False``)
        rather than terminating after one step.

        Eval default: ``continue_if_not_tool_call`` is true.
        """
        config = _make_config(max_steps=3, continue_if_not_tool_call=True)
        agent, client = _make_agent_and_client(config)
        agent.server_client.post.return_value = _dotjson_mock(_text_response("Hello!"))

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        output = res.json()["output"]

        assert agent.server_client.post.call_count == 3, (
            "loop should have run max_steps=3 model calls instead of breaking after first text response"
        )

        user_continues = [o for o in output if o["type"] == "message" and o["role"] == "user"]
        assert len(user_continues) >= 1, "no 'Continue.' user message injected"
        assert all(o["content"] == _V1_POLICY["no_tool_call_nudge"] for o in user_continues)

    def test_text_only_response_injects_the_v2_nudge(self) -> None:
        """The FABv2 profile names submit_final_result instead of sending a bare
        "Continue."; picking up the other profile's nudge is silent drift."""
        agent, client = _make_agent_and_client(_v2_config(max_steps=3))
        agent.server_client.post.return_value = _dotjson_mock(_text_response("Hello!"))

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200

        nudges = [o for o in res.json()["output"] if o["type"] == "message" and o["role"] == "user"]
        assert nudges, "no nudge injected after a text-only response"
        assert all(o["content"] == _V2_POLICY["no_tool_call_nudge"] for o in nudges)

    def test_nudge_is_configurable(self) -> None:
        agent, client = _make_agent_and_client(
            _make_config(max_steps=1, continue_if_not_tool_call=True, no_tool_call_nudge="Keep going.")
        )
        agent.server_client.post.return_value = _dotjson_mock(_text_response("Hello!"))

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        nudges = [o for o in res.json()["output"] if o["type"] == "message" and o["role"] == "user"]
        assert [o["content"] for o in nudges] == ["Keep going."]

    def test_continue_injection_stops_at_submit_final_result(self) -> None:
        """Continue.-loop must yield as soon as a done-tool fires -- otherwise
        the agent could keep looping past a legitimate terminal tool call.
        """
        config = _make_config(max_steps=5, continue_if_not_tool_call=True)
        agent, client = _make_agent_and_client(config)

        text_then_submit_responses = [
            _text_response("Thinking out loud..."),
            _tool_call_response("submit_final_result", json.dumps({"final_result": "42"})),
        ]
        model_mock = _dotjson_mock(*text_then_submit_responses)
        rs_mock = _dotjson_mock({"status": "ok"})

        agent.server_client.post = AsyncMock(side_effect=_route(model_mock, rs_mock))

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        output = res.json()["output"]
        fn_names = [o.get("name") for o in output if o["type"] == "function_call"]
        assert "submit_final_result" in fn_names, "done-tool should still terminate after Continue. injection"

    def test_tool_call_then_text_continues_until_max_steps(self) -> None:
        """Tool call → text response no longer terminates when
        ``continue_if_not_tool_call`` is true -- the loop injects
        ``Continue.`` and keeps going.  Bounded by max_steps here so the test
        doesn't run forever.
        """
        config = _make_config(max_steps=2, continue_if_not_tool_call=True)
        agent, client = _make_agent_and_client(config)

        tool_call_data = _tool_call_response("sec_filing_search", json.dumps({"ticker": "AAPL"}))
        text_data = _text_response("Here is the data.")
        tool_result = {"result": "filing data"}

        model_mock = _dotjson_mock(tool_call_data, text_data)
        rs_mock = _dotjson_mock(tool_result)

        agent.server_client.post = AsyncMock(side_effect=_route(model_mock, rs_mock))

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        output = res.json()["output"]
        types = [o["type"] for o in output]
        assert "function_call" in types
        assert "function_call_output" in types
        # Last item is the injected Continue. user message (would have been
        # the seed for a step-3 model call if max_steps had allowed it).
        assert output[-1]["type"] == "message"
        assert output[-1]["role"] == "user"
        assert output[-1]["content"] == "Continue."

    def test_done_tool_terminates_loop(self) -> None:
        """submit_final_result tool call exits the loop immediately."""
        agent, client = _make_agent_and_client()

        tool_call_data = _tool_call_response("submit_final_result", json.dumps({"final_result": "Revenue was $100B"}))
        tool_result = {"status": "ok"}

        model_mock = _dotjson_mock(tool_call_data)
        rs_mock = _dotjson_mock(tool_result)

        agent.server_client.post = AsyncMock(side_effect=_route(model_mock, rs_mock))

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        output = res.json()["output"]
        fn_names = [o.get("name") for o in output if o["type"] == "function_call"]
        assert "submit_final_result" in fn_names

    def test_max_steps_terminates_loop(self) -> None:
        """Loop exits after max_steps even if model keeps producing tool calls."""
        config = _make_config(max_steps=2)
        agent, client = _make_agent_and_client(config)

        tool_call_data = _tool_call_response("sec_filing_search", json.dumps({"ticker": "AAPL"}))
        tool_result = {"result": "data"}

        model_mock = _dotjson_mock(tool_call_data, tool_call_data)
        rs_mock = _dotjson_mock(tool_result)

        agent.server_client.post = AsyncMock(side_effect=_route(model_mock, rs_mock))

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        output = res.json()["output"]
        fn_calls = [o for o in output if o["type"] == "function_call"]
        assert len(fn_calls) == 2

    def test_exhausted_time_budget_prevents_any_model_call(self) -> None:
        """The budget is checked before the turn's query, so an already-expired
        budget must not pay for one more call."""
        agent, client = _make_agent_and_client(_v2_config(max_time_seconds=0))
        agent.server_client.post = AsyncMock()

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        assert res.json()["id"] == "error"
        assert agent.server_client.post.call_count == 0

    def test_time_budget_stops_loop_mid_run(self) -> None:
        agent, client = _make_agent_and_client(_v2_config(max_steps=5, max_time_seconds=0.05))

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

    def test_no_time_budget_under_the_v1_profile(self) -> None:
        """max_time_seconds: null must leave the loop turn-bounded, with no clock
        check able to cut a training rollout short."""
        agent, client = _make_agent_and_client(_make_config(max_steps=2))

        tool_call = _tool_call_response("sec_filing_search", json.dumps({"ticker": "AAPL"}))
        model_mock = _dotjson_mock(tool_call, tool_call)

        async def route_post(**kwargs):
            if kwargs["server_name"] == _MODEL_SERVER:
                await asyncio.sleep(0.02)
                return model_mock
            return _dotjson_mock({"result": "data"})

        agent.server_client.post = AsyncMock(side_effect=route_post)

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        assert res.json()["metadata"]["stop_reason"] == "max_turns"

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
        agent, client = _make_agent_and_client(_v2_config(**config))
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
        """The FABv2 profile aborts on retry exhaustion rather than letting the
        model continue without the data."""
        agent, client = _make_agent_and_client(_v2_config(max_steps=5))

        tool_call = _tool_call_response("price_history", json.dumps({"ticker": "AAPL"}))
        model_mock = _dotjson_mock(tool_call, tool_call)
        rs_mock = _dotjson_mock({"results": json.dumps({"error": "RetryExhaustedError: gave up after 5 attempts"})})
        agent.server_client.post = AsyncMock(side_effect=_route(model_mock, rs_mock))

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        model_calls = [c for c in agent.server_client.post.call_args_list if c.kwargs["server_name"] == _MODEL_SERVER]
        assert len(model_calls) == 1, "rollout should abort instead of starting a second turn"

    def test_ordinary_tool_error_does_not_abort(self) -> None:
        agent, client = _make_agent_and_client(_v2_config(max_steps=2))

        tool_call = _tool_call_response("price_history", json.dumps({"ticker": "AAPL"}))
        model_mock = _dotjson_mock(tool_call, tool_call)
        rs_mock = _dotjson_mock({"results": json.dumps({"error": "ConnectionError: server unavailable"})})
        agent.server_client.post = AsyncMock(side_effect=_route(model_mock, rs_mock))

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        model_calls = [c for c in agent.server_client.post.call_args_list if c.kwargs["server_name"] == _MODEL_SERVER]
        assert len(model_calls) == 2, "a plain tool failure is fed back to the model"

    def test_abort_error_is_fed_back_under_the_v1_profile(self) -> None:
        """With no abort types configured, the same error keeps the loop going."""
        agent, client = _make_agent_and_client(_make_config(max_steps=2))

        tool_call = _tool_call_response("sec_filing_search", json.dumps({"ticker": "AAPL"}))
        model_mock = _dotjson_mock(tool_call, tool_call)
        rs_mock = _dotjson_mock({"results": json.dumps({"error": "RetryExhaustedError: gave up after 5 attempts"})})
        agent.server_client.post = AsyncMock(side_effect=_route(model_mock, rs_mock))

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        model_calls = [c for c in agent.server_client.post.call_args_list if c.kwargs["server_name"] == _MODEL_SERVER]
        assert len(model_calls) == 2

    def test_model_call_timeout(self) -> None:
        """Model call timeout → loop terminates gracefully."""
        config = _make_config(model_call_timeout=0.01)
        agent, client = _make_agent_and_client(config)

        async def slow_post(**kwargs):
            await asyncio.sleep(10)

        agent.server_client.post = AsyncMock(side_effect=slow_post)

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        assert res.json()["id"] == "error"

    def test_tool_call_timeout_returns_error(self) -> None:
        """Tool call timeout → error JSON fed back to model, loop continues."""
        config = _make_config(tool_call_timeout=0.01, max_steps=2)
        agent, client = _make_agent_and_client(config)

        tool_call_data = _tool_call_response("sec_filing_search", json.dumps({"ticker": "AAPL"}))
        text_data = _text_response("I'll try something else.")

        model_mock = _dotjson_mock(tool_call_data, text_data)

        async def route_post(**kwargs):
            if kwargs["server_name"] == _MODEL_SERVER:
                return model_mock
            await asyncio.sleep(10)

        agent.server_client.post = AsyncMock(side_effect=route_post)

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        output = res.json()["output"]
        tool_outputs = [o for o in output if o["type"] == "function_call_output"]
        assert len(tool_outputs) >= 1
        error_payload = json.loads(tool_outputs[0]["output"])
        assert "timed out" in error_payload["error"]

    def test_tool_call_exception_returns_error(self) -> None:
        """Tool call exception → error JSON fed back to model, loop continues."""
        config = _make_config(max_steps=2)
        agent, client = _make_agent_and_client(config)

        tool_call_data = _tool_call_response("sec_filing_search", json.dumps({"ticker": "AAPL"}))
        text_data = _text_response("Something went wrong.")

        model_mock = _dotjson_mock(tool_call_data, text_data)

        def route_post(**kwargs):
            if kwargs["server_name"] == _MODEL_SERVER:
                return model_mock
            raise ConnectionError("server unavailable")

        agent.server_client.post = AsyncMock(side_effect=route_post)

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        output = res.json()["output"]
        tool_outputs = [o for o in output if o["type"] == "function_call_output"]
        assert len(tool_outputs) >= 1
        error_payload = json.loads(tool_outputs[0]["output"])
        assert "ConnectionError" in error_payload["error"]

    def test_model_error_terminates_loop(self) -> None:
        """Non-overflow model error → loop breaks."""
        agent, client = _make_agent_and_client()

        def fail_post(**kwargs):
            raise RuntimeError("internal server error")

        agent.server_client.post = AsyncMock(side_effect=fail_post)

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        assert res.json()["id"] == "error"

    def test_context_overflow_with_truncation(self) -> None:
        """Context overflow with truncate_on_overflow=True → retries after truncation.

        Need >=2 tool-call rounds so _truncate_oldest_exchange can drop one
        and keep another. Overflow on model call 3, recovery on model call 4
        via a done-tool so the loop terminates cleanly (under C1 a text
        response would keep looping with Continue. injection).
        """
        config = _make_config(truncate_on_overflow=True, max_steps=10)
        agent, client = _make_agent_and_client(config)

        tc1 = _tool_call_response("sec_filing_search", json.dumps({"ticker": "AAPL"}), call_id="c1")
        tc2 = _tool_call_response("sec_filing_search", json.dumps({"ticker": "MSFT"}), call_id="c2")
        submit = _tool_call_response("submit_final_result", json.dumps({"final_result": "Got it."}), call_id="c3")

        model_calls = {"n": 0}

        def route_post(**kwargs):
            if kwargs["server_name"] == _MODEL_SERVER:
                model_calls["n"] += 1
                if model_calls["n"] <= 2:
                    data = tc1 if model_calls["n"] == 1 else tc2
                    return _dotjson_mock(data)
                if model_calls["n"] == 3:
                    raise Exception("maximum context length is 8192 tokens")
                return _dotjson_mock(submit)
            return _dotjson_mock({"result": "filing data"})

        agent.server_client.post = AsyncMock(side_effect=route_post)

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        output = res.json()["output"]
        fn_names = [o.get("name") for o in output if o["type"] == "function_call"]
        assert "submit_final_result" in fn_names
        assert model_calls["n"] == 4

    def test_context_overflow_without_truncation_breaks(self) -> None:
        """Context overflow with truncate_on_overflow=False → loop breaks."""
        agent, client = _make_agent_and_client(_make_config(truncate_on_overflow=False))

        def fail_post(**kwargs):
            raise Exception("maximum context length is 8192 tokens")

        agent.server_client.post = AsyncMock(side_effect=fail_post)

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        assert res.json()["id"] == "error"

    def test_usage_accumulation(self) -> None:
        """Usage tokens accumulate across multiple model calls."""
        config = _make_config(max_steps=3)
        agent, client = _make_agent_and_client(config)

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

        resp1 = _reasoning_response() | {"usage": usage1}
        resp2 = _text_response("Done.") | {"usage": usage2}

        agent.server_client.post.return_value = _dotjson_mock(resp1, resp2)

        res = client.post("/v1/responses", json=_INPUT)
        assert res.status_code == 200
        u = res.json()["usage"]
        assert u["input_tokens"] == 110
        assert u["output_tokens"] == 220
        assert u["total_tokens"] == 330

    def test_string_input_converted_to_message(self) -> None:
        """String input is automatically wrapped in a user message.

        Bounded by max_steps=1 because under C1 a text-only response no
        longer terminates the loop on its own.
        """
        config = _make_config(max_steps=1)
        agent, client = _make_agent_and_client(config)
        agent.server_client.post.return_value = _dotjson_mock(_text_response("Hi!"))

        res = client.post("/v1/responses", json={"input": "hello"})
        assert res.status_code == 200

        post_kwargs = agent.server_client.post.call_args_list[0].kwargs
        body = post_kwargs["json"]
        assert isinstance(body.input[0], NeMoGymEasyInputMessage)
        assert body.input[0].content == "hello"

    def test_incomplete_details_max_tokens_breaks(self) -> None:
        """incomplete_details with reason=max_output_tokens → loop exits."""
        agent, client = _make_agent_and_client()

        resp = _text_response("partial") | {"incomplete_details": {"reason": "max_output_tokens"}}
        agent.server_client.post.return_value = _dotjson_mock(resp)

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

        body = FinanceAgentRunRequest.model_validate(
            {
                "responses_create_params": {"input": [{"role": "user", "content": "test"}]},
            }
        )
        req = MagicMock()
        req.cookies = {}
        result = await agent.run(req, body)
        assert result.reward == 0.0
        assert result.response.id == "error"


# ---------------------------------------------------------------------------
# Tests: Agent sessions (an Environment Server runs the loop)
# ---------------------------------------------------------------------------

_GRANT = DirectHTTPToolAccess(
    name="finance_sec_search.direct_http",
    required=True,
    base_url="http://resources:9000",
    cookies={"session": "seeded"},
    headers={"x-grant": "token"},
)


def _session_seed(**overrides) -> AgentSeedSessionRequest:
    return AgentSeedSessionRequest(
        agent_session_id="agent-session",
        episode_id=EpisodeId(rollout_id="rollout", attempt=0),
        task_id=TaskId(taskset="finance", task_id="0"),
        **overrides,
    )


def _tool_reply(body: str, *, cookies: dict | None = None, status: int = 200) -> MagicMock:
    reply = MagicMock()
    reply.status = status
    reply.cookies = cookies or {}
    reply.content.read = AsyncMock(return_value=body.encode())
    return reply


def _model_error_reply(status: int, body: bytes = b'{"error": "upstream"}') -> MagicMock:
    """A model server reply that fails the agent's own ``raise_for_status`` with ``status`` and ``body``."""
    url = URL("http://model/v1/responses")
    reply = MagicMock()
    reply.ok = False
    reply.status = status
    reply.content.read = AsyncMock(return_value=body)
    reply.raise_for_status = MagicMock(
        side_effect=ClientResponseError(
            RequestInfo(url=url, method="POST", headers=CIMultiDictProxy(CIMultiDict()), real_url=url),
            (),
            status=status,
            message="upstream",
        )
    )
    return reply


# What openai_model answers when the provider refuses a context overflow with a 400 and the model server does not
# propagate that status: a 500 whose body carries the provider's text.
_OVERFLOW_BODY = json.dumps(
    "Hit an exception in SimpleModelServer___policy_model calling an inner server: "
    '{"error": {"message": "This model\'s maximum context length is 128000 tokens."}}'
).encode()


def _session_client(model_mock, *, observability: bool = False, config: FinanceAgentConfig | None = None):
    agent, client = _make_agent_and_client(config)
    agent.server_client.global_config_dict = {"observability_enabled": observability}
    agent.server_client.post = AsyncMock(side_effect=_route(model_mock, AsyncMock()))
    return agent, client


def _seed(client: TestClient) -> None:
    seed = _session_seed(tool_accesses=[_GRANT])
    assert client.post("/v1/agent_sessions", json=seed.model_dump(mode="json")).is_success


def _close(client: TestClient) -> dict:
    seed = _session_seed()
    response = client.post(
        "/v1/agent_sessions/close",
        json={"agent_session_id": seed.agent_session_id, "episode_id": seed.episode_id.model_dump(mode="json")},
    )
    assert response.status_code == 200, response.text
    return response.json()


class TestAgentSessions:
    @pytest.mark.parametrize(
        ("unsupported", "reason"),
        [
            ({}, "Finance Agent needs exactly one direct HTTP tool access"),
            (
                {
                    "tool_accesses": [
                        _GRANT,
                        DirectHTTPToolAccess(name="second", required=True, base_url="http://second:9000"),
                    ]
                },
                "Finance Agent needs exactly one direct HTTP tool access",
            ),
            (
                {
                    "tool_accesses": [
                        _GRANT,
                        MCPToolAccess(
                            name="resources",
                            required=True,
                            connection=MCPStreamableHTTPConnection(url="http://resources:9000/mcp"),
                        ),
                    ]
                },
                "Finance Agent does not support required MCP tool access: resources",
            ),
            (
                {
                    "tool_accesses": [_GRANT],
                    "sandbox_access": SandboxAccess(
                        connection=DirectSandboxConnection(provider_config_ref="runtime", descriptor={}),
                        workdir="/work",
                    ),
                },
                "Finance Agent does not support sandbox access",
            ),
        ],
    )
    def test_unsupported_seed_is_refused_with_a_final_422(self, unsupported: dict, reason: str) -> None:
        agent, _ = _make_agent_and_client()
        assert check_agent_seed_rejected(agent.setup_webserver(), _session_seed(**unsupported)).startswith(reason)

    def test_tool_calls_go_through_the_grant_and_close_returns_the_cookie_jar(self, monkeypatch) -> None:
        model_mock = _dotjson_mock(
            _tool_call_response("parse_html_page", '{"url": "https://sec.gov/10k", "key": "k"}'),
            _tool_call_response("submit_final_result", '{"final_result": "42"}', call_id="call_2", resp_id="resp_2"),
        )
        agent, client = _session_client(model_mock)
        tool = AsyncMock(
            side_effect=[
                _tool_reply('{"results": "SUCCESS"}', cookies={"session": "after-parse"}),
                _tool_reply('{"results": "submitted"}'),
            ]
        )
        monkeypatch.setattr("responses_api_agents.finance_agent.app.http_request", tool)

        assert client.post(
            "/v1/agent_sessions", json=_session_seed(tool_accesses=[_GRANT]).model_dump(mode="json")
        ).is_success
        response = client.post("/v1/responses", json=_INPUT)

        assert response.status_code == 200, response.text
        assert response.json()["metadata"]["stop_reason"] == "done_tool"
        # The configured resources server has no session for this episode, so only the model is called by name.
        assert [call.kwargs["server_name"] for call in agent.server_client.post.await_args_list] == [_MODEL_SERVER] * 2
        assert [call.kwargs["url"] for call in tool.await_args_list] == [
            "http://resources:9000/parse_html_page",
            "http://resources:9000/submit_final_result",
        ]
        assert all(call.kwargs["headers"] == {"x-grant": "token"} for call in tool.await_args_list)
        assert [call.kwargs["cookies"] for call in tool.await_args_list] == [
            {"session": "seeded"},
            {"session": "after-parse"},
        ]
        # The resources cookie travels back through close, not on the activation's reply.
        assert "session" not in response.cookies
        closed = _close(client)
        assert closed["resources_cookies"] == {"session": "after-parse"}
        assert "agent_observations" not in closed or closed["agent_observations"] is None

    @pytest.mark.parametrize(
        ("error", "failure_kind", "status_code", "action"),
        [
            (
                ClientResponseError(MagicMock(), (), status=503, message="Service Unavailable"),
                "model_request_error",
                503,
                OutcomeAction.RETRY,
            ),
            (
                ClientResponseError(MagicMock(), (), status=400, message="Bad Request"),
                "model_request_error",
                400,
                OutcomeAction.VERIFY,
            ),
            (
                ClientResponseError(MagicMock(), (), status=401, message="Unauthorized"),
                "model_request_error",
                401,
                OutcomeAction.FAIL,
            ),
            (ClientConnectionError("connection refused"), "model_transport_error", None, OutcomeAction.RETRY),
            (RuntimeError("bad model reply"), "unknown", None, OutcomeAction.RETRY),
        ],
    )
    def test_model_failure_in_a_session_is_classified_not_zeroed(
        self, error: Exception, failure_kind: str, status_code: int | None, action: OutcomeAction
    ) -> None:
        agent, client = _session_client(AsyncMock())
        agent.server_client.post = AsyncMock(side_effect=error)

        assert client.post(
            "/v1/agent_sessions", json=_session_seed(tool_accesses=[_GRANT]).model_dump(mode="json")
        ).is_success
        reply = client.post("/v1/responses", json=_INPUT)

        assert reply.status_code == 200, reply.text
        response = NeMoGymResponse.model_validate(reply.json())
        assert response.status == "failed"
        assert response.metadata["stop_reason"] == "error"
        outcome = json.loads(response.metadata[AGENT_OUTCOME_METADATA_KEY])
        assert outcome.get("failure_kind") == failure_kind
        assert outcome.get("status_code") == status_code
        assert agent_outcome_policy(response) == action

    def test_model_timeout_in_a_session_is_a_transport_failure(self) -> None:
        agent, client = _session_client(AsyncMock(), config=_make_config(model_call_timeout=0.01))

        async def slow_post(**kwargs):
            await asyncio.sleep(1)

        agent.server_client.post = AsyncMock(side_effect=slow_post)
        assert client.post(
            "/v1/agent_sessions", json=_session_seed(tool_accesses=[_GRANT]).model_dump(mode="json")
        ).is_success
        response = NeMoGymResponse.model_validate(client.post("/v1/responses", json=_INPUT).json())

        assert json.loads(response.metadata[AGENT_OUTCOME_METADATA_KEY]) == {"failure_kind": "model_transport_error"}
        assert agent_outcome_policy(response) == OutcomeAction.RETRY

    @pytest.mark.parametrize("status", [500, 400])
    def test_context_overflow_in_a_session_is_graded_at_any_status(self, status: int) -> None:
        """An overflow is the model's own failure, so the episode is verified, not dispatched again until it fits."""
        agent, client = _session_client(AsyncMock())
        agent.server_client.post = AsyncMock(return_value=_model_error_reply(status, _OVERFLOW_BODY))
        _seed(client)

        response = NeMoGymResponse.model_validate(client.post("/v1/responses", json=_INPUT).json())

        assert response.status == "incomplete"
        assert response.error is None
        assert response.metadata["stop_reason"] == "error"
        assert AGENT_OUTCOME_METADATA_KEY not in response.metadata
        assert agent_outcome_policy(response) == OutcomeAction.VERIFY

    def test_model_failure_without_a_session_is_still_verified(self) -> None:
        """A legacy /run verifies whatever the loop returns; only a session reports the cause."""
        agent, client = _make_agent_and_client()
        agent.server_client.post = AsyncMock(
            side_effect=ClientResponseError(MagicMock(), (), status=503, message="Service Unavailable")
        )

        response = NeMoGymResponse.model_validate(client.post("/v1/responses", json=_INPUT).json())

        assert response.status != "failed"
        assert AGENT_OUTCOME_METADATA_KEY not in response.metadata
        assert agent_outcome_policy(response) == OutcomeAction.VERIFY

    def test_tool_abort_in_a_session_is_graded_like_the_legacy_run(self, monkeypatch) -> None:
        """An aborting tool error ends the loop as upstream does; the answer is graded, not retried."""
        model_mock = _dotjson_mock(_tool_call_response("web_search", '{"query": "revenue"}'))
        agent, client = _session_client(model_mock, config=_v2_config())
        abort = json.dumps({"results": json.dumps({"error": "RetryExhaustedError: Tavily failed after 3 attempts"})})
        monkeypatch.setattr(
            "responses_api_agents.finance_agent.app.http_request", AsyncMock(return_value=_tool_reply(abort))
        )

        assert client.post(
            "/v1/agent_sessions", json=_session_seed(tool_accesses=[_GRANT]).model_dump(mode="json")
        ).is_success
        response = NeMoGymResponse.model_validate(client.post("/v1/responses", json=_INPUT).json())

        assert response.metadata["stop_reason"] == "error"
        assert agent_outcome_policy(response) == OutcomeAction.VERIFY

    def test_session_returns_observations_when_model_calls_are_captured(self, monkeypatch) -> None:
        model_mock = _dotjson_mock(
            _tool_call_response("sec_filing_search", '{"ticker": "AAPL"}'),
            _tool_call_response("submit_final_result", '{"final_result": "42"}', call_id="call_2", resp_id="resp_2"),
        )
        agent, client = _session_client(model_mock, observability=True)
        monkeypatch.setattr(
            "responses_api_agents.finance_agent.app.http_request",
            AsyncMock(side_effect=[_tool_reply("not found", status=404), _tool_reply('{"results": "submitted"}')]),
        )

        assert client.post(
            "/v1/agent_sessions", json=_session_seed(tool_accesses=[_GRANT]).model_dump(mode="json")
        ).is_success
        assert client.post("/ng-rollout/rollout/v1/responses", json=_INPUT).status_code == 200

        # Model calls carry the rollout prefix, so captured calls join the episode.
        assert {call.kwargs["url_path"] for call in agent.server_client.post.await_args_list} == {
            "/ng-rollout/rollout/v1/responses"
        }
        records = _close(client)["agent_observations"]["records"]
        invocation = next(record for record in records if record["kind"] == "agent_invocation")
        assert invocation["status"] == "completed"
        assert invocation["stop_reason"] == "done_tool"
        assert [call["response_id"] for call in invocation["model_calls"]] == ["resp_1", "resp_2"]
        tool_calls = [record for record in records if record["kind"] == "tool_call"]
        assert [(call["tool_name"], call["status"], call["error_type"]) for call in tool_calls] == [
            ("sec_filing_search", "failed", "http_404"),
            ("submit_final_result", "completed", None),
        ]
