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
"""Tests for the Finance Agent v2 (FABv2) resource server.

These exercise the tools-only wrapper without hitting external services: each
upstream tool's network layer (and the retrieval / judge model servers) is
mocked. Requires the upstream `finance_agent` package to be importable (installed
via the resource server's requirements.txt).
"""

import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.finance_agent_v2.app import (
    FinanceAgentV2ResourcesServer,
    FinanceAgentV2ResourcesServerConfig,
    FinanceAgentV2VerifyRequest,
)


_PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompt_templates"
_TEST_SESSION_ID = "test-session"


def _prompt_fpaths() -> dict:
    return {
        "judge_prompt_template_fpath": str(_PROMPT_DIR / "finance_agent_v2_judge.yaml"),
        "retrieval_system_prompt_fpath": str(_PROMPT_DIR / "finance_agent_v2_retrieval.yaml"),
    }


def _make_server(**overrides) -> FinanceAgentV2ResourcesServer:
    cfg_kwargs = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="finance_agent_v2_test",
        tavily_api_key="dummy-tavily",
        sec_api_key="dummy-sec",
        pricing_data_api_key="dummy-tiingo",
        retrieval_model_server=ModelServerRef(type="responses_api_models", name="policy"),
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        reward_mode="binary",
        **_prompt_fpaths(),
    )
    cfg_kwargs.update(overrides)
    config = FinanceAgentV2ResourcesServerConfig(**cfg_kwargs)
    return FinanceAgentV2ResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _mock_request(session_id: str = _TEST_SESSION_ID) -> MagicMock:
    req = MagicMock()
    req.session = {SESSION_ID_KEY: session_id}
    return req


# ---------------------------------------------------------------------------
# verify() helpers (mirror the v1 construction pattern)
# ---------------------------------------------------------------------------


def _msg(text: str) -> dict:
    return {
        "id": "msg_1",
        "content": [{"annotations": [], "text": text, "type": "output_text"}],
        "role": "assistant",
        "status": "completed",
        "type": "message",
    }


def _tool_call(name: str, arguments: str) -> dict:
    return {
        "id": "tc_1",
        "call_id": "call_1",
        "name": name,
        "arguments": arguments,
        "type": "function_call",
        "status": "completed",
    }


def _make_response(*output_items) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp_test",
        created_at=0.0,
        model="test",
        object="response",
        output=list(output_items),
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )


def _make_verify_request(response: NeMoGymResponse, expected_answer=None, rubric=None) -> FinanceAgentV2VerifyRequest:
    return FinanceAgentV2VerifyRequest(
        question="What was revenue?",
        expected_answer=expected_answer,
        rubric=rubric,
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "user", "content": "What was revenue?"}]
        ),
        response=response,
    )


def _judge_response_json(text: str) -> str:
    return NeMoGymResponse(
        id="judge_resp",
        created_at=0.0,
        model="judge",
        object="response",
        output=[
            {
                "id": "judge_msg",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    ).model_dump_json()


# ============================================================================
# Initialization / tool registration
# ============================================================================


class TestInitialization:
    def test_server_instantiates(self) -> None:
        assert _make_server() is not None

    def test_all_tools_available_with_keys(self) -> None:
        server = _make_server()
        for name in [
            "calculator", "parse_html_page", "submit_final_result",
            "web_search", "edgar_search", "price_history", "retrieve_information",
        ]:
            assert server._tools.get(name) is not None, f"{name} should be available"

    def test_tools_unavailable_without_keys(self) -> None:
        server = _make_server(
            tavily_api_key=None, sec_api_key=None, pricing_data_api_key=None, retrieval_model_server=None
        )
        assert server._tools["web_search"] is None
        assert server._tools["edgar_search"] is None
        assert server._tools["price_history"] is None
        assert server._tools["retrieve_information"] is None
        # No-key tools remain available.
        assert server._tools["calculator"] is not None
        assert server._tools["parse_html_page"] is not None
        assert server._tools["submit_final_result"] is not None


# ============================================================================
# Tool dispatch
# ============================================================================


class TestToolDispatch:
    @pytest.mark.asyncio
    async def test_calculator(self) -> None:
        server = _make_server()
        out = await server._dispatch_tool("calculator", _mock_request(), {"expression": "(5000 - 3200) * 0.21"})
        assert out["results"] == str((5000 - 3200) * 0.21)

    @pytest.mark.asyncio
    async def test_unavailable_tool_returns_error(self) -> None:
        server = _make_server(pricing_data_api_key=None)
        out = await server._dispatch_tool(
            "price_history", _mock_request(), {"ticker": "AAPL", "start_date": "2024-01-01", "end_date": "2024-02-01", "asset_class": "equity"}
        )
        assert "not available" in json.loads(out["results"])["error"]

    @pytest.mark.asyncio
    async def test_time_budget_exhausted(self) -> None:
        server = _make_server(max_rollout_time_seconds=0.001)
        server._session_start_times[_TEST_SESSION_ID] = time.monotonic() - 10
        out = await server._dispatch_tool("calculator", _mock_request(), {"expression": "1+1"})
        assert "Time budget exhausted" in json.loads(out["results"])["error"]

    @pytest.mark.asyncio
    async def test_tool_exception_surfaced_as_error(self) -> None:
        server = _make_server()
        # RetrieveInformation with a prompt lacking {{key}} -> upstream returns a
        # ToolOutput with an error string (not a 500).
        out = await server._dispatch_tool("retrieve_information", _mock_request(), {"prompt": "no placeholder here"})
        assert "ERROR" in out["results"]

    @pytest.mark.asyncio
    async def test_parse_html_then_retrieve_share_state(self) -> None:
        """parse_html_page writes to the per-session state; retrieve_information reads it."""
        server = _make_server()
        req = _mock_request()

        # Mock the upstream HTML fetch so no network is hit.
        server._tools["parse_html_page"]._parse_html_page = AsyncMock(return_value="NVIDIA 10-K: shares outstanding 24.3 billion")
        parse_out = await server._dispatch_tool(
            "parse_html_page", req, {"url": "https://example.com/10k", "key": "nvda_10k"}
        )
        assert "SUCCESS" in parse_out["results"]
        # State was populated under the session.
        assert server._get_session_storage(_TEST_SESSION_ID)["nvda_10k"].startswith("NVIDIA 10-K")

        # Mock the retrieval LLM round-trip; the tool must find the shared key.
        server._run_retrieval = AsyncMock(
            return_value=SimpleNamespace(output_text_str="24.3 billion shares", metadata={})
        )
        retrieve_out = await server._dispatch_tool(
            "retrieve_information", req, {"prompt": "How many shares? {{nvda_10k}}"}
        )
        assert retrieve_out["results"] == "24.3 billion shares"
        # The substituted prompt (with the document text) reached the LLM.
        sent_prompt = server._run_retrieval.call_args.args[0]
        assert "shares outstanding 24.3 billion" in sent_prompt

    @pytest.mark.asyncio
    async def test_retrieve_missing_key_errors(self) -> None:
        server = _make_server()
        out = await server._dispatch_tool(
            "retrieve_information", _mock_request(), {"prompt": "Use {{missing}} please"}
        )
        assert "not found in the data storage" in out["results"]


# ============================================================================
# Session lifecycle
# ============================================================================


class TestSession:
    @pytest.mark.asyncio
    async def test_seed_resets_state(self) -> None:
        server = _make_server()
        req = _mock_request()
        server._get_session_storage(_TEST_SESSION_ID)["stale"] = "old"
        await server.seed_session(req, MagicMock())
        assert server._get_session_storage(_TEST_SESSION_ID) == {}
        assert _TEST_SESSION_ID in server._session_start_times


# ============================================================================
# verify(): legacy [[N]] judge
# ============================================================================


class TestVerifyLegacyJudge:
    def _server(self, reward_mode="binary"):
        server = _make_server(reward_mode=reward_mode)
        post_mock = MagicMock()
        server._post_mock = post_mock  # keep a handle for assertions
        return server, post_mock

    @pytest.mark.asyncio
    async def test_binary_full_credit(self) -> None:
        server, post_mock = self._server("binary")
        post_mock.read = AsyncMock(return_value=_judge_response_json("Matches exactly. The rating is: [[2]]"))
        server.server_client.post = AsyncMock(return_value=post_mock)

        resp = _make_response(_tool_call("submit_final_result", json.dumps({"final_result": "$391.0 billion"})))
        res = await server.verify(_mock_request(), _make_verify_request(resp, expected_answer="$391.0 billion"))
        assert res.reward == 1.0
        assert res.judge_rating == 2

    @pytest.mark.asyncio
    async def test_binary_partial_is_zero(self) -> None:
        server, post_mock = self._server("binary")
        post_mock.read = AsyncMock(return_value=_judge_response_json("Partly right. [[1]]"))
        server.server_client.post = AsyncMock(return_value=post_mock)

        resp = _make_response(_tool_call("submit_final_result", json.dumps({"final_result": "$391 billion"})))
        res = await server.verify(_mock_request(), _make_verify_request(resp, expected_answer="$391.0 billion"))
        assert res.reward == 0.0

    @pytest.mark.asyncio
    async def test_scaled_partial_is_half(self) -> None:
        server, post_mock = self._server("scaled")
        post_mock.read = AsyncMock(return_value=_judge_response_json("Partly right. [[1]]"))
        server.server_client.post = AsyncMock(return_value=post_mock)

        resp = _make_response(_tool_call("submit_final_result", json.dumps({"final_result": "$391 billion"})))
        res = await server.verify(_mock_request(), _make_verify_request(resp, expected_answer="$391.0 billion"))
        assert res.reward == 0.5

    @pytest.mark.asyncio
    async def test_no_submit_is_zero(self) -> None:
        server, _ = self._server("binary")
        server.server_client.post = AsyncMock()
        resp = _make_response(_msg("I think it's $391 billion but I won't submit."))
        res = await server.verify(_mock_request(), _make_verify_request(resp, expected_answer="$391.0 billion"))
        assert res.reward == 0.0
        server.server_client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_judge_no_verdict_is_flagged_not_silent_zero(self, monkeypatch) -> None:
        """Judge emits no [[N]] (e.g. token budget exhausted): reward 0.0 but
        judge_error set and judge_rating None, so it is distinguishable from a
        real [[0]] and can be filtered downstream."""
        import resources_servers.finance_agent_v2.app as app_mod

        async def _no_sleep(*_a, **_k):
            return None

        monkeypatch.setattr(app_mod.asyncio, "sleep", _no_sleep)

        server, post_mock = self._server("binary")
        post_mock.read = AsyncMock(return_value=_judge_response_json(""))  # empty / no [[N]]
        server.server_client.post = AsyncMock(return_value=post_mock)

        resp = _make_response(_tool_call("submit_final_result", json.dumps({"final_result": "$391.0 billion"})))
        res = await server.verify(_mock_request(), _make_verify_request(resp, expected_answer="$391.0 billion"))
        assert res.reward == 0.0
        assert res.judge_rating is None
        assert res.judge_error is not None


# ============================================================================
# verify(): unlabeled dry-run
# ============================================================================


class TestVerifyDryRun:
    @pytest.mark.asyncio
    async def test_dry_run_no_judge_no_labels(self) -> None:
        """Unlabeled dry-run: no judge server, no expected_answer -> reward 0."""
        server = _make_server(judge_model_server=None)
        resp = _make_response(_tool_call("submit_final_result", json.dumps({"final_result": "anything"})))
        res = await server.verify(_mock_request(), _make_verify_request(resp))
        assert res.reward == 0.0

    @pytest.mark.asyncio
    async def test_rubric_field_is_ignored_for_reward(self) -> None:
        """The dataset's rubric field is reference-only and must not affect scoring."""
        server = _make_server(reward_mode="binary")
        post_mock = MagicMock()
        post_mock.read = AsyncMock(return_value=_judge_response_json("Matches. [[2]]"))
        server.server_client.post = AsyncMock(return_value=post_mock)

        rubric = json.dumps([{"operator": "finance_agent_v2_operator", "criteria": "Revenue = $391.0 billion"}])
        resp = _make_response(_tool_call("submit_final_result", json.dumps({"final_result": "$391.0 billion"})))
        res = await server.verify(
            _mock_request(), _make_verify_request(resp, expected_answer="$391.0 billion", rubric=rubric)
        )
        # Reward comes purely from the [[N]] judge, not the rubric.
        assert res.reward == 1.0
        assert res.judge_rating == 2
