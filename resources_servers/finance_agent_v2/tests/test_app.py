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

import asyncio
import json
import time
from collections import Counter
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
from resources_servers.finance_agent_v2.cached_tools import (
    CachedParseHtmlPage,
    CachedPriceHistory,
    SecFilingSearch,
)


_PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompt_templates"
_TEST_SESSION_ID = "test-session"


def _prompt_fpaths() -> dict:
    return {
        "rubric_judge_prompt_template_fpath": str(_PROMPT_DIR / "finance_agent_v2_rubric_judge.yaml"),
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
        # Default tests run without the disk cache; cache-specific tests opt in.
        use_cache=False,
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


# ---------------------------------------------------------------------------
# Rubric judge helpers
# ---------------------------------------------------------------------------

# Deliberately chosen so no criterion text is a substring of the answer: the
# rendered prompt contains both, and several assertions below check which
# criterion a given prompt is about.
_ANSWER = "Total revenue reached $391.0B for the year and the stock reacted favorably."
_C1 = "Revenue was $391.0 billion"
_C2 = "Market sentiment was positive"
_C3 = "Free cash flow was $60.9 billion"

# Where the template puts the criterion. The judge stub keys off it so it can
# tell which criterion a concurrent call is grading.
_CRITERION_MARKER = "RUBRIC CRITERION (grade only this):"


def _verdict(score: int, evidence: str = "the relevant span", reason: str = "matches") -> str:
    return json.dumps({"extracted_evidence": evidence, "score": score, "reason": reason})


def _rubric(*criteria: str) -> str:
    return json.dumps([{"operator": "finance_agent_v2_operator", "criteria": c} for c in criteria])


def _submitted_request(rubric) -> FinanceAgentV2VerifyRequest:
    resp = _make_response(_tool_call("submit_final_result", json.dumps({"final_result": _ANSWER})))
    return _make_verify_request(resp, rubric=rubric)


class _JudgeStub:
    """Scripted judge endpoint, keyed by criterion rather than by call order.

    Criteria are judged concurrently, so calls do not arrive in a fixed order and
    an index-keyed script would be flaky. ``script`` maps criterion text to the
    replies its successive calls get, where a reply is an int (turned into a
    well-formed verdict), a str (returned verbatim, for parse-failure cases), or
    an exception instance (raised, for API-failure cases).
    """

    def __init__(self, script: dict) -> None:
        self._script = {criterion: list(replies) for criterion, replies in script.items()}
        self.calls: Counter = Counter()
        self.prompts: list[str] = []
        self.max_in_flight = 0
        self._in_flight = 0

    async def post(self, *, server_name, url_path, json) -> MagicMock:  # noqa: A002
        params = json
        prompt = params.input[0].content
        self.prompts.append(prompt)

        criterion_section = prompt.rsplit(_CRITERION_MARKER, 1)[-1]
        criterion = next((c for c in self._script if c in criterion_section), None)
        assert criterion is not None, f"judge called with an unscripted criterion: {criterion_section[:200]!r}"
        self.calls[criterion] += 1

        self._in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            # Yield so concurrent criteria actually overlap. Without this every
            # call would finish before the next started and the semaphore's
            # effect would be invisible to max_in_flight.
            for _ in range(3):
                await asyncio.sleep(0)

            queue = self._script[criterion]
            reply = queue.pop(0) if queue else "ran out of scripted replies"
            if isinstance(reply, BaseException):
                raise reply
            if isinstance(reply, int):
                reply = _verdict(reply)
            response = MagicMock()
            response.read = AsyncMock(return_value=_judge_response_json(reply))
            return response
        finally:
            self._in_flight -= 1


def _rubric_server(script: dict, **overrides) -> tuple[FinanceAgentV2ResourcesServer, _JudgeStub]:
    server = _make_server(**overrides)
    stub = _JudgeStub(script)
    server.server_client.post = stub.post
    return server, stub


@pytest.fixture
def no_sleep(monkeypatch):
    """Drop the retry backoff so failure-path tests do not sleep for real.

    Patches ``asyncio.sleep`` itself, so tests that rely on ``sleep(0)`` to
    interleave concurrent work (the semaphore test) must not use this.
    """
    import resources_servers.finance_agent_v2.app as app_mod

    async def _sleep(*_args, **_kwargs):
        return None

    monkeypatch.setattr(app_mod.asyncio, "sleep", _sleep)


# ============================================================================
# Initialization / tool registration
# ============================================================================


class TestInitialization:
    def test_server_instantiates(self) -> None:
        assert _make_server() is not None

    def test_all_tools_available_with_keys(self) -> None:
        server = _make_server()
        for name in [
            "calculator",
            "parse_html_page",
            "submit_final_result",
            "web_search",
            "edgar_search",
            "price_history",
            "retrieve_information",
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
# Tool surface / caching wiring
# ============================================================================


class TestToolSurface:
    def test_edgar_search_default_secfiling_absent(self) -> None:
        """Default eval surface: edgar_search present, sec_filing_search not registered."""
        server = _make_server()
        assert server._tools.get("edgar_search") is not None
        assert "sec_filing_search" not in server._tools

    def test_sec_filing_search_when_enabled(self) -> None:
        server = _make_server(enabled_sec_tools=["edgar_search", "sec_filing_search"])
        assert isinstance(server._tools.get("sec_filing_search"), SecFilingSearch)

    def test_cache_disabled_by_default_uses_plain_tools(self) -> None:
        server = _make_server()
        assert server._cache.enabled is False
        # Plain upstream classes (not the Cached* subclasses).
        assert not isinstance(server._tools["parse_html_page"], CachedParseHtmlPage)
        assert not isinstance(server._tools["price_history"], CachedPriceHistory)

    def test_cache_enabled_swaps_in_cached_tools(self, tmp_path) -> None:
        server = _make_server(use_cache=True, cache_dir=str(tmp_path))
        assert server._cache.enabled is True
        assert isinstance(server._tools["parse_html_page"], CachedParseHtmlPage)
        assert isinstance(server._tools["price_history"], CachedPriceHistory)


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
            "price_history",
            _mock_request(),
            {"ticker": "AAPL", "start_date": "2024-01-01", "end_date": "2024-02-01", "asset_class": "equity"},
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
        server._tools["parse_html_page"]._parse_html_page = AsyncMock(
            return_value="NVIDIA 10-K: shares outstanding 24.3 billion"
        )
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
# verify(): per-criterion rubric judge
# ============================================================================


class TestRubricParsing:
    @pytest.mark.parametrize(
        "rubric",
        [
            None,
            "",
            "not json",
            json.dumps({"criteria": "a dict, not a list"}),
            json.dumps([]),
            json.dumps([{"operator": "op"}]),  # no criteria key
            json.dumps([{"criteria": "   "}]),  # blank criteria
        ],
    )
    def test_unusable_rubrics_yield_no_criteria(self, rubric) -> None:
        assert FinanceAgentV2ResourcesServer._parse_rubric(rubric) == []

    def test_parses_criteria_and_operator(self) -> None:
        rubric = json.dumps(
            [
                {"operator": "finance_agent_v2_operator", "criteria": " Revenue was $391.0 billion "},
                {"criteria": "Sentiment was positive"},
            ]
        )
        assert FinanceAgentV2ResourcesServer._parse_rubric(rubric) == [
            {"criteria": "Revenue was $391.0 billion", "operator": "finance_agent_v2_operator"},
            {"criteria": "Sentiment was positive", "operator": None},
        ]


class TestScoreExtraction:
    @pytest.mark.parametrize("score", [0, 1])
    def test_bare_json(self, score) -> None:
        assert FinanceAgentV2ResourcesServer._extract_score(_verdict(score)) == score

    def test_fenced_json(self) -> None:
        assert FinanceAgentV2ResourcesServer._extract_score(f"Here:\n```json\n{_verdict(1)}\n```") == 1

    def test_last_object_wins(self) -> None:
        """A worked example quoted while reasoning must not outrank the verdict."""
        text = f"I recall the format {_verdict(0)} but for this answer: {_verdict(1)}"
        assert FinanceAgentV2ResourcesServer._extract_score(text) == 1

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "   ",
            "The criterion is satisfied.",  # no JSON at all
            '{"score": "pass"}',  # strings are not verdicts
            '{"score": 1.0}',  # nor floats
            '{"score": true}',  # nor bools
            '{"score": 2}',  # nor out-of-range ints
            '{"reason": "no score key"}',
            "{not valid json}",
        ],
    )
    def test_unusable_replies(self, text) -> None:
        assert FinanceAgentV2ResourcesServer._extract_score(text) is None


class TestVerifyRubricJudge:
    @pytest.mark.asyncio
    async def test_all_criteria_pass(self) -> None:
        server, stub = _rubric_server({_C1: [1, 1, 1], _C2: [1, 1, 1]})
        res = await server.verify(_mock_request(), _submitted_request(_rubric(_C1, _C2)))

        assert res.reward == 1.0
        assert (res.rubric_total, res.rubric_passed, res.rubric_unresolved) == (2, 2, 0)
        assert res.rubric_fraction == 1.0
        assert res.rubric_all_pass is True
        assert res.judge_error is None
        assert stub.calls[_C1] == 3
        assert all(j.unanimous for j in res.rubric_judgements)

    @pytest.mark.asyncio
    async def test_one_criterion_fails_zeroes_reward_but_keeps_partial_credit(self) -> None:
        server, _ = _rubric_server({_C1: [1, 1, 1], _C2: [0, 0, 0], _C3: [1, 1, 1]})
        res = await server.verify(_mock_request(), _submitted_request(_rubric(_C1, _C2, _C3)))

        assert res.reward == 0.0
        assert res.rubric_passed == 2
        assert res.rubric_fraction == pytest.approx(2 / 3)
        assert res.rubric_all_pass is False
        # Partial credit is the only thing separating this from a total miss, so
        # the reward alone must not be the whole record.
        assert res.judge_error is None
        by_criterion = {j.criteria: j for j in res.rubric_judgements}
        assert by_criterion[_C2].score == 0
        assert by_criterion[_C2].votes == [0, 0, 0]

    @pytest.mark.asyncio
    async def test_majority_vote_resolves_and_flags_disagreement(self) -> None:
        server, _ = _rubric_server({_C1: [1, 1, 0], _C2: [0, 0, 1]})
        res = await server.verify(_mock_request(), _submitted_request(_rubric(_C1, _C2)))

        by_criterion = {j.criteria: j for j in res.rubric_judgements}
        assert by_criterion[_C1].score == 1
        assert (by_criterion[_C1].votes_for, by_criterion[_C1].votes_against) == (2, 1)
        assert by_criterion[_C1].unanimous is False
        assert by_criterion[_C2].score == 0
        assert (by_criterion[_C2].votes_for, by_criterion[_C2].votes_against) == (1, 2)
        assert by_criterion[_C2].unanimous is False
        # A 2-1 pass still counts as a pass.
        assert res.rubric_passed == 1

    @pytest.mark.asyncio
    async def test_stops_at_required_successes(self) -> None:
        """The attempt cap is a ceiling, not a target: 3 clean verdicts end it."""
        server, stub = _rubric_server({_C1: [1] * 10})
        res = await server.verify(_mock_request(), _submitted_request(_rubric(_C1)))

        assert stub.calls[_C1] == 3
        assert res.rubric_judgements[0].attempts_used == 3

    @pytest.mark.asyncio
    async def test_unparseable_replies_are_retried(self, no_sleep) -> None:
        server, stub = _rubric_server({_C1: ["I think it passes.", '{"score": "pass"}', 1, 1, 1]})
        res = await server.verify(_mock_request(), _submitted_request(_rubric(_C1)))

        judgement = res.rubric_judgements[0]
        assert judgement.score == 1
        assert judgement.parse_failures == 2
        assert judgement.api_failures == 0
        assert judgement.attempts_used == 5
        assert stub.calls[_C1] == 5
        assert res.reward == 1.0

    @pytest.mark.asyncio
    async def test_api_errors_are_retried(self, no_sleep) -> None:
        server, _ = _rubric_server({_C1: [RuntimeError("connection reset"), TimeoutError(), 0, 0, 0]})
        res = await server.verify(_mock_request(), _submitted_request(_rubric(_C1)))

        judgement = res.rubric_judgements[0]
        assert judgement.score == 0
        assert judgement.api_failures == 2
        assert judgement.parse_failures == 0
        assert res.reward == 0.0
        assert res.judge_error is None  # resolved, just not satisfied

    @pytest.mark.asyncio
    async def test_empty_reply_counts_as_api_failure(self, no_sleep) -> None:
        """No visible text usually means the token budget went to hidden reasoning,
        which is an infrastructure failure rather than a bad verdict."""
        server, _ = _rubric_server({_C1: ["", 1, 1, 1]})
        res = await server.verify(_mock_request(), _submitted_request(_rubric(_C1)))

        assert res.rubric_judgements[0].api_failures == 1
        assert res.rubric_judgements[0].parse_failures == 0

    @pytest.mark.asyncio
    async def test_exhausted_attempts_leave_criterion_unresolved(self, no_sleep) -> None:
        """Never reaching 3 verdicts is a judge failure, not a miss: score stays
        null and judge_error is set so the row can be filtered out rather than
        read as a genuine zero."""
        server, stub = _rubric_server({_C1: [1, 1] + ["garbage"] * 20, _C2: [1, 1, 1]})
        res = await server.verify(_mock_request(), _submitted_request(_rubric(_C1, _C2)))

        assert stub.calls[_C1] == 10  # judge_max_attempts, then gives up
        by_criterion = {j.criteria: j for j in res.rubric_judgements}
        assert by_criterion[_C1].score is None
        assert by_criterion[_C1].votes == [1, 1]  # kept for debugging, not voted on
        assert by_criterion[_C1].unanimous is None
        assert by_criterion[_C1].error is not None
        assert res.rubric_unresolved == 1
        assert res.rubric_all_pass is False
        assert res.reward == 0.0
        assert res.judge_error is not None and "unresolved" in res.judge_error

    @pytest.mark.asyncio
    async def test_evidence_and_reason_recorded_per_vote(self) -> None:
        server, _ = _rubric_server({_C1: [_verdict(1, evidence="revenue of $391.0B", reason="exact match"), 1, 1]})
        res = await server.verify(_mock_request(), _submitted_request(_rubric(_C1)))

        judgement = res.rubric_judgements[0]
        assert judgement.evidence[0] == "revenue of $391.0B"
        assert judgement.reasons[0] == "exact match"
        assert len(judgement.evidence) == len(judgement.votes) == 3

    @pytest.mark.asyncio
    async def test_prompt_carries_question_answer_and_single_criterion(self) -> None:
        server, stub = _rubric_server({_C1: [1, 1, 1], _C2: [1, 1, 1]})
        await server.verify(_mock_request(), _submitted_request(_rubric(_C1, _C2)))

        prompts_for_c1 = [p for p in stub.prompts if _C1 in p]
        assert prompts_for_c1
        prompt = prompts_for_c1[0]
        assert "What was revenue?" in prompt
        assert _ANSWER in prompt
        # One criterion per call: the judge never sees the rest of the rubric.
        assert _C2 not in prompt
        # The guardrail that keeps the question from becoming extra requirements.
        assert "Use the question only for scope/disambiguation" in prompt
        # Placeholders were all substituted.
        for placeholder in ("{question}", "{generated_answer}", "{criterion}"):
            assert placeholder not in prompt

    @pytest.mark.asyncio
    async def test_no_submission_is_zero_without_judge_spend(self) -> None:
        server, stub = _rubric_server({_C1: [1, 1, 1]})
        request = _make_verify_request(
            _make_response(_msg("It's $391 billion but I won't submit.")), rubric=_rubric(_C1)
        )
        res = await server.verify(_mock_request(), request)

        assert res.reward == 0.0
        assert res.judge_error is None  # a real failure, not a judge problem
        assert res.rubric_judgements is None
        assert sum(stub.calls.values()) == 0

    @pytest.mark.asyncio
    async def test_missing_rubric_is_flagged_as_unscorable(self) -> None:
        server, stub = _rubric_server({})
        res = await server.verify(_mock_request(), _submitted_request(None))

        assert res.reward == 0.0
        assert res.judge_error is not None
        assert sum(stub.calls.values()) == 0

    @pytest.mark.asyncio
    async def test_no_judge_server_is_flagged_as_unscorable(self) -> None:
        server = _make_server(judge_model_server=None)
        res = await server.verify(_mock_request(), _submitted_request(_rubric(_C1)))

        assert res.reward == 0.0
        assert res.rubric_total == 1
        assert res.judge_error is not None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("concurrency", [1, 2, 4])
    async def test_concurrency_semaphore_is_respected(self, concurrency) -> None:
        criteria = [_C1, _C2, _C3, "Margin expanded"]
        server, stub = _rubric_server({c: [1, 1, 1] for c in criteria}, judge_max_concurrency=concurrency)
        res = await server.verify(_mock_request(), _submitted_request(_rubric(*criteria)))

        assert res.reward == 1.0
        assert stub.max_in_flight == concurrency


# ============================================================================
# Aggregate metrics
# ============================================================================


class TestAggregateMetrics:
    def _rollout(self, **overrides) -> dict:
        rollout = {
            "reward": 1.0,
            "rubric_total": 2,
            "rubric_passed": 2,
            "rubric_unresolved": 0,
            "rubric_fraction": 1.0,
            "rubric_all_pass": True,
            "judge_error": None,
            "rubric_judgements": [
                {"criteria": _C1, "score": 1, "unanimous": True, "attempts_used": 3},
                {"criteria": _C2, "score": 1, "unanimous": True, "attempts_used": 3},
            ],
        }
        rollout.update(overrides)
        return rollout

    def test_empty_input(self) -> None:
        assert _make_server().compute_metrics([]) == {}

    def test_pools_criteria_across_rollouts(self) -> None:
        partial = self._rollout(
            reward=0.0,
            rubric_passed=1,
            rubric_fraction=0.5,
            rubric_all_pass=False,
            rubric_judgements=[
                {"criteria": _C1, "score": 1, "unanimous": False, "attempts_used": 3, "parse_failures": 1},
                {"criteria": _C2, "score": 0, "unanimous": True, "attempts_used": 4, "api_failures": 1},
            ],
        )
        metrics = _make_server().compute_metrics([[self._rollout()], [partial]])

        assert metrics["mean/rubric_fraction"] == 0.75
        assert metrics["mean/rubric_all_pass"] == 0.5
        # Denominator is criteria, not questions.
        assert metrics["rubric/criteria_total"] == 4
        assert metrics["mean/criterion_pass_rate"] == 0.75
        assert metrics["mean/judge_disagreement_rate"] == 0.25
        assert metrics["rubric/parse_failures"] == 1
        assert metrics["rubric/api_failures"] == 1
        assert metrics["rubric/mean_attempts_per_criterion"] == pytest.approx(3.25)

    def test_unresolved_criteria_excluded_from_rates_but_counted(self) -> None:
        unresolved = self._rollout(
            reward=0.0,
            rubric_passed=1,
            rubric_unresolved=1,
            rubric_fraction=0.5,
            rubric_all_pass=False,
            judge_error="1/2 criteria unresolved",
            rubric_judgements=[
                {"criteria": _C1, "score": 1, "unanimous": True, "attempts_used": 3},
                {"criteria": _C2, "score": None, "unanimous": None, "attempts_used": 10},
            ],
        )
        metrics = _make_server().compute_metrics([[unresolved]])

        assert metrics["rubric/criteria_unresolved"] == 1
        assert metrics["rubric/criteria_resolved"] == 1
        assert metrics["mean/criterion_pass_rate"] == 1.0
        assert metrics["rubric/rollouts_with_judge_error"] == 1

    def test_no_submission_rollout_counted_separately(self) -> None:
        no_submit = {"reward": 0.0, "rubric_judgements": None, "judge_error": None}
        metrics = _make_server().compute_metrics([[no_submit]])

        assert metrics["rubric/rollouts_without_submission"] == 1
        assert metrics["rubric/criteria_total"] == 0
        assert "mean/criterion_pass_rate" not in metrics

    def test_key_metrics_promote_judge_health(self) -> None:
        agent_metrics = {
            "mean/reward": 0.5,
            "mean/rubric_fraction": 0.75,
            "std/reward": 0.1,
            "rubric/criteria_unresolved": 2,
            "rubric/rollouts_with_judge_error": 1,
            "rubric/rollouts_without_submission": 0,
            "rubric/parse_failures": 3,
        }
        key = _make_server().get_key_metrics(agent_metrics)

        assert key["mean/reward"] == 0.5
        assert key["mean/rubric_fraction"] == 0.75
        assert "std/reward" not in key
        # A run whose judge failed must not read as a low score.
        assert key["rubric/criteria_unresolved"] == 2
        assert key["rubric/rollouts_with_judge_error"] == 1
        assert "rubric/parse_failures" not in key
