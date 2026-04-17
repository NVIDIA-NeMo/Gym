# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from app import (
    LogScaleCQLResourcesServer,
    LogScaleCQLResourcesServerConfig,
    LogScaleCQLVerifyRequest,
    extract_cql_from_response,
    _parse_semantic_judge,
    _parse_execution_judge,
    compute_reward,
)
from container_engine import LogScaleContainerEngine


_RESPONSE_DEFAULTS = dict(
    id="resp-test",
    created_at=0,
    model="test-model",
    object="response",
    parallel_tool_calls=False,
    tool_choice="auto",
    tools=[],
)


def _make_response(text: str) -> NeMoGymResponse:
    out = NeMoGymResponseOutputMessage(
        id="msg-1",
        type="message",
        content=[NeMoGymResponseOutputText(annotations=[], text=text)],
    )
    return NeMoGymResponse(output=[out], **_RESPONSE_DEFAULTS)


def _make_empty_response() -> NeMoGymResponse:
    return NeMoGymResponse(output=[], **_RESPONSE_DEFAULTS)


def _make_response_no_content() -> NeMoGymResponse:
    out = NeMoGymResponseOutputMessage(id="msg-1", type="message", content=[])
    return NeMoGymResponse(output=[out], **_RESPONSE_DEFAULTS)


# ── CQL extraction ────────────────────────────────────────────────────

class TestExtractCqlFromResponse:
    def test_extract_from_message_output_text(self):
        response = _make_response("#event_simpleName=ProcessRollup2\n| groupBy(ImageFileName, function=count())")
        result = extract_cql_from_response(response)
        assert "ProcessRollup2" in result
        assert "groupBy" in result

    def test_extract_strips_markdown_fence(self):
        response = _make_response("```\n#event_simpleName=DnsRequest\n| head(10)\n```")
        result = extract_cql_from_response(response)
        assert "DnsRequest" in result
        assert "head" in result
        assert result.strip() == "#event_simpleName=DnsRequest\n| head(10)"

    def test_empty_response(self):
        response = _make_empty_response()
        result = extract_cql_from_response(response)
        assert result == ""

    def test_no_message_content(self):
        response = _make_response_no_content()
        result = extract_cql_from_response(response)
        assert result == ""


# ── Judge parsers ─────────────────────────────────────────────────────

class TestParseSemanticJudge:
    def test_parse_full(self):
        raw = "SEMANTIC_SCORE: 4\nSEMANTIC_REASONING: Query captures intent well."
        r = _parse_semantic_judge(raw)
        assert r["semantic_score"] == 4
        assert "intent" in r["semantic_reasoning"]

    def test_parse_missing_score(self):
        raw = "SEMANTIC_REASONING: Something happened."
        r = _parse_semantic_judge(raw)
        assert r["semantic_score"] is None


class TestParseExecutionJudge:
    def test_parse_full(self):
        raw = "EXECUTION_SCORE: 5\nEXECUTION_REASONING: Results are perfect."
        r = _parse_execution_judge(raw)
        assert r["execution_score"] == 5
        assert "perfect" in r["execution_reasoning"]

    def test_parse_missing_score(self):
        raw = "EXECUTION_REASONING: Empty results."
        r = _parse_execution_judge(raw)
        assert r["execution_score"] is None


# ── Reward computation ────────────────────────────────────────────────

class TestComputeReward:
    def test_perfect_scores(self):
        assert compute_reward(1, 5, 5) == pytest.approx(1.0)

    def test_all_zeros(self):
        assert compute_reward(0, None, None) == pytest.approx(0.0)

    def test_valid_but_no_judge(self):
        # validity=1, no semantic, no execution
        assert compute_reward(1, None, None) == pytest.approx(1 / 3)

    def test_mixed(self):
        # validity=1, semantic=3/5, execution=4/5
        expected = (1.0 + 3 / 5 + 4 / 5) / 3
        assert compute_reward(1, 3, 4) == pytest.approx(expected)

    def test_invalid_query(self):
        # validity=0, semantic=2, execution=1
        expected = (0.0 + 2 / 5 + 1 / 5) / 3
        assert compute_reward(0, 2, 1) == pytest.approx(expected)


# ── Container engine (unit tests for static helpers) ──────────────────

class TestContainerEngineFormatPreview:
    def test_empty_events(self):
        assert LogScaleContainerEngine._format_preview([]) == "(no results)"

    def test_skips_internal_fields(self):
        events = [{"ComputerName": "SRV-01", "@rawstring": "raw", "#repo": "sandbox", "aid": "abc"}]
        preview = LogScaleContainerEngine._format_preview(events)
        assert "SRV-01" in preview
        assert "@rawstring" not in preview
        assert "#repo" not in preview
        assert "aid" in preview

    def test_limits_rows(self):
        events = [{"x": str(i)} for i in range(100)]
        preview = LogScaleContainerEngine._format_preview(events, max_rows=3)
        lines = [l for l in preview.strip().split("\n") if l.strip()]
        assert len(lines) == 4  # header + 3 data rows


# ── Verify function ──────────────────────────────────────────────────


_RCP_DEFAULTS = dict(input=[{"role": "user", "content": "test"}])


def _make_verify_request(question: str, cql_text: str) -> LogScaleCQLVerifyRequest:
    response = _make_response(cql_text)
    return LogScaleCQLVerifyRequest(
        question=question,
        responses_create_params=_RCP_DEFAULTS,
        response=response,
    )


class _FakeServer:
    """Lightweight stand-in for LogScaleCQLResourcesServer that avoids Pydantic."""

    def __init__(self, validate_result=None, execute_result=None):
        self.config = MagicMock()
        self.config.logscale_url = "http://localhost:8080"
        self.config.repository = None
        self.config.judge_model_server = None
        self.config.judge_responses_create_params = None

        self._container_engine = MagicMock()
        self._container_engine.validate_query.return_value = validate_result or {
            "is_valid": True, "diagnostics": [],
        }
        self._container_engine.execute.return_value = execute_result or {
            "success": True, "n_rows": 1, "preview": "_count\n  42",
        }
        self._local_engine = None
        self._judge_api_key = ""
        self._judge_api_url = ""
        self._judge_model = ""
        self.server_client = None

    def _validate(self, cql):
        return self._container_engine.validate_query(cql)

    def _execute(self, cql):
        return self._container_engine.execute(cql)

    def _direct_llm_call(self, prompt):
        raise NotImplementedError("should be mocked")

    verify = LogScaleCQLResourcesServer.verify


class TestVerify:
    @pytest.mark.asyncio
    async def test_empty_question_returns_zero_reward(self):
        server = _FakeServer()
        body = _make_verify_request(question="", cql_text="count()")
        result = await server.verify(body)
        assert result.reward == 0.0

    @pytest.mark.asyncio
    async def test_empty_cql_returns_zero_reward(self):
        server = _FakeServer()
        body = _make_verify_request(question="How many events?", cql_text="")
        result = await server.verify(body)
        assert result.reward == 0.0
        assert result.validity_score == 0

    @pytest.mark.asyncio
    async def test_valid_query_no_judge(self):
        server = _FakeServer(
            validate_result={"is_valid": True, "diagnostics": []},
            execute_result={"success": True, "n_rows": 1, "preview": "_count\n  5000"},
        )
        body = _make_verify_request("How many events?", "count()")
        result = await server.verify(body)

        assert result.validity_score == 1
        assert result.extracted_cql == "count()"
        assert result.exec_success is True
        assert result.execution_score == 5
        assert result.semantic_score is None
        assert result.reward == pytest.approx(compute_reward(1, None, 5))

    @pytest.mark.asyncio
    async def test_invalid_query_no_judge(self):
        server = _FakeServer(
            validate_result={
                "is_valid": False,
                "diagnostics": [{"message": "syntax error", "severity": "error"}],
            },
            execute_result={"success": False, "n_rows": 0, "preview": "", "error": "parse error"},
        )
        body = _make_verify_request("Show events", "broken ||| syntax")
        result = await server.verify(body)

        assert result.validity_score == 0
        assert result.exec_success is False
        assert result.execution_score == 1
        assert result.reward == pytest.approx(compute_reward(0, None, 1))

    @pytest.mark.asyncio
    async def test_valid_query_with_direct_judge(self):
        server = _FakeServer(
            validate_result={"is_valid": True, "diagnostics": []},
            execute_result={"success": True, "n_rows": 5, "preview": "col1\n  a\n  b"},
        )
        server._judge_api_key = "fake-key"
        server._judge_api_url = "http://fake"
        server._judge_model = "fake-model"

        sem_response = "SEMANTIC_SCORE: 4\nSEMANTIC_REASONING: Captures intent well."
        exe_response = "EXECUTION_SCORE: 5\nEXECUTION_REASONING: Results are correct."

        with patch.object(server, "_direct_llm_call", side_effect=[sem_response, exe_response]):
            body = _make_verify_request("Show all events", "head(10)")
            result = await server.verify(body)

        assert result.validity_score == 1
        assert result.semantic_score == 4
        assert result.execution_score == 5
        assert result.reward == pytest.approx(compute_reward(1, 4, 5))

    @pytest.mark.asyncio
    async def test_judge_error_falls_back_gracefully(self):
        server = _FakeServer(
            validate_result={"is_valid": True, "diagnostics": []},
            execute_result={"success": True, "n_rows": 1, "preview": "_count\n  1"},
        )
        server._judge_api_key = "fake-key"
        server._judge_api_url = "http://fake"
        server._judge_model = "fake-model"

        with patch.object(server, "_direct_llm_call", side_effect=Exception("API timeout")):
            body = _make_verify_request("Count events", "count()")
            result = await server.verify(body)

        assert result.validity_score == 1
        assert result.semantic_score is None
        assert "error" in result.semantic_reasoning.lower()
        assert result.execution_score is None
        assert "error" in result.execution_reasoning.lower()
