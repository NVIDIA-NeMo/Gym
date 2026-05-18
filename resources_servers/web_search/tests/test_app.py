# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.web_search.app import (
    SearchRequest,
    WebSearchResourcesServer,
    WebSearchResourcesServerConfig,
    WebSearchVerifyRequest,
    _SEARCH_TYPE_ENDPOINTS,
    extract_boxed_answer,
    parse_judge_score,
    strip_thinking,
    verify_mcq,
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _msg(text: str, *, msg_id: str = "msg") -> NeMoGymResponseOutputMessage:
    return NeMoGymResponseOutputMessage(
        id=msg_id,
        content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
        role="assistant",
        status="completed",
        type="message",
    )


def _make_response(text: str, *, response_id: str = "resp") -> NeMoGymResponse:
    return NeMoGymResponse(
        id=response_id,
        created_at=0.0,
        model="dummy",
        object="response",
        output=[_msg(text)],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def _make_request(
    response_text: str,
    ground_truth,
    *,
    question: str = "What is something?",
    request_id: int = 1,
) -> WebSearchVerifyRequest:
    return WebSearchVerifyRequest(
        responses_create_params={"input": [{"role": "user", "content": question}]},
        response=_make_response(response_text, response_id=f"resp_test_{request_id}"),
        ground_truth=ground_truth,
        question=question,
    )


def _judge_response_json(text: str, *, response_id: str = "judge_resp") -> str:
    return NeMoGymResponse(
        id=response_id,
        created_at=0.0,
        model="judge",
        object="response",
        output=[_msg(text, msg_id="judge_msg")],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    ).model_dump_json()


def _wire_judge(server_client: MagicMock, judge_text: str) -> None:
    post_mock = MagicMock()
    post_mock.read = AsyncMock(return_value=_judge_response_json(judge_text))
    server_client.post = AsyncMock(return_value=post_mock)


def _make_config_no_judge() -> WebSearchResourcesServerConfig:
    """Config without a wired-up judge — natural_text falls back to substring."""
    return WebSearchResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        serper_api_key="test_key",
    )


def _make_config_with_judge() -> WebSearchResourcesServerConfig:
    template = str(
        Path(__file__).resolve().parents[1] / "prompt_templates/web_search_judge.txt"
    )
    return WebSearchResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        serper_api_key="test_key",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        judge_prompt_template_fpath=template,
    )


def _make_server(server_client=None, *, with_judge: bool = False) -> WebSearchResourcesServer:
    if server_client is None:
        server_client = MagicMock(spec=ServerClient)
    config = _make_config_with_judge() if with_judge else _make_config_no_judge()
    return WebSearchResourcesServer(config=config, server_client=server_client)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_strip_thinking_complete(self) -> None:
        assert strip_thinking(
            "<think>reasoning here</think>The answer is \\boxed{C}"
        ) == "The answer is \\boxed{C}"

    def test_strip_thinking_incomplete(self) -> None:
        assert strip_thinking("<think>reasoning without close") == ""

    def test_strip_thinking_no_tags(self) -> None:
        assert strip_thinking("The answer is \\boxed{A}") == "The answer is \\boxed{A}"

    def test_extract_boxed_answer(self) -> None:
        assert extract_boxed_answer("The answer is \\boxed{C}") == "C"
        assert extract_boxed_answer("No boxed here") is None

    def test_verify_mcq_correct(self) -> None:
        assert verify_mcq("The answer is \\boxed{C}", "C") == 1.0
        assert verify_mcq("The answer is \\boxed{c}", "C") == 1.0

    def test_verify_mcq_incorrect(self) -> None:
        assert verify_mcq("The answer is \\boxed{A}", "C") == 0.0

    def test_verify_mcq_fallback_pattern(self) -> None:
        assert verify_mcq("Answer: B", "B") == 1.0
        assert verify_mcq("Answer is (D)", "D") == 1.0

    def test_verify_mcq_with_thinking(self) -> None:
        assert verify_mcq("<think>let me think</think>The answer is \\boxed{B}", "B") == 1.0

    def test_parse_judge_score_one(self) -> None:
        assert parse_judge_score('{"score": 1}') == 1
        assert parse_judge_score('Some prose then {"score":1}.') == 1

    def test_parse_judge_score_zero(self) -> None:
        assert parse_judge_score('{"score": 0}') == 0

    def test_parse_judge_score_after_think(self) -> None:
        assert parse_judge_score('<think>weighing</think>{"score": 1}') == 1

    def test_parse_judge_score_unparseable(self) -> None:
        assert parse_judge_score("no json here") is None

    def test_search_type_default_is_general(self) -> None:
        # Locks in the RLVR-equivalent default; older versions used "search".
        assert SearchRequest(query="x").search_type == "general"

    def test_search_type_endpoint_map(self) -> None:
        # Faithful to RLVR's WebSearchEnvConfig-mentioned types.
        assert _SEARCH_TYPE_ENDPOINTS["general"] == "/search"
        assert _SEARCH_TYPE_ENDPOINTS["news"] == "/news"
        assert _SEARCH_TYPE_ENDPOINTS["scholar"] == "/scholar"
        # Inline-answer types fall through to /search.
        assert _SEARCH_TYPE_ENDPOINTS["weather"] == "/search"
        assert _SEARCH_TYPE_ENDPOINTS["sports"] == "/search"
        assert _SEARCH_TYPE_ENDPOINTS["stock"] == "/search"


# ---------------------------------------------------------------------------
# /verify, mcq path
# ---------------------------------------------------------------------------


class TestVerifyMcq:
    def test_correct_letter_yields_one(self) -> None:
        req = _make_request(
            "After thinking, \\boxed{C}",
            {"style": "mcq", "value": "C"},
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0
        assert result.ground_truth_style == "mcq"
        assert result.predicted_answer == "C"
        assert result.verification_failed is False
        assert result.judge_evaluation is None

    def test_wrong_letter_yields_zero(self) -> None:
        req = _make_request(
            "\\boxed{A}",
            {"style": "mcq", "value": "C"},
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 0.0
        assert result.predicted_answer == "A"

    def test_ground_truth_as_json_string(self) -> None:
        req = _make_request(
            "\\boxed{C}",
            json.dumps({"style": "mcq", "value": "C"}),
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0

    def test_unknown_style_marks_failed(self) -> None:
        req = _make_request("anything", {"style": "definitely_not_real", "value": "x"})
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 0.0
        assert result.verification_failed is True


# ---------------------------------------------------------------------------
# /verify, natural_text fallback (no judge configured)
# ---------------------------------------------------------------------------


class TestVerifyNaturalTextFallback:
    def test_substring_match_when_no_judge(self) -> None:
        req = _make_request(
            "The capital of Australia is Canberra.",
            {"style": "natural_text", "value": "Canberra"},
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 1.0
        assert result.judge_evaluation is None

    def test_substring_miss_when_no_judge(self) -> None:
        req = _make_request(
            "The capital is Sydney.",
            {"style": "natural_text", "value": "Canberra"},
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 0.0
        assert result.verification_failed is False

    def test_unclosed_think_returns_zero_no_failure(self) -> None:
        req = _make_request(
            "<think>still thinking",
            {"style": "natural_text", "value": "Canberra"},
        )
        result = asyncio.run(_make_server().verify(req))
        assert result.reward == 0.0
        assert result.verification_failed is False


# ---------------------------------------------------------------------------
# /verify, natural_text with LLM judge
# ---------------------------------------------------------------------------


class TestVerifyNaturalTextWithJudge:
    def test_judge_score_one(self) -> None:
        server_client = MagicMock(spec=ServerClient)
        _wire_judge(server_client, '{"score": 1}')
        rs = _make_server(server_client, with_judge=True)

        req = _make_request(
            "Ousmane Dembélé won the 2025 Ballon d'Or.",
            {"style": "natural_text", "value": "Ousmane Dembele"},
            question="Who won the men's Ballon d'Or in 2025?",
        )
        result = asyncio.run(rs.verify(req))
        assert result.reward == 1.0
        assert result.ground_truth_style == "natural_text"
        assert result.verification_failed is False
        assert result.judge_evaluation is not None
        assert result.judge_evaluation.response is not None
        assert server_client.post.call_count == 1

    def test_judge_score_zero(self) -> None:
        server_client = MagicMock(spec=ServerClient)
        _wire_judge(server_client, '{"score": 0}')
        rs = _make_server(server_client, with_judge=True)

        req = _make_request(
            "I am not sure who won.",
            {"style": "natural_text", "value": "Ousmane Dembele"},
            question="Who won the men's Ballon d'Or in 2025?",
        )
        result = asyncio.run(rs.verify(req))
        assert result.reward == 0.0
        assert result.verification_failed is False

    def test_judge_unparseable_marks_failed(self) -> None:
        server_client = MagicMock(spec=ServerClient)
        _wire_judge(server_client, "no json at all here")
        rs = _make_server(server_client, with_judge=True)

        req = _make_request(
            "Some answer.",
            {"style": "natural_text", "value": "Carlos Alcaraz"},
        )
        result = asyncio.run(rs.verify(req))
        assert result.reward == 0.0
        assert result.verification_failed is True
        assert result.judge_evaluation is not None
        assert result.judge_evaluation.response is not None  # call returned, just unparseable

    def test_judge_http_failure_marks_failed(self) -> None:
        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock(side_effect=RuntimeError("boom"))
        rs = _make_server(server_client, with_judge=True)

        req = _make_request(
            "Some answer.",
            {"style": "natural_text", "value": "Carlos Alcaraz"},
        )
        result = asyncio.run(rs.verify(req))
        assert result.reward == 0.0
        assert result.verification_failed is True
        assert result.judge_evaluation is not None
        assert result.judge_evaluation.response is None

    def test_unclosed_think_skips_judge_call(self) -> None:
        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock()
        rs = _make_server(server_client, with_judge=True)

        req = _make_request(
            "<think>incomplete reasoning",
            {"style": "natural_text", "value": "X"},
        )
        result = asyncio.run(rs.verify(req))
        assert result.reward == 0.0
        assert result.verification_failed is False
        server_client.post.assert_not_called()

    def test_judge_score_with_think_tag_in_judge_response(self) -> None:
        server_client = MagicMock(spec=ServerClient)
        _wire_judge(server_client, '<think>let me grade</think>{"score": 1}')
        rs = _make_server(server_client, with_judge=True)

        req = _make_request(
            "Carlos Alcaraz won the Australian Open 2026.",
            {"style": "natural_text", "value": "Carlos Alcaraz"},
        )
        result = asyncio.run(rs.verify(req))
        assert result.reward == 1.0

    def test_response_propagates_ground_truth_and_question(self) -> None:
        # Regression: rollout JSONL should carry ground_truth + question through.
        server_client = MagicMock(spec=ServerClient)
        _wire_judge(server_client, '{"score": 1}')
        rs = _make_server(server_client, with_judge=True)

        req = _make_request(
            "Carlos Alcaraz won the Australian Open 2026.",
            {"style": "natural_text", "value": "Carlos Alcaraz"},
            question="Who won the men's singles at the Australian Open 2026?",
        )
        result = asyncio.run(rs.verify(req))
        # The verify response, when serialized, must include ground_truth and question
        # so downstream tools (rollouts JSONL, reward profile) can join on them.
        dumped = result.model_dump()
        assert dumped["ground_truth"] == {"style": "natural_text", "value": "Carlos Alcaraz"}
        assert dumped["question"] == "Who won the men's singles at the Australian Open 2026?"


def _mock_httpx_post(response_json: dict, status_code: int = 200):
    """Build a context manager that patches ``httpx.AsyncClient`` with a
    captured-call mock returning the given JSON. Returns ``(patch, captured)``
    where ``captured`` is a list that gets ``(url, payload)`` appended on each
    call. Use ``with patch_obj:`` in tests.
    """
    captured: list[tuple[str, dict]] = []

    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()
    fake_resp.status_code = status_code
    fake_resp.json = MagicMock(return_value=response_json)

    async def _post(url, headers=None, json=None):  # noqa: A002 - shadowing intentional, matches httpx kw
        captured.append((url, json))
        return fake_resp

    fake_client = MagicMock()
    fake_client.post = _post

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return fake_client

        async def __aexit__(self, exc_type, exc, tb):
            return None

    return patch("httpx.AsyncClient", _FakeAsyncClient), captured


# ---------------------------------------------------------------------------
# /search routing (mocked httpx)
# ---------------------------------------------------------------------------


class TestSearchRouting:
    def test_general_routes_to_search_endpoint_with_in_en_locale(self) -> None:
        rs = _make_server()
        patch_obj, captured = _mock_httpx_post(
            {"organic": [{"title": "T", "snippet": "S", "link": "L"}]}
        )
        with patch_obj:
            req = SearchRequest(query="hello", search_type="general")
            result = asyncio.run(rs.search(req))
        assert len(captured) == 1
        url, payload = captured[0]
        assert url == "https://google.serper.dev/search"
        # No site filter for general; gl/hl India-tuned matching RLVR.
        assert payload == {"q": "hello", "num": 5, "gl": "in", "hl": "en"}
        # Structured JSON output (matches RLVR's _parse_results shape).
        parsed = json.loads(result.results)
        assert parsed["query"] == "hello"
        assert parsed["search_type"] == "general"
        assert parsed["total_results"] == 1
        assert parsed["results"][0] == {"position": 1, "title": "T", "url": "L", "snippet": "S"}

    def test_weather_appends_accuweather_site_filter(self) -> None:
        rs = _make_server()
        patch_obj, captured = _mock_httpx_post(
            {"organic": [{"title": "Weather in SF", "snippet": "72F", "link": "L"}]}
        )
        with patch_obj:
            req = SearchRequest(query="san francisco today", search_type="weather")
            result = asyncio.run(rs.search(req))
        url, payload = captured[0]
        assert url == "https://google.serper.dev/search"
        assert payload["q"] == "san francisco today site:accuweather.com"
        parsed = json.loads(result.results)
        assert parsed["search_type"] == "weather"

    def test_sports_appends_sports_site_filter(self) -> None:
        rs = _make_server()
        patch_obj, captured = _mock_httpx_post({"organic": []})
        with patch_obj:
            req = SearchRequest(query="ipl 2026 winner", search_type="sports")
            asyncio.run(rs.search(req))
        _, payload = captured[0]
        assert (
            payload["q"]
            == "ipl 2026 winner site:espn.in OR site:sofascore.com OR site:cricbuzz.com"
        )

    def test_stock_appends_stock_site_filter(self) -> None:
        rs = _make_server()
        patch_obj, captured = _mock_httpx_post({"organic": []})
        with patch_obj:
            req = SearchRequest(query="reliance share price", search_type="stock")
            asyncio.run(rs.search(req))
        _, payload = captured[0]
        assert (
            payload["q"]
            == "reliance share price site:screener.in OR site:moneycontrol.com"
        )

    def test_news_routes_to_news_endpoint_and_includes_date_source(self) -> None:
        rs = _make_server()
        patch_obj, captured = _mock_httpx_post(
            {
                "news": [
                    {
                        "title": "PSG draw Arsenal in UCL final",
                        "snippet": "Arsenal will face PSG in Budapest.",
                        "source": "BBC Sport",
                        "date": "2 days ago",
                        "link": "https://bbc.co.uk/foo",
                    }
                ]
            }
        )
        with patch_obj:
            req = SearchRequest(query="UCL final 2026", search_type="news")
            result = asyncio.run(rs.search(req))
        assert captured[0][0] == "https://google.serper.dev/news"
        # No site filter for news.
        assert captured[0][1]["q"] == "UCL final 2026"
        parsed = json.loads(result.results)
        assert parsed["search_type"] == "news"
        assert parsed["results"][0]["date"] == "2 days ago"
        assert parsed["results"][0]["source"] == "BBC Sport"

    def test_scholar_routes_to_scholar_endpoint_and_includes_citedBy(self) -> None:
        rs = _make_server()
        patch_obj, captured = _mock_httpx_post(
            {
                "organic": [
                    {
                        "title": "Attention is all you need",
                        "snippet": "We propose a new simple network architecture, the Transformer...",
                        "citedBy": 100000,
                        "link": "https://arxiv.org/abs/1706.03762",
                    }
                ]
            }
        )
        with patch_obj:
            req = SearchRequest(query="transformers paper", search_type="scholar")
            result = asyncio.run(rs.search(req))
        assert captured[0][0] == "https://google.serper.dev/scholar"
        # No site filter for scholar.
        assert captured[0][1]["q"] == "transformers paper"
        parsed = json.loads(result.results)
        assert parsed["search_type"] == "scholar"
        assert parsed["results"][0]["citedBy"] == 100000

    def test_num_results_capped_at_ten(self) -> None:
        rs = _make_server()
        patch_obj, captured = _mock_httpx_post({"organic": []})
        with patch_obj:
            req = SearchRequest(query="x", search_type="general", num_results=50)
            asyncio.run(rs.search(req))
        # Serper caps at 10 — RLVR enforces this client-side.
        assert captured[0][1]["num"] == 10

    def test_unknown_search_type_falls_back_to_general(self) -> None:
        rs = _make_server()
        patch_obj, captured = _mock_httpx_post({"organic": []})
        with patch_obj:
            req = SearchRequest(query="x", search_type="not_a_real_type")
            result = asyncio.run(rs.search(req))
        assert captured[0][0] == "https://google.serper.dev/search"
        assert captured[0][1]["q"] == "x"  # no site filter applied
        parsed = json.loads(result.results)
        assert parsed["total_results"] == 0
        assert parsed["results"] == []

    def test_missing_result_key_returns_no_results_message(self) -> None:
        # Server returned a response shape without the expected result key
        # (e.g. /news but data has no "news" array). RLVR's _parse_results
        # surfaces this as a structured "No X search results found" envelope.
        rs = _make_server()
        patch_obj, _ = _mock_httpx_post({"someUnrelatedField": []})
        with patch_obj:
            req = SearchRequest(query="x", search_type="news")
            result = asyncio.run(rs.search(req))
        parsed = json.loads(result.results)
        assert parsed["results"] == []
        assert parsed["search_type"] == "news"
        assert "No news search results found" in parsed["message"]

    def test_no_api_key_short_circuits(self) -> None:
        config = WebSearchResourcesServerConfig(
            host="0.0.0.0", port=8080, entrypoint="", name="", serper_api_key=""
        )
        rs = WebSearchResourcesServer(
            config=config, server_client=MagicMock(spec=ServerClient)
        )
        # No httpx mock — if the code tried to make a request it'd fail.
        # Also clear SERPER_API_KEY env var temporarily.
        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("SERPER_API_KEY", None)
            req = SearchRequest(query="x", search_type="general")
            result = asyncio.run(rs.search(req))
        assert "SERPER_API_KEY not configured" in result.results

