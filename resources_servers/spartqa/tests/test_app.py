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
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from pytest import approx

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.spartqa.app import (
    BOTH_LABEL,
    NONE_LABEL,
    SpartqaResourcesServer,
    SpartqaResourcesServerConfig,
    SpartqaVerifyRequest,
    _extract_answer,
    _label_key,
    _normalize,
    _response_text,
    _strip_reasoning,
    candidate_labels,
    match_label,
)


_EXAMPLE_JSONL = Path(__file__).resolve().parent.parent / "data" / "example.jsonl"


def _make_response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp",
        created_at=0.0,
        model="policy_model",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="msg",
                content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


def _config() -> SpartqaResourcesServerConfig:
    return SpartqaResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="spartqa")


def _make_request(
    text: str,
    *,
    target: str = "",
    options: list[str] | None = None,
    verifier_metadata: dict | None = None,
) -> SpartqaVerifyRequest:
    return SpartqaVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=_make_response(text),
        target=target,
        options=options or [],
        verifier_metadata=verifier_metadata,
    )


# A representative CO row: two story objects plus the two fixed labels.
_OPTIONS = ["a big blue square", "a small blue square"]


def _server() -> SpartqaResourcesServer:
    return SpartqaResourcesServer(config=_config(), server_client=MagicMock(spec=ServerClient))


# ── Pure helpers ─────────────────────────────────────────────────────────


class TestExtractAnswer:
    def test_pulls_phrase_after_final_answer(self) -> None:
        assert _extract_answer("Reasoning here.\nFinal answer: below the circle") == ("below the circle")

    def test_strips_think_reasoning(self) -> None:
        text = "<think>the star is north</think>Final answer: yes"
        assert _extract_answer(text) == "yes"

    def test_thinking_process_prefix_returns_last_line(self) -> None:
        # No explicit "Final answer:" marker; a multi-line "thinking process"
        # prefix means the answer is the last non-empty line.
        text = "Thinking process: I consider the layout\ngreen cup"
        assert _extract_answer(text) == "green cup"

    def test_ignores_template_echo_and_takes_last_answer(self) -> None:
        # Reasoning models often restate the prompt's format instruction
        # ("Final answer: <candidate answer>") mid-thought. The extractor must
        # skip that placeholder echo and return the real concluding answer,
        # not the first "Final answer:" line it sees.
        text = (
            "To answer I must end with 'Final answer: <candidate answer>'.\n"
            "Working through the layout, the yellow circle is below.\n"
            "Final answer: a big yellow circle that is touching the bottom edge of a block"
        )
        assert _extract_answer(text) == "a big yellow circle that is touching the bottom edge of a block"

    def test_multiple_final_answer_lines_takes_last(self) -> None:
        # A model that drafts an answer then revises it: the last line wins.
        text = "Final answer: a small blue shape\nOn reflection:\nFinal answer: both of them"
        assert _extract_answer(text) == "both of them"

    def test_empty_returns_empty(self) -> None:
        assert _extract_answer("   ") == ""


class TestStripReasoning:
    def test_removes_think_block(self) -> None:
        assert _strip_reasoning("<think>hidden</think>visible") == "visible"

    def test_passthrough(self) -> None:
        assert _strip_reasoning("plain") == "plain"


class TestNormalize:
    def test_lowercases_strips_punctuation_collapses_ws(self) -> None:
        assert _normalize("  The  Big, BLACK Square!! ") == "the big black square"

    def test_empty(self) -> None:
        assert _normalize("") == ""


# ── verify() ─────────────────────────────────────────────────────────────


class TestVerify:
    async def test_verbatim_gold_scores_one_and_exact(self) -> None:
        result = await _server().verify(
            _make_request("Final answer: a big blue square", target=_OPTIONS[0], options=_OPTIONS)
        )
        assert result.reward == approx(1.0)
        assert result.exact is True
        assert result.parsed is True
        assert result.extracted == "a big blue square"
        assert result.predicted_label == _OPTIONS[0]

    async def test_gold_inside_a_sentence_scores_one_but_not_exact(self) -> None:
        result = await _server().verify(
            _make_request(
                "Final answer: it must be a big blue square.",
                target=_OPTIONS[0],
                options=_OPTIONS,
            )
        )
        assert result.reward == approx(1.0)
        assert result.exact is False
        assert result.predicted_label == _OPTIONS[0]

    async def test_wrong_option_scores_zero(self) -> None:
        result = await _server().verify(
            _make_request("Final answer: a small blue square", target=_OPTIONS[0], options=_OPTIONS)
        )
        assert result.reward == approx(0.0)
        assert result.predicted_label == _OPTIONS[1]

    async def test_both_of_them_gold(self) -> None:
        result = await _server().verify(
            _make_request(f"Final answer: {BOTH_LABEL}", target=BOTH_LABEL, options=_OPTIONS)
        )
        assert result.reward == approx(1.0)
        assert result.exact is True

    async def test_none_of_them_gold(self) -> None:
        result = await _server().verify(
            _make_request(f"Final answer: {NONE_LABEL}", target=NONE_LABEL, options=_OPTIONS)
        )
        assert result.reward == approx(1.0)

    async def test_article_difference_still_matches(self) -> None:
        # qrels phrases and the story wording disagree on leading articles.
        result = await _server().verify(
            _make_request(
                "Final answer: the big blue square",
                target="a big blue square",
                options=_OPTIONS,
            )
        )
        assert result.reward == approx(1.0)
        assert result.exact is True

    async def test_alias_resolves_to_fixed_label(self) -> None:
        for alias, gold in (("Both", BOTH_LABEL), ("Neither", NONE_LABEL), ("DK", NONE_LABEL)):
            result = await _server().verify(_make_request(f"Final answer: {alias}", target=gold, options=_OPTIONS))
            assert result.reward == approx(1.0), alias
            assert result.exact is False, alias

    async def test_naming_a_single_option_does_not_earn_a_both_gold(self) -> None:
        # Regression: the retrieval qrels list "both of them" alongside each
        # object, so any-of matching used to credit this.
        for option in _OPTIONS:
            result = await _server().verify(
                _make_request(f"Final answer: {option}", target=BOTH_LABEL, options=_OPTIONS)
            )
            assert result.reward == approx(0.0), option

    async def test_echoing_both_options_is_ambiguous_and_scores_zero(self) -> None:
        # Regression: "X or Y" copied straight out of the question used to score
        # 1.0 on ~95% of the corpus via substring matching.
        text = f"Final answer: {_OPTIONS[0]} or {_OPTIONS[1]}"
        for gold in (_OPTIONS[0], _OPTIONS[1], BOTH_LABEL):
            result = await _server().verify(_make_request(text, target=gold, options=_OPTIONS))
            assert result.reward == approx(0.0), gold
            assert result.predicted_label == ""

    async def test_nested_options_resolve_to_the_most_specific(self) -> None:
        options = ["a triangle that is in block B", "a big blue triangle that is in block B"]
        result = await _server().verify(
            _make_request(
                "Final answer: a big blue triangle that is in block B",
                target=options[1],
                options=options,
            )
        )
        assert result.reward == approx(1.0)
        assert result.predicted_label == options[1]

    async def test_empty_output_reward_zero_no_raise(self) -> None:
        result = await _server().verify(_make_request("   ", target=_OPTIONS[0], options=_OPTIONS))
        assert result.reward == approx(0.0)
        assert result.parsed is False
        assert result.predicted_label == ""

    async def test_empty_target_scores_zero(self) -> None:
        result = await _server().verify(_make_request("Final answer: yes", target="", options=[]))
        assert result.reward == approx(0.0)

    async def test_fields_from_verifier_metadata(self) -> None:
        # The native driver drops the top-level ``options`` list; the candidate
        # set must be recoverable from verifier_metadata.
        result = await _server().verify(
            _make_request(
                f"Final answer: {BOTH_LABEL}",
                target="",
                options=[],
                verifier_metadata={"target": BOTH_LABEL, "options": _OPTIONS},
            )
        )
        assert result.reward == approx(1.0)

    async def test_options_missing_falls_back_to_target_only(self) -> None:
        result = await _server().verify(_make_request("Final answer: a big blue square", target=_OPTIONS[0]))
        assert result.reward == approx(1.0)


class TestLabelMatching:
    def test_candidate_labels_appends_fixed_labels_and_dedupes(self) -> None:
        assert candidate_labels(_OPTIONS) == [*_OPTIONS, BOTH_LABEL, NONE_LABEL]
        assert candidate_labels([BOTH_LABEL]) == [BOTH_LABEL, NONE_LABEL]

    def test_label_key_strips_articles_and_punctuation(self) -> None:
        assert _label_key("  The Big, BLUE Square! ") == "big blue square"
        assert _label_key("") == ""

    def test_match_label_returns_none_when_nothing_matches(self) -> None:
        assert match_label("a green cup", candidate_labels(_OPTIONS)) is None
        assert match_label("", candidate_labels(_OPTIONS)) is None


# ── compute_metrics() / get_key_metrics() ──────────────────────────────────


class TestComputeMetrics:
    def test_mean_and_rates(self) -> None:
        tasks = [
            [{"reward": 1.0, "exact": True, "parsed": True, "target": BOTH_LABEL, "predicted_label": BOTH_LABEL}],
            [
                {
                    "reward": 0.0,
                    "exact": False,
                    "parsed": True,
                    "target": BOTH_LABEL,
                    "predicted_label": "a big blue square",
                }
            ],
            [
                {
                    "reward": 1.0,
                    "exact": False,
                    "parsed": True,
                    "target": "a big blue square",
                    "predicted_label": "a big blue square",
                }
            ],
            [{"reward": 0.0, "exact": False, "parsed": False, "target": NONE_LABEL, "predicted_label": ""}],
        ]
        metrics = _server().compute_metrics(tasks)
        assert metrics["mean_reward"] == approx(0.5)
        assert metrics["count"] == 4
        assert metrics["exact_match_rate"] == approx(0.25)
        assert metrics["parse_rate"] == approx(0.75)
        assert metrics["label_resolve_rate"] == approx(0.75)
        assert metrics["accuracy_both_of_them"] == approx(0.5)
        assert metrics["accuracy_none_of_them"] == approx(0.0)

    def test_empty(self) -> None:
        assert _server().compute_metrics([]) == {}


class TestGetKeyMetrics:
    def test_selects_headline(self) -> None:
        out = _server().get_key_metrics({"mean_reward": 0.5, "exact_match_rate": 0.25, "parse_rate": 0.75, "count": 4})
        assert out == {"mean_reward": approx(0.5), "exact_match_rate": approx(0.25)}


# ── _response_text() ───────────────────────────────────────────────────────


class TestResponseText:
    def test_output_text_fast_path(self) -> None:
        assert _response_text(_make_response("hello")) == "hello"

    def test_fallback_joins_message_content(self) -> None:
        message = SimpleNamespace(
            type="message",
            content=[SimpleNamespace(text="a"), {"text": "b"}],
        )
        reasoning = SimpleNamespace(type="reasoning", content="ignored")
        response = SimpleNamespace(output_text=None, output=[reasoning, message])
        assert _response_text(response) == "ab"

    def test_fallback_string_content(self) -> None:
        message = SimpleNamespace(type="message", content="plain")
        response = SimpleNamespace(output_text="", output=[message])
        assert _response_text(response) == "plain"


# ── Acceptance / parity ────────────────────────────────────────────────────


class TestAcceptance:
    async def test_each_example_gold_scores_one(self) -> None:
        server = _server()
        rows = [json.loads(line) for line in _EXAMPLE_JSONL.read_text().splitlines() if line.strip()]
        assert len(rows) >= 5
        for row in rows:
            result = await server.verify(
                _make_request(
                    f"Final answer: {row['target']}",
                    target=row["target"],
                    options=row["options"],
                )
            )
            assert result.reward == approx(1.0), row["target"]

    async def test_example_rows_cover_both_and_none_golds(self) -> None:
        rows = [json.loads(line) for line in _EXAMPLE_JSONL.read_text().splitlines() if line.strip()]
        golds = {row["target"] for row in rows}
        assert BOTH_LABEL in golds
        assert NONE_LABEL in golds
        # Every row must offer exactly two story objects to choose between.
        assert all(len(row["options"]) == 2 for row in rows)

    async def test_echoing_the_question_options_never_scores(self) -> None:
        server = _server()
        rows = [json.loads(line) for line in _EXAMPLE_JSONL.read_text().splitlines() if line.strip()]
        for row in rows:
            text = f"Final answer: {row['options'][0]} or {row['options'][1]}"
            result = await server.verify(_make_request(text, target=row["target"], options=row["options"]))
            assert result.reward == approx(0.0), row["target"]
