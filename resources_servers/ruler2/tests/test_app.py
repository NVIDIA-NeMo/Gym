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
from unittest.mock import MagicMock

import pytest
from app import (
    RULER2_SUBTASKS,
    Ruler2ResourcesServer,
    Ruler2ResourcesServerConfig,
    Ruler2VerifyRequest,
    _default_parse,
    _soft_match,
    _wer,
    extract_mcq_letter,
    string_match_2steps_single,
    string_match_all_single,
    string_match_part_single,
)

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient


def _make_server() -> Ruler2ResourcesServer:
    config = Ruler2ResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
    return Ruler2ResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _make_response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp_test",
        created_at=0.0,
        model="dummy",
        object="response",
        output=[
            {
                "id": "msg_test",
                "content": [
                    {
                        "annotations": [],
                        "text": text,
                        "type": "output_text",
                    }
                ],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def _verify(server: Ruler2ResourcesServer, output_text: str, **kwargs):
    """Build a Ruler2VerifyRequest and call verify() synchronously."""
    body = Ruler2VerifyRequest(
        responses_create_params={"input": []},
        response=_make_response(output_text),
        **kwargs,
    )
    return asyncio.run(server.verify(body))


class TestSanity:
    def test_construct(self) -> None:
        _make_server()


class TestDefaultParse:
    def test_strip_non_printable_to_newline(self) -> None:
        # \x01 is a control character; it should become \n
        assert _default_parse("hello\x01world") == "hello\nworld"

    def test_leading_trailing_strip(self) -> None:
        assert _default_parse("   abc   ") == "abc"

    def test_empty_string(self) -> None:
        assert _default_parse("") == ""

    def test_combined(self) -> None:
        assert _default_parse(" \tA\x00B\x1fC\n ") == "A\nB\nC"


class TestWER:
    def test_identical(self) -> None:
        assert _wer("a b c", "a b c") == 0.0

    def test_complete_difference(self) -> None:
        assert _wer("a b c", "x y z") == 1.0

    def test_one_substitution(self) -> None:
        assert _wer("a b c", "a b d") == pytest.approx(1.0 / 3)

    def test_empty_reference_returns_inf(self) -> None:
        assert _wer("a", "") == float("inf")


class TestSoftMatch:
    def test_substring_returns_one(self) -> None:
        assert _soft_match("haystack with needle in it", "needle") == 1.0

    def test_case_insensitive(self) -> None:
        assert _soft_match("THE Needle", "needle") == 1.0

    def test_close_wer_above_substring(self) -> None:
        # Exact substring miss but close → WER fallback yields a positive score.
        score = _soft_match("foo bar baz", "foo bar quux")
        # 1 substitution out of 3 reference words → WER = 1/3 → 1 - WER = 2/3
        assert score == pytest.approx(2.0 / 3)

    def test_completely_unrelated(self) -> None:
        # No substring AND high WER → both branches yield ≤ 0
        assert _soft_match("foo bar", "the quick brown fox") == 0.0


class TestStringMatchAll:
    def test_all_refs_present_returns_one(self) -> None:
        assert string_match_all_single("foo bar baz", ["foo", "bar", "baz"]) == 1.0

    def test_partial_substring_average(self) -> None:
        score = string_match_all_single("foo bar", ["foo", "bar", "qux"])
        # foo: 1.0 ; bar: 1.0 ; qux: max(0, 1 - WER("foo bar","qux")) = 0 → avg 2/3
        assert score == pytest.approx(2.0 / 3)

    def test_empty_refs_returns_zero(self) -> None:
        assert string_match_all_single("anything", []) == 0.0


class TestStringMatchPart:
    def test_max_across_refs(self) -> None:
        score = string_match_part_single("answer is foo", ["foo", "totally unrelated zzz"])
        assert score == 1.0

    def test_strips_document_headers(self) -> None:
        pred = "Document 1:\nfoo content\n\nDocument 2:\nbar content\n\nthe answer"
        # The "Document N:" prefix segments are stripped, but the trailing
        # non-document text "the answer" remains.
        assert string_match_part_single(pred, ["answer"]) == 1.0

    def test_empty_refs_returns_zero(self) -> None:
        assert string_match_part_single("anything", []) == 0.0


class TestStringMatch2Steps:
    def test_uses_last_paragraph_only(self) -> None:
        pred = "the answer is foo\n\nbut now I'm rambling about cats"
        # Only the last paragraph is considered → "foo" is not present there
        # so substring match misses; WER fallback also yields 0 since the
        # last paragraph is much longer than ref.
        assert string_match_2steps_single(pred, ["foo"]) == string_match_all_single(
            "but now I'm rambling about cats", ["foo"]
        )

    def test_last_paragraph_matches(self) -> None:
        pred = "intro\n\nthe answer is foo"
        assert string_match_2steps_single(pred, ["foo"]) == 1.0

    def test_empty_refs_returns_zero(self) -> None:
        assert string_match_2steps_single("anything", []) == 0.0


class TestExtractMcqLetter:
    def test_extract_from_boxed(self) -> None:
        assert extract_mcq_letter("solving... \\boxed{B}") == "B"

    def test_extract_from_final_answer_regex(self) -> None:
        assert extract_mcq_letter("The final answer is C") == "C"

    def test_extract_from_answer_colon(self) -> None:
        assert extract_mcq_letter("blah blah\nAnswer: D") == "D"

    def test_extract_from_markdown_answer(self) -> None:
        assert extract_mcq_letter("**Answer:** A") == "A"

    def test_returns_none_when_no_match(self) -> None:
        assert extract_mcq_letter("no answer here, just words") is None

    def test_normalize_arabic_letter(self) -> None:
        assert extract_mcq_letter("\\boxed{أ}") == "A"

    def test_takes_last_letter_in_long_answer(self) -> None:
        assert extract_mcq_letter("\\boxed{The answer is B}") == "B"


class TestVerifyRuler2All:
    def test_substring_match_rewards_one(self) -> None:
        server = _make_server()
        out = asyncio.run(
            server.verify(
                Ruler2VerifyRequest(
                    responses_create_params={"input": []},
                    response=_make_response("foo bar baz"),
                    expected_answer=["foo", "bar"],
                    eval_type="ruler2",
                    match_type="all",
                )
            )
        )
        assert out.reward == 1.0
        assert out.predicted_answer == "foo bar baz"
        assert out.extracted_answer is None

    def test_partial_match_returns_fraction(self) -> None:
        server = _make_server()
        out = _verify(
            server,
            output_text="foo",
            expected_answer=["foo", "qux"],
            eval_type="ruler2",
            match_type="all",
        )
        # foo matches substring (1.0), qux: max(0, 1 - 1) = 0 → avg 0.5
        assert out.reward == 0.5


class TestVerifyRuler2Part:
    def test_part_max_across_refs(self) -> None:
        server = _make_server()
        out = _verify(
            server,
            output_text="here is the relevant doc text",
            expected_answer=["doc text", "totally different"],
            eval_type="ruler2",
            match_type="part",
        )
        assert out.reward == 1.0


class TestVerifyRuler2TwoSteps:
    def test_2steps_only_last_paragraph(self) -> None:
        server = _make_server()
        out = _verify(
            server,
            output_text="reasoning here that mentions foo\n\nfinal answer: foo",
            expected_answer=["foo"],
            eval_type="ruler2",
            match_type="2steps",
        )
        assert out.reward == 1.0


class TestVerifyMultichoice:
    def test_correct_letter_boxed(self) -> None:
        server = _make_server()
        out = _verify(
            server,
            output_text="step by step ... \\boxed{C}",
            expected_answer="C",
            eval_type="multichoice",
            match_type="all",
        )
        assert out.reward == 1.0
        assert out.extracted_answer == "C"

    def test_wrong_letter(self) -> None:
        server = _make_server()
        out = _verify(
            server,
            output_text="\\boxed{A}",
            expected_answer="B",
            eval_type="multichoice",
            match_type="all",
        )
        assert out.reward == 0.0
        assert out.extracted_answer == "A"

    def test_no_extraction_returns_zero(self) -> None:
        server = _make_server()
        out = _verify(
            server,
            output_text="nope",
            expected_answer="A",
            eval_type="multichoice",
            match_type="all",
        )
        assert out.reward == 0.0
        assert out.extracted_answer is None


class TestVerifyInvalidMatchType:
    def test_unknown_match_type_raises(self) -> None:
        server = _make_server()
        with pytest.raises(ValueError):
            _verify(
                server,
                output_text="x",
                expected_answer=["x"],
                eval_type="ruler2",
                match_type="bogus",
            )


class TestComputeMetrics:
    def test_empty(self) -> None:
        server = _make_server()
        assert server.compute_metrics([]) == {}

    def test_per_subtask_breakdown(self) -> None:
        server = _make_server()
        tasks = [
            [{"reward": 1.0, "task": "mk_niah_basic"}, {"reward": 0.0, "task": "mk_niah_basic"}],
            [{"reward": 1.0, "task": "qa_basic"}, {"reward": 1.0, "task": "qa_basic"}],
        ]
        metrics = server.compute_metrics(tasks)
        assert any(k.startswith("pass@1") for k in metrics)
        assert any(k.startswith("mk_niah_basic/pass@") for k in metrics)
        assert any(k.startswith("qa_basic/pass@") for k in metrics)

    def test_suite_avg_only_when_full_suite(self) -> None:
        server = _make_server()
        tasks = [[{"reward": 1.0, "task": "mk_niah_basic"}]]
        metrics = server.compute_metrics(tasks)
        assert not any(k.startswith("ruler2_suite_avg/") for k in metrics)

    def test_suite_avg_when_all_12_present(self) -> None:
        server = _make_server()
        tasks = [[{"reward": 1.0 if i % 2 == 0 else 0.5, "task": t}] for i, t in enumerate(RULER2_SUBTASKS)]
        metrics = server.compute_metrics(tasks)
        assert any(k.startswith("ruler2_suite_avg/") for k in metrics)


class TestGetKeyMetrics:
    def test_picks_highest_k(self) -> None:
        server = _make_server()
        agent_metrics = {
            "pass@1/accuracy": 50.0,
            "pass@4/accuracy": 80.0,
            "pass@1[avg-of-4]/accuracy": 60.0,
            "ruler2_suite_avg/pass@1/accuracy": 70.0,
            "mean/input_tokens": 100,
            "mean/output_tokens": 50,
        }
        key = server.get_key_metrics(agent_metrics)
        assert key.get("pass@4/accuracy") == 80.0
        assert key.get("pass@1[avg-of-4]/accuracy") == 60.0
        assert key.get("ruler2_suite_avg/pass@1/accuracy") == 70.0
        assert key.get("mean/input_tokens") == 100
