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
from unittest.mock import MagicMock

from pytest import approx

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.hotpotqa_qa.app import (
    HotpotQAQAResourcesServer,
    HotpotQAQAResourcesServerConfig,
    HotpotQAQAVerifyRequest,
    _task_should_remove,
)
from resources_servers.hotpotqa_qa.scoring import (
    answer_exact_match,
    answer_f1_score,
    is_correct,
    is_correct_strict,
    normalize_answer,
    normalize_gt,
    parse_generation,
)


def _make_response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="test",
        created_at=0.0,
        model="test_model",
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


def _make_server(report_filtered_metrics: bool = True) -> HotpotQAQAResourcesServer:
    config = HotpotQAQAResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        report_filtered_metrics=report_filtered_metrics,
    )
    return HotpotQAQAResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


# ──────────────────────────────────────────────────────────
# Pure scoring functions
# ──────────────────────────────────────────────────────────


class TestNormalizeAnswer:
    def test_lowercases(self) -> None:
        assert normalize_answer("Paris") == "paris"

    def test_strips_articles(self) -> None:
        assert normalize_answer("the Eiffel Tower") == "eiffel tower"

    def test_collapses_whitespace(self) -> None:
        assert normalize_answer("foo   bar") == "foo bar"

    def test_strips_punctuation(self) -> None:
        assert normalize_answer("Hello, world!") == "hello world"

    def test_combined(self) -> None:
        assert normalize_answer("  The   QUICK,   brown FOX!! ") == "quick brown fox"


class TestAnswerExactMatch:
    def test_match(self) -> None:
        assert answer_exact_match("Paris", "Paris") == approx(1.0)

    def test_match_after_normalization(self) -> None:
        assert answer_exact_match("the paris", "Paris") == approx(1.0)

    def test_mismatch(self) -> None:
        assert answer_exact_match("London", "Paris") == approx(0.0)

    def test_partial_no_match(self) -> None:
        assert answer_exact_match("Paris is the capital", "Paris") == approx(0.0)


class TestAnswerF1Score:
    def test_perfect_match(self) -> None:
        f1, p, r = answer_f1_score("Paris", "Paris")
        assert f1 == approx(1.0)
        assert p == approx(1.0)
        assert r == approx(1.0)

    def test_partial_overlap(self) -> None:
        # After SQuAD normalization: "the cat sat" -> "cat sat" (article stripped)
        # and "cat dog" -> "cat dog". Shared token: "cat".
        f1, p, r = answer_f1_score("the cat sat", "cat dog")
        assert 0 < f1 < 1
        assert p == approx(1.0 / 2)
        assert r == approx(1.0 / 2)

    def test_no_overlap(self) -> None:
        assert answer_f1_score("foo", "bar") == approx((0.0, 0.0, 0.0))

    def test_yes_special_case_mismatch(self) -> None:
        # Skills' rule: if either side normalizes to yes/no/noanswer and they
        # don't match, score is forced to zero.
        assert answer_f1_score("yes", "no") == approx((0.0, 0.0, 0.0))

    def test_yes_special_case_match(self) -> None:
        assert answer_f1_score("yes", "yes") == approx((1.0, 1.0, 1.0))

    def test_yes_special_case_prediction_only(self) -> None:
        # When the prediction normalizes to exactly "yes" but the ground truth
        # does NOT, Skills' special-case rule forces the score to zero — even
        # though there is shared-token overlap. This is the "model said yes
        # but the real answer was something else" guard.
        f1, _, _ = answer_f1_score("yes", "yes definitely")
        assert f1 == approx(0.0)

    def test_gt_yes_special_case_with_extra_tokens(self) -> None:
        # The mirror case: ground truth normalizes to "yes" but the prediction
        # does not. Skills' rule forces the score to zero on either side.
        # GT must normalize to one of {"yes", "no", "noanswer"} and prediction
        # must differ.
        f1, _, _ = answer_f1_score("yes definitely", "yes")
        assert f1 == approx(0.0)

    def test_no_token_overlap_returns_zero(self) -> None:
        # Distinct tokens with no overlap should yield (0, 0, 0).
        assert answer_f1_score("alpha beta", "gamma delta") == approx((0.0, 0.0, 0.0))


class TestParseGeneration:
    def test_simple_json(self) -> None:
        ans, sp = parse_generation('{"answer": "Paris"}')
        assert ans == "Paris"
        assert sp == []

    def test_json_with_supporting_facts(self) -> None:
        ans, sp = parse_generation('{"answer": "Paris", "supporting_facts": [["Title", 0]]}')
        assert ans == "Paris"
        assert sp == [["Title", 0]]

    def test_invalid_supporting_facts_filtered(self) -> None:
        ans, sp = parse_generation('{"answer": "x", "supporting_facts": [["t", "bad"], ["t2", 1]]}')
        assert ans == "x"
        assert sp == [["t2", 1]]

    def test_supporting_facts_not_list(self) -> None:
        # If the SP value isn't a list, an empty list is used.
        ans, sp = parse_generation('{"answer": "x", "supporting_facts": "not a list"}')
        assert ans == "x"
        assert sp == []

    def test_supporting_facts_invalid_int(self) -> None:
        # SP item with a non-coercible second element should be skipped.
        ans, sp = parse_generation('{"answer": "x", "supporting_facts": [["t", "abc"]]}')
        assert ans == "x"
        assert sp == []

    def test_skips_json_without_answer_key(self) -> None:
        # Brace-balanced JSON without an "answer" key is skipped.
        ans, _ = parse_generation('{"random": "json"} no_answer_object')
        assert ans == '{"random": "json"} no_answer_object'

    def test_skips_unparseable_json_candidate(self) -> None:
        # A candidate that looks like JSON but is malformed gets skipped.
        ans, _ = parse_generation('{not valid json} {"answer": "good"}')
        assert ans == "good"

    def test_markdown_fenced_json(self) -> None:
        ans, _ = parse_generation('Some reasoning.\n```json\n{"answer": "Paris"}\n```')
        assert ans == "Paris"

    def test_picks_last_json_when_multiple(self) -> None:
        # The model is prompted to put the JSON last, so the last valid
        # object containing an "answer" key wins.
        ans, _ = parse_generation('First thought: {"answer": "Wrong"}\nFinal: {"answer": "Right"}')
        assert ans == "Right"

    def test_skips_non_answer_json(self) -> None:
        ans, _ = parse_generation('{"unrelated": "x"} {"answer": "Final"}')
        assert ans == "Final"

    def test_falls_back_to_raw_text(self) -> None:
        ans, sp = parse_generation("just text no json")
        assert ans == "just text no json"
        assert sp == []

    def test_empty(self) -> None:
        assert parse_generation("") == ("", [])

    def test_non_string_answer_coerced(self) -> None:
        ans, _ = parse_generation('{"answer": 42}')
        assert ans == "42"


class TestNormalizeGt:
    def test_simple_keeps_original(self) -> None:
        info = normalize_gt("Paris")
        assert "Paris" in info["alternatives"]
        assert info["should_remove"] is False
        assert info["remove_reason"] == ""

    def test_strips_article(self) -> None:
        alts = normalize_gt("the Eiffel Tower")["alternatives"]
        assert "Eiffel Tower" in alts
        assert "the Eiffel Tower" in alts

    def test_number_word_to_digit(self) -> None:
        alts = normalize_gt("five")["alternatives"]
        assert "5" in alts
        assert "five" in alts

    def test_parentheses_handling(self) -> None:
        info = normalize_gt("Apple (fruit)")
        assert "Apple" in info["alternatives"]
        assert "fruit" in info["alternatives"]

    def test_ampersand_to_and(self) -> None:
        alts = normalize_gt("AT & T")["alternatives"]
        assert "AT and T" in alts

    def test_long_gt_flagged(self) -> None:
        info = normalize_gt("x" * 50)
        assert info["should_remove"] is True
        assert info["remove_reason"] == "gt_too_long"

    def test_multi_word_name_flagged(self) -> None:
        info = normalize_gt("John Doe Smith")
        assert info["should_remove"] is True
        assert info["remove_reason"] == "multi_word_name"

    def test_short_gt_kept(self) -> None:
        info = normalize_gt("yes")
        assert info["should_remove"] is False

    def test_strip_quotes(self) -> None:
        alts = normalize_gt('"foo bar"')["alternatives"]
        assert "foo bar" in alts

    def test_strip_number_commas(self) -> None:
        alts = normalize_gt("1,000")["alternatives"]
        assert "1000" in alts

    def test_strip_trailing_punct(self) -> None:
        alts = normalize_gt("Paris.")["alternatives"]
        assert "Paris" in alts

    def test_strip_abbrev_dots(self) -> None:
        alts = normalize_gt("U.S.A.")["alternatives"]
        # The trailing-punct rule and abbrev-dots rule may both fire; just
        # confirm at least one alternative drops the dots.
        assert "USA" in alts

    def test_hyphen_to_space(self) -> None:
        alts = normalize_gt("multi-word")["alternatives"]
        assert "multi word" in alts

    def test_and_to_ampersand(self) -> None:
        alts = normalize_gt("Hall and Oates")["alternatives"]
        assert "Hall & Oates" in alts

    def test_number_digit_to_word(self) -> None:
        alts = normalize_gt("5")["alternatives"]
        assert "five" in alts

    def test_two_word_kept(self) -> None:
        # Two-word strings are not flagged as multi-word names.
        info = normalize_gt("New York")
        assert info["should_remove"] is False

    def test_six_word_few_caps_kept(self) -> None:
        # 6 words but only 2 capitalized -> kept.
        info = normalize_gt("Just a normal Sentence here please")
        assert info["should_remove"] is False

    def test_six_word_many_caps_flagged(self) -> None:
        # 6 words with >=3 capitalized non-stopwords -> flagged.
        info = normalize_gt("John Mary Paul David Smith Jones")
        assert info["should_remove"] is True
        assert info["remove_reason"] == "multi_word_name"

    def test_seven_words_kept(self) -> None:
        # 7+ words is outside the multi-word-name rule.
        info = normalize_gt("a b c d e f g")
        assert info["should_remove"] is False

    def test_unicode_quotes_stripped(self) -> None:
        alts = normalize_gt("“hello”")["alternatives"]
        assert "hello" in alts

    def test_parens_inner_extracted(self) -> None:
        alts = normalize_gt("Apple (the fruit)")["alternatives"]
        # The inner content should be extracted as its own alt and then have
        # the article stripped.
        assert "fruit" in alts


class TestIsCorrect:
    def test_substring_match(self) -> None:
        assert is_correct(["Paris"], "The capital of France is Paris.") is True

    def test_alt_substring_match(self) -> None:
        # Lenient substring check — picks up "5" inside the model output
        # when the GT alternatives include the digit form.
        alts = normalize_gt("five")["alternatives"]
        assert is_correct(alts, "There are 5 apples on the table.") is True

    def test_no_match(self) -> None:
        assert is_correct(["Berlin"], "The capital of France is Paris.") is False

    def test_unicode_normalization(self) -> None:
        assert is_correct(["it's"], "it’s a test") is True


class TestIsCorrectStrict:
    def test_word_boundary_for_short(self) -> None:
        # The short alt "5" must match at a word boundary — "5x" should fail.
        assert is_correct_strict(["5"], "5") is True
        assert is_correct_strict(["5"], "5x") is False
        assert is_correct_strict(["5"], "the 5 boys") is True

    def test_long_answer_position_guard(self) -> None:
        # Long answer (>80 chars): match must start within first 40 chars.
        long_text = "x" * 50 + " Paris " + "y" * 50
        assert len(long_text) > 80
        assert long_text.find("Paris") > 40
        assert is_correct_strict(["Paris"], long_text) is False

    def test_long_answer_position_guard_passes_when_early(self) -> None:
        long_text = "Paris " + "y" * 100
        assert is_correct_strict(["Paris"], long_text) is True

    def test_falls_back_to_lenient_for_normal_alts(self) -> None:
        assert is_correct_strict(["Paris"], "I think Paris is correct") is True

    def test_no_match(self) -> None:
        assert is_correct_strict(["Berlin"], "Paris") is False

    def test_empty_alt_skipped(self) -> None:
        assert is_correct_strict(["", "Paris"], "Paris") is True


# ──────────────────────────────────────────────────────────
# Server.verify endpoint
# ──────────────────────────────────────────────────────────


class TestVerify:
    async def test_exact_match_full_score(self) -> None:
        server = _make_server()
        request = HotpotQAQAVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_response('{"answer": "Paris"}'),
            expected_answer="Paris",
        )
        result = await server.verify(request)
        assert result.reward == approx(1.0)
        assert result.answer_em == approx(1.0)
        assert result.answer_f1 == approx(1.0)
        assert result.is_correct == approx(1.0)
        assert result.is_correct_strict == approx(1.0)
        assert result.extracted_answer == "Paris"

    async def test_substring_match_no_em(self) -> None:
        # The model emits a fully-formed sentence around the answer, so
        # SQuAD EM is 0 but token F1 + substring-based scores are positive.
        server = _make_server()
        request = HotpotQAQAVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_response('{"answer": "The capital is Paris"}'),
            expected_answer="Paris",
        )
        result = await server.verify(request)
        assert result.answer_em == approx(0.0)
        assert result.answer_f1 > 0.0
        assert result.is_correct == approx(1.0)
        assert result.is_correct_strict == approx(1.0)
        # reward is the strict score.
        assert result.reward == approx(1.0)

    async def test_complete_mismatch_zero(self) -> None:
        server = _make_server()
        request = HotpotQAQAVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_response('{"answer": "London"}'),
            expected_answer="Paris",
        )
        result = await server.verify(request)
        assert result.reward == approx(0.0)
        assert result.answer_em == approx(0.0)
        assert result.answer_f1 == approx(0.0)
        assert result.is_correct == approx(0.0)
        assert result.is_correct_strict == approx(0.0)

    async def test_unparseable_response_uses_raw_text(self) -> None:
        # When no JSON is found, the verifier falls back to scoring the raw text.
        server = _make_server()
        request = HotpotQAQAVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_response("Paris"),
            expected_answer="Paris",
        )
        result = await server.verify(request)
        assert result.answer_em == approx(1.0)
        assert result.extracted_answer == "Paris"

    async def test_empty_response(self) -> None:
        server = _make_server()
        empty_response = NeMoGymResponse(
            id="t",
            created_at=0.0,
            model="m",
            object="response",
            output=[],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )
        request = HotpotQAQAVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=empty_response,
            expected_answer="Paris",
        )
        result = await server.verify(request)
        assert result.reward == approx(0.0)
        assert result.extracted_answer is None

    async def test_unreliable_gt_marked(self) -> None:
        server = _make_server()
        request = HotpotQAQAVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_response('{"answer": "anything"}'),
            expected_answer="John Doe Smith",
        )
        result = await server.verify(request)
        assert result.gt_should_remove is True
        assert result.gt_remove_reason == "multi_word_name"

    async def test_alternative_matching(self) -> None:
        # GT "five" — model says "5" — alt-aware substring match should fire.
        server = _make_server()
        request = HotpotQAQAVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_response('{"answer": "5"}'),
            expected_answer="five",
        )
        result = await server.verify(request)
        assert result.is_correct == approx(1.0)


# ──────────────────────────────────────────────────────────
# Aggregate metrics: filtered + unfiltered
# ──────────────────────────────────────────────────────────


def _verify_record(
    *,
    expected_answer: str,
    extracted_answer: str,
    answer_em: float = 0.0,
    answer_f1: float = 0.0,
    is_correct_score: float = 0.0,
    is_correct_strict_score: float = 0.0,
    gt_should_remove: bool = False,
    task_index: int = 0,
    rollout_index: int = 0,
) -> dict:
    return {
        "task_index": task_index,
        "rollout_index": rollout_index,
        "expected_answer": expected_answer,
        "extracted_answer": extracted_answer,
        "answer_em": answer_em,
        "answer_f1": answer_f1,
        "is_correct": is_correct_score,
        "is_correct_strict": is_correct_strict_score,
        "gt_should_remove": gt_should_remove,
        "gt_remove_reason": "multi_word_name" if gt_should_remove else "",
        "reward": is_correct_strict_score,
    }


class TestComputeMetricsFilteredAndUnfiltered:
    def test_filtered_excludes_unreliable_gt(self) -> None:
        server = _make_server(report_filtered_metrics=True)

        # Two tasks: task 0 has reliable GT (Paris), task 1 has unreliable GT
        # (multi-word name flagged). Both rollouts on task 0 are correct,
        # both on task 1 are incorrect.
        tasks = [
            [
                _verify_record(
                    expected_answer="Paris",
                    extracted_answer="Paris",
                    answer_em=1.0,
                    answer_f1=1.0,
                    is_correct_score=1.0,
                    is_correct_strict_score=1.0,
                    task_index=0,
                    rollout_index=0,
                ),
            ],
            [
                _verify_record(
                    expected_answer="John Doe Smith",
                    extracted_answer="someone else",
                    gt_should_remove=True,
                    task_index=1,
                    rollout_index=0,
                ),
            ],
        ]
        metrics = server.compute_metrics(tasks)
        assert metrics["unfiltered_num_tasks"] == 2
        assert metrics["filtered_num_tasks"] == 1

        # Unfiltered: pass@1 over both tasks -> 50% on is_correct_strict.
        assert metrics["pass@1[avg-of-1]/is_correct_strict"] == approx(50.0)
        # Filtered: only task 0 (correct), so 100%.
        assert metrics["filtered_pass@1[avg-of-1]/is_correct_strict"] == approx(100.0)

    def test_filtered_metrics_disabled(self) -> None:
        server = _make_server(report_filtered_metrics=False)
        tasks = [
            [
                _verify_record(
                    expected_answer="Paris",
                    extracted_answer="Paris",
                    is_correct_strict_score=1.0,
                ),
            ],
        ]
        metrics = server.compute_metrics(tasks)
        # No filtered_* keys when disabled.
        assert all(not k.startswith("filtered_") for k in metrics)

    def test_filtered_num_tasks_zero(self) -> None:
        server = _make_server(report_filtered_metrics=True)
        tasks = [
            [
                _verify_record(
                    expected_answer="John Doe Smith",
                    extracted_answer="x",
                    gt_should_remove=True,
                ),
            ],
        ]
        metrics = server.compute_metrics(tasks)
        assert metrics["filtered_num_tasks"] == 0
        assert metrics["unfiltered_num_tasks"] == 1

    def test_task_should_remove_helper(self) -> None:
        ok_task = [_verify_record(expected_answer="Paris", extracted_answer="Paris")]
        bad_task = [_verify_record(expected_answer="A B C", extracted_answer="x", gt_should_remove=True)]
        assert _task_should_remove(ok_task) is False
        assert _task_should_remove(bad_task) is True

    def test_get_key_metrics_picks_highest_k(self) -> None:
        server = _make_server()
        agent_metrics = {
            "mean/input_tokens": 100.0,
            "mean/output_tokens": 50.0,
            "pass@1/is_correct_strict": 70.0,
            "pass@2/is_correct_strict": 80.0,
            "pass@1[avg-of-2]/is_correct_strict": 75.0,
            "majority@2/is_correct_strict": 78.0,
            "filtered_pass@2/is_correct_strict": 82.0,
        }
        key = server.get_key_metrics(agent_metrics)
        assert "mean/input_tokens" in key
        assert "mean/output_tokens" in key
        assert "pass@2/is_correct_strict" in key
        assert "pass@1[avg-of-2]/is_correct_strict" in key
        assert "majority@2/is_correct_strict" in key
        assert "filtered_pass@2/is_correct_strict" in key

    def test_compute_metrics_empty_tasks(self) -> None:
        server = _make_server()
        assert server.compute_metrics([]) == {
            "unfiltered_num_tasks": 0,
            "filtered_num_tasks": 0,
        }


class TestServerInstantiation:
    def test_default_filtered_metrics_on(self) -> None:
        server = _make_server()
        assert server.config.report_filtered_metrics is True

    def test_custom_disable_filter(self) -> None:
        server = _make_server(report_filtered_metrics=False)
        assert server.config.report_filtered_metrics is False


class TestUnicodeNormalization:
    def test_em_dash_normalized(self) -> None:
        # An em dash in the model output should be matched against a hyphen
        # alternative in the GT alternatives.
        assert is_correct(["co-pilot"], "the co—pilot of the plane") is True

    def test_nbsp_normalized(self) -> None:
        # Non-breaking spaces in the model output should be matched as plain
        # spaces.
        assert is_correct(["foo bar"], "foo bar") is True


class TestScoreFn:
    def test_score_fn_filters_unknown_keys(self) -> None:
        # _hotpotqa_score_fn only forwards the four canonical channels.
        scores = HotpotQAQAResourcesServer._hotpotqa_score_fn(
            {
                "answer_em": 1.0,
                "answer_f1": 0.5,
                "is_correct": 1.0,
                "is_correct_strict": 0.0,
                "extra_field": 99.0,
            }
        )
        assert set(scores) == {"answer_em", "answer_f1", "is_correct", "is_correct_strict"}
        assert scores["answer_em"] == 1.0
        assert scores["is_correct_strict"] == 0.0

    def test_score_fn_handles_missing_keys(self) -> None:
        # Tolerates a partial dict.
        scores = HotpotQAQAResourcesServer._hotpotqa_score_fn({"answer_em": 1.0})
        assert scores == {"answer_em": 1.0}
