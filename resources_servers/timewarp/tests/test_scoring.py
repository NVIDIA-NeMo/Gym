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
"""TimeWarp's verifier tests (src/tests/timewarp/test_evaluators.py at commit 4978e69), ported.

Every case scores a synthetic final answer through ``TimewarpVerifier.verify``, the path a
rollout takes, so the expectations double as a regression check that the port matches upstream.
The judge-selection tests are not ported: Gym chooses the judge through its model-server config.
"""

from decimal import Decimal

import pytest

from nemo_gym.openai_utils import NeMoGymResponse
from resources_servers.timewarp.app import TimewarpVerifier, TimewarpVerifyRequest
from resources_servers.timewarp.normalization import extract_numbers, find_entry, first_sentence, normalize_text
from resources_servers.timewarp.scoring import DETERMINISTIC_EVAL_TYPES, parse_judge_verdict, strip_thinking


def _response(answer: str) -> NeMoGymResponse:
    return NeMoGymResponse.model_validate(
        {
            "id": "response",
            "created_at": 0,
            "model": "test",
            "object": "response",
            "output": [
                {
                    "id": "message",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": answer, "annotations": []}],
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        }
    )


async def score(eval_block: dict, answer: str, intent: str = "test intent") -> float:
    request = TimewarpVerifyRequest(
        responses_create_params={"input": [{"role": "user", "content": intent}]},
        response=_response(answer),
        verifier_metadata=eval_block,
        intent=intent,
    )
    return (await TimewarpVerifier().verify(request)).reward


def string_eval(**references) -> dict:
    return {"eval_types": ["string_match"], "reference_answers": references}


def number_eval(scope: str = "full", **spec) -> dict:
    spec.setdefault("scope", scope)
    return {"eval_types": ["number_match"], "reference_answers": {"number_match": spec}}


def list_eval(**spec) -> dict:
    return {"eval_types": ["list_match"], "reference_answers": {"list_match": spec}}


class TestNormalizeText:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("  Biology  ", "biology"),
            ('"Biology."', "biology"),
            ("**Biology**", "biology"),
            ("`Biology`", "biology"),
            ("Beyoncé", "beyonce"),
            ("Quest Lumaflex™ Band", "quest lumaflex band"),
            ("multi\n  line\ttext", "multi line text"),
            ("“smart quotes”", "smart quotes"),
            ("$9.99", "$9.99"),
            ("45%", "45%"),
            (None, ""),
        ],
    )
    def test_canonicalizes(self, raw, expected):
        assert normalize_text(raw) == expected

    def test_preserves_internal_punctuation(self):
        assert normalize_text("The U.S. government") == "the u.s. government"


class TestFirstSentence:
    def test_splits_on_terminator(self):
        assert first_sentence("Yes. It is mentioned in Biology.") == "Yes"

    def test_does_not_split_decimals(self):
        assert first_sentence("The price is 9.99 dollars") == "The price is 9.99 dollars"

    def test_splits_on_newline(self):
        assert first_sentence("No\nExplanation follows") == "No"

    @pytest.mark.parametrize(
        "raw",
        [
            "**Yes.** There is no other article.",
            '"Yes." There is no other article.',
            "*Yes.* There is no other article.",
            "`Yes.` There is no other article.",
        ],
    )
    def test_splits_through_closing_markdown(self, raw):
        assert normalize_text(first_sentence(raw)) == "yes"


class TestWordBoundaryContainment:
    @pytest.mark.parametrize(
        "answer, reference, matches",
        [
            ("The answer is 10", "10", True),
            ("The answer is 100", "10", False),
            ("It happened in 2010", "10", False),
            ("There are 10, not 11", "10", True),
            ("The article is in the north wing", "no", False),
            ("No, it is not listed", "no", True),
            ("It is not listed", "no", False),
            ("Send an e-mail", "e mail", True),
            ("Send an e mail", "e-mail", True),
            ("Hong Kong's population", "hong kong", True),
            ("**Biology** is the answer", "biology", True),
        ],
    )
    def test_boundaries(self, answer, reference, matches):
        assert (find_entry(answer, reference) is not None) is matches


class TestExtractNumbers:
    @pytest.mark.parametrize(
        "text, expected_subset",
        [
            ("$1,234.56", ["1234.56"]),
            ("Over 57.7 million people", ["57700000.0"]),
            ("7,000,000", ["7000000"]),
            ("Thirteen articles", ["13"]),
            ("one hundred two", ["102"]),
            ("The 13th of May", ["13"]),
            ("-5 degrees", ["-5"]),
            ("no numbers here", []),
        ],
    )
    def test_extraction(self, text, expected_subset):
        found = extract_numbers(text)
        for expected in expected_subset:
            assert any(value == Decimal(expected) for value in found), (text, found)
        if not expected_subset:
            assert found == []

    def test_ambiguous_suffix_yields_both_readings(self):
        found = extract_numbers("a 5m wingspan")
        assert Decimal(5) in found and Decimal(5_000_000) in found


class TestStringMatch:
    async def test_exact_match_tolerates_formatting(self):
        block = string_eval(exact_match="Biology")
        assert await score(block, "  **Biology.**  ") == 1.0
        assert await score(block, "Chemistry") == 0.0

    async def test_exact_match_rejects_verbose_answers(self):
        assert await score(string_eval(exact_match="Biology"), "The answer is Biology") == 0.0

    async def test_must_include_requires_all_entries(self):
        block = string_eval(must_include=["biology", "physics"])
        assert await score(block, "Both Biology and Physics mention it.") == 1.0
        assert await score(block, "Only Biology mentions it.") == 0.0

    async def test_or_separator_tolerates_missing_spaces(self):
        block = string_eval(must_include=["kangaroos|OR|kangaroo"])
        assert await score(block, "I found kangaroos.") == 1.0
        assert await score(block, "I found wombats.") == 0.0

    async def test_must_exclude(self):
        block = string_eval(must_include=["biology"], must_exclude=["both", "neither"])
        assert await score(block, "It is mentioned in Biology only.") == 1.0
        assert await score(block, "It is mentioned in both articles, Biology and Physics.") == 0.0

    async def test_yes_no_with_first_sentence_scope(self):
        block = string_eval(must_include=["yes"], must_exclude=["no"], scope="first_sentence")
        assert await score(block, "Yes. There is no other article covering it.") == 1.0
        assert await score(block, "**Yes.** There is no other article covering it.") == 1.0
        assert await score(block, "No. The article does not exist.") == 0.0
        assert await score(block, "Yes, although it is not in the related pages list.") == 1.0
        unscoped = string_eval(must_include=["yes"], must_exclude=["no"])
        assert await score(unscoped, "Yes. There is no other article covering it.") == 0.0

    async def test_regex_entry(self):
        block = string_eval(must_include=["^yes.*$"], scope="first_sentence")
        assert await score(block, "Yes, both articles mention it. Details follow.") == 1.0
        assert await score(block, "No, neither does.") == 0.0

    async def test_numeric_lookalike_is_rejected(self):
        block = string_eval(must_include=["10"])
        assert await score(block, "There are 100 articles.") == 0.0
        assert await score(block, "There are 10 articles.") == 1.0

    async def test_empty_spec_raises(self):
        with pytest.raises(ValueError, match="at least one of"):
            await score(string_eval(), "anything")


class TestNumberMatch:
    async def test_formatting_independence(self):
        block = number_eval(value=7000000)
        for answer in ["7,000,000", "7000000", "7 million", "The difference is 7 million people."]:
            assert await score(block, answer) == 1.0, answer
        assert await score(block, "70 million") == 0.0

    async def test_relative_tolerance_for_hedged_golds(self):
        block = number_eval(value=7000000, rel_tolerance=0.1)
        assert await score(block, "around 7.2 million") == 1.0
        assert await score(block, "about 9 million") == 0.0

    async def test_currency(self):
        block = number_eval(value=9.99, abs_tolerance=0.01)
        assert await score(block, "It costs $9.99") == 1.0
        assert await score(block, "It costs $99.90") == 0.0

    async def test_multiple_required_values(self):
        block = number_eval(values=[5, 1.8])
        assert await score(block, "The wingspan is 5 metres and the height 1.8 metres.") == 1.0
        assert await score(block, "The wingspan is 5 metres.") == 0.0

    async def test_missing_value_raises(self):
        with pytest.raises(ValueError, match="'value' or 'values'"):
            await score(number_eval(), "13")


class TestListMatch:
    ANIMALS = [["koalas |OR| koala"], ["kangaroos |OR| kangaroo"], ["wombats |OR| wombat"]]

    async def test_unordered_requires_every_item(self):
        block = list_eval(items=self.ANIMALS)
        assert await score(block, "Wombats, koalas and kangaroos live there.") == 1.0
        assert await score(block, "Koalas and kangaroos live there.") == 0.0

    async def test_ordered_enforces_sequence(self):
        block = list_eval(items=[["periodic table"], ["atomic number"], ["proton"]], ordered=True)
        assert await score(block, "Periodic table, then atomic number, then proton.") == 1.0
        assert await score(block, "Proton, then atomic number, then periodic table.") == 0.0

    async def test_forbidden_entries(self):
        block = list_eval(items=[["koalas"]], forbidden=["dingoes"])
        assert await score(block, "Koalas.") == 1.0
        assert await score(block, "Koalas and dingoes.") == 0.0

    async def test_empty_items_raises(self):
        with pytest.raises(ValueError, match="non-empty 'items'"):
            await score(list_eval(items=[]), "anything")


class TestRouter:
    async def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            await score({"eval_types": ["program_html"], "reference_answers": {}}, "x")

    async def test_empty_eval_types_raises(self):
        with pytest.raises(ValueError, match="at least one evaluator"):
            await score({"eval_types": [], "reference_answers": {}}, "x")

    async def test_combined_types_are_anded(self):
        block = {
            "eval_types": ["string_match", "number_match"],
            "reference_answers": {"must_include": ["food and drug administration"], "number_match": {"value": 1906}},
        }
        assert await score(block, "The Food and Drug Administration was formed in 1906.") == 1.0
        assert await score(block, "The Food and Drug Administration was formed in 1907.") == 0.0
        assert await score(block, "The agency was formed in 1906.") == 0.0

    async def test_legacy_exact_match_type_lowercases_and_strips_quotes(self):
        block = {"eval_types": ["exact_match"], "reference_answers": {"exact_match": "Eric Adams"}}
        assert await score(block, '"eric   ADAMS"') == 1.0
        assert await score(block, "Eric Adams is the mayor") == 0.0

    def test_deterministic_type_list_excludes_llm_judge(self):
        assert "llm_judge" not in DETERMINISTIC_EVAL_TYPES
        assert set(DETERMINISTIC_EVAL_TYPES) >= {"string_match", "number_match", "list_match"}


class TestJudgeVerdict:
    @pytest.mark.parametrize(
        "verdict, expected",
        [
            ("correct", 1.0),
            ("Correct", 1.0),
            ("incorrect", 0.0),
            ("partially correct", 0.0),
            ("The answer is incorrect.", 0.0),
            ("'correct'", 1.0),
            ("I cannot tell", 0.0),
        ],
    )
    def test_negative_verdicts_win_over_the_correct_substring(self, verdict, expected):
        assert parse_judge_verdict(verdict) == expected


class TestStripThinking:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("<think>Texas? No.</think>Alaska.", "Alaska."),
            ("<thinking>hmm</thinking> Alaska", "Alaska"),
            ("reasoning the template opened</think>Alaska", "Alaska"),
            ("Alaska", "Alaska"),
        ],
    )
    def test_reasoning_never_counts_as_the_answer(self, raw, expected):
        assert strip_thinking(raw) == expected
