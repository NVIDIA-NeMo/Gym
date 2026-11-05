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

from types import SimpleNamespace

import pytest

from resources_servers.opencode_search.app import (
    SearchResourcesServerConfig,
    _last_assistant_text,
    parse_judge_verdict,
)


def _text_part(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="output_text", text=text)


def _assistant_message(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="message", role="assistant", content=[_text_part(text)])


class TestParseJudgeVerdict:
    def test_parses_yes(self):
        is_correct, extracted, parsed_ok = parse_judge_verdict(
            "extracted_final_answer: Paris\nreasoning: matches\ncorrect: yes\nconfidence: 90%"
        )
        assert (is_correct, extracted, parsed_ok) == (True, "Paris", True)

    def test_parses_no(self):
        is_correct, _, parsed_ok = parse_judge_verdict("extracted_final_answer: Berlin\ncorrect: no")
        assert parsed_ok is True
        assert is_correct is False

    def test_missing_verdict_is_unparsed(self):
        assert parse_judge_verdict("I think the answer looks right.") == (False, "", False)
        assert parse_judge_verdict("") == (False, "", False)

    def test_last_match_wins_over_think_block_echo(self):
        """A reasoning judge echoes the rubric mid-think; only the final verdict counts."""
        text = (
            "<think>the rubric says answer correct: yes if they match, "
            "but these differ</think>\n"
            "extracted_final_answer: Lyon\ncorrect: no"
        )
        is_correct, _, parsed_ok = parse_judge_verdict(text)
        assert parsed_ok is True
        assert is_correct is False

    def test_word_boundary_rejects_substring_match(self):
        """'correct: yesterday' is not a yes verdict."""
        assert parse_judge_verdict("correct: yesterday the model answered") == (False, "", False)

    def test_template_echo_is_unparsed(self):
        """A judge that repeats the template instead of grading must not score the row."""
        text = (
            "extracted_final_answer: The final exact answer extracted from the [response]. "
            "Put the extracted answer as 'None'...\ncorrect: yes"
        )
        assert parse_judge_verdict(text) == (False, "", False)

    def test_case_insensitive(self):
        is_correct, _, parsed_ok = parse_judge_verdict("Correct: YES")
        assert (is_correct, parsed_ok) == (True, True)


class TestLastAssistantText:
    def test_takes_last_assistant_message_not_concatenation(self):
        response = SimpleNamespace(
            output=[
                _assistant_message("let me search for that"),
                SimpleNamespace(type="function_call", role=None),
                _assistant_message("Exact Answer: 42"),
            ]
        )
        assert _last_assistant_text(response) == "Exact Answer: 42"

    def test_skips_trailing_reasoning_item(self):
        """Reasoning items carry no content; they must not mask the final answer."""
        response = SimpleNamespace(
            output=[
                _assistant_message("Exact Answer: 42"),
                SimpleNamespace(type="reasoning", role=None, summary=["thinking"]),
            ]
        )
        assert _last_assistant_text(response) == "Exact Answer: 42"

    def test_empty_output(self):
        assert _last_assistant_text(SimpleNamespace(output=[])) == ""
        assert _last_assistant_text(SimpleNamespace(output=None)) == ""


class TestJudgeRetryLadder:
    """The retry schedule is a WALL-CLOCK budget, not just a politeness delay.

    These assertions replace a TestJudgeBackoff that checked an exponential helper
    (`_judge_backoff_s`, 1/2/4/8/16 capped at 120). That helper no longer exists: the ladder was
    deliberately flattened to a fixed delay to match the legacy grader. The test outlived the
    code and broke collection for the whole module, so it is restated against what actually runs.
    """

    def test_ladder_is_fixed_delay_not_exponential(self):
        cfg = SearchResourcesServerConfig.model_fields
        assert cfg["judge_retry_delay_seconds"].default == 30.0
        assert cfg["judge_max_attempts"].default == 4

    def test_a_dead_judge_cannot_eat_the_wall(self):
        """One rollout against a black-hole judge must stay far under a 4h job wall.

        This is the property the ladder exists for. The schedule it replaced (15 attempts at 300s
        each) burned ~90 min per rollout, so a dead endpoint starved the run of completed rows
        while every sample still scored. Worst case is one full attempt at the per-attempt
        timeout, then each retry paying the delay plus the shorter retry timeout.
        """
        cfg = SearchResourcesServerConfig.model_fields
        attempts = cfg["judge_max_attempts"].default
        worst_case_s = cfg["judge_timeout_seconds"].default + (attempts - 1) * (
            cfg["judge_retry_delay_seconds"].default + cfg["judge_retry_timeout_seconds"].default
        )
        assert worst_case_s == pytest.approx(750.0)
        assert worst_case_s < 0.25 * 4 * 3600
