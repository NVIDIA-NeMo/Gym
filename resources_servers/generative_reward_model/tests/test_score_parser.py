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

import json

from pytest import approx

from resources_servers.generative_reward_model.score_parser import extract_scores


def _make_output(rubric_evals, overall, fenced=False):
    """Helper to build a model output string."""
    obj = {"rubric_evaluations": rubric_evals, "overall": overall}
    text = json.dumps(obj)
    if fenced:
        return f"```json\n{text}\n```"
    return text


def _rubric(rubric_id, s1, s2, ranking):
    return {
        "rubric_id": rubric_id,
        "response_1_analysis": f"Analysis R1 for rubric {rubric_id}",
        "response_2_analysis": f"Analysis R2 for rubric {rubric_id}",
        "score_1": s1,
        "score_2": s2,
        "ranking": ranking,
    }


def _overall(s1, s2, ranking):
    return {
        "response_1_analysis": "Overall R1 analysis",
        "response_2_analysis": "Overall R2 analysis",
        "score_1": s1,
        "score_2": s2,
        "ranking": ranking,
    }


class TestExtractScores:
    def test_single_rubric_fenced(self) -> None:
        output = _make_output([_rubric(1, 5, 2, 1)], _overall(5, 2, 1), fenced=True)
        result = extract_scores(output)
        assert result is not None
        assert result.score_1 == approx(5.0)
        assert result.score_2 == approx(2.0)
        assert result.ranking == approx(1.0)
        assert len(result.rubric_scores) == 1
        assert result.rubric_scores[0].rubric_id == 1
        assert result.rubric_scores[0].score_1 == approx(5.0)

    def test_single_rubric_bare(self) -> None:
        output = _make_output([_rubric(1, 4, 3, 2)], _overall(4, 3, 2))
        result = extract_scores(output)
        assert result is not None
        assert result.score_1 == approx(4.0)
        assert result.ranking == approx(2.0)
        assert len(result.rubric_scores) == 1

    def test_multiple_rubrics(self) -> None:
        output = _make_output(
            [_rubric(1, 5, 1, 1), _rubric(2, 3, 4, 5), _rubric(3, 4, 4, 3)],
            _overall(4, 3, 2),
        )
        result = extract_scores(output)
        assert result is not None
        assert result.score_1 == approx(4.0)
        assert result.score_2 == approx(3.0)
        assert result.ranking == approx(2.0)
        assert len(result.rubric_scores) == 3
        assert result.rubric_scores[0].rubric_id == 1
        assert result.rubric_scores[1].rubric_id == 2
        assert result.rubric_scores[2].score_1 == approx(4.0)

    def test_nested_json_with_surrounding_text(self) -> None:
        """Parser should handle nested JSON even without fences."""
        inner = _make_output([_rubric(1, 5, 2, 1)], _overall(5, 2, 1))
        output = f"Here is my evaluation:\n\n{inner}\n\nThat concludes my analysis."
        result = extract_scores(output)
        assert result is not None
        assert result.score_1 == approx(5.0)
        assert len(result.rubric_scores) == 1

    def test_no_json_returns_none(self) -> None:
        assert extract_scores("I cannot parse this.") is None

    def test_empty_string_returns_none(self) -> None:
        assert extract_scores("") is None

    def test_missing_overall_returns_none(self) -> None:
        obj = {"rubric_evaluations": [_rubric(1, 5, 2, 1)]}
        assert extract_scores(json.dumps(obj)) is None

    def test_missing_rubric_evaluations_returns_none(self) -> None:
        obj = {"overall": _overall(5, 2, 1)}
        assert extract_scores(json.dumps(obj)) is None

    def test_empty_rubric_evaluations_returns_none(self) -> None:
        obj = {"rubric_evaluations": [], "overall": _overall(5, 2, 1)}
        assert extract_scores(json.dumps(obj)) is None

    def test_rubric_missing_score_key_returns_none(self) -> None:
        bad_rubric = {"rubric_id": 1, "score_1": 5, "ranking": 1}  # missing score_2
        obj = {"rubric_evaluations": [bad_rubric], "overall": _overall(5, 2, 1)}
        assert extract_scores(json.dumps(obj)) is None

    def test_overall_missing_score_key_returns_none(self) -> None:
        bad_overall = {"score_1": 5, "ranking": 1}  # missing score_2
        obj = {"rubric_evaluations": [_rubric(1, 5, 2, 1)], "overall": bad_overall}
        assert extract_scores(json.dumps(obj)) is None

    def test_invalid_score_value_returns_none(self) -> None:
        bad_rubric = _rubric(1, "bad", 2, 1)
        obj = {"rubric_evaluations": [bad_rubric], "overall": _overall(5, 2, 1)}
        assert extract_scores(json.dumps(obj)) is None

    def test_old_flat_format_returns_none(self) -> None:
        """Old flat format without rubric_evaluations/overall should fail."""
        output = json.dumps(
            {
                "response_1_analysis": "A",
                "response_2_analysis": "B",
                "score_1": 5,
                "score_2": 1,
                "ranking": 1,
            }
        )
        assert extract_scores(output) is None

    def test_repeated_verdict_blocks_return_none(self) -> None:
        """A degenerate policy repeating its verdict block must be a parse failure."""
        block = _make_output([_rubric(1, 5, 2, 1)], _overall(5, 2, 1), fenced=True)
        assert extract_scores(f"{block}</think>{block}") is None
        assert extract_scores("</think>".join([block] * 15)) is None

    def test_repeated_bare_dicts_return_none(self) -> None:
        one = _make_output([_rubric(1, 5, 2, 1)], _overall(5, 2, 1))
        two = _make_output([_rubric(1, 3, 3, 4)], _overall(3, 3, 4))
        assert extract_scores(f"{one}\n\n{two}") is None

    def test_extra_json_dict_in_prose_returns_none(self) -> None:
        """Any second valid JSON dict alongside the verdict is a parse failure."""
        verdict = _make_output([_rubric(1, 5, 2, 1)], _overall(5, 2, 1), fenced=True)
        assert extract_scores(f'Note: {{"foo": 1}}\n{verdict}') is None

    def test_trailing_truncated_repeat_still_parses(self) -> None:
        """A truncated (invalid) second block is not a valid dict, so the first parses."""
        block = _make_output([_rubric(1, 5, 2, 1)], _overall(5, 2, 1), fenced=True)
        truncated_repeat = block[: len(block) // 2]
        result = extract_scores(f"{block}</think>{truncated_repeat}")
        assert result is not None
        assert result.score_1 == approx(5.0)
