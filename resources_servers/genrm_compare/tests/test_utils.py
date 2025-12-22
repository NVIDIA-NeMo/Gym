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
import pytest
from pytest import approx

from resources_servers.genrm_compare.utils import (
    GenRMOutputParseError,
    aggregate_scores,
    extract_from_response_obj,
    extract_output_text,
    generate_comparison_pairs,
    parse_genrm_output,
    apply_length_bonuses,
    EMPTY_OUTPUT_PLACEHOLDER,
)


class TestGenerateComparisonPairs:
    """Tests for generate_comparison_pairs function."""

    def test_circular_strategy_3_responses(self) -> None:
        """Circular strategy with 3 responses: (0,1), (1,2), (2,0)."""
        pairs = generate_comparison_pairs("circular", 3)
        assert pairs == [(0, 1), (1, 2), (2, 0)]

    def test_circular_strategy_4_responses(self) -> None:
        """Circular strategy with 4 responses: (0,1), (1,2), (2,3), (3,0)."""
        pairs = generate_comparison_pairs("circular", 4)
        assert pairs == [(0, 1), (1, 2), (2, 3), (3, 0)]

    def test_circular_strategy_2_responses(self) -> None:
        """Circular strategy with 2 responses: (0,1), (1,0)."""
        pairs = generate_comparison_pairs("circular", 2)
        assert pairs == [(0, 1), (1, 0)]

    def test_all_pairs_strategy_3_responses(self) -> None:
        """All pairs strategy with 3 responses: C(3,2) = 3 pairs."""
        pairs = generate_comparison_pairs("all_pairs", 3)
        assert pairs == [(0, 1), (0, 2), (1, 2)]

    def test_all_pairs_strategy_4_responses(self) -> None:
        """All pairs strategy with 4 responses: C(4,2) = 6 pairs."""
        pairs = generate_comparison_pairs("all_pairs", 4)
        assert pairs == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    def test_unsupported_strategy_raises(self) -> None:
        """Unsupported strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown comparison strategy"):
            generate_comparison_pairs("unknown", 3)

    def test_less_than_2_responses_raises(self) -> None:
        """Less than 2 responses raises ValueError."""
        with pytest.raises(ValueError, match="Need at least 2 responses"):
            generate_comparison_pairs("circular", 1)


class TestParseGenRMOutput:
    """Tests for parse_genrm_output function."""

    def test_valid_json_fenced(self) -> None:
        """Parse JSON from fenced code block."""
        output = """Here's my evaluation:
```json
{"score_1": 4, "score_2": 3, "ranking": 2}
```
"""
        score_1, score_2, ranking = parse_genrm_output(output, 3.0, 3.5)
        assert score_1 == approx(4.0)
        assert score_2 == approx(3.0)
        assert ranking == approx(2.0)

    def test_valid_json_unfenced(self) -> None:
        """Parse JSON from unfenced block."""
        output = 'The result is {"score_1": 5, "score_2": 2, "ranking": 1}'
        score_1, score_2, ranking = parse_genrm_output(output, 3.0, 3.5)
        assert score_1 == approx(5.0)
        assert score_2 == approx(2.0)
        assert ranking == approx(1.0)

    def test_partial_json_uses_defaults(self) -> None:
        """Missing keys use default values."""
        output = '{"score_1": 4}'
        score_1, score_2, ranking = parse_genrm_output(output, 3.0, 3.5)
        assert score_1 == approx(4.0)
        assert score_2 == approx(3.0)  # default
        assert ranking == approx(3.5)  # default

    def test_no_json_returns_defaults(self) -> None:
        """No JSON returns all defaults."""
        output = "This is just plain text without any JSON."
        score_1, score_2, ranking = parse_genrm_output(output, 3.0, 3.5)
        assert score_1 == approx(3.0)
        assert score_2 == approx(3.0)
        assert ranking == approx(3.5)

    def test_invalid_json_returns_defaults(self) -> None:
        """Invalid JSON returns defaults."""
        output = '{"score_1": invalid}'
        score_1, score_2, ranking = parse_genrm_output(output, 3.0, 3.5)
        assert score_1 == approx(3.0)
        assert score_2 == approx(3.0)
        assert ranking == approx(3.5)

    def test_raise_on_fail(self) -> None:
        """raise_on_fail=True raises GenRMOutputParseError."""
        output = "No JSON here"
        with pytest.raises(GenRMOutputParseError):
            parse_genrm_output(output, 3.0, 3.5, raise_on_fail=True)

    def test_multiple_json_uses_last_valid(self) -> None:
        """Multiple JSON blocks uses the last valid one."""
        output = '{"score_1": 1} more text {"score_1": 5, "score_2": 4, "ranking": 3}'
        score_1, score_2, ranking = parse_genrm_output(output, 3.0, 3.5)
        assert score_1 == approx(5.0)
        assert score_2 == approx(4.0)
        assert ranking == approx(3.0)

    def test_pretty_printed_json(self) -> None:
        """Parse pretty-printed JSON."""
        output = """
{
    "score_1": 4,
    "score_2": 3,
    "ranking": 2
}
"""
        score_1, score_2, ranking = parse_genrm_output(output, 3.0, 3.5)
        assert score_1 == approx(4.0)
        assert score_2 == approx(3.0)
        assert ranking == approx(2.0)


class TestExtractFromResponseObj:
    """Tests for extract_from_response_obj function."""

    def test_extract_output_text_only(self) -> None:
        """Extract output text from message type."""
        response_obj = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "The answer is 42."}
                    ]
                }
            ]
        }
        reasoning, output = extract_from_response_obj(response_obj)
        assert reasoning == ""
        assert output == "The answer is 42."

    def test_extract_reasoning_and_output(self) -> None:
        """Extract both reasoning and output."""
        response_obj = {
            "output": [
                {
                    "type": "reasoning",
                    "summary": [{"text": "Let me think step by step."}]
                },
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "Final answer: 42"}
                    ]
                }
            ]
        }
        reasoning, output = extract_from_response_obj(response_obj)
        assert reasoning == "Let me think step by step."
        assert output == "Final answer: 42"

    def test_empty_response_obj(self) -> None:
        """Empty response object returns empty strings."""
        response_obj = {}
        reasoning, output = extract_from_response_obj(response_obj)
        assert reasoning == ""
        assert output == ""

    def test_non_dict_response_obj(self) -> None:
        """Non-dict response object returns empty strings."""
        reasoning, output = extract_from_response_obj("not a dict")
        assert reasoning == ""
        assert output == ""


class TestExtractOutputText:
    """Tests for extract_output_text function."""

    def test_extract_text(self) -> None:
        """Extract output text from response object."""
        response_obj = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Hello world"}]
                }
            ]
        }
        text = extract_output_text(response_obj)
        assert text == "Hello world"

    def test_empty_returns_placeholder(self) -> None:
        """Empty output returns placeholder."""
        response_obj = {"output": []}
        text = extract_output_text(response_obj)
        assert text == EMPTY_OUTPUT_PLACEHOLDER

    def test_whitespace_only_returns_placeholder(self) -> None:
        """Whitespace-only output returns placeholder."""
        response_obj = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "   "}]
                }
            ]
        }
        text = extract_output_text(response_obj)
        assert text == EMPTY_OUTPUT_PLACEHOLDER


class TestAggregateScores:
    """Tests for aggregate_scores function."""

    def _make_response_obj(self, output_text: str) -> dict:
        """Helper to create a minimal Response API object."""
        return {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": output_text}]
                }
            ]
        }

    def test_simple_tiebreaker_no_tie(self) -> None:
        """Aggregate scores without tiebreaker."""
        # Two comparisons: (0,1) and (1,2) with clear winners
        comparison_results = [
            (5.0, 3.0, 2.0),  # Response 0 wins against 1
            (4.0, 2.0, 1.0),  # Response 1 wins against 2
        ]
        comparison_metadata = [(0, 1, 0), (1, 2, 0)]
        response_objs = [
            self._make_response_obj("Answer 0"),
            self._make_response_obj("Answer 1"),
            self._make_response_obj("Answer 2"),
        ]

        final_scores, metrics, base_scores, bonuses = aggregate_scores(
            comparison_results=comparison_results,
            comparison_metadata=comparison_metadata,
            response_objs=response_objs,
            aggregator_method="simple_tiebreaker",
            default_score=3.0,
            reasoning_bonus=0.0,
            answer_bonus=0.0,
            top_percentile=0.2,
            group_reasoning_length_penalty_coeff=0.0,
            group_answer_length_penalty_coeff=0.0,
        )

        # Response 0: score 5 from one comparison -> avg = 5
        # Response 1: score 3 + 4 from two comparisons -> avg = 3.5
        # Response 2: score 2 from one comparison -> avg = 2
        assert final_scores[0] == approx(5.0)
        assert final_scores[1] == approx(3.5)
        assert final_scores[2] == approx(2.0)

    def test_simple_tiebreaker_with_tie(self) -> None:
        """Aggregate scores with tiebreaker activation."""
        # Tied scores with ranking deciding winner
        comparison_results = [
            (3.0, 3.0, 2.0),  # Tied, ranking=2 means response_1 (idx 0) is better
        ]
        comparison_metadata = [(0, 1, 0)]
        response_objs = [
            self._make_response_obj("Answer 0"),
            self._make_response_obj("Answer 1"),
        ]

        final_scores, metrics, base_scores, bonuses = aggregate_scores(
            comparison_results=comparison_results,
            comparison_metadata=comparison_metadata,
            response_objs=response_objs,
            aggregator_method="simple_tiebreaker",
            default_score=3.0,
            reasoning_bonus=0.0,
            answer_bonus=0.0,
            top_percentile=0.2,
            group_reasoning_length_penalty_coeff=0.0,
            group_answer_length_penalty_coeff=0.0,
        )

        # Tiebreaker: adjustment = 3.5 - 2.0 = 1.5
        # Response 0: 3.0 + 1.5 = 4.5
        # Response 1: 3.0 - 1.5 = 1.5
        assert final_scores[0] == approx(4.5)
        assert final_scores[1] == approx(1.5)
        assert metrics["tiebreak_usage_rate"] == approx(1.0)

    def test_unsupported_aggregator_raises(self) -> None:
        """Unsupported aggregator method raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported aggregator_method"):
            aggregate_scores(
                comparison_results=[],
                comparison_metadata=[],
                response_objs=[],
                aggregator_method="unknown",
                default_score=3.0,
                reasoning_bonus=0.0,
                answer_bonus=0.0,
                top_percentile=0.2,
                group_reasoning_length_penalty_coeff=0.0,
                group_answer_length_penalty_coeff=0.0,
            )

    def test_no_comparisons_returns_default(self) -> None:
        """No comparisons for a response returns default score."""
        response_objs = [
            self._make_response_obj("Answer 0"),
        ]

        final_scores, metrics, base_scores, bonuses = aggregate_scores(
            comparison_results=[],
            comparison_metadata=[],
            response_objs=response_objs,
            aggregator_method="simple_tiebreaker",
            default_score=3.0,
            reasoning_bonus=0.0,
            answer_bonus=0.0,
            top_percentile=0.2,
            group_reasoning_length_penalty_coeff=0.0,
            group_answer_length_penalty_coeff=0.0,
        )

        assert final_scores[0] == approx(3.0)


class TestApplyLengthBonuses:
    """Tests for apply_length_bonuses function."""

    def _make_response_obj(self, output_text: str, reasoning_text: str = "") -> dict:
        """Helper to create a Response API object with reasoning and output."""
        output = []
        if reasoning_text:
            output.append({
                "type": "reasoning",
                "summary": [{"text": reasoning_text}]
            })
        output.append({
            "type": "message",
            "content": [{"type": "output_text", "text": output_text}]
        })
        return {"output": output}

    def test_answer_bonus_shortest_among_top_gets_bonus(self) -> None:
        """Bonus goes to the shortest answer within the top scorers; others unchanged."""
        scores = [5.0, 5.0, 4.0]  # Top scorers: idx 0 and 1 (both 5.0)
        response_objs = [
            self._make_response_obj("a"),                   # 1 char (shortest and top)
            self._make_response_obj("much longer texttt"),    # longer but also top scorer
            self._make_response_obj("mid length here"),     # not top scorer, longer than idx0
        ]

        adjusted, bonuses = apply_length_bonuses(
            scores=scores,
            response_objs=response_objs,
            reasoning_bonus=0.0,
            answer_bonus=0.5,
            top_percentile=0.67,  # Top ~2 of 3
            group_reasoning_length_penalty_coeff=0.0,
            group_answer_length_penalty_coeff=0.0,
        )

        # Only idx 0 is shortest among the top scorers -> gets the bonus
        assert adjusted == approx([5.5, 5.0, 4.0])
        assert bonuses == approx([0.5, 0.0, 0.0])

    def test_no_bonus_when_disabled(self) -> None:
        """No bonuses applied when all bonus configs are 0."""
        scores = [5.0, 4.0]
        response_objs = [
            self._make_response_obj("short"),
            self._make_response_obj("much longer text here"),
        ]

        adjusted, bonuses = apply_length_bonuses(
            scores=scores,
            response_objs=response_objs,
            reasoning_bonus=0.0,
            answer_bonus=0.0,
            top_percentile=0.2,
            group_reasoning_length_penalty_coeff=0.0,
            group_answer_length_penalty_coeff=0.0,
        )

        assert adjusted[0] == approx(5.0)
        assert adjusted[1] == approx(4.0)
        assert bonuses == [0.0, 0.0]
