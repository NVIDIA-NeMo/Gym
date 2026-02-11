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
    EMPTY_OUTPUT_PLACEHOLDER,
    GenRMOutputParseError,
    aggregate_scores,
    extract_from_response_obj,
    extract_output_text,
    generate_comparison_pairs,
    parse_genrm_output,
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
        """No JSON found returns default values."""
        output = "I think response 1 is better"
        score_1, score_2, ranking = parse_genrm_output(output, 2.5, 4.0)
        assert score_1 == approx(2.5)
        assert score_2 == approx(2.5)
        assert ranking == approx(4.0)

    def test_invalid_json_returns_defaults(self) -> None:
        """Invalid JSON returns default values."""
        output = '{"score_1": "not a number"}'
        score_1, score_2, ranking = parse_genrm_output(output, 3.0, 3.5)
        assert score_1 == approx(3.0)
        assert score_2 == approx(3.0)
        assert ranking == approx(3.5)

    def test_raise_on_fail_raises_exception(self) -> None:
        """raise_on_fail=True raises GenRMOutputParseError."""
        output = "No JSON here"
        with pytest.raises(GenRMOutputParseError):
            parse_genrm_output(output, 3.0, 3.5, raise_on_fail=True)


class TestExtractFromResponseObj:
    """Tests for extract_from_response_obj and extract_output_text."""

    def test_extract_output_text_from_message(self) -> None:
        """Extract output text from message type."""
        response_obj = {"output": [{"type": "message", "content": [{"type": "output_text", "text": "Hello world"}]}]}
        output = extract_output_text(response_obj)
        assert output == "Hello world"

    def test_extract_reasoning_and_output(self) -> None:
        """Extract both reasoning and output."""
        response_obj = {
            "output": [
                {
                    "type": "reasoning",
                    "summary": [
                        {"text": "Let me think... ", "type": "summary_text"},
                        {"text": "The answer is clear.", "type": "summary_text"},
                    ],
                },
                {"type": "message", "content": [{"type": "output_text", "text": "Paris"}]},
            ]
        }
        reasoning, output = extract_from_response_obj(response_obj)
        assert reasoning == "Let me think... The answer is clear."
        assert output == "Paris"

    def test_extract_empty_returns_placeholder(self) -> None:
        """Empty output returns placeholder."""
        response_obj = {"output": []}
        output = extract_output_text(response_obj)
        assert output == EMPTY_OUTPUT_PLACEHOLDER


class TestAggregateScores:
    """Tests for aggregate_scores function."""

    def test_simple_aggregation_no_ties(self) -> None:
        """Simple aggregation without ties."""
        # Two responses: R0 vs R1
        # R0 gets score 4, R1 gets score 2
        comparison_results = [(4.0, 2.0, 1.0)]  # R0 much better
        comparison_metadata = [(0, 1, 0)]  # pair (0,1), judge 0
        response_objs = [{"output": []}, {"output": []}]

        rewards, metrics, base_rewards, bonuses = aggregate_scores(
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

        assert rewards[0] == approx(4.0)
        assert rewards[1] == approx(2.0)

    def test_tiebreaker_applied(self) -> None:
        """Tiebreaker uses ranking when scores are equal."""
        # Both get score 3, but ranking=2 means R0 is better
        comparison_results = [(3.0, 3.0, 2.0)]  # tie, but R0 better (ranking < 3.5)
        comparison_metadata = [(0, 1, 0)]
        response_objs = [{"output": []}, {"output": []}]

        rewards, metrics, base_rewards, bonuses = aggregate_scores(
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

        # Adjustment = 3.5 - 2.0 = 1.5
        # score_1 = 3.0 + 1.5 = 4.5
        # score_2 = 3.0 - 1.5 = 1.5
        assert rewards[0] == approx(4.5)
        assert rewards[1] == approx(1.5)
        assert metrics["tiebreak_usage_rate"] == approx(1.0)

    def test_circular_strategy_integration(self) -> None:
        """Test circular strategy with 3 responses."""
        # Circular: (0,1), (1,2), (2,0)
        # R0 vs R1: R0=4, R1=2
        # R1 vs R2: R1=3, R2=5
        # R2 vs R0: R2=3, R0=4
        comparison_results = [
            (4.0, 2.0, 1.0),  # R0 > R1
            (3.0, 5.0, 5.0),  # R2 > R1
            (3.0, 4.0, 4.0),  # R0 > R2
        ]
        comparison_metadata = [(0, 1, 0), (1, 2, 0), (2, 0, 0)]
        response_objs = [{"output": []}, {"output": []}, {"output": []}]

        rewards, metrics, base_rewards, bonuses = aggregate_scores(
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

        # R0: (4+4)/2 = 4.0
        # R1: (2+3)/2 = 2.5
        # R2: (5+3)/2 = 4.0
        assert rewards[0] == approx(4.0)
        assert rewards[1] == approx(2.5)
        assert rewards[2] == approx(4.0)
