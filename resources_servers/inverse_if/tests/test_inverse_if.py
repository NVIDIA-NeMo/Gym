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

from resources_servers.inverse_if.app import (
    AggregationMode,
    InverseIFConfig,
    RubricEvaluation,
    _extract_verdict,
)
from resources_servers.inverse_if.dataset_preprocess import (
    _extract_rubrics,
    _normalise_rubric_item,
)


# ---------------------------------------------------------------------------
# Verdict extraction tests
# ---------------------------------------------------------------------------


class TestExtractVerdict:
    """Tests for JSON-based verdict extraction from judge responses."""

    def test_json_in_code_block(self):
        """Standard case: JSON inside a fenced code block."""
        text = 'Some analysis...\n```json\n{"result": "PASS", "explanation": "All criteria met."}\n```'
        verdict, explanation = _extract_verdict(text)
        assert verdict == "PASS"
        assert explanation == "All criteria met."

    def test_json_in_code_block_fail(self):
        """Standard case: FAIL verdict in a fenced code block."""
        text = '```json\n{"result": "FAIL", "explanation": "Missing exclamation mark."}\n```'
        verdict, explanation = _extract_verdict(text)
        assert verdict == "FAIL"
        assert explanation == "Missing exclamation mark."

    def test_json_without_code_block(self):
        """JSON object in plain text without fences."""
        text = 'Here is my evaluation:\n{"result": "PASS", "explanation": "Looks good."}'
        verdict, explanation = _extract_verdict(text)
        assert verdict == "PASS"
        assert explanation == "Looks good."

    def test_keyword_fallback_pass(self):
        """Fallback to keyword scanning when no valid JSON found."""
        text = "The student met all requirements.\nPASS"
        verdict, _ = _extract_verdict(text)
        assert verdict == "PASS"

    def test_keyword_fallback_fail(self):
        """Fallback to keyword scanning for FAIL."""
        text = "Criterion not satisfied.\nFAIL"
        verdict, _ = _extract_verdict(text)
        assert verdict == "FAIL"

    def test_default_to_fail(self):
        """When nothing is extractable, default to FAIL."""
        text = "I'm not sure what to say here."
        verdict, _ = _extract_verdict(text)
        assert verdict == "FAIL"

    def test_case_insensitive_json(self):
        """JSON result field should work with different casing."""
        text = '{"result": "pass", "explanation": "ok"}'
        verdict, _ = _extract_verdict(text)
        assert verdict == "PASS"

    def test_json_with_extra_text_around(self):
        """JSON embedded in surrounding prose."""
        text = (
            "Let me evaluate this carefully.\n\n"
            "The student wrote three sentences correctly.\n\n"
            '```json\n{"result": "PASS", "explanation": "All three sentences present."}\n```\n\n'
            "That concludes my evaluation."
        )
        verdict, explanation = _extract_verdict(text)
        assert verdict == "PASS"
        assert "three sentences" in explanation


# ---------------------------------------------------------------------------
# Rubric normalisation tests
# ---------------------------------------------------------------------------


class TestRubricNormalisation:
    """Tests for normalising inconsistent rubric key names."""

    def test_standard_criteria_key(self):
        """Most common format: {"id": "C1", "criteria": "..."}."""
        item = {"id": "C1", "criteria": "Does it have three sentences?"}
        result = _normalise_rubric_item(item)
        assert result == {"id": "C1", "criteria": "Does it have three sentences?"}

    def test_numbered_criteria_key(self):
        """Numbered variant: {"id": "C2", "criteria2": "..."}."""
        item = {"id": "C2", "criteria2": "Is the formatting correct?"}
        result = _normalise_rubric_item(item)
        assert result == {"id": "C2", "criteria": "Is the formatting correct?"}

    def test_spaced_criteria_key(self):
        """Space variant: {"id": "C1", "criteria 1": "..."}."""
        item = {"id": "C1", "criteria 1": "Does it start with However?"}
        result = _normalise_rubric_item(item)
        assert result == {"id": "C1", "criteria": "Does it start with However?"}

    def test_rule_key(self):
        """Rule variant: {"id": "C1", "rule": "..."}."""
        item = {"id": "C1", "rule": "No spaces allowed"}
        result = _normalise_rubric_item(item)
        assert result == {"id": "C1", "criteria": "No spaces allowed"}

    def test_question_key(self):
        """Question variant: {"id": "C1", "question": "..."}."""
        item = {"id": "C1", "question": "Are all words uppercase?"}
        result = _normalise_rubric_item(item)
        assert result == {"id": "C1", "criteria": "Are all words uppercase?"}

    def test_extract_rubrics_from_top_level(self):
        """Rubrics extracted from the top-level rubrics array."""
        task = {
            "rubrics": [
                {"id": "C1", "criteria": "First criterion"},
                {"id": "C2", "criteria2": "Second criterion"},
            ],
            "messages": [],
        }
        result = _extract_rubrics(task)
        assert len(result) == 2
        assert result[0] == {"id": "C1", "criteria": "First criterion"}
        assert result[1] == {"id": "C2", "criteria": "Second criterion"}

    def test_extract_rubrics_fallback_to_response_reference(self):
        """Rubrics parsed from response_reference when rubrics array is empty."""
        import json

        criteria_json = json.dumps([
            {"id": "C1", "criteria": "From response_reference"},
        ])
        task = {
            "rubrics": [],
            "messages": [
                {"role": "response_reference", "content": criteria_json},
            ],
        }
        result = _extract_rubrics(task)
        assert len(result) == 1
        assert result[0]["criteria"] == "From response_reference"

    def test_extract_rubrics_fallback_with_prose_preamble(self):
        """Rubrics extracted from response_reference that has a prose preamble."""
        content = (
            'For earning 1 point (PASS), the student must PASS:\n\n'
            '{\n    "id": "C1",\n    "criteria": "First criterion"\n  },\n\n'
            '  {\n    "id": "C2",\n    "criteria": "Second criterion"\n  }\n\n'
            'End of criteria.'
        )
        task = {
            "rubrics": [],
            "messages": [
                {"role": "response_reference", "content": content},
            ],
        }
        result = _extract_rubrics(task)
        assert len(result) == 2
        assert result[0] == {"id": "C1", "criteria": "First criterion"}
        assert result[1] == {"id": "C2", "criteria": "Second criterion"}


# ---------------------------------------------------------------------------
# Score aggregation tests
# ---------------------------------------------------------------------------


class TestAggregation:
    """Tests for score aggregation."""

    @staticmethod
    def _make_config() -> InverseIFConfig:
        """Create a minimal InverseIFConfig for testing."""
        from nemo_gym.config_types import ModelServerRef
        from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming

        return InverseIFConfig(
            host="",
            port=0,
            entrypoint="",
            name="test",
            judge_model_server=ModelServerRef(type="responses_api_models", name="test"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            default_judge_prompt_template="",
            default_judge_system_prompt="",
        )

    def create_evaluations(self, scores: list[float]) -> list[RubricEvaluation]:
        """Create mock evaluations with given scores."""
        return [
            RubricEvaluation(
                criterion_id=f"C{i + 1}",
                criteria=f"Criterion {i + 1}",
                judge_prompt="...",
                judge_response="...",
                verdict="PASS" if s >= 0.99 else "FAIL",
                explanation="",
                score=s,
            )
            for i, s in enumerate(scores)
        ]

    def test_aggregation_modes(self):
        """Test various aggregation modes."""
        from unittest.mock import MagicMock

        from nemo_gym.server_utils import ServerClient
        from resources_servers.inverse_if.app import InverseIFServer

        config = self._make_config()
        mock_client = MagicMock(spec=ServerClient)
        server = InverseIFServer.model_construct(config=config, server_client=mock_client)
        evaluations = self.create_evaluations([1.0, 0.0, 1.0])

        # Test MEAN
        config.aggregation_mode = AggregationMode.MEAN
        assert server._aggregate_scores(evaluations) == pytest.approx(2.0 / 3.0)

        # Test MIN
        config.aggregation_mode = AggregationMode.MIN
        assert server._aggregate_scores(evaluations) == 0.0

        # Test MAX
        config.aggregation_mode = AggregationMode.MAX
        assert server._aggregate_scores(evaluations) == 1.0

        # Test ALL (not all pass)
        config.aggregation_mode = AggregationMode.ALL
        assert server._aggregate_scores(evaluations) == 0.0

        # Test ANY (at least one passes)
        config.aggregation_mode = AggregationMode.ANY
        assert server._aggregate_scores(evaluations) == 1.0

    def test_empty_evaluations(self):
        """Empty list should return 0.0."""
        from unittest.mock import MagicMock

        from nemo_gym.server_utils import ServerClient
        from resources_servers.inverse_if.app import InverseIFServer

        config = self._make_config()
        mock_client = MagicMock(spec=ServerClient)
        server = InverseIFServer.model_construct(config=config, server_client=mock_client)
        assert server._aggregate_scores([]) == 0.0

    def test_all_pass(self):
        """All criteria passing should give 1.0 for all aggregation modes."""
        from unittest.mock import MagicMock

        from nemo_gym.server_utils import ServerClient
        from resources_servers.inverse_if.app import InverseIFServer

        config = self._make_config()
        mock_client = MagicMock(spec=ServerClient)
        server = InverseIFServer.model_construct(config=config, server_client=mock_client)
        evaluations = self.create_evaluations([1.0, 1.0, 1.0])

        for mode in AggregationMode:
            config.aggregation_mode = mode
            assert server._aggregate_scores(evaluations) == 1.0, f"Failed for mode {mode}"
