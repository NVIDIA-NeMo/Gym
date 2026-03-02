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

from resources_servers.multichallenge.app import (
    AggregationMode,
    MultiChallengeConfig,
    RubricEvaluation,
    _build_context_from_messages,
    _extract_verdict,
)


class TestMultiChallenge:
    """Tests for MultiChallenge environment utilities."""

    def test_extract_verdict_yes(self):
        """Test extracting YES verdict."""
        response = "After analysis, the model correctly addressed the user's allergy. [[YES]]"
        verdict = _extract_verdict(response, "[[YES]]", "[[NO]]")
        assert verdict == "YES"

    def test_extract_verdict_no(self):
        """Test extracting NO verdict."""
        response = "The model failed to remember the allergy. [[NO]]"
        verdict = _extract_verdict(response, "[[YES]]", "[[NO]]")
        assert verdict == "NO"

    def test_extract_verdict_fallback(self):
        """Test fallback when no label present."""
        response = "The model did well.\nYES"
        verdict = _extract_verdict(response, "[[YES]]", "[[NO]]")
        assert verdict == "YES"

    def test_extract_verdict_last_wins(self):
        """Test that last label wins when both present."""
        response = "Initially [[YES]] but actually [[NO]]"
        verdict = _extract_verdict(response, "[[YES]]", "[[NO]]")
        assert verdict == "NO"

    def test_build_context_excludes_thinking(self):
        """Test that thinking messages are excluded from context."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "thinking", "content": "Processing..."},
            {"role": "assistant", "content": "Hi there!"},
        ]
        context = _build_context_from_messages(messages, exclude_thinking=True)
        assert "Processing" not in context
        assert "[USER]: Hello" in context
        assert "[ASSISTANT]: Hi there!" in context

    def test_build_context_includes_thinking(self):
        """Test that thinking messages can be included."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "thinking", "content": "Processing..."},
            {"role": "assistant", "content": "Hi there!"},
        ]
        context = _build_context_from_messages(messages, exclude_thinking=False)
        assert "[THINKING]: Processing" in context


class TestAggregation:
    """Tests for score aggregation."""

    def create_evaluations(self, scores: list[float]) -> list[RubricEvaluation]:
        """Create mock evaluations with given scores."""
        return [
            RubricEvaluation(
                question=f"Q{i}",
                pass_criteria="YES",
                judge_prompt="...",
                judge_response="...",
                verdict="YES" if s >= 0.99 else "NO",
                score=s,
                weight=1.0,
            )
            for i, s in enumerate(scores)
        ]

    def test_aggregation_modes(self):
        """Test various aggregation modes."""
        from unittest.mock import MagicMock

        from nemo_gym.config_types import ModelServerRef
        from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
        from nemo_gym.server_utils import ServerClient
        from resources_servers.multichallenge.app import MultiChallengeServer

        config = MultiChallengeConfig(
            host="",
            port=0,
            entrypoint="",
            name="test",
            judge_model_server=ModelServerRef(type="responses_api_models", name="test"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )

        # Create a proper mock that passes pydantic validation
        mock_client = MagicMock(spec=ServerClient)
        server = MultiChallengeServer.model_construct(config=config, server_client=mock_client)
        evaluations = self.create_evaluations([1.0, 0.5, 0.0])

        # Test MEAN
        config.aggregation_mode = AggregationMode.MEAN
        assert server._aggregate_scores(evaluations) == pytest.approx(0.5)

        # Test MIN
        config.aggregation_mode = AggregationMode.MIN
        assert server._aggregate_scores(evaluations) == 0.0

        # Test MAX
        config.aggregation_mode = AggregationMode.MAX
        assert server._aggregate_scores(evaluations) == 1.0

        # Test ALL (only first passes)
        config.aggregation_mode = AggregationMode.ALL
        assert server._aggregate_scores(evaluations) == 0.0

        # Test ANY (first passes)
        config.aggregation_mode = AggregationMode.ANY
        assert server._aggregate_scores(evaluations) == 1.0
