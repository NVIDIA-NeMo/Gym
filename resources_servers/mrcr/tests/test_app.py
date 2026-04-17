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

import pytest

from nemo_gym.server_utils import ServerClient
from resources_servers.mrcr.app import (
    MRCRResourcesServer,
    MRCRResourcesServerConfig,
    _grade,
)


class TestSanity:
    def test_sanity(self) -> None:
        config = MRCRResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        MRCRResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


class TestGrade:
    """Tests for the MRCR grading function.

    Gold reference: nemo_skills.evaluation.evaluator.mrcr.eval_mrcr.grade
    and the official openai/mrcr grading function.
    """

    def test_missing_prefix_returns_zero(self) -> None:
        """No matter how similar, missing the prefix yields 0.0."""
        score = _grade(
            response="hello world",
            expected_answer="xyz123hello world",
            random_string_to_prepend="xyz123",
        )
        assert score == 0.0

    def test_exact_match_with_prefix_returns_one(self) -> None:
        score = _grade(
            response="xyz123The quick brown fox",
            expected_answer="xyz123The quick brown fox",
            random_string_to_prepend="xyz123",
        )
        assert score == 1.0

    def test_partial_match_returns_intermediate_ratio(self) -> None:
        score = _grade(
            response="xyz123The quick brown fox",
            expected_answer="xyz123The quick black fox",
            random_string_to_prepend="xyz123",
        )
        assert 0.0 < score < 1.0

    def test_prefix_stripped_before_matching(self) -> None:
        """Different prefix lengths shouldn't skew the ratio — both sides are stripped."""
        # Post-strip both sides are "hello" exactly → ratio 1.0
        score = _grade(
            response="PFXhello",
            expected_answer="PFXhello",
            random_string_to_prepend="PFX",
        )
        assert score == 1.0

    def test_empty_response_no_prefix_returns_zero(self) -> None:
        score = _grade(
            response="",
            expected_answer="xyz123hello",
            random_string_to_prepend="xyz123",
        )
        assert score == 0.0

    def test_prefix_only_response(self) -> None:
        """Response is just the prefix with nothing after — after stripping,
        compare '' to 'hello' → ratio = 0.0."""
        score = _grade(
            response="xyz123",
            expected_answer="xyz123hello",
            random_string_to_prepend="xyz123",
        )
        assert score == 0.0


class TestScoreFn:
    def test_score_fn_returns_accuracy_equals_reward(self) -> None:
        scores = MRCRResourcesServer._score_fn({"reward": 0.73})
        assert scores == {"accuracy": 0.73}

    def test_score_fn_handles_zero(self) -> None:
        scores = MRCRResourcesServer._score_fn({"reward": 0.0})
        assert scores == {"accuracy": 0.0}

    def test_score_fn_handles_one(self) -> None:
        scores = MRCRResourcesServer._score_fn({"reward": 1.0})
        assert scores == {"accuracy": 1.0}


class TestComputeMetrics:
    """Tests for metrics aggregation."""

    @pytest.fixture
    def server(self) -> MRCRResourcesServer:
        config = MRCRResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        return MRCRResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def test_compute_metrics_empty(self, server: MRCRResourcesServer) -> None:
        assert server.compute_metrics([]) == {}

    def test_compute_metrics_includes_pass_at_k(self, server: MRCRResourcesServer) -> None:
        tasks = [
            [{"reward": 1.0, "n_needles": 2}, {"reward": 0.5, "n_needles": 2}],
            [{"reward": 0.8, "n_needles": 4}, {"reward": 0.6, "n_needles": 4}],
        ]
        metrics = server.compute_metrics(tasks)
        assert "pass@1/accuracy" in metrics
        assert "pass@2/accuracy" in metrics
        assert "pass@1[avg-of-2]/accuracy" in metrics

    def test_compute_metrics_includes_subset_breakdown(self, server: MRCRResourcesServer) -> None:
        """Per-needle-count subset breakdown should appear as `{n_needles}/pass@k/...`."""
        tasks = [
            [{"reward": 1.0, "n_needles": 2}, {"reward": 0.5, "n_needles": 2}],
            [{"reward": 0.8, "n_needles": 4}, {"reward": 0.6, "n_needles": 4}],
        ]
        metrics = server.compute_metrics(tasks)
        # compute_subset_metrics emits keys like "2/pass@1[avg-of-2]/accuracy"
        assert any(k.startswith("2/pass@") for k in metrics)
        assert any(k.startswith("4/pass@") for k in metrics)

    def test_compute_metrics_no_majority(self, server: MRCRResourcesServer) -> None:
        """majority@k is skipped because MRCR has no discrete answer_key."""
        tasks = [[{"reward": 1.0, "n_needles": 2}, {"reward": 0.5, "n_needles": 2}]]
        metrics = server.compute_metrics(tasks)
        assert not any(k.startswith("majority@") for k in metrics)


class TestGetKeyMetrics:
    @pytest.fixture
    def server(self) -> MRCRResourcesServer:
        config = MRCRResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        return MRCRResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def test_get_key_metrics_picks_highest_k(self, server: MRCRResourcesServer) -> None:
        agent_metrics = {
            "pass@1/accuracy": 50.0,
            "pass@2/accuracy": 70.0,
            "pass@4/accuracy": 80.0,
            "pass@1[avg-of-4]/accuracy": 60.0,
            "mean/input_tokens": 1000,
            "mean/output_tokens": 200,
        }
        key = server.get_key_metrics(agent_metrics)
        assert key["pass@4/accuracy"] == 80.0
        assert key["pass@1[avg-of-4]/accuracy"] == 60.0
        assert key["mean/input_tokens"] == 1000
        assert key["mean/output_tokens"] == 200
        # Lower-k entries should not be in the key set
        assert "pass@1/accuracy" not in key
        assert "pass@2/accuracy" not in key
