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

import math
from unittest.mock import MagicMock

import pytest

from nemo_gym.server_utils import ServerClient
from resources_servers.graphwalks.app import (
    GraphWalksResourcesServer,
    GraphWalksResourcesServerConfig,
    _f1_score,
    _parse_node_list,
)


class TestSanity:
    def test_sanity(self) -> None:
        config = GraphWalksResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        GraphWalksResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


class TestParseNodeList:
    """Tests for the Final-Answer line parser.

    Reference: https://huggingface.co/datasets/openai/graphwalks
    """

    def test_parses_well_formed_list(self) -> None:
        nodes, failed = _parse_node_list("...\nFinal Answer: [node_1, node_2, node_3]")
        assert nodes == ["node_1", "node_2", "node_3"]
        assert failed is False

    def test_empty_list_is_valid(self) -> None:
        """`Final Answer: []` is a valid no-nodes answer, not a parse failure."""
        nodes, failed = _parse_node_list("Final Answer: []")
        assert nodes == []
        assert failed is False

    def test_only_uses_last_line(self) -> None:
        text = "Final Answer: [decoy]\nbecause I said so\nFinal Answer: [real]"
        nodes, failed = _parse_node_list(text)
        assert nodes == ["real"]
        assert failed is False

    def test_skips_trailing_blank_lines(self) -> None:
        nodes, failed = _parse_node_list("Final Answer: [a, b]\n\n   \n")
        assert nodes == ["a", "b"]
        assert failed is False

    def test_missing_format_fails(self) -> None:
        nodes, failed = _parse_node_list("The answer is node_42.")
        assert nodes == []
        assert failed is True

    def test_blank_response_fails(self) -> None:
        nodes, failed = _parse_node_list("")
        assert nodes == []
        assert failed is True

    def test_strips_whitespace_inside_list(self) -> None:
        nodes, failed = _parse_node_list("Final Answer: [  a  ,b ,   c]")
        assert nodes == ["a", "b", "c"]
        assert failed is False

    def test_drops_empty_items(self) -> None:
        """Trailing commas / double commas should not produce empty entries."""
        nodes, failed = _parse_node_list("Final Answer: [a,, b,]")
        assert nodes == ["a", "b"]
        assert failed is False


class TestF1Score:
    def test_parse_failed_is_zero(self) -> None:
        assert _f1_score({"a"}, {"a"}, parse_failed=True) == 0.0

    def test_both_empty_is_one(self) -> None:
        assert _f1_score(set(), set(), parse_failed=False) == 1.0

    def test_predicted_empty_expected_nonempty(self) -> None:
        assert _f1_score(set(), {"a"}, parse_failed=False) == 0.0

    def test_predicted_nonempty_expected_empty(self) -> None:
        assert _f1_score({"a"}, set(), parse_failed=False) == 0.0

    def test_exact_match_is_one(self) -> None:
        assert _f1_score({"a", "b"}, {"a", "b"}, parse_failed=False) == 1.0

    def test_no_overlap_is_zero(self) -> None:
        assert _f1_score({"a"}, {"b"}, parse_failed=False) == 0.0

    def test_partial_overlap(self) -> None:
        # P=1/2, R=1/2 → F1=0.5
        assert math.isclose(_f1_score({"a", "b"}, {"a", "c"}, parse_failed=False), 0.5)

    def test_unequal_sizes(self) -> None:
        # predicted={a,b,c}, expected={a}; P=1/3, R=1 → F1=0.5
        assert math.isclose(_f1_score({"a", "b", "c"}, {"a"}, parse_failed=False), 0.5)


class TestScoreFn:
    def test_score_fn_returns_accuracy_equals_reward(self) -> None:
        assert GraphWalksResourcesServer._score_fn({"reward": 0.73}) == {"accuracy": 0.73}

    def test_score_fn_handles_zero(self) -> None:
        assert GraphWalksResourcesServer._score_fn({"reward": 0.0}) == {"accuracy": 0.0}

    def test_score_fn_handles_one(self) -> None:
        assert GraphWalksResourcesServer._score_fn({"reward": 1.0}) == {"accuracy": 1.0}


class TestComputeMetrics:
    @pytest.fixture
    def server(self) -> GraphWalksResourcesServer:
        config = GraphWalksResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        return GraphWalksResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def test_compute_metrics_empty(self, server: GraphWalksResourcesServer) -> None:
        assert server.compute_metrics([]) == {}

    def test_compute_metrics_includes_pass_at_k(self, server: GraphWalksResourcesServer) -> None:
        tasks = [
            [{"reward": 1.0, "problem_type": "parents"}, {"reward": 0.5, "problem_type": "parents"}],
            [{"reward": 0.8, "problem_type": "bfs"}, {"reward": 0.6, "problem_type": "bfs"}],
        ]
        metrics = server.compute_metrics(tasks)
        assert "pass@1/accuracy" in metrics
        assert "pass@2/accuracy" in metrics
        assert "pass@1[avg-of-2]/accuracy" in metrics

    def test_compute_metrics_includes_subset_breakdown(self, server: GraphWalksResourcesServer) -> None:
        """Per-problem-type subset should appear as `problem_type=<value>/...`."""
        tasks = [
            [{"reward": 1.0, "problem_type": "parents"}, {"reward": 0.5, "problem_type": "parents"}],
            [{"reward": 0.8, "problem_type": "bfs"}, {"reward": 0.6, "problem_type": "bfs"}],
        ]
        metrics = server.compute_metrics(tasks)
        assert any(k.startswith("problem_type=parents/pass@") for k in metrics)
        assert any(k.startswith("problem_type=bfs/pass@") for k in metrics)
        # Bare "<value>/..." keys must NOT leak through from compute_subset_metrics.
        assert not any(k.startswith(("parents/", "bfs/")) for k in metrics)

    def test_compute_metrics_no_majority(self, server: GraphWalksResourcesServer) -> None:
        """majority@k is skipped because F1 has no discrete answer_key."""
        tasks = [[{"reward": 1.0, "problem_type": "parents"}, {"reward": 0.5, "problem_type": "parents"}]]
        metrics = server.compute_metrics(tasks)
        assert not any(k.startswith("majority@") for k in metrics)


class TestGetKeyMetrics:
    @pytest.fixture
    def server(self) -> GraphWalksResourcesServer:
        config = GraphWalksResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        return GraphWalksResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def test_get_key_metrics_picks_highest_k(self, server: GraphWalksResourcesServer) -> None:
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
