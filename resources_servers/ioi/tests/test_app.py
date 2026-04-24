# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the IOI resource server (CCC subclass).

Covers the parts the subclass actually owns: the static ``_score_fn``, the
metadata-cap lookup, and the IOI-shape ``compute_metrics`` /
``get_key_metrics`` aggregation. The inherited verify / sandbox path is
exercised by ``competitive_coding_challenges``' own tests.
"""

from types import SimpleNamespace

import pytest
from app import IOIResourcesServer


# ---------------------------------------------------------------------------
# _score_fn
# ---------------------------------------------------------------------------


class TestScoreFn:
    def test_empty_tcr_scores_zero(self):
        assert IOIResourcesServer._score_fn({"details": {"test_case_results": {}}}) == {"accuracy": 0.0}

    def test_missing_details_scores_zero(self):
        assert IOIResourcesServer._score_fn({}) == {"accuracy": 0.0}

    def test_any_zero_subtask_blocks_accuracy(self):
        result = {"details": {"test_case_results": {"01-a": {"score": 6.0}, "02-b": {"score": 0.0}}}}
        assert IOIResourcesServer._score_fn(result) == {"accuracy": 0.0}

    def test_all_positive_subtasks_score_one(self):
        result = {"details": {"test_case_results": {"01-a": {"score": 6.0}, "02-b": {"score": 13.0}}}}
        assert IOIResourcesServer._score_fn(result) == {"accuracy": 1.0}

    def test_none_score_treated_as_zero(self):
        result = {"details": {"test_case_results": {"01-a": {"score": None}}}}
        assert IOIResourcesServer._score_fn(result) == {"accuracy": 0.0}


# ---------------------------------------------------------------------------
# _lookup_subtask_cap
# ---------------------------------------------------------------------------


def _make_server_with_evaluator(metadata: dict) -> IOIResourcesServer:
    """Build a bare server instance with a stub evaluator; skip FastAPI setup."""
    server = IOIResourcesServer.model_construct()
    server._evaluator = SimpleNamespace(
        get_problem_metadata=lambda problem_id, competition_id: metadata[problem_id],
    )
    return server


class TestLookupSubtaskCap:
    def test_returns_zero_when_no_evaluator(self):
        server = IOIResourcesServer.model_construct()
        server._evaluator = None
        assert server._lookup_subtask_cap("ioi24", "nile", "01-equal") == 0.0

    def test_returns_subtask_score(self):
        server = _make_server_with_evaluator(
            {"nile": {"subtasks": {"01-equal": {"score": 6.0}, "02-permutation": {"score": 13.0}}}}
        )
        assert server._lookup_subtask_cap("ioi24", "nile", "01-equal") == 6.0
        assert server._lookup_subtask_cap("ioi24", "nile", "02-permutation") == 13.0

    def test_returns_zero_for_unknown_subtask(self):
        server = _make_server_with_evaluator({"nile": {"subtasks": {"01-equal": {"score": 6.0}}}})
        assert server._lookup_subtask_cap("ioi24", "nile", "unknown") == 0.0

    def test_swallows_evaluator_error(self):
        def _raise(*_, **__):
            raise ValueError("problem not found")

        server = IOIResourcesServer.model_construct()
        server._evaluator = SimpleNamespace(get_problem_metadata=_raise)
        assert server._lookup_subtask_cap("ioi24", "nile", "01-equal") == 0.0


# ---------------------------------------------------------------------------
# compute_metrics — the core IOI-shape aggregation
# ---------------------------------------------------------------------------


def _make_rollout(problem_id: str, subtask: str, tcr: dict, competition_id: str = "ioi24") -> dict:
    return {
        "problem_id": problem_id,
        "competition_id": competition_id,
        "subtask": subtask,
        "extracted_code": "<stub code>",
        "details": {"test_case_results": tcr},
    }


@pytest.fixture
def server_with_nile_metadata():
    return _make_server_with_evaluator(
        {
            "nile": {
                "subtasks": {
                    "01-equal": {"score": 6.0},
                    "02-permutation": {"score": 13.0},
                }
            },
            "message": {"subtasks": {"01-len64": {"score": 10.0}, "02-full": {"score": 90.0}}},
        }
    )


class TestComputeMetrics:
    def test_empty_tasks_returns_zero_total(self, server_with_nile_metadata):
        metrics = server_with_nile_metadata.compute_metrics([])
        assert metrics["ioi_total_score"] == 0
        assert metrics["per_problem_subtask_scores"] == {}

    def test_sums_max_per_subtask_across_rollouts(self, server_with_nile_metadata):
        # Two rollouts for Nile: both evaluate every Nile subtask (CCC per-problem semantics).
        # Rollout A gets subtask 01-equal fully, misses 02-permutation.
        # Rollout B misses 01-equal, gets 02-permutation fully.
        # Pooled max: 6 + 13 = 19.
        rollout_a = _make_rollout("nile", "01-equal", {"01-equal": {"score": 6.0}, "02-permutation": {"score": 0.0}})
        rollout_b = _make_rollout(
            "nile", "02-permutation", {"01-equal": {"score": 0.0}, "02-permutation": {"score": 13.0}}
        )
        metrics = server_with_nile_metadata.compute_metrics([[rollout_a], [rollout_b]])
        assert metrics["ioi_total_score"] == 19
        per = metrics["per_problem_subtask_scores"]["nile"]
        assert per["total"] == {"score": 19.0, "max_score": 19.0}
        assert per["subtasks"]["01-equal"] == {"score": 6.0, "max_score": 6.0}
        assert per["subtasks"]["02-permutation"] == {"score": 13.0, "max_score": 13.0}

    def test_sums_across_problems(self, server_with_nile_metadata):
        nile_rollout = _make_rollout(
            "nile", "01-equal", {"01-equal": {"score": 6.0}, "02-permutation": {"score": 13.0}}
        )
        message_rollout = _make_rollout(
            "message", "01-len64", {"01-len64": {"score": 10.0}, "02-full": {"score": 0.0}}
        )
        metrics = server_with_nile_metadata.compute_metrics([[nile_rollout], [message_rollout]])
        # Nile 6+13 + Message 10 = 29
        assert metrics["ioi_total_score"] == 29
        assert set(metrics["per_problem_subtask_scores"].keys()) == {"nile", "message"}

    def test_falls_back_to_ioi_id_when_problem_id_missing(self, server_with_nile_metadata):
        # Some rollout records carry only ioi_id instead of problem_id.
        rollout = {
            "ioi_id": "nile",
            "competition_id": "ioi24",
            "subtask": "01-equal",
            "extracted_code": "x",
            "details": {"test_case_results": {"01-equal": {"score": 6.0}}},
        }
        metrics = server_with_nile_metadata.compute_metrics([[rollout]])
        assert metrics["ioi_total_score"] == 6

    def test_skips_rollouts_without_problem_id(self, server_with_nile_metadata):
        rollout = {"details": {"test_case_results": {"01-equal": {"score": 6.0}}}}
        metrics = server_with_nile_metadata.compute_metrics([[rollout]])
        assert metrics["ioi_total_score"] == 0


# ---------------------------------------------------------------------------
# get_key_metrics
# ---------------------------------------------------------------------------


class TestGetKeyMetrics:
    def test_promotes_ioi_total_score(self, server_with_nile_metadata):
        agent_metrics = {
            "ioi_total_score": 123,
            "pass@1[avg-of-2]/accuracy": 0.5,
            "pass@2/accuracy": 1.0,
        }
        key = server_with_nile_metadata.get_key_metrics(agent_metrics)
        assert key["ioi_total_score"] == 123

    def test_omits_ioi_total_score_when_absent(self, server_with_nile_metadata):
        key = server_with_nile_metadata.get_key_metrics({})
        assert "ioi_total_score" not in key
