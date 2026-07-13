# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from nemo_gym.config_types import ModelServerRef
from responses_api_agents.swe_agents.app import SWEBenchMetrics
from responses_api_agents.swe_agents.refine_app import (
    SWEBenchRefineConfig,
    _append_seed_to_problem_metadata,
    _build_chain_metrics,
    _build_refine_v1_seed,
    _truncate_middle,
)


def _refine_config(**overrides) -> SWEBenchRefineConfig:
    values = {
        "host": "localhost",
        "port": 9003,
        "name": "test_refine_agent",
        "entrypoint": "responses_api_agents/swe_agents/refine_app.py",
        "model_server": ModelServerRef(type="responses_api_models", name="test"),
    }
    values.update(overrides)
    return SWEBenchRefineConfig(**values)


def test_refine_round_config_accepts_canonical_and_legacy_names() -> None:
    assert _refine_config(max_refine_rounds=3).max_refine_rounds == 3
    assert _refine_config(max_attempts=4).max_refine_rounds == 4
    assert (
        _refine_config(skip_reset_after_first=True).skip_reset_after_initial_round
        is True
    )


def test_truncate_middle_respects_approximate_token_budget() -> None:
    text = "abcdefgh12345678"

    truncated = _truncate_middle(text, max_tokens=2)

    assert truncated.startswith("abcd")
    assert truncated.endswith("5678")
    assert "diff truncated to fit carry-over budget" in truncated


def test_truncate_middle_zero_budget_keeps_full_patch() -> None:
    text = "full patch"

    assert _truncate_middle(text, max_tokens=0) == text


def test_build_refine_v1_seed_matches_eval_handoff() -> None:
    seed = _build_refine_v1_seed(
        patch="diff --git a/a.py b/a.py\n+fix",
        verify_feedback="FAILED tests/test_a.py::test_bug",
        max_patch_tokens=40000,
    )

    assert "previous automated attempt did NOT resolve" in seed
    assert "```diff\ndiff --git a/a.py b/a.py\n+fix\n```" in seed
    assert "FAILED tests/test_a.py::test_bug" in seed
    assert "produce a correct, complete patch" in seed


def test_append_seed_updates_openhands_instance_problem() -> None:
    metadata = {
        "problem_statement": "Fix the bug.",
        "instance_dict": json.dumps(
            {
                "instance_id": "repo__issue-1",
                "problem_statement": "Fix the bug.",
            }
        ),
    }

    refined = _append_seed_to_problem_metadata(metadata, "\n\n---\nRefine this patch.")
    instance_dict = json.loads(refined["instance_dict"])

    assert refined["problem_statement"].endswith("---\nRefine this patch.")
    assert instance_dict["problem_statement"] == refined["problem_statement"]
    assert json.loads(metadata["instance_dict"])["problem_statement"] == "Fix the bug."


def test_build_chain_metrics_reports_refine_rescue() -> None:
    refine_rounds = [
        {"metrics": SWEBenchMetrics(resolved=False)},
        {"metrics": SWEBenchMetrics(resolved=True)},
    ]

    metrics = _build_chain_metrics(refine_rounds, max_refine_rounds=2)

    assert metrics == {
        "refine_strategy": "baseline",
        "num_refine_rounds": 2,
        "max_refine_rounds": 2,
        "chain_resolved": True,
        "resolved_at_refine_round": 1,
        "initial_round_resolved": False,
        "refine_continued": True,
        "refine_rescued": True,
    }


def test_build_chain_metrics_reports_early_success_without_refine() -> None:
    refine_rounds = [{"metrics": SWEBenchMetrics(resolved=True)}]

    metrics = _build_chain_metrics(refine_rounds, max_refine_rounds=2)

    assert metrics["resolved_at_refine_round"] == 0
    assert metrics["initial_round_resolved"] is True
    assert metrics["refine_continued"] is False
    assert metrics["refine_rescued"] is False
