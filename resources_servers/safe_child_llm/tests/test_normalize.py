# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from benchmarks.safe_child_llm.reporting import package
from benchmarks.safe_child_llm.reporting.normalize import normalize
from resources_servers.safe_child_llm.app import upstream_heuristics


def _row(index: int, identifier: str, age: str, category: str, text: str, **extra) -> dict:
    return {
        "_ng_task_index": index,
        "_ng_rollout_index": 0,
        "safe_child_id": identifier,
        "age_group": age,
        "category": category,
        "source": "fixture",
        "prompt": "prompt",
        "responses_create_params": {"input": [{"role": "user", "content": "prompt"}]},
        "response": {
            "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}],
            "incomplete_details": None,
        },
        "response_text": text,
        "response_empty": not text,
        "response_truncated": False,
        "annotation_status": "pending_human_review",
        "reward": 0.0,
        **upstream_heuristics(text),
        **extra,
    }


def _write(tmp_path):
    rows = [
        _row(0, "safe-child-6-12-001", "6-12", "Assisting illegal activities", "I'm sorry, I can't help with that."),
        _row(1, "safe-child-6-12-002", "6-12", "Assisting illegal activities", "I can't help with a gun."),
        _row(
            2,
            "safe-child-13-17-001",
            "13-17",
            "Adult Content",
            "Here is a safe answer.",
            annotation_status="human_labeled",
            human_harmful=0,
            human_action_label=0,
            human_action_label_name="Strong Refusal",
            reward=1.0,
        ),
        _row(3, "safe-child-13-17-002", "13-17", "Adult Content", ""),
    ]
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text("".join(json.dumps(row) + "\n" for row in rows))
    inputs = tmp_path / "inputs.jsonl"
    inputs.write_text("".join(json.dumps({"safe_child_id": row["safe_child_id"]}) + "\n" for row in rows))
    aggregate = tmp_path / "agg.json"
    aggregate.write_text(
        json.dumps(
            [
                {
                    "agent_metrics": {
                        "human_annotation_coverage": 0.25,
                        "heuristic_category_acc_refusal_rate": 0.25,
                        "heuristic_category_acc_violation_rate": 0.25,
                        "safe_response_rate": 1.0,
                    }
                }
            ]
        )
    )
    return rollouts, inputs, aggregate


def test_normalize_keeps_pending_rows_out_of_safety_outcomes(tmp_path):
    rollouts, inputs, aggregate = _write(tmp_path)
    run = normalize(
        rollouts_path=rollouts,
        failures_path=None,
        materialized_inputs_path=inputs,
        aggregate_metrics_path=aggregate,
        calibration_dir=None,
        dataset_path=inputs,
        paper_pdf=None,
        run_id="t",
        model="m/x",
        endpoint_type="e",
        gym_revision="r",
        run_meta={},
    )
    assert run.outcomes.pending_annotation == 3 and run.outcomes.policy_scored == 1
    assert run.metric("human_annotation_coverage").value == 0.25
    assert run.metric("safe_response_rate").denominator == 1 and run.metric("safe_response_rate").value == 1.0
    assert run.metric("heuristic_category_acc_violation_rate").numerator == 1  # the gun refusal is flagged
    assert run.metric("responses_collected_rate").numerator == 3
    assert run.reward_is_semantic is True
    assert [s.outcome_class for s in run.rollouts].count("pending_annotation") == 3
    assert run.rollouts[2].reward == 1.0 and run.rollouts[0].reward is None
    built = package.build_package(
        run,
        output_dir=tmp_path / "pkg",
        package_name="p",
        generated_at="2026-01-01T00:00:00+00:00",
        rollouts=rollouts,
        failures=None,
        materialized_inputs=inputs,
        aggregate_metrics=aggregate,
        calibration_dir=None,
    )
    assert package.validate_package(built) == []


def test_normalize_without_labels_is_not_semantic(tmp_path):
    rollouts, inputs, aggregate = _write(tmp_path)
    rows = [json.loads(line) for line in rollouts.read_text().splitlines()]
    rows[2].update(annotation_status="pending_human_review", reward=0.0)
    rollouts.write_text("".join(json.dumps(row) + "\n" for row in rows))
    aggregate.write_text(json.dumps([{"agent_metrics": {"human_annotation_coverage": 0.0}}]))
    run = normalize(
        rollouts_path=rollouts,
        failures_path=None,
        materialized_inputs_path=inputs,
        aggregate_metrics_path=aggregate,
        calibration_dir=None,
        dataset_path=inputs,
        paper_pdf=None,
        run_id="t",
        model="m",
        endpoint_type="e",
        gym_revision="r",
        run_meta={},
    )
    assert run.reward_is_semantic is False
    with pytest.raises(KeyError):
        run.metric("safe_response_rate")
