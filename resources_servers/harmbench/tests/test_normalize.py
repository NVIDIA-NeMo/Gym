# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from benchmarks.harmbench.reporting import package
from benchmarks.harmbench.reporting.normalize import normalize


def _row(
    index: int, identifier: str, functional: str, semantic: str, label: int, text: str, generation: str, **extra
) -> dict:
    return {
        "_ng_task_index": index,
        "_ng_rollout_index": 0,
        "harmbench_id": identifier,
        "behavior": f"behavior {identifier}",
        "context": "ctx" if functional == "contextual" else "",
        "functional_category": functional,
        "semantic_category": semantic,
        "responses_create_params": {"input": [{"role": "user", "content": f"behavior {identifier}"}]},
        "response": {
            "output": [
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": generation}]}
            ],
            "incomplete_details": None,
        },
        "generation": generation,
        "generation_for_classifier": generation,
        "classifier_label": label,
        "classifier_text": text,
        "classifier_prompt_variant": functional,
        "classifier_prompt_sha256": "0" * 64,
        "generation_empty": not generation,
        "generation_truncated": False,
        "generation_token_count": len(generation.split()) + 1,
        "generation_clipped": False,
        "reward": 1.0 if label == 0 else 0.0,
        **extra,
    }


def _write(tmp_path):
    rows = [
        _row(0, "b1", "standard", "cybercrime_intrusion", 1, "Yes", "here is the code"),
        _row(1, "b2", "standard", "cybercrime_intrusion", 0, "No", "I cannot help"),
        _row(2, "b3", "contextual", "illegal", 0, "No", ""),
        _row(3, "b4", "contextual", "illegal", -1, "Maybe", "some text", generation_clipped=True),
    ]
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text("".join(json.dumps(row) + "\n" for row in rows))
    inputs = tmp_path / "inputs.jsonl"
    inputs.write_text("".join(json.dumps({"harmbench_id": row["harmbench_id"]}) + "\n" for row in rows))
    aggregate = tmp_path / "agg.json"
    aggregate.write_text(
        json.dumps(
            [
                {
                    "agent_metrics": {
                        "attack_success_rate": 0.25,
                        "attack_success_rate_valid_only": 1 / 3,
                        "classifier_valid_rate": 0.75,
                        "attack_success_rate/functional_category/standard": 0.5,
                    }
                }
            ]
        )
    )
    return rollouts, inputs, aggregate


def test_normalize_uses_upstream_denominator_and_separates_judge_missing(tmp_path):
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
    assert run.metric("attack_success_rate").value == 0.25 and run.metric("attack_success_rate").denominator == 4
    assert run.metric("attack_success_rate_valid_only").denominator == 3
    assert run.outcomes.judge_missing == 1 and run.outcomes.policy_scored == 3
    assert run.metric("generation_empty_rate").numerator == 1 and run.metric("generation_clipped_rate").numerator == 1
    assert run.metric("attack_success_rate/semantic_category/illegal").value == 0.0
    assert [s.outcome_class for s in run.rollouts] == [
        "policy_scored",
        "policy_scored",
        "policy_scored",
        "judge_missing",
    ]
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
    d1 = json.loads((built / "blade/d1-metrics.json").read_text())
    assert d1["outcome_classes"]["judge_missing"] == 1 and d1["pass_at_1"] == pytest.approx(2 / 3)


def test_normalize_rejects_disagreeing_aggregate(tmp_path):
    rollouts, inputs, aggregate = _write(tmp_path)
    aggregate.write_text(json.dumps([{"agent_metrics": {"attack_success_rate": 0.5}}]))
    with pytest.raises(ValueError, match="do not reconcile"):
        normalize(
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
