# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

from nemo_gym.eval_dataset_validation import validate_eval_dataset


def _write_manifest(path: Path, *, min_tasks: int = 2) -> None:
    path.write_text(
        f"""schema_version: 1
name: test-eval
use_case: test representative evaluation
dataset:
  path: tasks.jsonl
  task_id_fields: [task_id, seed]
split:
  field: split
  allowed: [test]
quality:
  min_tasks: {min_tasks}
  coverage:
    - field: category
      required_values: [a, b]
      min_count_per_value: 1
    - field: difficulty
      min_distinct: 2
  edge_cases:
    tag_field: eval_tags
    required: [recovery]
  recommended_fields: [provenance]
""",
        encoding="utf-8",
    )


def _task(task_id: str, category: str, difficulty: str, *, seed: int = 1) -> dict:
    return {
        "task_id": task_id,
        "seed": seed,
        "split": "test",
        "category": category,
        "difficulty": difficulty,
        "eval_tags": ["recovery"],
        "provenance": "fixture-v1",
        "responses_create_params": {"input": [{"role": "user", "content": task_id}]},
        "agent_ref": {"type": "responses_api_agents", "name": "fixture_agent"},
    }


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_representative_eval_passes(tmp_path: Path) -> None:
    manifest = tmp_path / "eval.yaml"
    _write_manifest(manifest)
    _write_rows(tmp_path / "tasks.jsonl", [_task("one", "a", "easy"), _task("two", "b", "hard")])

    result = validate_eval_dataset(manifest)

    assert result.passed
    assert result.rows == 2
    assert result.diagnostics == ()


def test_structural_and_coverage_failures_are_actionable(tmp_path: Path) -> None:
    manifest = tmp_path / "eval.yaml"
    _write_manifest(manifest, min_tasks=3)
    duplicate = _task("one", "a", "easy")
    duplicate["split"] = "train"
    duplicate["reward"] = 1
    duplicate.pop("eval_tags")
    _write_rows(tmp_path / "tasks.jsonl", [duplicate, duplicate])

    result = validate_eval_dataset(manifest)
    codes = {diagnostic.code for diagnostic in result.diagnostics}

    assert not result.passed
    assert {
        "rollout_fields",
        "duplicate_task_identity",
        "held_out_split",
        "task_count",
        "coverage_values",
        "coverage_distinct",
        "edge_case_coverage",
    }.issubset(codes)
    assert any(
        diagnostic.line == 2 for diagnostic in result.diagnostics if diagnostic.code == "duplicate_task_identity"
    )


def test_tales_example_is_smoke_data_not_eval_ready() -> None:
    manifest = Path("resources_servers/tales/eval/eval_readiness.yaml")

    result = validate_eval_dataset(manifest)
    codes = {diagnostic.code for diagnostic in result.diagnostics}

    assert not result.passed
    assert result.rows == 5
    assert {"held_out_split", "task_count", "coverage_count", "coverage_distinct", "edge_case_coverage"}.issubset(
        codes
    )
