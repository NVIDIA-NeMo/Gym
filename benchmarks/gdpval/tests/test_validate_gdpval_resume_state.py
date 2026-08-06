# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import subprocess
import sys
from pathlib import Path


GDPVAL_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = GDPVAL_ROOT / "slurm" / "validate_gdpval_resume_state.py"
MODEL_NAME = "my-model-0"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _build_partial_state(tmp_path: Path) -> tuple[list[str], Path, Path]:
    launch = tmp_path / "input" / "launch.jsonl"
    rollouts = tmp_path / "results" / "rollouts.jsonl"
    materialized = tmp_path / "results" / "rollouts_materialized_inputs.jsonl"
    failures = tmp_path / "results" / "rollouts_failures.jsonl"
    run_config = tmp_path / "run_config.sha256"
    deliverables = tmp_path / "deliverables"

    launch_rows = [{"task_id": f"AQ-{index:05d}"} for index in range(100)]
    materialized_rows = [
        {
            "task_id": row["task_id"],
            "_ng_task_index": index,
            "_ng_rollout_index": 0,
            "agent_ref": {"name": "gdpval_stirrup_agent"},
        }
        for index, row in enumerate(launch_rows)
    ]
    task_id = launch_rows[0]["task_id"]
    task_dir = deliverables / f"task_{task_id}" / "repeat_0"
    task_dir.mkdir(parents=True)
    (task_dir / "finish_params.json").write_text('{"paths": []}\n', encoding="utf-8")
    rollout_rows = [
        {
            "task_id": task_id,
            "_ng_task_index": 0,
            "_ng_rollout_index": 0,
            "agent_ref": {"name": "gdpval_stirrup_agent"},
            "execute_only": True,
            "reward": None,
            "judge_response": None,
            "response": {"model": MODEL_NAME, "error": None},
            "deliverables_dir": str(task_dir),
        }
    ]

    _write_jsonl(launch, launch_rows)
    _write_jsonl(materialized, materialized_rows)
    _write_jsonl(rollouts, rollout_rows)
    failures.write_text("", encoding="utf-8")
    run_config.write_text("a" * 64 + "\n", encoding="utf-8")
    launch_sha = hashlib.sha256(launch.read_bytes()).hexdigest()
    args = [
        sys.executable,
        str(VALIDATOR),
        "--launch-input",
        str(launch),
        "--rollouts",
        str(rollouts),
        "--materialized-inputs",
        str(materialized),
        "--failures",
        str(failures),
        "--run-config",
        str(run_config),
        "--expected-launch-sha256",
        launch_sha,
        "--expected-model",
        MODEL_NAME,
        "--deliverables-dir",
        str(deliverables),
    ]
    return args, rollouts, task_dir


def test_partial_resume_validator_accepts_matching_gym_keys_and_finish_marker(tmp_path):
    args, _, _ = _build_partial_state(tmp_path)

    result = subprocess.run(args, check=False, capture_output=True, text=True)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "1/100 completed rows" in result.stdout


def test_partial_resume_validator_rejects_task_mapped_to_wrong_gym_key(tmp_path):
    args, rollouts, _ = _build_partial_state(tmp_path)
    row = json.loads(rollouts.read_text(encoding="utf-8"))
    row["_ng_task_index"] = 1
    _write_jsonl(rollouts, [row])

    result = subprocess.run(args, check=False, capture_output=True, text=True)

    assert result.returncode == 1
    assert "materialized input maps to" in result.stdout


def test_partial_resume_validator_requires_cached_finish_marker(tmp_path):
    args, _, task_dir = _build_partial_state(tmp_path)
    (task_dir / "finish_params.json").unlink()

    result = subprocess.run(args, check=False, capture_output=True, text=True)

    assert result.returncode == 1
    assert "cached rollout is missing" in result.stdout
