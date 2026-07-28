# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate that a partial AfterQuery rollout can be resumed safely."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


class ResumeStateError(ValueError):
    """Raised when cached rollout state is unsafe to resume."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ResumeStateError(f"{path}: blank line {line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ResumeStateError(f"{path}: row {line_number} is not an object")
            rows.append(value)
    return rows


def _rollout_key(row: dict[str, Any], *, label: str) -> tuple[int, int]:
    task_index = row.get("_ng_task_index")
    rollout_index = row.get("_ng_rollout_index")
    if (
        isinstance(task_index, bool)
        or not isinstance(task_index, int)
        or isinstance(rollout_index, bool)
        or not isinstance(rollout_index, int)
    ):
        raise ResumeStateError(f"{label}: invalid Gym task/rollout key")
    if not 0 <= task_index < 100 or rollout_index != 0:
        raise ResumeStateError(f"{label}: unexpected Gym key ({task_index}, {rollout_index})")
    return task_index, rollout_index


def validate_partial_resume_state(
    *,
    launch_input: Path,
    rollouts: Path,
    materialized_inputs: Path,
    failures: Path | None,
    run_config: Path,
    expected_launch_sha256: str,
    expected_model: str,
    deliverables_dir: Path,
) -> int:
    actual_hash = hashlib.sha256(launch_input.read_bytes()).hexdigest()
    if actual_hash != expected_launch_sha256:
        raise ResumeStateError(f"launch hash mismatch: {actual_hash}")

    expected_rows = _read_jsonl(launch_input)
    expected_ids = [str(row.get("task_id")) for row in expected_rows]
    if len(expected_ids) != 100 or len(set(expected_ids)) != 100:
        raise ResumeStateError("launch input does not contain 100 unique task IDs")

    materialized = _read_jsonl(materialized_inputs)
    if len(materialized) != 100:
        raise ResumeStateError("materialized input does not contain 100 rows")
    materialized_by_key: dict[tuple[int, int], str] = {}
    for row_number, row in enumerate(materialized, 1):
        key = _rollout_key(row, label=f"materialized row {row_number}")
        if key in materialized_by_key:
            raise ResumeStateError(f"materialized inputs duplicate Gym key {key}")
        task_id = str(row.get("task_id"))
        if task_id != expected_ids[key[0]]:
            raise ResumeStateError(f"materialized Gym key {key} maps to {task_id}, expected {expected_ids[key[0]]}")
        agent_ref = row.get("agent_ref")
        if not isinstance(agent_ref, dict) or agent_ref.get("name") != "gdpval_stirrup_agent":
            raise ResumeStateError(f"{task_id}: materialized input has an unexpected agent")
        materialized_by_key[key] = task_id
    expected_keys = {(index, 0) for index in range(100)}
    if set(materialized_by_key) != expected_keys:
        raise ResumeStateError("materialized Gym keys do not cover the 100-task batch")

    outputs = _read_jsonl(rollouts)
    if len(outputs) > 100:
        raise ResumeStateError("partial rollout output contains more than 100 rows")
    output_keys: set[tuple[int, int]] = set()
    for row_number, row in enumerate(outputs, 1):
        key = _rollout_key(row, label=f"rollout row {row_number}")
        if key in output_keys:
            raise ResumeStateError(f"partial rollout output duplicates Gym key {key}")
        output_keys.add(key)
        task_id = str(row.get("task_id"))
        if materialized_by_key.get(key) != task_id:
            raise ResumeStateError(
                f"rollout Gym key {key} maps to {task_id}, materialized input maps to {materialized_by_key.get(key)}"
            )
        if row.get("execute_only") is not True:
            raise ResumeStateError(f"{task_id}: execute_only is not true")
        if row.get("reward") is not None or row.get("judge_response") is not None:
            raise ResumeStateError(f"{task_id}: judge fields are populated")
        agent_ref = row.get("agent_ref")
        if not isinstance(agent_ref, dict) or agent_ref.get("name") != "gdpval_stirrup_agent":
            raise ResumeStateError(f"{task_id}: rollout has an unexpected agent")
        response = row.get("response")
        if not isinstance(response, dict) or response.get("error") is not None:
            raise ResumeStateError(f"{task_id}: rollout has an invalid response")
        if response.get("model") != expected_model:
            raise ResumeStateError(f"{task_id}: response model mismatch")

        expected_dir = (deliverables_dir / f"task_{task_id}" / "repeat_0").resolve()
        reported_dir = row.get("deliverables_dir")
        if not isinstance(reported_dir, str) or Path(reported_dir).resolve() != expected_dir:
            raise ResumeStateError(f"{task_id}: deliverables directory mismatch")
        finish_marker = expected_dir / "finish_params.json"
        if not finish_marker.is_file():
            raise ResumeStateError(f"{task_id}: cached rollout is missing {finish_marker}")
        finish_value = json.loads(finish_marker.read_text(encoding="utf-8"))
        if not isinstance(finish_value, dict):
            raise ResumeStateError(f"{task_id}: finish marker is not an object")

    if failures is not None and failures.is_file():
        _read_jsonl(failures)

    fingerprint = run_config.read_text(encoding="utf-8").strip()
    if re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None:
        raise ResumeStateError("run_config.sha256 is missing or malformed")

    return len(outputs)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-input", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--materialized-inputs", type=Path, required=True)
    parser.add_argument("--failures", type=Path)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--expected-launch-sha256", required=True)
    parser.add_argument("--expected-model", required=True)
    parser.add_argument("--deliverables-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        completed = validate_partial_resume_state(
            launch_input=args.launch_input,
            rollouts=args.rollouts,
            materialized_inputs=args.materialized_inputs,
            failures=args.failures,
            run_config=args.run_config,
            expected_launch_sha256=args.expected_launch_sha256,
            expected_model=args.expected_model,
            deliverables_dir=args.deliverables_dir,
        )
    except (OSError, json.JSONDecodeError, ResumeStateError) as exc:
        print(f"ERROR: partial rollout state is not resumable: {exc}")
        return 1
    print(f"Partial full state is resumable: {completed}/100 completed rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
