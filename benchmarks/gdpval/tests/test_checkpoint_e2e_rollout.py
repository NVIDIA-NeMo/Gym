# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

from benchmarks.gdpval.hsg.checkpoint_e2e import rollout_shard_coverage as coverage


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "benchmarks" / "gdpval" / "hsg" / "checkpoint_e2e"
LIFECYCLE = PACKAGE / "rollout_lifecycle.sh"
ROLLOUT = PACKAGE / "gdpval_rollout.sbatch"


def _write_dataset(path: Path, task_ids: list[str]) -> Path:
    path.write_text(
        "".join(json.dumps({"task_id": task_id}) + "\n" for task_id in task_ids),
        encoding="utf-8",
    )
    return path


def _write_marker(root: Path, task_id: str, payload: str = "{}\n") -> Path:
    marker = root / f"task_{task_id}" / "repeat_0" / "finish_params.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(payload, encoding="utf-8")
    return marker


def _run_lifecycle(body: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            "-c",
            f"""
set -euo pipefail
source {LIFECYCLE!s}
{body}
""",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_cleanup_dead_pids_is_idempotent_and_non_failing() -> None:
    result = _run_lifecycle(
        """
kill_calls=0
kill() { kill_calls=$((kill_calls + 1)); return 1; }
gym_pid=12345
serve_pids=(23456)
gdpval_rollout_cleanup
gdpval_rollout_cleanup
printf 'kill_calls=%s cleanup_done=%s\\n' "$kill_calls" "$GDPVAL_ROLLOUT_CLEANUP_DONE"
"""
    )

    assert result.returncode == 0, (result.stdout, result.stderr)
    assert result.stdout == "kill_calls=3 cleanup_done=true\n"


def test_exit_trap_preserves_success_with_already_exited_gym() -> None:
    result = _run_lifecycle(
        """
true &
gym_pid=$!
wait "$gym_pid"
serve_pids=()
trap gdpval_rollout_on_exit EXIT
exit 0
"""
    )

    assert result.returncode == 0, (result.stdout, result.stderr)


def test_exit_trap_preserves_original_failure_code() -> None:
    result = _run_lifecycle(
        """
kill() { return 1; }
gym_pid=12345
serve_pids=()
trap gdpval_rollout_on_exit EXIT
exit 37
"""
    )

    assert result.returncode == 37, (result.stdout, result.stderr)


def test_materialized_dataset_is_private_idempotent_and_drift_checked(tmp_path: Path) -> None:
    immutable_dir = tmp_path / "immutable-shards"
    immutable_dir.mkdir()
    source = _write_dataset(immutable_dir / "shard_00.jsonl", ["task-a"])
    destination = tmp_path / "rollout_s00" / "input" / "dataset.jsonl"
    command = f"gdpval_rollout_materialize_dataset {shlex.quote(str(source))} {shlex.quote(str(destination))}"

    first = _run_lifecycle(command)
    second = _run_lifecycle(command)
    (destination.parent / "dataset_prepare.jsonl").write_text("generated\n", encoding="utf-8")

    assert first.returncode == second.returncode == 0
    assert destination.read_bytes() == source.read_bytes()
    assert [path.name for path in immutable_dir.iterdir()] == [source.name]

    destination.chmod(0o600)
    destination.write_text('{"task_id":"drift"}\n', encoding="utf-8")
    drift = _run_lifecycle(command)
    assert drift.returncode == 64
    assert "materialized rollout dataset drift" in drift.stderr


def test_shard_coverage_ignores_completed_sibling_tasks_in_shared_root(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path / "shard.jsonl", ["shard-a", "shard-b"])
    deliverables = tmp_path / "deliverables"
    _write_marker(deliverables, "shard-a")
    _write_marker(deliverables, "shard-b", "null\n")
    _write_marker(deliverables, "sibling-1")
    _write_marker(deliverables, "sibling-2")

    report = coverage.shard_coverage(dataset, deliverables)

    assert report == {
        "status": "PASS",
        "dataset": str(dataset.resolve()),
        "deliverables": str(deliverables.resolve()),
        "expected": 2,
        "completed": 2,
        "missing": [],
        "invalid": {},
        "shared_markers": 4,
    }
    assert coverage.main(["--dataset", str(dataset), "--deliverables", str(deliverables)]) == 0


def test_shard_coverage_fails_for_missing_or_malformed_own_marker(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path / "shard.jsonl", ["good", "bad", "missing"])
    deliverables = tmp_path / "deliverables"
    _write_marker(deliverables, "good")
    _write_marker(deliverables, "bad", "[]\n")
    _write_marker(deliverables, "sibling")

    report = coverage.shard_coverage(dataset, deliverables)

    assert report["status"] == "INCOMPLETE"
    assert report["completed"] == 1
    assert report["missing"] == ["bad", "missing"]
    assert report["invalid"] == {
        "bad": "not_an_object_or_null",
        "missing": "missing_or_nonregular",
    }
    assert coverage.main(["--dataset", str(dataset), "--deliverables", str(deliverables)]) == 1


def test_rollout_wrapper_uses_safe_exit_and_exact_shard_postcondition() -> None:
    script = ROLLOUT.read_text(encoding="utf-8")

    assert '"${ROLLOUT_PACKAGE_DIR:?' in script
    assert 'SCRIPT_DIR="$(cd -P -- "$ROLLOUT_PACKAGE_DIR"' in script
    assert "${BASH_SOURCE[0]}" not in script
    assert 'source "$ROLLOUT_LIFECYCLE_SH"' in script
    assert "trap gdpval_rollout_on_exit EXIT" in script
    assert "gdpval_rollout_cleanup" in script
    assert "trap cleanup EXIT" not in script
    assert '--dataset "$DATASET" --deliverables "$PERSIST_DELIVERABLES_DIR"' in script
    assert 'WORKING_DATASET="$RUN_DIR/input/dataset.jsonl"' in script
    assert "jsonl_fpath: $WORKING_DATASET" in script
    assert "jsonl_fpath: $DATASET" not in script
    assert 'replica_dir="$RUN_DIR/replica_$r/rotation_$ROTATION"' in script
    assert 'OUTPUT_DIR="$replica_dir"' in script
    assert 'find "${replica_dirs[$r]}/server_info"' in script
    assert 'find "$RUN_DIR/replica_$r/server_info"' not in script
    assert "if (( gym_rc != 0 )); then" in script
    assert 'exit "$coverage_rc"' in script
    assert 'find "$PERSIST_DELIVERABLES_DIR" -name finish_params.json' not in script


def test_spooled_rollout_resolves_helpers_from_explicit_package(tmp_path: Path) -> None:
    spooled = tmp_path / "slurm_script"
    spooled.write_bytes(ROLLOUT.read_bytes())

    result = subprocess.run(
        ["bash", str(spooled)],
        cwd=tmp_path,
        env={**os.environ, "ROLLOUT_PACKAGE_DIR": str(PACKAGE)},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode != 0
    assert "RUN_DIR" in result.stderr
    assert "rollout lifecycle helper is unreadable" not in result.stderr
    assert "rollout shard coverage helper is unreadable" not in result.stderr
