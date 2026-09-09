# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from benchmarks.gdpval.hsg.checkpoint_e2e import rollout_shard_coverage as coverage


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "benchmarks" / "gdpval" / "hsg" / "checkpoint_e2e"
LIFECYCLE = PACKAGE / "rollout_lifecycle.sh"
ROLLOUT = PACKAGE / "gdpval_rollout.sbatch"
MARS_HELPER = PACKAGE / "mars_node_local.sh"


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


def test_materialized_dataset_preserves_reference_urls_across_resume(tmp_path: Path) -> None:
    source = tmp_path / "immutable.jsonl"
    contents = (
        '{ "task_id": "reference-task", "reference_file_urls": '
        '["/lustre/references/report.pdf", "file:///lustre/references/input.csv"] }\n\n'
    ).encode()
    source.write_bytes(contents)
    for rotation in range(2):
        destination = tmp_path / f"rotation-{rotation}" / "input" / "dataset.jsonl"
        result = _run_lifecycle(
            f"gdpval_rollout_materialize_dataset {shlex.quote(str(source))} {shlex.quote(str(destination))}"
        )

        assert result.returncode == 0, (result.stdout, result.stderr)
        assert destination.read_bytes() == contents
    assert source.read_bytes() == contents


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
    assert 'gdpval_rollout_materialize_dataset "$DATASET" "$RUN_DIR/input/dataset.jsonl"' in script
    assert "WORKING_DATASET=$MARS_JOB_ROOT/input/dataset.jsonl" in MARS_HELPER.read_text(encoding="utf-8")
    assert "jsonl_fpath: $WORKING_DATASET" in script
    assert "jsonl_fpath: $DATASET" not in script
    assert 'replica_dir="$RUN_DIR/replica_$r/rotation_$ROTATION"' in script
    assert 'OUTPUT_DIR="$replica_dir"' in script
    assert 'find "${replica_dirs[$r]}/server_info"' in script
    assert 'find "$RUN_DIR/replica_$r/server_info"' not in script
    assert "if (( gym_rc != 0 )); then" in script
    assert 'exit "$coverage_rc"' in script
    assert 'find "$PERSIST_DELIVERABLES_DIR" -name finish_params.json' not in script


def test_rollout_wrapper_launches_verified_local_runtime_before_serving() -> None:
    script = ROLLOUT.read_text(encoding="utf-8")
    helper = MARS_HELPER.read_text(encoding="utf-8")

    assert 'mars_stage_rollout_gym "$TREE" "$ROLLOUT_GYM_REVISION"' in helper
    assert "TREE=$MARS_GYM" in helper
    assert "AGENT_SIF=$MARS_GDPVAL_SIF" in helper
    assert "export GDPVAL_CONTAINER_PATH=$AGENT_SIF" in script
    assert "LOCAL_COMPONENT_VENVS=$MARS_JOB_ROOT/component_venvs" in helper
    assert 'mars_stage_file "$RUN_DIR/input/dataset.jsonl" "$WORKING_DATASET"' in helper
    assert 'setsid "$MARS_PYTHON" "$GYM_ENTRYPOINT" eval run' in script
    assert '++uv_venv_dir="$LOCAL_COMPONENT_VENVS"' in script
    assert '++uv_cache_dir="$UV_CACHE_DIR"' in script
    assert '++python_version="$MARS_PYTHON"' in script
    assert '"${MAX_OUTPUT_TOKENS:-262144}" --resume' in script
    assert "++reuse_existing_data_preparation=" not in script
    assert 'python3 "$ROLLOUT_SHARD_COVERAGE_PY"' not in script
    assert script.index("mars_prepare_rollout_runtime") < script.index('bash "$POLICY_SERVE_SCRIPT" &')


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


@pytest.mark.parametrize("filesystem", ["ext4", "lustre", "lustre_lite"])
def test_rollout_rotations_get_fresh_local_workspaces_and_reject_lustre(filesystem: str) -> None:
    # Use a short scratch prefix because the production helper also enforces
    # Ray's socket path limit. Only the cluster mount name is relocated.
    with tempfile.TemporaryDirectory(prefix="rt-", dir="/private/tmp" if sys.platform == "darwin" else "/tmp") as name:
        scratch = Path(name)
        helper = scratch / "mars_node_local.sh"
        helper.write_text(
            (PACKAGE / "mars_node_local.sh").read_text(encoding="utf-8").replace("/raid/scratch/", f"{scratch}/"),
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                "bash",
                "-c",
                r"""
set -euo pipefail
source "$1"
stat() { printf '%s\n' "$FAKE_FILESYSTEM"; }
sha256sum() { "$TEST_PYTHON" -c 'import hashlib,sys; print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())'; }
mars_init rollout-123 rollout-0 /lustre/campaign
printf '%s\n' "$MARS_JOB_ROOT" "$TMPDIR" "$UV_CACHE_DIR"
mars_init rollout-123 rollout-1 /lustre/campaign
printf '%s\n' "$MARS_JOB_ROOT" "$TMPDIR" "$UV_CACHE_DIR"
""",
                "rotation-test",
                str(helper),
            ],
            env={
                **os.environ,
                "SLURM_JOB_ID": "123",
                "SLURM_JOB_USER": "t",
                "CHECKPOINT_E2E_NODE_LOCAL_BASE": str(scratch / "base"),
                "TEST_PYTHON": sys.executable,
                "FAKE_FILESYSTEM": filesystem,
            },
            text=True,
            capture_output=True,
            check=False,
        )

        if filesystem != "ext4":
            assert result.returncode == 64, (result.stdout, result.stderr)
            assert "resolves to Lustre" in result.stderr
            assert result.stdout == ""
            return

        assert result.returncode == 0, (result.stdout, result.stderr)
        paths = [Path(line) for line in result.stdout.splitlines()]
        assert len(paths) == 6
        first_root, first_tmp, first_cache, second_root, second_tmp, second_cache = paths
        assert first_root != second_root
        assert first_root.name == "123-rollout-0"
        assert second_root.name == "123-rollout-1"
        assert first_tmp.is_relative_to(first_root)
        assert first_cache.is_relative_to(first_root)
        assert second_tmp.is_relative_to(second_root)
        assert second_cache.is_relative_to(second_root)
        assert all(path.is_dir() and path.is_relative_to(scratch) for path in paths)


@pytest.mark.parametrize("outcome", ["success", "sync-failure", "source-drift"])
def test_rollout_gym_builds_a_fresh_locked_environment_and_never_adopts_a_partial_install(
    tmp_path: Path, outcome: str
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for name, contents in (
        (".python-version", "3.13.14\n"),
        ("uv.lock", "version = 1\n"),
        ("pyproject.toml", "[project]\nname = 'fixture'\n"),
    ):
        (source / name).write_text(contents, encoding="utf-8")
    (source / ".venv").mkdir()
    (source / ".venv" / "stale-source-environment").touch()
    job = tmp_path / "job"
    events = tmp_path / "sync-events"
    result = subprocess.run(
        [
            "bash",
            "-c",
            r"""
set -euo pipefail
source "$1"
mars_validate_source_dir() { [[ -d $1 ]]; }
git() {
    case $3 in
        rev-parse) printf '%s\n' "$REVISION" ;;
        diff) [[ $TEST_OUTCOME != source-drift ]] ;;
        archive) tar -c -C "$SOURCE" .python-version uv.lock pyproject.toml ;;
        *) return 97 ;;
    esac
}
uv_stub() {
    [[ ! -e $UV_PROJECT_ENVIRONMENT ]] || return 88
    [[ -z ${PYTHONHOME+x} && -z ${PYTHONPATH+x} && -z ${NEMO_GYM_EXTRA_ROOTS+x} && -z ${VIRTUAL_ENV+x} ]] || return 89
    printf '%s\n' "$PWD" "$UV_PROJECT_ENVIRONMENT" "$*" > "$EVENTS"
    mkdir -p "$UV_PROJECT_ENVIRONMENT/bin" "$UV_PYTHON_INSTALL_DIR"
    [[ $TEST_OUTCOME != sync-failure ]] || { touch "$UV_PROJECT_ENVIRONMENT/partial"; return 29; }
    touch "$UV_PYTHON_INSTALL_DIR/python"
    chmod 0700 "$UV_PYTHON_INSTALL_DIR/python"
    ln -s "$UV_PYTHON_INSTALL_DIR/python" "$UV_PROJECT_ENVIRONMENT/bin/python"
}
MARS_UV=uv_stub
if mars_stage_rollout_gym "$SOURCE" "$REVISION"; then first=0; else first=$?; fi
if mars_stage_rollout_gym "$SOURCE" "$REVISION"; then second=0; else second=$?; fi
printf 'first=%s second=%s\n' "$first" "$second"
[[ -z ${UV_PROJECT_ENVIRONMENT+x} ]]
""",
            "stage-gym-test",
            str(MARS_HELPER),
        ],
        env={
            **os.environ,
            "SOURCE": str(source),
            "MARS_JOB_ROOT": str(job),
            "REVISION": "a" * 40,
            "TEST_OUTCOME": outcome,
            "EVENTS": str(events),
            "PYTHONHOME": "/lustre/stale-python",
            "PYTHONPATH": "/lustre/stale-modules",
            "NEMO_GYM_EXTRA_ROOTS": "/lustre/stale-components",
            "VIRTUAL_ENV": "/lustre/stale-environment",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, (result.stdout, result.stderr)
    expected_code = {"success": 0, "sync-failure": 29, "source-drift": 64}[outcome]
    assert result.stdout == f"first={expected_code} second=64\n"
    staged = job / "source" / "gym"
    if outcome == "source-drift":
        assert not events.exists()
        assert "runtime sources differ from the pinned commit" in result.stderr
        return

    assert events.read_text().splitlines() == [
        str(staged),
        str(staged / ".venv"),
        "sync --frozen --no-dev --managed-python --python 3.13.14",
    ]
    assert not (staged / ".venv" / "stale-source-environment").exists()
    assert (source / ".venv" / "stale-source-environment").exists()
    assert "refusing stale rollout source" in result.stderr
    assert (staged / ".checkpoint_e2e_revision").exists() is (outcome == "success")
    if outcome == "success":
        assert (staged / ".venv/bin/python").resolve().is_relative_to(job / "python")


def test_rollout_runtime_resets_inherited_reference_caches_before_starting_python(tmp_path: Path) -> None:
    job = tmp_path / "local-job"
    result = subprocess.run(
        [
            "bash",
            "-c",
            r"""
set -euo pipefail
source "$1"
mars_stage_rollout_gym() {
    [[ -d $HF_HOME && -d $HF_DATASETS_CACHE && -d $GDPVAL_REF_FILES_DIR ]] || return 68
    printf '%s\n' "$HF_HOME" "$HF_DATASETS_CACHE" "$GDPVAL_REF_FILES_DIR"
    return 67
}
if mars_prepare_rollout_runtime; then exit 1; else [[ $? == 67 ]]; fi
""",
            "cache-reset-test",
            str(MARS_HELPER),
        ],
        env={
            **os.environ,
            "MARS_JOB_ROOT": str(job),
            "MARS_USER": "test",
            "MARS_UV_DIR": str(job / "bin"),
            "SLURM_JOB_ID": "123",
            "TREE": "/lustre/source",
            "ROLLOUT_GYM_REVISION": "a" * 40,
            "HF_HOME": "/lustre/shared/huggingface",
            "HF_DATASETS_CACHE": "/lustre/shared/datasets",
            "GDPVAL_REF_FILES_DIR": "/lustre/shared/references",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, (result.stdout, result.stderr)
    assert result.stdout.splitlines() == [
        str(job / "cache/huggingface"),
        str(job / "cache/huggingface/datasets"),
        str(job / "tmp/reference_files"),
    ]
