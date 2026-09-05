# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "hsg" / "checkpoint_e2e" / "transition_receipt.py"
PACKAGE_IDENTITY_SCRIPT = SCRIPT.with_name("prepare_existing_campaign.py")
SPEC = importlib.util.spec_from_file_location("checkpoint_e2e_transition_receipt", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
TRANSITION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRANSITION)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _fixture(tmp_path: Path) -> dict[str, object]:
    fingerprint = "a" * 64
    task_ids = [f"task-{index:03d}" for index in range(220)]
    tail_ids = task_ids[-3:]
    stage0 = [
        {
            "stage_index": 0,
            "task_id": task_ids[0],
            "_ng_rollout_index": 0,
            "verify_cache_namespace": fingerprint,
        }
    ]
    stage1 = [
        {
            "stage_index": 1,
            "task_id": task_id,
            "_ng_rollout_index": 0,
            "verify_cache_namespace": fingerprint,
            "value": index,
        }
        for index, task_id in enumerate(task_ids)
    ]
    pre_tail = tmp_path / "pre-tail.jsonl"
    final_output = tmp_path / "gdpval_aav2.jsonl"
    _jsonl(pre_tail, stage0 + stage1[:-3])
    _jsonl(final_output, stage0 + stage1)

    old_manifest = tmp_path / "old-runtime.json"
    new_manifest = tmp_path / "new-runtime.json"
    _json(
        old_manifest,
        {
            "schema": "gdpval.transport-runtime.v2",
            "revision": "1" * 40,
            "output_sha256": {"resources_servers/gdpval/app.py": "2" * 64},
        },
    )
    _json(
        new_manifest,
        {
            "schema": TRANSITION.NEW_RUNTIME_SCHEMA,
            "revision": "1" * 40,
            "output_sha256": {"resources_servers/gdpval/app.py": "3" * 64},
        },
    )

    package = tmp_path / "package"
    package.mkdir()
    (package / "VERSION").write_text("1.4.11\n", encoding="utf-8")
    (package / "judge.sbatch").write_text("#!/bin/bash\ntrue\n", encoding="utf-8")

    seed = tmp_path / "seed_receipt.json"
    _json(
        seed,
        {
            "status": "READY",
            "applied": True,
            "plan_fingerprint": fingerprint,
            "target_stage1_rows_before": 0,
            "imported_stage1_rows": 217,
            "stage1_rows_after": 217,
            "stage1_rows_remaining": 3,
            "target_output_sha256_before": "4" * 64,
            "target_output_sha256_after": _sha256(pre_tail),
            "imported_task_ids": task_ids[:-3],
        },
    )

    result = tmp_path / "strict-result.json"
    _json(
        result,
        {
            "status": "PASS",
            "output": str(final_output.resolve()),
            "rows": 265,
            "stage0_tasks": 45,
            "stage1_tasks": 220,
            "stage0_trials": 180,
            "stage1_trials": 880,
            "invalid": 0,
            "stage0_partial": False,
            "stage0_elo": 1200.0,
            "stage0_normalized_elo": 0.35,
            "eval_elo": 1288.25,
            "normalized_elo": 0.394125,
            "top4": ["ref-a", "ref-b", "ref-c", "ref-d"],
        },
    )
    return {
        "fingerprint": fingerprint,
        "tail_ids": tail_ids,
        "pre_tail": pre_tail,
        "final_output": final_output,
        "old_manifest": old_manifest,
        "new_manifest": new_manifest,
        "package": package,
        "seed": seed,
        "result": result,
    }


def _arguments(fixture: dict[str, object], output: Path, *, pre_tail: bool = True) -> list[str]:
    arguments = [
        "--old-runtime-manifest",
        str(fixture["old_manifest"]),
        "--new-runtime-manifest",
        str(fixture["new_manifest"]),
        "--package-root",
        str(fixture["package"]),
        "--seed-receipt",
        str(fixture["seed"]),
        "--frozen-fingerprint",
        str(fixture["fingerprint"]),
        "--slurm-job-id",
        "6532001",
        "--final-output",
        str(fixture["final_output"]),
        "--final-result",
        str(fixture["result"]),
        "--output",
        str(output),
    ]
    if pre_tail:
        arguments.extend(("--pre-tail-output", str(fixture["pre_tail"])))
    for task_id in fixture["tail_ids"]:
        arguments.extend(("--tail-task-id", str(task_id)))
    return arguments


def test_publishes_bound_transition_and_exact_three_row_delta(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "transition.json"

    assert TRANSITION.main(_arguments(fixture, output)) == 0
    assert TRANSITION.main(_arguments(fixture, output)) == 0

    receipt = json.loads(output.read_text(encoding="utf-8"))
    package_identity = json.loads(
        subprocess.check_output(
            [sys.executable, str(PACKAGE_IDENTITY_SCRIPT), "identify-package", "--package", str(fixture["package"])],
            text=True,
        )
    )
    assert receipt["schema"] == TRANSITION.SCHEMA
    assert receipt["runtime_transition"]["old"]["schema"] == "gdpval.transport-runtime.v2"
    assert receipt["runtime_transition"]["new"]["schema"] == "gdpval.transport-runtime.v3"
    assert receipt["package"]["version"] == "1.4.11"
    assert receipt["package"]["inventory_sha256"] == package_identity["inventory_sha256"]
    assert receipt["seed"]["pre_tail_output"]["sha256"] == _sha256(fixture["pre_tail"])
    assert receipt["tail"]["task_ids"] == sorted(fixture["tail_ids"])
    assert receipt["tail"]["slurm_job_ids"] == ["6532001"]
    assert receipt["final"]["output"]["sha256"] == _sha256(fixture["final_output"])
    assert receipt["final"]["result"]["sha256"] == _sha256(fixture["result"])
    assert receipt["final"]["result_fields"]["eval_elo"] == 1288.25
    assert stat.S_IMODE(output.stat().st_mode) == 0o400
    assert output.with_suffix(".json.sha256").is_file()


def test_without_snapshot_still_binds_seed_declared_pre_tail_sha(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "transition.json"

    assert TRANSITION.main(_arguments(fixture, output, pre_tail=False)) == 0

    receipt = json.loads(output.read_text(encoding="utf-8"))
    seed = json.loads(Path(fixture["seed"]).read_text(encoding="utf-8"))
    assert receipt["seed"]["pre_tail_output"] == {"sha256": seed["target_output_sha256_after"]}


def test_preserves_multiple_slurm_attempts_in_supplied_order(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "transition.json"
    arguments = _arguments(fixture, output)
    job_flag = arguments.index("--slurm-job-id")
    arguments[job_flag : job_flag + 2] = ["--slurm-job-id", "6532088", "--slurm-job-id", "6532265"]

    assert TRANSITION.main(arguments) == 0

    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["tail"]["slurm_job_ids"] == ["6532088", "6532265"]


def test_rejects_duplicate_slurm_attempt_ids(tmp_path: Path, capsys) -> None:
    fixture = _fixture(tmp_path)
    arguments = _arguments(fixture, tmp_path / "transition.json")
    job_flag = arguments.index("--slurm-job-id")
    arguments[job_flag : job_flag + 2] = ["--slurm-job-id", "6532088", "--slurm-job-id", "6532088"]

    assert TRANSITION.main(arguments) == 64
    assert "must be unique and ordered" in capsys.readouterr().err


def test_accepts_old_v3_runtime_manifest(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    old_manifest = Path(fixture["old_manifest"])
    old = json.loads(old_manifest.read_text(encoding="utf-8"))
    old["schema"] = "gdpval.transport-runtime.v3"
    _json(old_manifest, old)

    output = tmp_path / "transition.json"
    assert TRANSITION.main(_arguments(fixture, output)) == 0
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["runtime_transition"]["old"]["schema"] == "gdpval.transport-runtime.v3"


def test_rejects_new_v2_runtime_manifest(tmp_path: Path, capsys) -> None:
    fixture = _fixture(tmp_path)
    new_manifest = Path(fixture["new_manifest"])
    new = json.loads(new_manifest.read_text(encoding="utf-8"))
    new["schema"] = "gdpval.transport-runtime.v2"
    _json(new_manifest, new)

    assert TRANSITION.main(_arguments(fixture, tmp_path / "transition.json")) == 64
    assert "new runtime manifest schema" in capsys.readouterr().err


def test_rejects_any_non_tail_delta_when_snapshot_is_available(tmp_path: Path, capsys) -> None:
    fixture = _fixture(tmp_path)
    final_rows = [json.loads(line) for line in Path(fixture["final_output"]).read_text().splitlines()]
    final_rows[10]["value"] = "mutated"
    _jsonl(Path(fixture["final_output"]), final_rows)

    assert TRANSITION.main(_arguments(fixture, tmp_path / "transition.json")) == 64
    assert "changed or removed a pre-tail row" in capsys.readouterr().err


def test_rejects_seed_receipt_with_wrong_fingerprint(tmp_path: Path, capsys) -> None:
    fixture = _fixture(tmp_path)
    seed_path = Path(fixture["seed"])
    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    seed["plan_fingerprint"] = "b" * 64
    _json(seed_path, seed)

    assert TRANSITION.main(_arguments(fixture, tmp_path / "transition.json")) == 64
    assert "fingerprint differs" in capsys.readouterr().err
