# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "hsg" / "checkpoint_e2e" / "seed_compatible_judgments.py"
SPEC = importlib.util.spec_from_file_location("seed_compatible_judgments", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SEED = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SEED)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _valid_row(
    task_id: str,
    *,
    task_index: int,
    reference: str,
    namespace: str,
    deliverables_dir: Path,
) -> dict:
    return {
        "task_id": task_id,
        "stage_index": 1,
        "_ng_task_index": task_index,
        "_ng_rollout_index": 0,
        "reference_ids": [reference],
        "deliverables_dir": str(deliverables_dir),
        "verify_cache_namespace": namespace,
        "invalid_judge_response": False,
        "response": {},
        "judge_response": {
            "error": None,
            "scoring_error": None,
            "ref_errors": {},
            "total_judged": 4,
            "total_invalid": 0,
            "judge_panel": [{"name": name, "model": "judge", "weight": 1.0} for name in SEED.EXPECTED_JUDGE_MODELS],
        },
    }


def _fixture(tmp_path: Path, *, target_row_index: int | None) -> tuple[Path, Path, Path, str, str]:
    target = tmp_path / "target"
    source = tmp_path / "source"
    deliverables = tmp_path / "candidate"
    target.mkdir()
    source.mkdir()
    deliverables.mkdir()

    task_ids = [f"task-{index:03d}" for index in range(220)]
    imported_task = task_ids[57]
    plan_order = task_ids[57:] + task_ids[:57]
    reference = "deepseek_v4_pro"
    namespace = "a" * 64
    assignments = {task_id: reference for task_id in plan_order}
    _write_jsonl(
        target / "preprocessed_datasets" / "benchmark.jsonl",
        [{"task_id": task_id} for task_id in task_ids],
    )
    _write_jsonl(
        target / "gdpval_aav2_multistage_state.jsonl",
        [
            {
                "stage_index": 1,
                "status": "planned",
                "task_ids": plan_order,
                "task_reference_ids": assignments,
                "fingerprint": namespace,
            }
        ],
    )
    task_deliverables = deliverables / f"task_{imported_task}" / "repeat_0"
    task_deliverables.mkdir(parents=True)
    row = _valid_row(
        imported_task,
        task_index=999 if target_row_index is None else target_row_index,
        reference=reference,
        namespace=namespace,
        deliverables_dir=task_deliverables,
    )
    _write_jsonl(target / "gdpval_aav2.jsonl", [] if target_row_index is None else [row])
    _write_jsonl(source / "gdpval_aav2.jsonl", [row] if target_row_index is None else [])
    return target, source, deliverables, imported_task, namespace


def test_import_rebinds_to_preprocessed_dataset_index_not_stage_plan_position(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    target, source, deliverables, imported_task, _ = _fixture(tmp_path, target_row_index=None)
    preview = tmp_path / "preview.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--target-judge",
            str(target),
            "--source-judge",
            str(source),
            "--target-deliverables",
            str(deliverables),
            "--preview-output",
            str(preview),
        ],
    )

    assert SEED.main() == 0

    summary = json.loads(capsys.readouterr().out)
    rows = [json.loads(line) for line in preview.read_text().splitlines()]
    assert summary["status"] == "READY"
    assert summary["imported_stage1_rows"] == 1
    assert summary["reindexed_existing_stage1_rows"] == 0
    assert rows[0]["task_id"] == imported_task
    assert rows[0]["_ng_task_index"] == 57
    assert rows[0]["_ng_task_index"] != 0  # Its position in the frozen Stage-1 plan.


def test_apply_supersedes_wrong_seed_with_new_backup_and_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    target, source, deliverables, imported_task, _ = _fixture(tmp_path, target_row_index=0)
    prior_backup = target / "seed_backup_prior"
    prior_backup.mkdir()
    (prior_backup / "marker").write_text("preserve me", encoding="utf-8")
    (target / "seed_receipt_prior.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--target-judge",
            str(target),
            "--source-judge",
            str(source),
            "--target-deliverables",
            str(deliverables),
            "--apply",
        ],
    )

    assert SEED.main() == 0

    summary = json.loads(capsys.readouterr().out)
    rows = [json.loads(line) for line in (target / "gdpval_aav2.jsonl").read_text().splitlines()]
    backup = Path(summary["backup_dir"])
    old_rows = [json.loads(line) for line in (backup / "gdpval_aav2.jsonl").read_text().splitlines()]
    assert summary["status"] == "READY"
    assert summary["imported_stage1_rows"] == 0
    assert summary["reindexed_existing_stage1_rows"] == 1
    assert summary["reindexed_task_ids"] == [imported_task]
    assert rows[0]["_ng_task_index"] == 57
    assert old_rows[0]["_ng_task_index"] == 0
    assert (prior_backup / "marker").read_text(encoding="utf-8") == "preserve me"
    assert (target / "seed_receipt_prior.json").exists()
    receipts = list(target.glob("seed_receipt_*.json"))
    assert len(receipts) == 2


def test_rejects_preprocessed_identity_drift_before_writing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target, source, deliverables, _, _ = _fixture(tmp_path, target_row_index=None)
    rows = [
        json.loads(line) for line in (target / "preprocessed_datasets" / "benchmark.jsonl").read_text().splitlines()
    ]
    rows[-1]["task_id"] = "unexpected-task"
    _write_jsonl(target / "preprocessed_datasets" / "benchmark.jsonl", rows)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--target-judge",
            str(target),
            "--source-judge",
            str(source),
            "--target-deliverables",
            str(deliverables),
        ],
    )

    with pytest.raises(SEED.SeedError, match="task identities do not match"):
        SEED.main()
