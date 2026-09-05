# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from benchmarks.gdpval.hsg.checkpoint_e2e import campaign, judge_state


FINGERPRINT = "a" * 64


def test_sanitizer_contract_constants_match_final_validator() -> None:
    assert judge_state.EXPECTED_JUDGE_MODELS == campaign.EXPECTED_JUDGE_MODELS
    assert judge_state.EXPECTED_FINAL_STAGE_INDEX == 1
    assert judge_state.EXPECTED_STAGE_ROW_COUNTS == {
        0: campaign.STAGE0_PLANNED_TASKS,
        1: campaign.STAGE1_TASKS,
    }


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path.resolve()


def _valid_row(
    deliverables: Path,
    *,
    stage: int,
    task_index: int,
    task_id: str,
    fingerprint: str = FINGERPRINT,
    reference: str = "ref-a",
) -> dict:
    repeat = deliverables / f"task_{task_id}" / "repeat_0"
    repeat.mkdir(parents=True, exist_ok=True)
    (repeat / "answer.txt").write_text("answer\n", encoding="utf-8")
    (repeat / "finish_params.json").write_text("{}\n", encoding="utf-8")
    matchup = {
        "ref_id": reference,
        "ref_repeat": "repeat_0",
        "winner": "B",
        "win_count_a": 1,
        "win_count_b": 2,
        "tie_count": 1,
        "task_count": 4,
        "invalid_count": 0,
        "trial_judges": ["gpt-5.5", "gemini-3.1-pro", "claude-opus-4.8", "gpt-5.5"],
        "per_judge": {
            "gpt-5.5": {
                "win_count_a": 0,
                "win_count_b": 1,
                "tie_count": 1,
                "trials": 2,
                "invalid_count": 0,
            },
            "gemini-3.1-pro": {
                "win_count_a": 1,
                "win_count_b": 0,
                "tie_count": 0,
                "trials": 1,
                "invalid_count": 0,
            },
            "claude-opus-4.8": {
                "win_count_a": 0,
                "win_count_b": 1,
                "tie_count": 0,
                "trials": 1,
                "invalid_count": 0,
            },
        },
    }
    return {
        "task_id": task_id,
        "deliverables_dir": str(repeat.resolve()),
        "reference_ids": [reference],
        "verify_cache_namespace": fingerprint,
        "stage_index": stage,
        "expected_final_stage_index": 1,
        "expected_stage_row_count": 45 if stage == 0 else 220,
        "_ng_task_index": task_index,
        "_ng_rollout_index": 0,
        "verify_mode": "comparison",
        "reward": 1.0,
        "invalid_judge_response": False,
        "response": {"error": None},
        "per_reference": {
            reference: {
                "wins": 2,
                "losses": 1,
                "ties": 1,
                "reference_elo": 1000.0,
            }
        },
        "judge_response": {
            "error": None,
            "per_reference": {
                reference: {
                    "wins": 2,
                    "losses": 1,
                    "ties": 1,
                    "reference_elo": 1000.0,
                    "ref_repeat_count": 1,
                }
            },
            "per_ref_repeat": [matchup],
            "total_wins": 2,
            "total_losses": 1,
            "total_ties": 1,
            "total_judged": 4,
            "total_invalid": 0,
            "reference_count": 1,
            "ref_repeat_count": 1,
            "ref_errors": {},
            "av_routed": False,
            "judge_panel": [
                {"name": name, "model": model, "weight": 1.0}
                for name, model in judge_state.EXPECTED_JUDGE_MODELS.items()
            ],
            "per_judge": matchup["per_judge"],
        },
        "total_wins": 2,
        "total_losses": 1,
        "total_ties": 1,
    }


def _cache_path(deliverables: Path, row: dict) -> Path:
    key = judge_state._verify_cache_key(row["reference_ids"], row["verify_cache_namespace"])
    return deliverables / f"task_{row['task_id']}" / f"repeat_{row['_ng_rollout_index']}_verify_response_{key}.json"


def _write_cache(deliverables: Path, row: dict) -> Path:
    path = _cache_path(deliverables, row)
    path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    return path.resolve()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, list[dict], list[Path]]:
    deliverables = (tmp_path / "deliverables").resolve()
    deliverables.mkdir()
    rows = [
        _valid_row(deliverables, stage=0, task_index=0, task_id="valid"),
        _valid_row(deliverables, stage=0, task_index=1, task_id="invalid-flag"),
        _valid_row(deliverables, stage=1, task_index=2, task_id="error"),
        _valid_row(deliverables, stage=1, task_index=3, task_id="short"),
    ]
    rows[1]["invalid_judge_response"] = True
    rows[2]["judge_response"]["ref_errors"] = {"ref-a": ["upstream timeout"]}
    short = rows[3]["judge_response"]["per_ref_repeat"][0]
    short["win_count_b"] = 1
    short["task_count"] = 3
    short["trial_judges"] = short["trial_judges"][:3]
    short["per_judge"].pop("claude-opus-4.8")
    rows[3]["judge_response"]["total_wins"] = 1
    rows[3]["judge_response"]["total_judged"] = 3
    rows[3]["total_wins"] = 1

    caches = [_write_cache(deliverables, row) for row in rows]
    # An invalid cache belonging to another scientific fingerprint is not part
    # of this transaction and must remain untouched.
    old = _valid_row(
        deliverables,
        stage=1,
        task_index=99,
        task_id="old-run",
        fingerprint="b" * 64,
    )
    old["invalid_judge_response"] = True
    caches.append(_write_cache(deliverables, old))

    output = _jsonl(tmp_path / "judge" / "gdpval_aav2.jsonl", rows)
    journal = _jsonl(
        tmp_path / "judge" / "gdpval_aav2_multistage_state.jsonl",
        [
            {"stage_index": 0, "status": "planned", "fingerprint": FINGERPRINT},
            {"stage_index": 1, "status": "planned", "fingerprint": FINGERPRINT},
        ],
    )
    state_root = (tmp_path / "judge-state").resolve()
    return output, journal, deliverables, state_root, rows, caches


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_audit_accepts_raw_verify_cache_without_output_wrapper_receipts(tmp_path: Path) -> None:
    deliverables = (tmp_path / "deliverables").resolve()
    deliverables.mkdir()
    row = _valid_row(deliverables, stage=1, task_index=7, task_id="valid")
    cache_row = json.loads(json.dumps(row))
    cache_row.pop("expected_final_stage_index")
    cache_row.pop("expected_stage_row_count")
    cache = _write_cache(deliverables, cache_row)
    output = _jsonl(tmp_path / "judge" / "gdpval_aav2.jsonl", [row])
    journal = _jsonl(
        tmp_path / "judge" / "gdpval_aav2_multistage_state.jsonl",
        [{"stage_index": 1, "status": "planned", "fingerprint": FINGERPRINT}],
    )

    report = judge_state.audit(
        output=output,
        journal=journal,
        deliverables=deliverables,
        max_attempts=3,
    )

    assert report["status"] == "CLEAN"
    assert report["active_cache_count"] == 1
    assert report["quarantine_count"] == 0
    assert cache.is_file()
    assert _read_rows(output) == [row]


def test_cache_without_output_receipts_still_exposes_real_trial_defect(tmp_path: Path) -> None:
    deliverables = (tmp_path / "deliverables").resolve()
    deliverables.mkdir()
    row = _valid_row(deliverables, stage=1, task_index=7, task_id="poisoned")
    row["judge_response"]["total_judged"] = 2
    cache_row = json.loads(json.dumps(row))
    cache_row.pop("expected_final_stage_index")
    cache_row.pop("expected_stage_row_count")
    cache = _write_cache(deliverables, cache_row)
    output = _jsonl(tmp_path / "judge" / "gdpval_aav2.jsonl", [row])
    journal = _jsonl(
        tmp_path / "judge" / "gdpval_aav2_multistage_state.jsonl",
        [{"stage_index": 1, "status": "planned", "fingerprint": FINGERPRINT}],
    )
    state_root = (tmp_path / "judge-state").resolve()

    report = judge_state.audit(
        output=output,
        journal=journal,
        deliverables=deliverables,
        max_attempts=3,
    )

    assert report["status"] == "SANITIZATION_REQUIRED"
    reasons = next(iter(report["reasons"].values()))
    assert "non_four_total_judged" in reasons
    assert "final_stage_receipt_mismatch" not in reasons
    assert "stage_row_count_receipt_mismatch" not in reasons

    result = judge_state.sanitize(
        output=output,
        journal=journal,
        deliverables=deliverables,
        state_root=state_root,
    )
    assert result["status"] == "SANITIZED"
    assert _read_rows(output) == []
    assert not cache.exists()


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("expected_final_stage_index", 0, "final_stage_receipt_mismatch"),
        ("expected_stage_row_count", 45, "stage_row_count_receipt_mismatch"),
    ],
)
def test_output_wrapper_receipts_remain_strict(tmp_path: Path, field: str, value: int, reason: str) -> None:
    deliverables = (tmp_path / "deliverables").resolve()
    deliverables.mkdir()
    row = _valid_row(deliverables, stage=1, task_index=7, task_id="bad-receipt")
    row[field] = value
    cache_row = json.loads(json.dumps(row))
    cache_row.pop("expected_final_stage_index")
    cache_row.pop("expected_stage_row_count")

    assert reason in judge_state._comparison_defects(
        row,
        label="main output",
        require_output_receipts=True,
    )
    assert reason not in judge_state._comparison_defects(
        cache_row,
        label="raw verify cache",
        require_output_receipts=False,
    )


def test_audit_preserves_valid_legacy_row_from_completed_frozen_stage(tmp_path: Path) -> None:
    deliverables = (tmp_path / "deliverables").resolve()
    deliverables.mkdir()
    row = _valid_row(deliverables, stage=0, task_index=7, task_id="legacy")
    cache = _write_cache(deliverables, row)
    row.pop("verify_cache_namespace")
    cache_row = json.loads(cache.read_text(encoding="utf-8"))
    cache_row.pop("verify_cache_namespace")
    cache.write_text(json.dumps(cache_row) + "\n", encoding="utf-8")
    output = _jsonl(tmp_path / "judge" / "gdpval_aav2.jsonl", [row])
    journal = _jsonl(
        tmp_path / "judge" / "gdpval_aav2_multistage_state.jsonl",
        [
            {
                "stage_index": 0,
                "status": "planned",
                "fingerprint": FINGERPRINT,
                "task_ids": ["legacy"],
                "task_reference_ids": {"legacy": "ref-a"},
            },
            {"stage_index": 0, "status": "complete", "fingerprint": FINGERPRINT},
        ],
    )

    report = judge_state.audit(
        output=output,
        journal=journal,
        deliverables=deliverables,
        max_attempts=3,
    )

    assert report["status"] == "CLEAN"
    assert _read_rows(output) == [row]
    assert cache.is_file()


def test_audit_rejects_missing_namespace_outside_completed_frozen_stage(tmp_path: Path) -> None:
    deliverables = (tmp_path / "deliverables").resolve()
    deliverables.mkdir()
    row = _valid_row(deliverables, stage=0, task_index=7, task_id="unfrozen")
    row.pop("verify_cache_namespace")
    output = _jsonl(tmp_path / "judge" / "gdpval_aav2.jsonl", [row])
    journal = _jsonl(
        tmp_path / "judge" / "gdpval_aav2_multistage_state.jsonl",
        [{"stage_index": 0, "status": "planned", "fingerprint": FINGERPRINT}],
    )

    with pytest.raises(judge_state.JudgeStateError, match="completed frozen stage"):
        judge_state.audit(
            output=output,
            journal=journal,
            deliverables=deliverables,
            max_attempts=3,
        )


def _set_nested(row: dict, path: tuple[str | int, ...], value: object) -> None:
    target: object = row
    for component in path[:-1]:
        target = target[component]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]


@pytest.mark.parametrize(
    ("path", "value", "reason"),
    [
        (("invalid_judge_response",), 0, "invalid_judge_response"),
        (("error",), {}, "top_level_error"),
        (("response", "error"), False, "response_error"),
        (("judge_response", "error"), {}, "judge_error"),
        (("judge_response", "ref_errors"), None, "reference_errors"),
        (("judge_response", "ref_errors"), [], "reference_errors"),
        (("judge_response", "total_judged"), 4.0, "non_four_total_judged"),
        (("judge_response", "total_invalid"), False, "invalid_total"),
        (("reference_ids",), ["ref-a", "ref-b"], "reference_assignment_malformed"),
        (("per_reference",), {}, "top_level_per_reference_malformed"),
        (("per_reference", "ref-a", "wins"), True, "top_level_per_reference_tally_malformed"),
        (("per_reference", "ref-a", "reference_elo"), float("inf"), "reference_elo_malformed"),
        (("judge_response", "judge_panel"), [], "judge_panel_malformed"),
        (("judge_response", "per_judge"), None, "top_level_per_judge_malformed"),
        (("judge_response", "per_ref_repeat"), [], "no_single_matchup_trial_evidence"),
        (
            ("judge_response", "per_ref_repeat", 0, "ref_repeat"),
            "repeat_1",
            "matchup_reference_or_repeat_mismatch",
        ),
        (
            ("judge_response", "per_ref_repeat", 0, "trial_judges", 1),
            "gemini-3.1",
            "invalid_four_trial_judge_schedule",
        ),
        (("judge_response", "per_ref_repeat", 0, "invalid_count"), False, "matchup_invalid_trials"),
        (("judge_response", "per_ref_repeat", 0, "win_count_b"), 1, "matchup_non_four_trials"),
        (
            ("judge_response", "judge_panel", 0, "model"),
            "openai/wrong-model",
            "judge_panel_model_or_weight_mismatch",
        ),
        (
            ("judge_response", "judge_panel", 1, "weight"),
            0.5,
            "judge_panel_model_or_weight_mismatch",
        ),
        (
            ("judge_response", "judge_panel", 2, "name"),
            "claude-4.8",
            "judge_panel_unexpected_member",
        ),
        (
            ("judge_response", "per_judge", "unexpected-judge"),
            {"trials": 0, "invalid_count": 0},
            "top_level_per_judge_malformed",
        ),
        (
            ("judge_response", "per_judge", "gpt-5.5", "invalid_count"),
            1,
            "top_level_per_judge_invalid_trials",
        ),
        (
            ("judge_response", "per_judge", "gpt-5.5", "trials"),
            1,
            "top_level_per_judge_non_four_trials",
        ),
        (("total_wins",), 1, "top_level_tally_mismatch"),
    ],
    ids=[
        "invalid-marker-zero",
        "top-error-empty-object",
        "response-error-false",
        "judge-error-empty-object",
        "reference-errors-none",
        "reference-errors-list",
        "total-judged-float",
        "total-invalid-bool",
        "multiple-references",
        "missing-top-per-reference",
        "boolean-reference-count",
        "nonfinite-reference-elo",
        "missing-panel",
        "missing-top-per-judge",
        "missing-matchup",
        "wrong-repeat",
        "unexpected-trial-judge",
        "boolean-matchup-invalid-count",
        "matchup-vote-count",
        "panel-model",
        "panel-weight",
        "panel-name",
        "top-per-judge-name",
        "top-per-judge-invalid",
        "top-per-judge-trial-count",
        "outcome-tally",
    ],
)
def test_final_row_gate_defects_are_quarantined_before_resume(
    tmp_path: Path,
    path: tuple[str | int, ...],
    value: object,
    reason: str,
) -> None:
    deliverables = (tmp_path / "deliverables").resolve()
    deliverables.mkdir()
    row = _valid_row(deliverables, stage=1, task_index=7, task_id="poisoned")
    _set_nested(row, path, value)
    cache = _write_cache(deliverables, row)
    output = _jsonl(tmp_path / "judge" / "gdpval_aav2.jsonl", [row])
    journal = _jsonl(
        tmp_path / "judge" / "gdpval_aav2_multistage_state.jsonl",
        [{"stage_index": 1, "status": "planned", "fingerprint": FINGERPRINT}],
    )
    state_root = (tmp_path / "judge-state").resolve()

    audit = judge_state.audit(
        output=output,
        journal=journal,
        deliverables=deliverables,
        max_attempts=3,
    )
    assert audit["status"] == "SANITIZATION_REQUIRED"
    assert reason in next(iter(audit["reasons"].values()))

    result = judge_state.sanitize(
        output=output,
        journal=journal,
        deliverables=deliverables,
        state_root=state_root,
    )
    assert result["status"] == "SANITIZED"
    assert _read_rows(output) == []
    assert not cache.exists()
    failures = _read_rows(output.with_name("gdpval_aav2_failures.jsonl"))
    assert len(failures) == 1
    assert failures[0]["_ng_failure_class"] == "judge_invalid"
    assert reason in failures[0]["_ng_judge_state_sanitizer"]["reasons"]


def test_sanitize_quarantines_only_bad_active_caches_and_publishes_retry_state(tmp_path: Path) -> None:
    output, journal, deliverables, state_root, rows, caches = _fixture(tmp_path)
    output_before = output.read_bytes()
    bad_payloads = {path: path.read_bytes() for path in caches[1:4]}

    audit = judge_state.audit(
        output=output,
        journal=journal,
        deliverables=deliverables,
        max_attempts=3,
    )
    assert audit["status"] == "SANITIZATION_REQUIRED"
    assert audit["active_cache_count"] == 4
    assert audit["quarantine_count"] == 3
    assert audit["remove_output_rows"] == 3

    result = judge_state.sanitize(
        output=output,
        journal=journal,
        deliverables=deliverables,
        state_root=state_root,
    )
    assert result["status"] == "SANITIZED"
    assert [row["task_id"] for row in _read_rows(output)] == ["valid"]
    assert caches[0].is_file()
    assert caches[4].is_file()
    assert all(not path.exists() for path in caches[1:4])

    failures = output.with_name("gdpval_aav2_failures.jsonl")
    sidecar = _read_rows(failures)
    assert len(sidecar) == 3
    assert {row["task_id"] for row in sidecar} == {"invalid-flag", "error", "short"}
    assert all(row["_ng_failure_class"] == "judge_invalid" for row in sidecar)
    assert all(row["reuse_cached_deliverable"] is True for row in sidecar)
    assert all("_ng_failure_terminal" not in row for row in sidecar)
    assert all(row["_ng_attempt_index"] == 0 for row in sidecar)

    receipt_path = Path(result["receipt"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "COMMITTED"
    assert receipt["output"]["removed_rows"] == 3
    assert Path(receipt["output"]["backup"]).read_bytes() == output_before
    for cache in receipt["quarantined_caches"]:
        original = Path(cache["original"])
        quarantined = Path(cache["quarantine"])
        assert quarantined.read_bytes() == bad_payloads[original]
        assert hashlib.sha256(quarantined.read_bytes()).hexdigest() == cache["sha256"]
    digest = json.loads(receipt_path.with_name("receipt.json.sha256.json").read_text(encoding="utf-8"))
    assert digest["sha256"] == hashlib.sha256(receipt_path.read_bytes()).hexdigest()

    # A second invocation is clean and cannot consume another retry attempt.
    sidecar_before = failures.read_bytes()
    second = judge_state.sanitize(
        output=output,
        journal=journal,
        deliverables=deliverables,
        state_root=state_root,
    )
    assert second["status"] == "CLEAN"
    assert failures.read_bytes() == sidecar_before


def test_valid_main_row_disagreeing_with_bad_cache_fails_before_live_mutation(tmp_path: Path) -> None:
    output, journal, deliverables, state_root, rows, caches = _fixture(tmp_path)
    # Make the main row valid while leaving its cached copy explicitly invalid.
    rows[1]["invalid_judge_response"] = False
    _jsonl(output, rows)
    before = output.read_bytes()
    cache_before = caches[1].read_bytes()

    with pytest.raises(judge_state.JudgeStateError, match="disagrees with a valid main-output row"):
        judge_state.sanitize(
            output=output,
            journal=journal,
            deliverables=deliverables,
            state_root=state_root,
        )

    assert output.read_bytes() == before
    assert caches[1].read_bytes() == cache_before
    assert not output.with_name("gdpval_aav2_failures.jsonl").exists()
    assert not (state_root / "transactions").exists()


def test_sidecar_attempt_that_would_gate_resume_is_rejected_without_mutation(tmp_path: Path) -> None:
    output, journal, deliverables, state_root, _, caches = _fixture(tmp_path)
    identity = {"stage_index": 0, "_ng_task_index": 1, "_ng_rollout_index": 0}
    failures = _jsonl(
        output.with_name("gdpval_aav2_failures.jsonl"),
        [
            {**identity, "_ng_failure_class": "judge_invalid"},
            {**identity, "_ng_failure_class": "judge_invalid"},
        ],
    )
    output_before = output.read_bytes()
    failures_before = failures.read_bytes()
    cache_before = caches[1].read_bytes()

    with pytest.raises(judge_state.JudgeStateError, match="would max out"):
        judge_state.sanitize(
            output=output,
            journal=journal,
            deliverables=deliverables,
            state_root=state_root,
        )

    assert output.read_bytes() == output_before
    assert failures.read_bytes() == failures_before
    assert caches[1].read_bytes() == cache_before
    assert not (state_root / "transactions").exists()


def test_byte_identical_bad_retry_gets_a_distinct_second_transaction(tmp_path: Path) -> None:
    output, journal, deliverables, state_root, rows, _ = _fixture(tmp_path)
    first = judge_state.sanitize(
        output=output,
        journal=journal,
        deliverables=deliverables,
        state_root=state_root,
    )
    assert first["status"] == "SANITIZED"

    # Model the pinned resume returning the same cached judge payload again.
    # The prior sidecar is now different even though output/cache bytes match.
    invalid = rows[1]
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(invalid, sort_keys=True) + "\n")
    _write_cache(deliverables, invalid)

    second = judge_state.sanitize(
        output=output,
        journal=journal,
        deliverables=deliverables,
        state_root=state_root,
    )
    assert second["status"] == "SANITIZED"
    assert second["transaction_id"] != first["transaction_id"]
    failures = _read_rows(output.with_name("gdpval_aav2_failures.jsonl"))
    repeated = [row for row in failures if row["task_id"] == "invalid-flag"]
    assert [row["_ng_attempt_index"] for row in repeated] == [0, 1]


def test_interrupted_cache_move_recovers_the_same_transaction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output, journal, deliverables, state_root, _, caches = _fixture(tmp_path)
    real_replace = os.replace
    crashed = False

    def replace_then_crash(source: os.PathLike | str, destination: os.PathLike | str) -> None:
        nonlocal crashed
        destination_path = Path(destination)
        real_replace(source, destination)
        if not crashed and "quarantine" in destination_path.parts:
            crashed = True
            raise OSError("simulated process loss after atomic cache quarantine")

    monkeypatch.setattr(judge_state.os, "replace", replace_then_crash)
    with pytest.raises(OSError, match="simulated process loss"):
        judge_state.sanitize(
            output=output,
            journal=journal,
            deliverables=deliverables,
            state_root=state_root,
        )
    assert crashed
    assert len(_read_rows(output.with_name("gdpval_aav2_failures.jsonl"))) == 3
    assert sum(path.exists() for path in caches[1:4]) == 2
    # The main output still gates the retry until recovery completes.
    assert len(_read_rows(output)) == 4

    monkeypatch.setattr(judge_state.os, "replace", real_replace)
    recovered = judge_state.sanitize(
        output=output,
        journal=journal,
        deliverables=deliverables,
        state_root=state_root,
    )
    assert recovered["status"] == "RECOVERED"
    assert Path(recovered["receipt"]).is_file()
    assert [row["task_id"] for row in _read_rows(output)] == ["valid"]
    assert len(_read_rows(output.with_name("gdpval_aav2_failures.jsonl"))) == 3
    assert all(not path.exists() for path in caches[1:4])


def test_cli_defaults_to_the_multistage_journal_beside_output(tmp_path: Path, capsys) -> None:
    output, _, deliverables, _, _, _ = _fixture(tmp_path)
    rc = judge_state.main(
        [
            "audit",
            "--output",
            str(output),
            "--deliverables",
            str(deliverables),
        ]
    )
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "SANITIZATION_REQUIRED"
