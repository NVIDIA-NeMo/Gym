# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from benchmarks.gdpval.hsg.checkpoint_e2e.campaign import (
    CampaignError,
    _calculate_mle_elo,
    _elo_evidence_sha256,
    coverage_report,
    locate_campaign,
    main,
    prepare_campaign,
    validate_result,
    verify_campaign,
    write_residue,
)


def _checkpoint(root: Path) -> Path:
    checkpoint = root / "iter_0002000" / "hf"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text('{"model_type":"test"}\n', encoding="utf-8")
    (checkpoint / "tokenizer.json").write_text('{"version":"1.0"}\n', encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"small-test-weight-placeholder")
    return checkpoint.resolve()


def _dataset(root: Path, count: int) -> tuple[Path, list[bytes]]:
    reference = root / "reference.txt"
    reference.write_text("source\n", encoding="utf-8")
    lines = [
        (
            json.dumps(
                {
                    "task_id": f"task-{index:03d}",
                    "prompt": f"task {index}",
                    "reference_file_urls": [str(reference.resolve())],
                },
                sort_keys=True,
            )
            + "\n"
        ).encode()
        for index in range(count)
    ]
    path = root / "dataset.jsonl"
    path.write_bytes(b"".join(lines))
    return path, lines


def _complete(deliverables: Path, task_id: str) -> None:
    repeat = deliverables / f"task_{task_id}" / "repeat_0"
    repeat.mkdir(parents=True)
    (repeat / "finish_params.json").write_text('{"reason":"done"}\n', encoding="utf-8")


def test_prepare_locate_verify_and_idempotent_modulo_shards(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoint = _checkpoint(tmp_path)
    dataset, lines = _dataset(tmp_path, 7)
    campaign_root = tmp_path / "campaigns"
    profile = tmp_path / "profile.json"

    location = locate_campaign(checkpoint, campaign_root)
    prepared = prepare_campaign(
        checkpoint=checkpoint,
        dataset_path=dataset,
        campaign_root=campaign_root,
        shards=3,
        expected_tasks=7,
        profile_out=profile,
    )
    assert prepared["run_id"] == location["run_id"]
    assert prepared["run_dir"] == location["run_dir"]

    run_dir = Path(prepared["run_dir"])
    assert (run_dir / "shards" / "shard_00_of_03.jsonl").read_bytes() == b"".join(lines[0::3])
    assert (run_dir / "shards" / "shard_01_of_03.jsonl").read_bytes() == b"".join(lines[1::3])
    assert (run_dir / "shards" / "shard_02_of_03.jsonl").read_bytes() == b"".join(lines[2::3])
    for artifact in (
        run_dir / "campaign.json",
        run_dir / "campaign.json.sha256.json",
        run_dir / "reference_assets_fingerprint.json",
        run_dir / "shards" / "manifest.json",
        profile,
    ):
        assert stat.S_IMODE(artifact.stat().st_mode) == 0o400

    assert verify_campaign(run_dir) == {
        "status": "PASS",
        "run_id": prepared["run_id"],
        "run_dir": str(run_dir),
        "dataset_rows": 7,
        "shards": 3,
    }
    # Publishing an identical campaign is a no-op rather than an overwrite.
    assert (
        prepare_campaign(
            checkpoint=checkpoint,
            dataset_path=dataset,
            campaign_root=campaign_root,
            shards=3,
            expected_tasks=7,
            profile_out=profile,
        )["campaign_sha256"]
        == prepared["campaign_sha256"]
    )

    assert main(["locate", "--checkpoint", str(checkpoint), "--campaign-root", str(campaign_root)]) == 0
    stdout = capsys.readouterr().out.splitlines()
    assert stdout == [f"RUN_ID={prepared['run_id']}", f"RUN_DIR={run_dir}"]


def test_reference_assets_are_hashed_once_stat_checked_and_finally_rehashed(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    dataset, _ = _dataset(tmp_path, 2)
    prepared = prepare_campaign(
        checkpoint=checkpoint,
        dataset_path=dataset,
        campaign_root=tmp_path / "campaigns",
        shards=1,
        expected_tasks=2,
    )
    run_dir = Path(prepared["run_dir"])
    reference = tmp_path / "reference.txt"
    original_stat = reference.stat()

    reference.write_text("tamper\n", encoding="utf-8")
    with pytest.raises(CampaignError, match="reference asset stat drift"):
        verify_campaign(run_dir)

    # Routine preflights deliberately avoid rereading large attachments. The
    # final rehash still catches an adversarial same-size/same-mtime mutation.
    reference.touch()
    reference.write_text("source\n", encoding="utf-8")
    reference.write_text("tamper\n", encoding="utf-8")
    reference.chmod(original_stat.st_mode & 0o777)
    os.utime(reference, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    assert verify_campaign(run_dir)["status"] == "PASS"
    with pytest.raises(CampaignError, match="reference asset content drift"):
        verify_campaign(run_dir, rehash_reference_assets=True)


def test_prepare_refuses_input_and_artifact_drift(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    dataset, original_lines = _dataset(tmp_path, 3)
    campaign_root = tmp_path / "campaigns"
    prepared = prepare_campaign(
        checkpoint=checkpoint,
        dataset_path=dataset,
        campaign_root=campaign_root,
        shards=2,
        expected_tasks=3,
    )

    rows = [json.loads(line) for line in dataset.read_bytes().splitlines()]
    rows[0]["prompt"] = "drifted"
    dataset.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(CampaignError, match="immutable artifact drift"):
        prepare_campaign(
            checkpoint=checkpoint,
            dataset_path=dataset,
            campaign_root=campaign_root,
            shards=2,
            expected_tasks=3,
        )

    dataset.write_bytes(b"".join(original_lines))
    shard = Path(prepared["run_dir"]) / "shards" / "shard_00_of_02.jsonl"
    shard.chmod(0o600)
    shard.write_bytes(b"{}\n")
    shard.chmod(0o400)
    with pytest.raises(CampaignError, match="modulo partition"):
        verify_campaign(Path(prepared["run_dir"]))


def test_verify_rejects_gym_prepare_pollution_beside_immutable_shards(tmp_path: Path) -> None:
    prepared = prepare_campaign(
        checkpoint=_checkpoint(tmp_path),
        dataset_path=_dataset(tmp_path, 3)[0],
        campaign_root=tmp_path / "campaigns",
        shards=2,
        expected_tasks=3,
    )
    shards = Path(prepared["run_dir"]) / "shards"
    (shards / "shard_00_of_02_prepare.jsonl").write_text("generated\n", encoding="utf-8")

    with pytest.raises(CampaignError, match="unexpected shard artifacts"):
        verify_campaign(Path(prepared["run_dir"]))


def test_prepare_requires_resolved_hf_checkpoint_and_local_references(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    dataset, _ = _dataset(tmp_path, 1)
    with pytest.raises(CampaignError, match="absolute path"):
        locate_campaign(Path("relative/checkpoint"), tmp_path)

    (checkpoint / "tokenizer.json").unlink()
    with pytest.raises(CampaignError, match="tokenizer"):
        prepare_campaign(
            checkpoint=checkpoint,
            dataset_path=dataset,
            campaign_root=tmp_path / "campaigns",
            shards=1,
            expected_tasks=1,
        )

    (checkpoint / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    dataset.write_text(
        json.dumps({"task_id": "task-000", "reference_file_urls": ["https://example.test/ref.pdf"]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(CampaignError, match="remote rather than local"):
        prepare_campaign(
            checkpoint=checkpoint,
            dataset_path=dataset,
            campaign_root=tmp_path / "campaigns",
            shards=1,
            expected_tasks=1,
        )


def test_coverage_and_residue_preserve_canonical_raw_rows(tmp_path: Path) -> None:
    dataset, lines = _dataset(tmp_path, 5)
    deliverables = tmp_path / "deliverables"
    deliverables.mkdir()
    _complete(deliverables, "task-000")
    _complete(deliverables, "task-003")

    coverage = coverage_report(dataset_path=dataset, deliverables=deliverables, expected_tasks=5)
    assert coverage["status"] == "INCOMPLETE"
    assert coverage["missing"] == ["task-001", "task-002", "task-004"]
    assert coverage["extra"] == []

    output = tmp_path / "residue.jsonl"
    shards = tmp_path / "residue-shards"
    report = write_residue(
        dataset_path=dataset,
        deliverables=deliverables,
        output=output,
        shards_dir=shards,
        max_shards=2,
        expected_tasks=5,
    )
    assert report["missing"] == 3
    assert report["shards"] == 2
    assert output.read_bytes() == lines[1] + lines[2] + lines[4]
    assert (shards / "shard_00_of_02.jsonl").read_bytes() == lines[1] + lines[4]
    assert (shards / "shard_01_of_02.jsonl").read_bytes() == lines[2]
    assert stat.S_IMODE(output.stat().st_mode) == 0o400

    # Exact repeat is idempotent; pre-existing drift is never replaced.
    write_residue(
        dataset_path=dataset,
        deliverables=deliverables,
        output=output,
        shards_dir=shards,
        max_shards=2,
        expected_tasks=5,
    )
    output.chmod(0o600)
    output.write_bytes(b"drift\n")
    output.chmod(0o400)
    with pytest.raises(CampaignError, match="immutable artifact drift"):
        write_residue(
            dataset_path=dataset,
            deliverables=deliverables,
            output=output,
            shards_dir=shards,
            max_shards=2,
            expected_tasks=5,
        )


def test_residue_cli_plain_output_prints_scalar_missing_count(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset, _ = _dataset(tmp_path, 5)
    deliverables = tmp_path / "deliverables"
    deliverables.mkdir()
    _complete(deliverables, "task-000")
    _complete(deliverables, "task-003")

    assert (
        main(
            [
                "residue",
                "--dataset",
                str(dataset),
                "--deliverables",
                str(deliverables),
                "--output",
                str(tmp_path / "residue.jsonl"),
                "--shards-dir",
                str(tmp_path / "residue-shards"),
                "--max-shards",
                "2",
                "--expected-tasks",
                "5",
            ]
        )
        == 0
    )
    stdout = capsys.readouterr().out
    assert "status=PASS" in stdout
    assert "missing=3" in stdout
    assert "shards=2" in stdout


def test_coverage_requires_repeat_zero_finish_marker(tmp_path: Path) -> None:
    dataset, _ = _dataset(tmp_path, 2)
    deliverables = tmp_path / "deliverables"
    deliverables.mkdir()
    _complete(deliverables, "task-000")
    repeat_nine = deliverables / "task_task-001" / "repeat_9"
    repeat_nine.mkdir(parents=True)
    (repeat_nine / "finish_params.json").write_text('{"reason":"done"}\n', encoding="utf-8")

    coverage = coverage_report(dataset_path=dataset, deliverables=deliverables, expected_tasks=2)

    assert coverage["status"] == "INCOMPLETE"
    assert coverage["completed"] == 1
    assert coverage["missing"] == ["task-001"]


def test_coverage_accepts_null_repeat_zero_finish_marker(tmp_path: Path) -> None:
    dataset, _ = _dataset(tmp_path, 1)
    marker = tmp_path / "deliverables" / "task_task-000" / "repeat_0" / "finish_params.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("null\n", encoding="utf-8")

    report = coverage_report(dataset_path=dataset, deliverables=tmp_path / "deliverables", expected_tasks=1)

    assert report["status"] == "PASS"
    assert report["completed"] == 1


@pytest.mark.parametrize("payload", ["[]\n", '"scalar"\n', "17\n", "true\n"])
def test_coverage_rejects_nonterminal_json_finish_marker_shapes(tmp_path: Path, payload: str) -> None:
    dataset, _ = _dataset(tmp_path, 1)
    marker = tmp_path / "deliverables" / "task_task-000" / "repeat_0" / "finish_params.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(payload, encoding="utf-8")

    with pytest.raises(CampaignError, match="finish marker is neither a JSON object nor null"):
        coverage_report(dataset_path=dataset, deliverables=tmp_path / "deliverables", expected_tasks=1)


def test_coverage_rejects_malformed_repeat_zero_finish_marker(tmp_path: Path) -> None:
    dataset, _ = _dataset(tmp_path, 1)
    marker = tmp_path / "deliverables" / "task_task-000" / "repeat_0" / "finish_params.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("{broken\n", encoding="utf-8")

    with pytest.raises(CampaignError, match="invalid finish marker"):
        coverage_report(dataset_path=dataset, deliverables=tmp_path / "deliverables", expected_tasks=1)


def _result_artifacts(
    root: Path,
    *,
    stage0_omitted: set[int] | None = None,
    stage0_outcome_status: str | None = None,
    partial_policy: dict | None = None,
    stage0_failure_classes: dict[int, str] | None = None,
) -> tuple[Path, Path]:
    output = root / "rollouts.jsonl"
    rows = []
    stage0_omitted = stage0_omitted or set()
    stage0_failure_classes = stage0_failure_classes or {}
    stage0_indices = [index for index in range(45) if index not in stage0_omitted]
    judge_models = {
        "gpt-5.5": "openai/openai/gpt-5.5",
        "gemini-3.1-pro": "gcp/google/gemini-3.1-pro-preview",
        "claude-opus-4.8": "aws/anthropic/bedrock-claude-opus-4-8",
    }
    judge_names = list(judge_models)
    stage_references = {
        0: [f"reference-{index}" for index in range(9)],
        1: [f"reference-{index}" for index in range(4)],
    }
    for stage, indices in ((0, stage0_indices), (1, list(range(220)))):
        for index in indices:
            reference = stage_references[stage][index % len(stage_references[stage])]
            trial_judges = [judge_names[(index + trial) % len(judge_names)] for trial in range(4)]
            per_judge = {
                name: {
                    "wins": trial_judges.count(name),
                    "losses": 0,
                    "ties": 0,
                    "trials": trial_judges.count(name),
                    "invalid_count": 0,
                }
                for name in set(trial_judges)
            }
            rows.append(
                {
                    "verify_mode": "comparison",
                    "stage_index": stage,
                    "expected_final_stage_index": 1,
                    "expected_stage_row_count": 45 if stage == 0 else 220,
                    "_ng_task_index": index,
                    "_ng_rollout_index": 0,
                    "task_id": f"task-{index:03d}",
                    "reference_ids": [reference],
                    "total_wins": 4,
                    "total_losses": 0,
                    "total_ties": 0,
                    "per_reference": {
                        reference: {
                            "wins": 4,
                            "losses": 0,
                            "ties": 0,
                            "reference_elo": 1000.0 + 10.0 * (index % len(stage_references[stage])),
                        }
                    },
                    "response": {"error": None},
                    "judge_response": {
                        "error": None,
                        "scoring_error": None,
                        "ref_errors": {},
                        "total_judged": 4,
                        "total_invalid": 0,
                        "total_wins": 4,
                        "total_losses": 0,
                        "total_ties": 0,
                        "ref_repeat_count": 1,
                        "av_routed": False,
                        "judge_panel": [
                            {"name": name, "model": model, "weight": 1.0} for name, model in judge_models.items()
                        ],
                        "per_judge": per_judge,
                        "per_ref_repeat": [
                            {
                                "ref_id": reference,
                                "ref_repeat": "repeat_0",
                                "trial_judges": trial_judges,
                                "win_count_a": 0,
                                "win_count_b": 4,
                                "tie_count": 0,
                                "invalid_count": 0,
                                "task_count": 4,
                                "per_judge": {name: dict(counts) for name, counts in per_judge.items()},
                            }
                        ],
                    },
                }
            )
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    fingerprint = "f" * 64
    journal = root / "rollouts_multistage_state.jsonl"
    stage_task_ids = {
        0: [f"task-{index:03d}" for index in range(45)],
        1: [f"task-{index:03d}" for index in range(220)],
    }
    assignments = {
        stage: {
            task_id: stage_references[stage][index % len(stage_references[stage])]
            for index, task_id in enumerate(stage_task_ids[stage])
        }
        for stage in (0, 1)
    }
    journal_rows = [
        {
            "stage_index": 0,
            "status": "planned",
            "task_ids": stage_task_ids[0],
            "reference_ids": stage_references[0],
            "task_reference_ids": assignments[0],
            "fingerprint": fingerprint,
        }
    ]
    if stage0_omitted:
        journal_rows.append(
            {
                "stage_index": 0,
                "status": "attempt_dispositions",
                "attempts": [
                    {
                        "_ng_task_index": index,
                        "_ng_rollout_index": 0,
                        "_ng_failure_class": stage0_failure_classes.get(index, "timeout_exceeded"),
                        "_ng_no_persist": False,
                    }
                    for index in sorted(stage0_omitted)
                ],
                "fingerprint": fingerprint,
            }
        )
    outcome_status = stage0_outcome_status or ("partial_complete" if stage0_omitted else "complete")
    if outcome_status == "partial_complete":
        planned_per_reference = {
            reference_id: sum(value == reference_id for value in assignments[0].values())
            for reference_id in stage_references[0]
        }
        successful_assignments = {
            task_id: assignments[0][task_id] for task_id in (f"task-{index:03d}" for index in stage0_indices)
        }
        successful_per_reference = {
            reference_id: sum(value == reference_id for value in successful_assignments.values())
            for reference_id in stage_references[0]
        }
        policy = partial_policy or {
            "min_success_fraction": 0.9,
            "min_per_reference_success_fraction": 0.5,
            "min_successful_rows_per_reference": 1,
            "newly_waivable_failure_classes": ["timeout_exceeded", "transient"],
        }
        journal_rows.append(
            {
                "stage_index": 0,
                "status": "partial_complete",
                "included_keys": [[index, 0] for index in stage0_indices],
                "omitted_keys": [[index, 0] for index in sorted(stage0_omitted)],
                "accepted_unresolved_keys": [[index, 0] for index in sorted(stage0_omitted)],
                "already_resolved_omitted_keys": [],
                "evidence_sha256": _elo_evidence_sha256(row for row in rows if row["stage_index"] == 0),
                "success_fraction": len(stage0_indices) / 45,
                "persisted_success_fraction": len(stage0_indices) / 45,
                "per_reference": {
                    reference_id: {
                        "planned": planned_per_reference[reference_id],
                        "successful": successful_per_reference[reference_id],
                        "judged": successful_per_reference[reference_id],
                        "success_fraction": (
                            successful_per_reference[reference_id] / planned_per_reference[reference_id]
                        ),
                    }
                    for reference_id in stage_references[0]
                },
                "policy": policy,
                "fingerprint": fingerprint,
            }
        )
    else:
        journal_rows.append({"stage_index": 0, "status": outcome_status, "fingerprint": fingerprint})
    journal_rows.extend(
        [
            {
                "stage_index": 1,
                "status": "planned",
                "task_ids": stage_task_ids[1],
                "reference_ids": stage_references[1],
                "task_reference_ids": assignments[1],
                "fingerprint": fingerprint,
            },
            {"stage_index": 1, "status": "complete", "fingerprint": fingerprint},
        ]
    )
    journal.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in journal_rows), encoding="utf-8")
    stage_fits: dict[int, tuple[float, float]] = {}
    for stage in (0, 1):
        pooled: dict[str, dict[str, float]] = {}
        for row in (row for row in rows if row["stage_index"] == stage):
            reference_id = row["reference_ids"][0]
            counts = row["per_reference"][reference_id]
            entry = pooled.setdefault(
                reference_id,
                {
                    "wins": 0.0,
                    "losses": 0.0,
                    "ties": 0.0,
                    "reference_elo": float(counts["reference_elo"]),
                },
            )
            for field in ("wins", "losses", "ties"):
                entry[field] += counts[field]
        fit = _calculate_mle_elo(
            [
                (
                    counts["reference_elo"],
                    counts["wins"],
                    counts["losses"],
                    counts["ties"],
                )
                for counts in pooled.values()
            ]
        )
        assert fit is not None
        stage_fits[stage] = fit
    wins = sum(row["total_wins"] for row in rows)
    losses = sum(row["total_losses"] for row in rows)
    ties = sum(row["total_ties"] for row in rows)
    judged = wins + losses + ties
    metrics = {
        "comparison/eval_elo": stage_fits[1][0],
        "comparison/normalized_elo": stage_fits[1][1],
        "comparison/num_stages": 2,
        "comparison/stage_0/num_tasks": len(stage0_indices),
        "comparison/stage_1/num_tasks": 220,
        "comparison/judged": judged,
        "comparison/wins": wins,
        "comparison/losses": losses,
        "comparison/ties": ties,
        "comparison/stage_0/eval_elo": stage_fits[0][0],
        "comparison/stage_0/normalized_elo": stage_fits[0][1],
        "comparison/stage_1/eval_elo": stage_fits[1][0],
        "comparison/stage_1/normalized_elo": stage_fits[1][1],
        "comparison/headline_stage_index": 1,
        "comparison/expected_final_stage_declared_rows": len(rows),
        "comparison/expected_final_stage_consistent": 1,
        "comparison/expected_final_stage_index": 1,
        "comparison/final_stage_present": 1,
        "comparison/final_stage_complete": 1,
        "comparison/final_stage_fit": 1,
        "comparison/final_stage_degraded": 0,
        "comparison/observed_final_stage_row_count": 220,
        "comparison/expected_final_stage_row_count_consistent": 1,
        "comparison/expected_final_stage_row_count": 220,
    }
    (root / "rollouts_aggregate_metrics.json").write_text(
        json.dumps([{"agent_metrics": metrics, "key_metrics": metrics, "group_level_metrics": []}]) + "\n",
        encoding="utf-8",
    )
    return output, journal


def _journal_records(journal: Path) -> list[dict]:
    return [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]


def _write_journal(journal: Path, records: list[dict]) -> None:
    journal.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")


def _attempt_disposition(
    task_index: int,
    *,
    failure_class: str = "timeout_exceeded",
    no_persist: bool = False,
) -> dict:
    return {
        "_ng_task_index": task_index,
        "_ng_rollout_index": 0,
        "_ng_failure_class": failure_class,
        "_ng_no_persist": no_persist,
    }


def test_result_requires_complete_two_stage_four_trial_judging_and_numeric_elo(tmp_path: Path) -> None:
    output, journal = _result_artifacts(tmp_path)
    report = validate_result(output=output, journal=journal)
    assert report["rows"] == 265
    assert report["stage0_tasks"] == 45
    assert report["stage1_tasks"] == 220
    assert report["stage1_trials"] == 880
    assert report["invalid"] == 0
    metrics = json.loads((tmp_path / "rollouts_aggregate_metrics.json").read_text(encoding="utf-8"))
    assert report["eval_elo"] == metrics[0]["agent_metrics"]["comparison/eval_elo"]

    rows = output.read_text(encoding="utf-8").splitlines()
    broken = json.loads(rows[-1])
    broken["judge_response"]["total_invalid"] = 1
    rows[-1] = json.dumps(broken)
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(CampaignError, match="total_invalid"):
        validate_result(output=output, journal=journal)


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("verify_mode", "not a comparison result"),
        ("failure_marker", "contains a failure marker"),
        ("top_error_zero", "top-level error"),
        ("response_error_false", "response error"),
        ("judge_error_empty", "judge or reference errors"),
        ("scoring_error_zero", "judge or reference errors"),
        ("matchup_vote_string", "matchup win_count_b"),
        ("matchup_negative_balanced_vote", "matchup win_count_a"),
        ("matchup_task_count_missing", "matchup task_count"),
        ("matchup_raw_response_count", "exactly four raw responses"),
        ("matchup_per_judge_missing", "matchup has a malformed per-judge tally"),
        ("matchup_per_judge_string_trial", "matchup judge .* trials"),
        ("top_per_judge_string_trial", "judge .* trials"),
        ("top_per_judge_bool_invalid", "judge .* invalid_count"),
        ("judge_matchup_tally_mismatch", "judge outcome counts differ from its matchup"),
        ("row_judge_tally_mismatch", "outcome counts differ from judge_response"),
        ("row_total_string", "total_wins"),
    ],
)
def test_result_rejects_row_defects_rejected_by_cache_sanitizer(
    tmp_path: Path,
    defect: str,
    message: str,
) -> None:
    output, journal = _result_artifacts(tmp_path)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    row = rows[0]
    judge = row["judge_response"]
    matchup = judge["per_ref_repeat"][0]
    judge_name = next(iter(judge["per_judge"]))

    if defect == "verify_mode":
        row["verify_mode"] = "pointwise"
    elif defect == "failure_marker":
        row["_ng_failure_class"] = "timeout_exceeded"
    elif defect == "top_error_zero":
        row["error"] = 0
    elif defect == "response_error_false":
        row["response"]["error"] = False
    elif defect == "judge_error_empty":
        judge["error"] = {}
    elif defect == "scoring_error_zero":
        judge["scoring_error"] = 0
    elif defect == "matchup_vote_string":
        matchup["win_count_b"] = "4"
    elif defect == "matchup_negative_balanced_vote":
        matchup["win_count_a"] = -1
        matchup["win_count_b"] = 5
    elif defect == "matchup_task_count_missing":
        matchup.pop("task_count")
    elif defect == "matchup_raw_response_count":
        matchup["raw_responses"] = [{}, {}, {}]
    elif defect == "matchup_per_judge_missing":
        matchup.pop("per_judge")
    elif defect == "matchup_per_judge_string_trial":
        matchup["per_judge"][judge_name]["trials"] = str(matchup["per_judge"][judge_name]["trials"])
    elif defect == "top_per_judge_string_trial":
        judge["per_judge"][judge_name]["trials"] = str(judge["per_judge"][judge_name]["trials"])
    elif defect == "top_per_judge_bool_invalid":
        judge["per_judge"][judge_name]["invalid_count"] = False
    elif defect == "judge_matchup_tally_mismatch":
        judge["total_wins"] = 3
        judge["total_losses"] = 1
    elif defect == "row_judge_tally_mismatch":
        row["total_wins"] = 3
        row["total_losses"] = 1
    elif defect == "row_total_string":
        row["total_wins"] = "4"
    else:  # pragma: no cover - protects this adversarial-case table itself.
        raise AssertionError(f"unknown defect: {defect}")

    output.write_text("".join(json.dumps(value) + "\n" for value in rows), encoding="utf-8")
    with pytest.raises(CampaignError, match=message):
        validate_result(output=output, journal=journal)


def test_embedded_mle_matches_gdpval_runtime_algorithm() -> None:
    from resources_servers.gdpval.comparison import calculate_mle_elo

    battles = [
        (987.25, 7, 3, 2),
        (1210.0, 4, 8, 1),
        (1432.5, 11, 5, 4),
    ]

    assert _calculate_mle_elo(battles) == calculate_mle_elo(battles)


def test_result_rejects_consistently_tampered_aggregate_elos(tmp_path: Path) -> None:
    output, journal = _result_artifacts(tmp_path)
    metrics_path = tmp_path / "rollouts_aggregate_metrics.json"
    document = json.loads(metrics_path.read_text(encoding="utf-8"))
    for metrics_name in ("agent_metrics", "key_metrics"):
        metrics = document[0][metrics_name]
        for key in (
            "comparison/eval_elo",
            "comparison/stage_0/eval_elo",
            "comparison/stage_1/eval_elo",
        ):
            metrics[key] += 100.0
        for key in (
            "comparison/normalized_elo",
            "comparison/stage_0/normalized_elo",
            "comparison/stage_1/normalized_elo",
        ):
            metrics[key] += 0.05
    metrics_path.write_text(json.dumps(document) + "\n", encoding="utf-8")

    with pytest.raises(CampaignError, match="row evidence recomputes"):
        validate_result(output=output, journal=journal)


def test_result_rejects_inconsistent_reference_elo_rows(tmp_path: Path) -> None:
    output, journal = _result_artifacts(tmp_path)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    row = next(
        row
        for row in rows
        if row["stage_index"] == 0 and row["task_id"] == "task-009" and row["reference_ids"] == ["reference-0"]
    )
    row["per_reference"]["reference-0"]["reference_elo"] += 1.0
    output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(CampaignError, match="inconsistent reference_elo"):
        validate_result(output=output, journal=journal)


def test_result_requires_exact_unique_three_judge_panel_receipt(tmp_path: Path) -> None:
    output, journal = _result_artifacts(tmp_path)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    panel_member = rows[0]["judge_response"]["judge_panel"][0]
    rows[0]["judge_response"]["judge_panel"] = [dict(panel_member) for _ in range(3)]
    output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(CampaignError, match="duplicate panel member"):
        validate_result(output=output, journal=journal)


def test_result_accepts_and_counts_exact_gemini_only_av_route(tmp_path: Path) -> None:
    output, journal = _result_artifacts(tmp_path)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    row = rows[0]
    judge = row["judge_response"]
    matchup = judge["per_ref_repeat"][0]
    gemini = {
        "name": "gemini-3.1-pro",
        "model": "gcp/google/gemini-3.1-pro-preview",
        "weight": 1.0,
    }
    counts = {
        "wins": 4,
        "losses": 0,
        "ties": 0,
        "trials": 4,
        "invalid_count": 0,
    }
    matchup_counts = {
        "win_count_a": 0,
        "win_count_b": 4,
        "tie_count": 0,
        "trials": 4,
        "invalid_count": 0,
    }
    judge["av_routed"] = True
    judge["judge_panel"] = [gemini]
    judge["per_judge"] = {"gemini-3.1-pro": counts}
    matchup["trial_judges"] = ["gemini-3.1-pro"] * 4
    matchup["per_judge"] = {"gemini-3.1-pro": matchup_counts}
    output.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")

    report = validate_result(output=output, journal=journal)

    assert report["av_routed_rows"] == 1
    assert report["stage0_av_routed_rows"] == 1
    assert report["stage1_av_routed_rows"] == 0


def test_result_rejects_gemini_only_panel_without_av_route(tmp_path: Path) -> None:
    output, journal = _result_artifacts(tmp_path)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    rows[0]["judge_response"]["judge_panel"] = [
        {
            "name": "gemini-3.1-pro",
            "model": "gcp/google/gemini-3.1-pro-preview",
            "weight": 1.0,
        }
    ]
    output.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")

    with pytest.raises(CampaignError, match="three-member panel"):
        validate_result(output=output, journal=journal)


def test_result_accepts_bounded_stage0_partial_with_strict_final_stage(tmp_path: Path) -> None:
    output, journal = _result_artifacts(tmp_path, stage0_omitted={41, 42, 43, 44})

    report = validate_result(output=output, journal=journal)

    assert report["rows"] == 261
    assert report["stage0_tasks"] == 41
    assert report["stage1_tasks"] == 220
    assert report["stage0_trials"] == 164
    assert report["stage1_trials"] == 880
    assert report["stage0_partial"] is True
    metrics = json.loads((tmp_path / "rollouts_aggregate_metrics.json").read_text(encoding="utf-8"))
    assert report["eval_elo"] == metrics[0]["agent_metrics"]["comparison/eval_elo"]


def test_result_accepts_persisted_transient_stage0_omission_with_strict_final_stage(tmp_path: Path) -> None:
    output, journal = _result_artifacts(
        tmp_path,
        stage0_omitted={44},
        stage0_failure_classes={44: "transient"},
    )

    report = validate_result(output=output, journal=journal)

    assert report["stage0_tasks"] == 44
    assert report["stage0_partial"] is True
    assert report["stage1_tasks"] == 220
    assert report["stage1_trials"] == 880


@pytest.mark.parametrize(
    ("failure_class", "no_persist"),
    [
        ("provider_auth_failure", False),
        ("timeout_exceeded", True),
        ("transient", True),
    ],
)
def test_result_rejects_accepted_unresolved_key_without_latest_persisted_waivable_failure(
    tmp_path: Path,
    failure_class: str,
    no_persist: bool,
) -> None:
    output, journal = _result_artifacts(tmp_path, stage0_omitted={41, 42, 43, 44})
    records = _journal_records(journal)
    dispositions = next(record for record in records if record["status"] == "attempt_dispositions")
    attempt = next(attempt for attempt in dispositions["attempts"] if attempt["_ng_task_index"] == 41)
    attempt["_ng_failure_class"] = failure_class
    attempt["_ng_no_persist"] = no_persist
    _write_journal(journal, records)

    with pytest.raises(CampaignError, match="latest disposition is not a persisted waivable failure"):
        validate_result(output=output, journal=journal)


def test_result_rejects_accepted_unresolved_key_without_disposition(tmp_path: Path) -> None:
    output, journal = _result_artifacts(tmp_path, stage0_omitted={41, 42, 43, 44})
    records = _journal_records(journal)
    dispositions = next(record for record in records if record["status"] == "attempt_dispositions")
    dispositions["attempts"] = [attempt for attempt in dispositions["attempts"] if attempt["_ng_task_index"] != 41]
    _write_journal(journal, records)

    with pytest.raises(CampaignError, match="accepted unresolved key .* has no attempt disposition"):
        validate_result(output=output, journal=journal)


def test_result_uses_latest_stage_scoped_attempt_disposition(tmp_path: Path) -> None:
    output, journal = _result_artifacts(tmp_path, stage0_omitted={41, 42, 43, 44})
    records = _journal_records(journal)
    partial_index = next(index for index, record in enumerate(records) if record["status"] == "partial_complete")
    fingerprint = records[0]["fingerprint"]
    records[partial_index:partial_index] = [
        {
            "stage_index": 0,
            "status": "attempt_dispositions",
            "attempts": [_attempt_disposition(41, failure_class="provider_auth_failure")],
            "fingerprint": fingerprint,
        },
        {
            "stage_index": 0,
            "status": "attempt_dispositions",
            "attempts": [_attempt_disposition(41)],
            "fingerprint": fingerprint,
        },
        {
            # A disposition for the same key in Stage1 must not overwrite the
            # Stage0 evidence used by its partial-completion receipt.
            "stage_index": 1,
            "status": "attempt_dispositions",
            "attempts": [_attempt_disposition(41, failure_class="provider_auth_failure")],
            "fingerprint": fingerprint,
        },
    ]
    _write_journal(journal, records)

    assert validate_result(output=output, journal=journal)["stage0_partial"] is True


def test_result_rejects_when_later_disposition_overrides_waivable_failure(tmp_path: Path) -> None:
    output, journal = _result_artifacts(tmp_path, stage0_omitted={41, 42, 43, 44})
    records = _journal_records(journal)
    partial_index = next(index for index, record in enumerate(records) if record["status"] == "partial_complete")
    records.insert(
        partial_index,
        {
            "stage_index": 0,
            "status": "attempt_dispositions",
            "attempts": [_attempt_disposition(41, failure_class="provider_auth_failure")],
            "fingerprint": records[0]["fingerprint"],
        },
    )
    _write_journal(journal, records)

    with pytest.raises(CampaignError, match="latest disposition is not a persisted waivable failure"):
        validate_result(output=output, journal=journal)


def test_result_replays_restart_semantics_for_attempt_dispositions(tmp_path: Path) -> None:
    output, journal = _result_artifacts(tmp_path, stage0_omitted={41, 42, 43, 44})
    records = _journal_records(journal)
    fingerprint = records[0]["fingerprint"]
    partial = next(
        record for record in records if record["stage_index"] == 0 and record["status"] == "partial_complete"
    )
    stage1_plan = next(record for record in records if record["stage_index"] == 1 and record["status"] == "planned")
    stage1_complete = next(
        record for record in records if record["stage_index"] == 1 and record["status"] == "complete"
    )
    records.extend(
        [
            {"stage_index": 0, "status": "restart_from_stage", "fingerprint": fingerprint},
            {"stage_index": 0, "status": "restart_cleanup_complete", "fingerprint": fingerprint},
            partial,
            stage1_plan,
            stage1_complete,
        ]
    )
    _write_journal(journal, records)

    # Pinned PR #2588 keeps dispositions for the restarted stage itself while
    # invalidating later stages; the original Stage0 timeouts remain effective.
    assert validate_result(output=output, journal=journal)["stage0_partial"] is True


def test_result_does_not_require_timeout_for_already_resolved_omission(tmp_path: Path) -> None:
    output, journal = _result_artifacts(tmp_path, stage0_omitted={41, 42, 43, 44})
    records = _journal_records(journal)
    partial = next(record for record in records if record["status"] == "partial_complete")
    partial["accepted_unresolved_keys"].remove([44, 0])
    partial["already_resolved_omitted_keys"].append([44, 0])
    dispositions = next(record for record in records if record["status"] == "attempt_dispositions")
    dispositions["attempts"] = [attempt for attempt in dispositions["attempts"] if attempt["_ng_task_index"] != 44]
    _write_journal(journal, records)

    assert validate_result(output=output, journal=journal)["stage0_partial"] is True


def test_result_rejects_duplicate_attempt_disposition_key_in_one_record(tmp_path: Path) -> None:
    output, journal = _result_artifacts(tmp_path, stage0_omitted={41, 42, 43, 44})
    records = _journal_records(journal)
    dispositions = next(record for record in records if record["status"] == "attempt_dispositions")
    dispositions["attempts"].append(dict(dispositions["attempts"][0]))
    _write_journal(journal, records)

    with pytest.raises(CampaignError, match="attempt dispositions contain duplicate key"):
        validate_result(output=output, journal=journal)


@pytest.mark.parametrize("attempts", [[], {}, [None]])
def test_result_rejects_malformed_attempt_disposition_records(tmp_path: Path, attempts: object) -> None:
    output, journal = _result_artifacts(tmp_path, stage0_omitted={41, 42, 43, 44})
    records = _journal_records(journal)
    dispositions = next(record for record in records if record["status"] == "attempt_dispositions")
    dispositions["attempts"] = attempts
    _write_journal(journal, records)

    with pytest.raises(CampaignError, match="malformed attempt dispositions|attempt 0 is not an object"):
        validate_result(output=output, journal=journal)


def test_result_rejects_stage0_below_ninety_percent(tmp_path: Path) -> None:
    output, journal = _result_artifacts(tmp_path, stage0_omitted={40, 41, 42, 43, 44})

    with pytest.raises(CampaignError, match="result coverage"):
        validate_result(output=output, journal=journal)


@pytest.mark.parametrize(
    ("omitted", "outcome_status", "message"),
    [
        ({41, 42, 43, 44}, "complete", "must be partial_complete"),
        (set(), "partial_complete", "must be complete"),
    ],
)
def test_result_requires_partial_outcome_only_for_incomplete_stage0(
    tmp_path: Path, omitted: set[int], outcome_status: str, message: str
) -> None:
    output, journal = _result_artifacts(
        tmp_path,
        stage0_omitted=omitted,
        stage0_outcome_status=outcome_status,
    )

    with pytest.raises(CampaignError, match=message):
        validate_result(output=output, journal=journal)


@pytest.mark.parametrize(
    "policy",
    [
        {
            "min_success_fraction": 0.8,
            "min_per_reference_success_fraction": 0.5,
            "min_successful_rows_per_reference": 1,
            "newly_waivable_failure_classes": ["timeout_exceeded", "transient"],
        },
        {
            "min_success_fraction": 0.9,
            "min_per_reference_success_fraction": 0.5,
            "min_successful_rows_per_reference": 1,
            "newly_waivable_failure_classes": ["timeout_exceeded"],
        },
        {
            "min_success_fraction": 0.9,
            "min_per_reference_success_fraction": 0.5,
            "min_successful_rows_per_reference": 1,
            "newly_waivable_failure_classes": ["transient"],
        },
    ],
)
def test_result_rejects_stage0_partial_policy_drift(tmp_path: Path, policy: dict) -> None:
    output, journal = _result_artifacts(
        tmp_path,
        stage0_omitted={41, 42, 43, 44},
        partial_policy=policy,
    )

    with pytest.raises(CampaignError, match="Stage0 (policy|partial outcome permits)"):
        validate_result(output=output, journal=journal)


def test_result_rejects_partial_key_partition_drift(tmp_path: Path) -> None:
    output, journal = _result_artifacts(tmp_path, stage0_omitted={41, 42, 43, 44})
    records = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    partial = next(record for record in records if record["status"] == "partial_complete")
    partial["included_keys"][-1] = [41, 0]
    journal.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    with pytest.raises(CampaignError, match="included_keys"):
        validate_result(output=output, journal=journal)


def test_result_rejects_partial_evidence_hash_drift(tmp_path: Path) -> None:
    output, journal = _result_artifacts(tmp_path, stage0_omitted={41, 42, 43, 44})
    records = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    partial = next(record for record in records if record["status"] == "partial_complete")
    partial["evidence_sha256"] = "0" * 64
    journal.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    with pytest.raises(CampaignError, match="included ELO evidence"):
        validate_result(output=output, journal=journal)


def test_result_rejects_partial_per_reference_floor_failure(tmp_path: Path) -> None:
    output, journal = _result_artifacts(tmp_path, stage0_omitted={0, 9, 18, 27})

    with pytest.raises(CampaignError, match="below its partial-completion floor"):
        validate_result(output=output, journal=journal)


@pytest.mark.parametrize(
    ("metric", "value", "message"),
    [
        ("comparison/stage_0/num_tasks", 45, "Stage0 metric task count"),
        ("comparison/judged", 1060, "comparison/judged"),
    ],
)
def test_result_rejects_partial_metric_count_drift(tmp_path: Path, metric: str, value: int, message: str) -> None:
    output, journal = _result_artifacts(tmp_path, stage0_omitted={41, 42, 43, 44})
    metrics_path = tmp_path / "rollouts_aggregate_metrics.json"
    document = json.loads(metrics_path.read_text(encoding="utf-8"))
    document[0]["agent_metrics"][metric] = value
    document[0]["key_metrics"][metric] = value
    metrics_path.write_text(json.dumps(document) + "\n", encoding="utf-8")

    with pytest.raises(CampaignError, match=message):
        validate_result(output=output, journal=journal)


def test_result_keeps_stage1_exact_when_stage0_is_partial(tmp_path: Path) -> None:
    output, journal = _result_artifacts(tmp_path, stage0_omitted={41, 42, 43, 44})
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if not (row["stage_index"] == 1 and row["task_id"] == "task-219")]
    output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(CampaignError, match="stage1=219"):
        validate_result(output=output, journal=journal)


def test_result_requires_complete_stage1_as_final_elo_headline(tmp_path: Path) -> None:
    output, journal = _result_artifacts(tmp_path)
    metrics_path = tmp_path / "rollouts_aggregate_metrics.json"
    document = json.loads(metrics_path.read_text(encoding="utf-8"))
    document[0]["agent_metrics"]["comparison/headline_stage_index"] = 0
    document[0]["key_metrics"]["comparison/headline_stage_index"] = 0
    metrics_path.write_text(json.dumps(document) + "\n", encoding="utf-8")

    with pytest.raises(CampaignError, match="headline_stage_index"):
        validate_result(output=output, journal=journal)


def test_result_optionally_binds_stage1_to_canonical_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output, journal = _result_artifacts(tmp_path)
    dataset, _ = _dataset(tmp_path, 220)

    report = validate_result(output=output, journal=journal, dataset=dataset, expected_tasks=220)
    assert report["stage1_tasks"] == 220
    assert (
        main(
            [
                "result",
                "--output",
                str(output),
                "--journal",
                str(journal),
                "--dataset",
                str(dataset),
                "--expected-tasks",
                "220",
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["stage1_tasks"] == 220

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    next(row for row in rows if row["stage_index"] == 1 and row["task_id"] == "task-219")["task_id"] = "other"
    output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(CampaignError, match="canonical dataset"):
        validate_result(output=output, journal=journal, dataset=dataset, expected_tasks=220)
