import json
from pathlib import Path

import pytest

from benchmarks.biomysterybench.shard_official import (
    DEFAULT_EXPECTED,
    LEGACY_AGENT_FAILURE_NORMALIZATION,
    ComparisonError,
    _normalize_legacy_policy_evidence,
    merge_shards,
    prepare_shards,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_prepare_and_merge_preserve_official_indices(tmp_path: Path) -> None:
    expected = [json.loads(line) for line in DEFAULT_EXPECTED.read_text().splitlines()]
    seeded = [
        {
            "id": expected[0]["id"],
            "_ng_task_index": 0,
            "_ng_rollout_index": 2,
            "responses_create_params": expected[0]["responses_create_params"],
        },
        {
            "id": expected[17]["id"],
            "_ng_task_index": 17,
            "_ng_rollout_index": 4,
            "responses_create_params": expected[17]["responses_create_params"],
        },
    ]
    existing = tmp_path / "existing.jsonl"
    _write_jsonl(existing, seeded)
    data_root = tmp_path / "data"
    manifest = prepare_shards(
        DEFAULT_EXPECTED,
        tmp_path / "shards",
        shard_count=16,
        existing_rollouts_path=existing,
        data_root=data_root,
    )

    assert manifest["expected_rollout_count"] == 495
    assert manifest["seeded_rollout_count"] == 2
    assert sum(shard["task_count"] for shard in manifest["shards"]) == 99
    assert (
        sum(
            len(
                [
                    json.loads(line)
                    for line in (
                        tmp_path / f"shards/shard-{shard['shard_index']:02d}/rollouts_materialized_inputs.jsonl"
                    )
                    .read_text()
                    .splitlines()
                ]
            )
            for shard in manifest["shards"]
        )
        == 495
    )

    first_seed = json.loads((tmp_path / "shards/shard-00/rollouts.jsonl").read_text())
    assert first_seed["_ng_task_index"] == 0
    assert first_seed["responses_create_params"]["metadata"]["data_dir"] == str(
        (data_root / expected[0]["id"]).resolve()
    )
    second_shard = 17 % 16
    second_seed = json.loads((tmp_path / f"shards/shard-{second_shard:02d}/rollouts.jsonl").read_text())
    assert second_seed["_ng_task_index"] == 1

    first_materialized = [
        json.loads(line)
        for line in (tmp_path / "shards/shard-00/rollouts_materialized_inputs.jsonl").read_text().splitlines()
    ]
    first_preprocessed = [
        json.loads(line)
        for line in (tmp_path / "shards/shard-00/preprocessed_datasets/benchmark.jsonl").read_text().splitlines()
    ]
    assert [(row["_ng_task_index"], row["_ng_rollout_index"]) for row in first_materialized[:5]] == [
        (0, rollout_index) for rollout_index in range(5)
    ]
    assert len(first_preprocessed) == manifest["shards"][0]["expected_rollout_count"]
    assert all("_ng_task_index" not in row and "_ng_rollout_index" not in row for row in first_preprocessed)
    assert all(row["agent_ref"]["name"] == "biomysterybench_claude_code" for row in first_preprocessed)
    assert all(row["agent_ref"]["name"] == "biomysterybench_claude_code" for row in first_materialized)
    assert all(
        row["responses_create_params"]["metadata"]["data_dir"] == str((data_root / row["id"]).resolve())
        for row in first_materialized
    )

    merged = merge_shards(DEFAULT_EXPECTED, tmp_path / "shards", tmp_path / "merged.jsonl")
    assert [(row["_ng_task_index"], row["_ng_rollout_index"]) for row in merged] == [(0, 2), (17, 4)]

    for shard_path in (tmp_path / "shards").glob("shard-*/rollouts.jsonl"):
        rejudged_rows = [json.loads(line) for line in shard_path.read_text().splitlines()]
        for row in rejudged_rows:
            row.update(
                reward=1.0,
                resolved=True,
                extracted_answer="answer",
                expected_answer="rubric",
                verdict="YES",
                judge_output="Judgement: YES",
                invalid_judge_response=False,
                cheat_detected=False,
                cheat_evidence=[],
            )
        _write_jsonl(shard_path.with_name("rollouts_kimi_rejudged_8192.jsonl"), rejudged_rows)
    rejudged = merge_shards(
        DEFAULT_EXPECTED,
        tmp_path / "shards",
        tmp_path / "rejudged.jsonl",
        rollout_name="rollouts_kimi_rejudged_8192.jsonl",
        policy_name="rollouts.jsonl",
    )
    assert [(row["_ng_task_index"], row["_ng_rollout_index"]) for row in rejudged] == [(0, 2), (17, 4)]
    assert all(row["reward"] == 1.0 and row["verdict"] == "YES" for row in rejudged)


def test_prepare_rejects_duplicate_existing_keys(tmp_path: Path) -> None:
    expected = json.loads(DEFAULT_EXPECTED.read_text().splitlines()[0])
    row = {"id": expected["id"], "_ng_task_index": 0, "_ng_rollout_index": 0}
    existing = tmp_path / "existing.jsonl"
    _write_jsonl(existing, [row, row])
    with pytest.raises(ComparisonError, match="duplicate key"):
        prepare_shards(DEFAULT_EXPECTED, tmp_path / "shards", shard_count=16, existing_rollouts_path=existing)


def test_merge_can_require_complete_coverage(tmp_path: Path) -> None:
    prepare_shards(DEFAULT_EXPECTED, tmp_path / "shards", shard_count=16)
    with pytest.raises(ComparisonError, match="missing 495 keys"):
        merge_shards(
            DEFAULT_EXPECTED,
            tmp_path / "shards",
            tmp_path / "merged.jsonl",
            require_complete=True,
        )


def test_merge_rejects_rollout_name_with_directory(tmp_path: Path) -> None:
    with pytest.raises(ComparisonError, match="must be a filename"):
        merge_shards(
            DEFAULT_EXPECTED,
            tmp_path / "shards",
            tmp_path / "merged.jsonl",
            rollout_name="../rollouts.jsonl",
        )


def test_normalizes_only_unambiguous_legacy_agent_success() -> None:
    row = {
        "response": {"output": [{"type": "message"}]},
        "mask_sample": False,
        "agent_timed_out": False,
        "container_timed_out": False,
        "sandbox_failed": False,
        "agent_metrics": {
            "mask_sample": False,
            "agent_timed_out": False,
            "container_timed_out": False,
            "sandbox_failed": False,
            "agent_run_time": 12.5,
        },
    }

    _normalize_legacy_policy_evidence(row)

    assert row["agent_failed"] is False
    assert row["agent_metrics"]["agent_failed"] is False
    assert row["_ng_policy_evidence_normalizations"] == [LEGACY_AGENT_FAILURE_NORMALIZATION]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row["response"].update(output=[]),
        lambda row: row.update(mask_sample=None),
        lambda row: row["agent_metrics"].update(agent_run_time=None),
        lambda row: row["agent_metrics"].update(sandbox_failed=True),
        lambda row: row.update(agent_failed=True),
    ],
)
def test_does_not_normalize_ambiguous_legacy_agent_state(mutation) -> None:
    row = {
        "response": {"output": [{"type": "message"}]},
        "mask_sample": False,
        "agent_timed_out": False,
        "container_timed_out": False,
        "sandbox_failed": False,
        "agent_metrics": {
            "mask_sample": False,
            "agent_timed_out": False,
            "container_timed_out": False,
            "sandbox_failed": False,
            "agent_run_time": 12.5,
        },
    }
    mutation(row)

    _normalize_legacy_policy_evidence(row)

    assert "_ng_policy_evidence_normalizations" not in row
    assert row.get("agent_failed") is not False
    assert row["agent_metrics"].get("agent_failed") is not False
