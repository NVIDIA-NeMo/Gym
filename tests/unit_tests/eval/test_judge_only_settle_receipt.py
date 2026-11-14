# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The judge-only recovery job settles the selected saved results through the run's own settle path, refreshes the
run's coverage, aggregate metrics and quality summary, and ends with a receipt; a job that fails leaves a failed one."""

import hashlib
import json
from pathlib import Path

import pytest

from nemo_gym.cli import eval as eval_cli
from nemo_gym.eval.rollout_reverification import (
    RolloutReverificationConfig,
    RolloutReverificationHelper,
    due_judge_failures,
    read_settle_receipt,
)
from nemo_gym.rollouts.rollout_collection import RolloutAggregationHelper
from nemo_gym.rollouts.rollout_records import materialized_path_for
from nemo_gym.rollouts.rollout_store import RolloutStore
from tests.unit_tests.eval.test_reverify_saved_results import (
    ANSWER,
    _run,
    _selected,
    ungraded,
    verify_transport_masked,
)
from tests.unit_tests.eval.test_reverify_saved_results import verifier as verifier  # noqa: F401 - fixture


def _job_config(output: Path, nonce: str = "judge-1", **extra) -> RolloutReverificationConfig:
    """The configuration the judge-only job's `gym eval reverify` command carries."""
    folder = output.parent / "reverification" / nonce
    return RolloutReverificationConfig.model_validate(
        {
            "rollouts_jsonl_fpath": str(output),
            "materialized_inputs_jsonl_fpath": str(materialized_path_for(output)),
            "output_jsonl_fpath": str(folder / output.name),
            "judge_failed_only": True,
            "settle_into_source": True,
            "skip_spent_judge_failures": True,
            "disable_aggregation": True,
            "settle_receipt_fpath": str(folder / "receipt.json"),
        }
        | extra
    )


def _snapshot(output: Path) -> str:
    names = (
        "rollouts.jsonl",
        "rollouts_failures.jsonl",
        "rollouts_manifest.json",
        "rollouts_materialized_inputs.jsonl",
        "rollouts_resolved_config.yaml",
        "rollouts_attempts.jsonl",
    )
    digests = {}
    for name in names:
        path = output.parent / name
        digests[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    return json.dumps(digests, sort_keys=True)


@pytest.fixture
def aggregation(monkeypatch):
    """The aggregate-metrics recomputation of the source run, recorded instead of run."""
    calls = []

    async def run_from_config(self, config):
        calls.append(config)

    monkeypatch.setattr(RolloutAggregationHelper, "run_from_config", run_from_config)
    return calls


def test_the_receipt_flag_needs_settle_into_source(tmp_path) -> None:
    with pytest.raises(ValueError, match="settle_receipt_fpath"):
        RolloutReverificationConfig.model_validate(
            {
                "rollouts_jsonl_fpath": "r.jsonl",
                "materialized_inputs_jsonl_fpath": "m.jsonl",
                "output_jsonl_fpath": "o.jsonl",
                "judge_failed_only": True,
                "settle_receipt_fpath": "receipt.json",
            }
        )


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"judge_failed_only": False}, "`settle_into_source` is only valid together with `judge_failed_only`"),
        (
            {"settle_into_source": False, "skip_spent_judge_failures": False, "settle_receipt_fpath": None},
            "A source snapshot requires settle_into_source",
        ),
    ],
)
def test_source_settlement_requires_explicit_recovery_mode(tmp_path, overrides, message) -> None:
    output = _run(tmp_path, [verify_transport_masked(0), ungraded(1)])
    (output.parent / "rollouts_resolved_config.yaml").write_text("fixture: true\n")
    snapshot = _snapshot(output)
    before = {path.name: path.read_bytes() for path in output.parent.iterdir() if path.is_file()}
    assert _job_config(output, settle_source_snapshot_json=snapshot).settle_into_source

    with pytest.raises(ValueError, match=message):
        _job_config(output, settle_source_snapshot_json=snapshot, **overrides)

    assert {path.name: path.read_bytes() for path in output.parent.iterdir() if path.is_file()} == before
    assert not (output.parent / "reverification").exists()


async def test_the_pass_records_what_it_selected_judged_and_settled(tmp_path, verifier) -> None:  # noqa: F811
    output = _run(tmp_path, [verify_transport_masked(0), ungraded(1, judge_response="A")])
    helper = RolloutReverificationHelper()
    await helper.run_from_config(_job_config(output))
    assert helper.settlement == {"selected": 1, "unsupported": 0, "judged": 1, "settled": 1, "withdrawn": 0}
    assert verifier["posted"] == [("rs", "/verify")]
    selected, store = _selected(output)
    assert selected[list(selected)[0]]["reward"] == 1.0 and store.coverage()["masked"] == 0


async def test_a_reverification_without_a_grade_is_recorded_as_withdrawn(tmp_path, verifier) -> None:  # noqa: F811
    output = _run(tmp_path, [verify_transport_masked(0)])
    verifier["grade"] = lambda json: json | {"reward": 0.0, "mask_sample": True}
    helper = RolloutReverificationHelper()
    await helper.run_from_config(_job_config(output))
    assert helper.settlement == {"selected": 1, "unsupported": 0, "judged": 1, "settled": 0, "withdrawn": 1}


async def test_an_unsupported_verifier_is_recorded_as_left_as_it_was(tmp_path, verifier) -> None:  # noqa: F811
    output = _run(tmp_path, [verify_transport_masked(0)])
    verifier["mode"] = "unsupported"
    helper = RolloutReverificationHelper()
    await helper.run_from_config(_job_config(output))
    assert helper.settlement == {"selected": 1, "unsupported": 1, "judged": 0, "settled": 0, "withdrawn": 0}


def test_the_job_settles_refreshes_the_run_and_ends_with_a_receipt(tmp_path, monkeypatch, verifier, aggregation):  # noqa: F811
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    output = _run(tmp_path, [verify_transport_masked(0), ungraded(1, judge_response="A")])
    (output.parent / "rollouts_resolved_config.yaml").write_text("fixture: true\n")
    config = _job_config(output, settle_source_snapshot_json=_snapshot(output))
    run_config = {"config_paths": ["fixture.yaml"], "health_check_workers": 1}

    eval_cli._reverify_with_receipt(run_config, config)

    receipt = read_settle_receipt(Path(config.settle_receipt_fpath))
    assert receipt["version"] == 1 and receipt["state"] == "completed" and receipt["job_id"] == "4242"
    assert receipt["source"] == str(output) and receipt["error"] is None
    assert {k: receipt[k] for k in ("selected", "judged", "settled", "withdrawn", "unsupported")} == {
        "selected": 1,
        "judged": 1,
        "settled": 1,
        "withdrawn": 0,
        "unsupported": 0,
    }
    assert receipt["due_after"] == 0 and receipt["started_at"] <= receipt["ended_at"]
    # Coverage follows the settlement, the metrics are recomputed in place and the quality summary is rewritten
    # bound to the settled rows.
    coverage = json.loads((output.parent / "rollouts_coverage.json").read_text())
    assert coverage["masked"] == 0 and coverage["allocation_id"] == "4242"
    assert receipt["coverage"]["masked"] == 0 and receipt["coverage"]["expected"] == 2
    assert len(aggregation) == 1
    assert aggregation[0].input_glob == str(output) and aggregation[0].output_jsonl_fpath == str(output)
    assert aggregation[0].merge_shards is False
    assert receipt["metrics_recomputed"] is True and receipt["health_summary_recomputed"] is True
    summary = json.loads((output.parent / "quality_summary.json").read_text())
    digests = summary["run"]["provenance"]["rollout_artifact_digests"][0]
    assert digests["rollouts_jsonl_fpath"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert summary["run"]["artifacts"]["expected"] == 2
    assert due_judge_failures(output) == []


def test_a_job_that_cannot_judge_leaves_a_failed_receipt_and_the_run_as_it_was(
    tmp_path, verifier, aggregation
) -> None:  # noqa: F811
    output = _run(tmp_path, [verify_transport_masked(0)])
    (output.parent / "rollouts_resolved_config.yaml").write_text("fixture: true\n")
    stale = json.loads(_snapshot(output))
    stale["rollouts.jsonl"] = "0" * 64
    config = _job_config(output, settle_source_snapshot_json=json.dumps(stale))
    before = output.read_bytes()

    with pytest.raises(Exception, match="changed while queued"):
        eval_cli._reverify_with_receipt({"config_paths": ["fixture.yaml"]}, config)

    receipt = read_settle_receipt(Path(config.settle_receipt_fpath))
    assert receipt["state"] == "failed" and "changed while queued" in receipt["error"]
    assert receipt["settled"] == 0 and receipt["metrics_recomputed"] is False
    assert output.read_bytes() == before and verifier["posted"] == [] and aggregation == []
    assert len(due_judge_failures(output)) == 1
    store = RolloutStore.read(output, import_legacy=False)
    assert store.manifest.pending_rejudge == {}
    assert list(_selected(output)[0].values())[0]["response"] == ANSWER


def test_a_health_check_that_reads_the_store_selection_gets_the_settled_history_as_is(
    tmp_path, monkeypatch, verifier, aggregation
):  # noqa: F811
    """A health check that selects the Store's attempts itself (NVIDIA-NeMo/Gym#3881) takes no selected projection:
    the settled run's quality summary is rewritten through it from the run's own history."""
    from nemo_gym.rollouts import rollout_health

    calls = []

    def store_selected_health(
        rollout_paths,
        *,
        output_dir=None,
        workers=None,
        ignored_checks=(),
        limits=None,
        limits_source="config",
        run_coverage=None,
        source_histories=None,
        model_call_capture_dirs=(),
    ):
        calls.append({"rollout_paths": rollout_paths, "output_dir": output_dir, "run_coverage": run_coverage})
        return "health result"

    monkeypatch.setattr(rollout_health, "run_health_checks", store_selected_health)
    monkeypatch.setattr(rollout_health, "format_health_report", lambda result: f"report of {result}")
    output = _run(tmp_path, [verify_transport_masked(0), ungraded(1, judge_response="A")])
    (output.parent / "rollouts_resolved_config.yaml").write_text("fixture: true\n")
    config = _job_config(output, settle_source_snapshot_json=_snapshot(output))

    eval_cli._reverify_with_receipt({"config_paths": ["fixture.yaml"], "health_check_workers": 1}, config)

    receipt = read_settle_receipt(Path(config.settle_receipt_fpath))
    assert receipt["state"] == "completed" and receipt["settled"] == 1
    assert receipt["health_summary_recomputed"] is True
    [call] = calls
    assert call["rollout_paths"] == output and call["output_dir"] == output.parent
    assert call["run_coverage"][0]["masked"] == 0
    assert not (Path(config.settle_receipt_fpath).parent / "selected_rollouts.jsonl").exists()


@pytest.mark.parametrize("failing_step", [None, "recompute", "refresh"])
def test_the_settled_run_refreshes_or_removes_its_completion_files(
    tmp_path, monkeypatch, aggregation, failing_step
) -> None:
    """The judge-only job recomputes the run-wide metrics in place, so the completion files follow them or go."""
    output = tmp_path / "rollouts.jsonl"
    complete_dir = tmp_path / "rollouts_agent_complete"
    complete_dir.mkdir()
    (complete_dir / "alpha.json").write_text("{}")
    refreshed = []

    async def failing_aggregation(self, config):
        raise OSError("disk full")

    def refresh(path, global_config_dict):
        if failing_step == "refresh":
            raise OSError("disk full")
        refreshed.append(path)

    if failing_step == "recompute":
        monkeypatch.setattr(RolloutAggregationHelper, "run_from_config", failing_aggregation)
    monkeypatch.setattr("nemo_gym.rollouts.rollout_collection.refresh_agent_complete_files", refresh)

    eval_cli._refresh_settled_source(
        {"config_paths": ["fixture.yaml"], "disable_health_check": True}, output, workdir=tmp_path / "work"
    )

    assert refreshed == ([output] if failing_step is None else [])
    assert complete_dir.exists() is (failing_step is None)
