# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The judge-only recovery job settles the selected saved results through the run's own settle path, refreshes the
run's coverage, aggregate metrics and quality summary, and ends with a receipt; a job that fails leaves a failed one."""

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from nemo_gym.cli import eval as eval_cli
from nemo_gym.config.config_types import ConfigError
from nemo_gym.eval.rollout_reverification import (
    RolloutReverificationConfig,
    RolloutReverificationHelper,
    due_judge_failures,
    read_settle_receipt,
)
from nemo_gym.rollouts import rollout_collection
from nemo_gym.rollouts.rollout_collection import RolloutAggregationHelper, refresh_run_status
from nemo_gym.rollouts.rollout_records import materialized_path_for
from nemo_gym.rollouts.rollout_recovery import run_lock
from nemo_gym.rollouts.rollout_store import RolloutStore
from nemo_gym.utils.path_utils import batch_status_path_for
from tests.unit_tests.eval.test_rejudge_saved_judge_failures import _run as _run_with_judge_failures
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


def test_the_settled_run_recounts_its_run_status(tmp_path, verifier, aggregation) -> None:  # noqa: F811
    """The run status counts the rows as collection left them; the job settles rows after that, so it recounts them."""
    output, _ = _run_with_judge_failures(tmp_path, judge_failures=1)
    (output.parent / "rollouts_resolved_config.yaml").write_text("fixture: true\n")
    status = batch_status_path_for(output)
    status.write_text('{"schema_version": "1", "tracking": "enabled", "members": {}}')
    refresh_run_status(output)
    before = json.loads(status.read_text())["members"]["agent"]
    config = _job_config(output, settle_source_snapshot_json=_snapshot(output))

    eval_cli._reverify_with_receipt({"config_paths": ["fixture.yaml"], "health_check_workers": 1}, config)

    after = json.loads(status.read_text())["members"]["agent"]
    assert (before["completed_rollout_count"], before["failures_by_class"]) == (2, {"judge_failed": 1})
    assert (after["completed_rollout_count"], after["failures_by_class"]) == (3, {})


def test_the_recount_keeps_an_aggregation_status_written_while_it_scanned(tmp_path, monkeypatch) -> None:
    """Aggregation rewrites the status under the same lock, so the recount reads it there, not before its scan."""
    output, _ = _run_with_judge_failures(tmp_path, judge_failures=1)
    status = batch_status_path_for(output)
    status.write_text('{"schema_version": "1", "tracking": "enabled", "members": {}}')
    refresh_run_status(output)
    real_lock = rollout_collection.run_status_lock

    @contextmanager
    def lock_after_aggregation(status_fpath):
        with real_lock(status_fpath):
            aggregated = json.loads(status_fpath.read_text())
            aggregated["members"]["agent"]["aggregation_status"] = "complete"
            status_fpath.write_text(json.dumps(aggregated))
            yield

    monkeypatch.setattr(rollout_collection, "run_status_lock", lock_after_aggregation)

    refresh_run_status(output)

    assert json.loads(status.read_text())["members"]["agent"]["aggregation_status"] == "complete"


def test_a_job_refused_the_run_never_writes_its_run_status(tmp_path, verifier, aggregation) -> None:  # noqa: F811
    """Another writer owns the run, so a refused job must not replace that writer's status with its own snapshot."""
    output, _ = _run_with_judge_failures(tmp_path, judge_failures=1)
    status = batch_status_path_for(output)
    status.write_text('{"schema_version": "1", "tracking": "enabled", "members": {}}')
    refresh_run_status(output)
    owned = status.read_bytes()

    with run_lock(output), pytest.raises(ConfigError, match="Another collector owns this run"):
        eval_cli._reverify_with_receipt({"config_paths": ["fixture.yaml"]}, _job_config(output))

    assert status.read_bytes() == owned


def test_a_job_whose_run_changed_while_queued_never_writes_its_run_status(tmp_path, verifier, aggregation) -> None:  # noqa: F811
    """The run that replaced the queued one keeps the status its own collection wrote."""
    output, _ = _run_with_judge_failures(tmp_path, judge_failures=1)
    (output.parent / "rollouts_resolved_config.yaml").write_text("fixture: true\n")
    stale = json.loads(_snapshot(output))
    stale["rollouts.jsonl"] = "0" * 64
    status = batch_status_path_for(output)
    status.write_text('{"schema_version": "1", "tracking": "enabled", "members": {}}')
    replacement = status.read_bytes()

    with pytest.raises(ConfigError, match="changed while queued"):
        eval_cli._reverify_with_receipt(
            {"config_paths": ["fixture.yaml"]}, _job_config(output, settle_source_snapshot_json=json.dumps(stale))
        )

    assert status.read_bytes() == replacement


def test_a_job_through_a_rollout_alias_recounts_the_run_status(tmp_path, verifier, aggregation) -> None:  # noqa: F811
    """The status sits beside the resolved run, which settlement writes, not beside the alias the job was given."""
    run = tmp_path / "real"
    run.mkdir()
    output, _ = _run_with_judge_failures(run, judge_failures=1)
    (output.parent / "rollouts_resolved_config.yaml").write_text("fixture: true\n")
    alias = tmp_path / "alias.jsonl"
    alias.symlink_to(output)
    status = batch_status_path_for(output)
    status.write_text('{"schema_version": "1", "tracking": "enabled", "members": {}}')
    refresh_run_status(output)
    config = _job_config(alias, settle_source_snapshot_json=_snapshot(output)).model_copy(
        update={"materialized_inputs_jsonl_fpath": str(materialized_path_for(output))}
    )

    eval_cli._reverify_with_receipt({"config_paths": ["fixture.yaml"], "health_check_workers": 1}, config)

    after = json.loads(status.read_text())["members"]["agent"]
    assert (after["completed_rollout_count"], after["failures_by_class"]) == (3, {})
    assert not batch_status_path_for(alias).exists()


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


def test_the_quality_summary_alone_can_be_rewritten_from_the_saved_rows(tmp_path, aggregation) -> None:  # noqa: F811
    output = _run(tmp_path, [verify_transport_masked(0), ungraded(1, judge_response="A")])

    refreshed = eval_cli._refresh_settled_source(
        {"config_paths": ["fixture.yaml"], "health_check_workers": 1}, output, workdir=output.parent, metrics=False
    )

    assert refreshed == {"metrics_recomputed": False, "health_summary_recomputed": True}
    assert aggregation == []
    summary = json.loads((output.parent / "quality_summary.json").read_text())
    digests = summary["run"]["provenance"]["rollout_artifact_digests"][0]
    assert digests["rollouts_jsonl_fpath"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert summary["run"]["artifacts"]["expected"] == 2


def test_an_incomplete_strict_pass_rewrites_the_quality_summary_from_the_rows_it_settled(
    tmp_path, monkeypatch, verifier, aggregation
) -> None:  # noqa: F811
    """The summary written before the pass counts the masked row as unscored. The pass settles it and then loses its
    connection, so it returns no count: the summary reports the Store's settled counts and the digest of the rows as
    saved, and no aggregate metrics are computed."""
    from types import SimpleNamespace

    from aiohttp import ServerDisconnectedError

    output = _run(tmp_path, [verify_transport_masked(0), ungraded(1, judge_response="A")])
    (output.parent / "rollouts_resolved_config.yaml").write_text("fixture: true\n")
    run_config = {"config_paths": ["fixture.yaml"], "health_check_workers": 1}
    eval_cli._refresh_settled_source(run_config, output, workdir=output.parent, metrics=False)
    summary_path = output.parent / "quality_summary.json"
    assert json.loads(summary_path.read_text())["run"]["artifacts"]["scored"] == 1

    async def settle_for_real(config, source):
        helper = RolloutReverificationHelper()
        await helper.run_from_config(_job_config(source))
        raise ServerDisconnectedError()

    monkeypatch.setattr("nemo_gym.eval.rollout_reverification.rejudge_saved_judge_failures", settle_for_real)
    monkeypatch.setattr(eval_cli, "_saved_run_complete", lambda path: False)  # the strict run stays incomplete

    eval_cli._rejudge_saved_judge_failures(
        run_config, output, SimpleNamespace(disable_aggregation=False), complete_only=True
    )

    coverage = RolloutStore.read(output, import_legacy=False).coverage()
    summary = json.loads(summary_path.read_text())
    artifacts = summary["run"]["artifacts"]
    assert coverage["successful"] == 2 and coverage["masked"] == 0
    assert (artifacts["scored"], artifacts["failed"]["total"], artifacts["expected"]) == (2, 0, coverage["expected"])
    digests = summary["run"]["provenance"]["rollout_artifact_digests"][0]
    assert digests["rollouts_jsonl_fpath"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert aggregation == []  # an incomplete strict run publishes no score
