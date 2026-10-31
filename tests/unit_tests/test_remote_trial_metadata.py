# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session trial metadata derived by one helper, locally and inside the serialized remote Store reader."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from nemo_gym.orchestration.executors.slurm import _remote_managed_evidence, _remote_managed_reader_script
from nemo_gym.orchestration.recovery import _managed_trial_metadata, select_managed_evidence
from nemo_gym.rollout_recovery import RunManifest
from nemo_gym.rollout_store import RolloutStore
from tests.unit_tests.test_managed_recovery_selection import collect, saved_bytes
from tests.unit_tests.test_remote_managed_recovery import ProjectionConnection, config


QUALITY = {
    "provenance": {
        "servers": {
            "agent": {"kind": "responses_api_agents", "implementation": "simple_agent", "resources_server": "res"}
        }
    }
}
REPO = Path(__file__).resolve().parents[2]


def prepare_session(job, *, config_text="{}", result_type="single_agent_turn_legacy"):
    artifacts = job / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    rows = [{"_ng_task_index": 0, "_ng_rollout_index": 0, "task_id": "dataset-task"}]
    source = job / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def prepare():
        inputs = json.loads(json.dumps(rows))
        return inputs, RunManifest.create(source, inputs, {}, {}).model_copy(update={"max_rollout_attempts": 3})

    with RolloutStore.start_or_resume(artifacts / "rollouts.jsonl", prepare, resume=False) as store:
        row = dict(rows[0])
        store.record_dispatch(row)
        store.record_outcome(
            row
            | {
                "reward": 1.0,
                "agent_ref": {"name": "agent"},
                "_ng_result_type": result_type,
                "_ng_environment_server": "env",
                # The collector projects routing keys only for rows that carry a trajectory, as real results do.
                "ng_trajectory": {"task_id": "dataset-task", "rollout_id": "0-0"},
            }
        )
    if config_text is not None:
        (artifacts / "rollouts_resolved_config.yaml").write_text(config_text)


def collected(job):
    return collect(job).model_copy(update={"quality": QUALITY})


@pytest.mark.parametrize(
    "config_text,expected",
    [
        ("{}", False),
        ("checkpoint: {enabled: false}", False),
        ("checkpoint: {enabled: true, control_auth_token: fixture}", True),
        ("[]", None),
        ("checkpoint: [bad]", None),
        ("checkpoint: {enabled: true}", None),
        (None, None),
    ],
)
def test_local_reader_derives_checkpoint_and_task_from_bound_sources(tmp_path, config_text, expected):
    prepare_session(tmp_path, config_text=config_text)
    selected = select_managed_evidence(collected(tmp_path), tmp_path)
    assert selected.managed_reader_error is None
    assert selected.checkpoint_enabled is expected
    assert selected.results[0].get("materialized_task_id") == ("dataset-task" if expected is not None else None)


def test_helper_keeps_results_and_reports_unknown_without_a_fact(tmp_path):
    prepare_session(tmp_path)
    c = collected(tmp_path)
    artifacts = tmp_path / "artifacts"
    store = RolloutStore.read(artifacts / "rollouts.jsonl", import_legacy=False)
    results = select_managed_evidence(c, tmp_path).results
    plain = [{k: v for k, v in row.items() if k != "materialized_task_id"} for row in results]
    stale = dict(c.source_digests) | {"rollouts_resolved_config.yaml": "0" * 64}
    foreign = [row | {"_ng_task_index": 7} for row in plain]
    with pytest.raises(ValueError, match="changed"):
        _managed_trial_metadata(artifacts, stale, QUALITY, plain, store)  # config bytes are not the snapshotted ones
    for args in (
        (artifacts, c.source_digests, QUALITY, plain, object()),  # a Store without inputs_for
        (artifacts, c.source_digests, QUALITY, foreign, store),  # a row outside the Store inventory
        (artifacts, c.source_digests, {}, plain, store),  # no provenance for the session agent
    ):
        assert _managed_trial_metadata(*args) == (args[3], None)


def test_other_methods_keep_their_projection(tmp_path):
    prepare_session(tmp_path, result_type="legacy_agent")
    selected = select_managed_evidence(collected(tmp_path), tmp_path)
    assert selected.checkpoint_enabled is False
    assert "materialized_task_id" not in selected.results[0]


def run_serialized_reader(worker: Path, job: Path, raw) -> str:
    ref = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=worker, text=True).strip()
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    result = subprocess.run(
        [sys.executable, "-c", _remote_managed_reader_script(), str(job), ref],
        cwd=worker,
        input=raw.model_dump_json(),
        capture_output=True,
        text=True,
        env=env,
    )
    return result.stdout + result.stderr


@pytest.mark.skipif(
    subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True).stdout.strip() != "",
    reason="the serialized reader requires a clean pinned checkout",
)
@pytest.mark.parametrize(
    "config_text,expected", [("{}", False), ("checkpoint: {enabled: true, control_auth_token: x}", True)]
)
def test_serialized_reader_result_is_accepted_by_the_caller(tmp_path, config_text, expected):
    prepare_session(tmp_path, config_text=config_text)
    before = saved_bytes(tmp_path)
    raw = collected(tmp_path)
    output = run_serialized_reader(REPO, tmp_path, raw)
    result = _remote_managed_evidence(ProjectionConnection(output), raw, tmp_path, config())
    assert result.managed_reader_error is None
    assert result.checkpoint_enabled is expected
    assert result.results[0]["materialized_task_id"] == "dataset-task"
    assert saved_bytes(tmp_path) == before


def test_caller_still_rejects_other_replaced_evidence(tmp_path):
    prepare_session(tmp_path)
    raw = collected(tmp_path)
    selected = select_managed_evidence(raw, tmp_path).model_copy(update={"score": 1.0})
    result = _remote_managed_evidence(
        ProjectionConnection("__GYM_MANAGED:" + selected.model_dump_json()), raw, tmp_path, config()
    )
    assert result.managed_selection is None
    assert result.checkpoint_enabled is None


def test_missing_caller_helper_source_stops_before_bootstrap(tmp_path, monkeypatch):
    from nemo_gym.orchestration.executors import slurm

    prepare_session(tmp_path)
    raw = collected(tmp_path)

    def missing_source():
        raise OSError("source unavailable")

    monkeypatch.setattr(slurm, "_remote_managed_reader_script", missing_source)
    conn = ProjectionConnection(AssertionError("must not connect"))
    result = slurm._remote_managed_evidence(conn, raw, tmp_path, config())
    assert result.managed_selection is None
    assert result.managed_reader_error
    assert conn.commands is None
