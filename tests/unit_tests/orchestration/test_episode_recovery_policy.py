# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The planner and resumed Store obey one saved policy, including unrouted failures."""

import hashlib
import json
import subprocess
import sys

import pytest

from nemo_gym.core.failure_kinds import AGENT_RUN_ERROR, KILL_SHAPED
from nemo_gym.core.outcome_policy import OutcomeAction
from nemo_gym.orchestration.recovery import (
    COLLECT_SCRIPT,
    CollectedEvidence,
    decide_from_collected,
    select_managed_evidence,
)
from nemo_gym.rollouts.recovery_policy import (
    EPISODE_OUTCOME_KEY,
    RecoveryOutcomePolicyConfig,
    episode_outcome,
    episode_recovery_action,
)
from nemo_gym.rollouts.rollout_recovery import RunManifest
from nemo_gym.rollouts.rollout_store import RolloutStore


@pytest.mark.parametrize("diagnostic", [False, True])
@pytest.mark.parametrize("action", ["retry", "fail"])
@pytest.mark.parametrize("delivery", ["delivered", "possibly_delivered"])
def test_saved_policy_controls_unknown_failure_in_planner_and_resume(tmp_path, diagnostic, action, delivery):
    job = tmp_path / "job" / "bench"
    output = job / "artifacts" / "rollouts.jsonl"
    output.parent.mkdir(parents=True)
    policy = RecoveryOutcomePolicyConfig(default_action=action)
    # Use the launcher's real saved-policy layout; no injected Store policy or pending mock.
    (job.parent / "resolved-config.yaml").write_text(
        json.dumps(
            {
                "driver": {
                    "benchmarks": {"bench": {"resumable": {"recovery_outcome_policy": policy.model_dump(mode="json")}}}
                }
            }
        )
    )
    inputs = [{"_ng_task_index": 0, "_ng_rollout_index": 0}]
    source = job / "input.jsonl"
    source.write_text(json.dumps(inputs[0]) + "\n")

    def prepare():
        return inputs, RunManifest.create(source, inputs, {}, {}).model_copy(update={"max_rollout_attempts": 2})

    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        row = dict(inputs[0])
        store.allocate_attempt(row)
        failure = row | {
            "_ng_failure_class": AGENT_RUN_ERROR,
            "_ng_unrouted_run_error": True,
            "_ng_failure_http_status": 500,
            "_ng_failure_record": {
                "source": "collector",
                "delivery": delivery,
                "http_status": 500,
                "run_id": store.manifest.run_id,
                "episode_id": {"rollout_id": "0-0", "attempt": row["_ng_attempt_index"]},
                "failure": {
                    "terminal": False,
                    "failure_reason": "Unknown execution failure",
                    "failure_kind": AGENT_RUN_ERROR,
                },
            },
        }
        if diagnostic:
            store.record_unstarted(failure | {"_ng_omitted_rollout": True})
        else:
            store.record_outcome(failure)
        assert bool(store.pending()) is (action == "retry")
    saved = json.loads(output.with_name("rollouts_failures.jsonl").read_text())
    assert saved[EPISODE_OUTCOME_KEY]["action"] == action
    # Resume consumes the decision even if configuration changed after the boundary saved it.
    changed = RecoveryOutcomePolicyConfig(default_action="fail" if action == "retry" else "retry")
    (job.parent / "resolved-config.yaml").write_text(
        json.dumps(
            {
                "driver": {
                    "benchmarks": {
                        "bench": {"resumable": {"recovery_outcome_policy": changed.model_dump(mode="json")}}
                    }
                }
            }
        )
    )
    before = output.with_name("rollouts_failures.jsonl").read_bytes()
    raw = subprocess.run(
        [sys.executable, "-I", "-S", "-c", COLLECT_SCRIPT, str(job)],
        capture_output=True,
        text=True,
        check=True,
    )
    collected = select_managed_evidence(CollectedEvidence.model_validate_json(raw.stdout), job)
    assert collected.managed_reader_error is None
    decision = decide_from_collected(
        collected, max_infra_retries=3, max_walltime_s=None, live_job=False, outcome_policy=changed
    )
    assert decision.action.value == ("resubmit" if action == "retry" else "stop"), decision.reason
    assert bool(RolloutStore.read(output).pending()) is (action == "retry")
    with RolloutStore.start_or_resume(output, prepare, resume=True) as resumed:
        pending = resumed.pending()
        assert bool(pending) is (action == "retry")
        if pending:
            resumed.record_dispatch(pending[0])
            resumed.record_outcome(pending[0] | {"reward": 1.0})
    assert output.with_name("rollouts_failures.jsonl").read_bytes() == before
    if action == "retry":
        assert RolloutStore.read(output).coverage()["successful"] == 1
        assert RolloutStore.read(output).pending() == []


def test_interruption_action_is_a_policy_rule_not_a_store_veto():
    row = {"_ng_failure_class": KILL_SHAPED}
    assert episode_recovery_action(row, policy=RecoveryOutcomePolicyConfig()) == OutcomeAction.FAIL
    assert (
        episode_recovery_action(row, policy=RecoveryOutcomePolicyConfig(by_failure_kind={KILL_SHAPED: "retry"}))
        == OutcomeAction.RETRY
    )


@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("action", ["retry", "fail", "verify"])
def test_live_and_reopened_store_preserve_decisions_and_budget(tmp_path, restart, action):
    output = tmp_path / "artifacts" / "rollouts.jsonl"
    output.parent.mkdir()
    inputs = [{"_ng_task_index": 0, "_ng_rollout_index": 0}]
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps(inputs[0]) + "\n")

    def prepare():
        return inputs, RunManifest.create(source, inputs, {}, {}).model_copy(update={"max_rollout_attempts": 1})

    policy = RecoveryOutcomePolicyConfig(default_action=action)
    failure = {"_ng_failure_class": AGENT_RUN_ERROR, "_ng_unrouted_run_error": True}
    selected = episode_outcome(failure, policy=policy)
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        row = dict(inputs[0])
        store.record_dispatch(row)
        store.record_outcome(row | failure | {EPISODE_OUTCOME_KEY: selected.model_dump(mode="json")})
        reader = RolloutStore.read(output) if restart else store
        assert reader.pending() == []  # A retry decision does not replenish an exhausted failure budget.
        assert reader.failures()[0][EPISODE_OUTCOME_KEY] == selected.model_dump(mode="json")
        assert reader.coverage()["failure_budget_exhausted"] == int(action == "retry")


@pytest.mark.parametrize("action", ["retry", "fail"])
async def test_collector_persists_unknown_failure_and_resume_obeys_it(tmp_path, monkeypatch, action):
    from unittest.mock import AsyncMock

    from nemo_gym.rollouts.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper
    from nemo_gym.rollouts.rollout_recovery import IncompleteEvaluationError
    from tests.unit_tests.rollouts.test_rollout_collection import FakeResponse, failing_row, install_fake_server_client

    job = tmp_path / "job" / "bench"
    job.mkdir(parents=True)
    output = job / "artifacts" / "rollouts.jsonl"
    source = job / "input.jsonl"
    source.write_text(json.dumps(failing_row(0)) + "\n")
    policy = RecoveryOutcomePolicyConfig(default_action=action)
    (job.parent / "resolved-config.yaml").write_text(
        json.dumps(
            {
                "driver": {
                    "benchmarks": {"bench": {"resumable": {"recovery_outcome_policy": policy.model_dump(mode="json")}}}
                }
            }
        )
    )
    post = AsyncMock(side_effect=lambda **kwargs: FakeResponse(200, {"_ng_failure_class": AGENT_RUN_ERROR}))
    client = install_fake_server_client(monkeypatch, post)
    monkeypatch.setattr(
        "nemo_gym.rollouts.rollout_collection.get_global_config_dict", lambda: client.global_config_dict
    )
    config = RolloutCollectionConfig(
        input_jsonl_fpath=str(source),
        output_jsonl_fpath=str(output),
        route_failures_to_sidecar=True,
        require_complete=False,
        disable_aggregation=True,
        disable_health_check=True,
    )
    for resume in (False, True):
        try:
            await RolloutCollectionHelper().run_from_config(config.model_copy(update={"resume_from_cache": resume}))
        except IncompleteEvaluationError:
            pass
        reader = RolloutStore.read(output)
        assert reader.failures()[0][EPISODE_OUTCOME_KEY]["action"] == action
    assert post.await_count == (2 if action == "retry" else 1)


def test_a_saved_decision_is_not_reinterpreted_and_invalid_metadata_is_not_a_retry():
    from nemo_gym.config.config_types import ConfigError

    row = {"_ng_failure_class": AGENT_RUN_ERROR}
    selected = episode_outcome(row, policy=RecoveryOutcomePolicyConfig(default_action="fail"))
    assert (
        episode_outcome(
            row | {EPISODE_OUTCOME_KEY: selected.model_dump(mode="json")}, policy=RecoveryOutcomePolicyConfig()
        )
        == selected
    )
    with pytest.raises(ConfigError, match="Invalid saved episode decision for rollout"):
        episode_outcome(row | {EPISODE_OUTCOME_KEY: {"action": "typo"}}, policy=RecoveryOutcomePolicyConfig())


@pytest.mark.parametrize("action", ["verify", "fail"])
@pytest.mark.parametrize("saved", [False, True])
async def test_collector_saved_answer_decision_survives_recovery(tmp_path, monkeypatch, action, saved):
    from tests.unit_tests.orchestration.test_recovery_saved_generation import persist_producer_result, response

    job = tmp_path / "job" / "bench"
    job.parent.mkdir()
    policy = RecoveryOutcomePolicyConfig(by_failure_kind={"judge_failed": action})
    (job.parent / "resolved-config.yaml").write_text(
        json.dumps(
            {
                "driver": {
                    "benchmarks": {"bench": {"resumable": {"recovery_outcome_policy": policy.model_dump(mode="json")}}}
                }
            }
        )
    )
    result = {
        "_ng_failure_class": "judge_failed",
        "failure_kind": "judge_failed",
        "response": response() if saved else {},
    }
    decision, store, projected = await persist_producer_result(job, monkeypatch, result)
    assert store.failures()[0][EPISODE_OUTCOME_KEY]["action"] == action
    assert decision.action.value == ("reverify" if saved and action == "verify" else "stop")
    assert store.pending() == []


@pytest.mark.parametrize("spent", [1, 2])
def test_saved_terminal_timeout_retry_agrees_with_planner_and_keeps_budget(tmp_path, spent):
    output = tmp_path / "artifacts" / "rollouts.jsonl"
    output.parent.mkdir()
    inputs = [{"_ng_task_index": 0, "_ng_rollout_index": 0}]
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps(inputs[0]) + "\n")

    def prepare():
        return inputs, RunManifest.create(source, inputs, {}, {}).model_copy(
            update={"max_rollout_attempts": 2, "retry_terminal_timeouts": True}
        )

    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        for _ in range(spent):
            row = dict(inputs[0])
            store.allocate_attempt(row)
            store.record_outcome(row | {"_ng_failure_class": "timeout_exceeded", "_ng_failure_terminal": True})
        assert bool(store.pending(retry_terminal_timeouts=False)) is (spent < 2)
        assert store.failures()[0][EPISODE_OUTCOME_KEY]["action"] == "retry"
    raw = subprocess.run(
        [sys.executable, "-I", "-S", "-c", COLLECT_SCRIPT, str(tmp_path)], capture_output=True, text=True, check=True
    )
    collected = select_managed_evidence(CollectedEvidence.model_validate_json(raw.stdout), tmp_path)
    decision = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    assert decision.evidence[0].state.value == ("retryable" if spent < 2 else "exhausted")
    assert decision.evidence[0].attempts == spent
    assert bool(RolloutStore.read(output).pending()) is (spent < 2)


@pytest.mark.parametrize("case", ["intact", "bad_hash", "missing_timings"])
def test_raw_and_projected_patch_recovery_agree(tmp_path, case):
    from nemo_gym.rollouts.recovery_policy import _saved_model_patch
    from tests.unit_tests.orchestration.test_eval_resubmit_decision import collect_with_score, ident, write_job_dir

    row = ident(
        0,
        reward=0.0,
        model_patch="patch",
        model_patch_sha256=hashlib.sha256(b"patch").hexdigest(),
        patch_collection_time_s=0.1,
        verification_time_s=0.2,
    )
    if case == "bad_hash":
        row["model_patch_sha256"] = "incorrect"
    elif case == "missing_timings":
        del row["verification_time_s"]
    job = write_job_dir(tmp_path / "job", n=1, results=[row], failures=[])
    [projected] = collect_with_score(job, "[]").results
    assert _saved_model_patch(row) is (case == "intact")
    assert _saved_model_patch(projected) == _saved_model_patch(row)
