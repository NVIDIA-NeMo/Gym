# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The in-place re-judgment of a run's saved judge failures at the end of its collection."""

from pathlib import Path
from unittest.mock import Mock

import orjson
import pytest

from nemo_gym import rollout_reverification
from nemo_gym.config.config_types import ConfigError
from nemo_gym.rollout_reverification import (
    RolloutReverificationConfig,
    bind_judge_failures,
    due_judge_failures,
    rejudge_saved_judge_failures,
)
from nemo_gym.rollouts.rollout_records import logical_rollout_id, materialized_path_for
from nemo_gym.rollouts.rollout_recovery import RunManifest
from nemo_gym.rollouts.rollout_store import RolloutStore


ANSWER = {"object": "response", "id": "resp_saved", "status": "completed", "output": []}


def _run(tmp_path: Path, *, judge_failures: int) -> tuple[Path, list[dict]]:
    """A managed run of three rollouts: the first ``judge_failures`` failed to judge, the others scored 1.0."""
    rows = [{"_ng_task_index": i, "_ng_rollout_index": 0, "agent_ref": {"name": "agent"}} for i in range(3)]
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))
    prepare = Mock(
        side_effect=lambda: (rows, RunManifest.create(source, rows, {}, {"agent": {"responses_api_agents": {}}}))
    )
    output = tmp_path / "rollouts.jsonl"
    failures = []
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        for index, row in enumerate(store.pending()):
            store.record_dispatch(row)
            if index < judge_failures:
                failure = row | {
                    "_ng_failure_class": "judge_failed",
                    "_ng_failure_id": f"judge-{index}",
                    "failure_kind": "judge_failed",
                    "mask_sample": True,
                    "reward": 0.0,
                    "response": ANSWER,
                }
                store.record_outcome(failure)
                failures.append(failure)
            else:
                store.record_outcome(row | {"reward": 1.0, "response": ANSWER})
    return output, failures


def _settle(output: Path, failure: dict, reward: float) -> None:
    """What the reverify helper does for one judged failure: admit, then settle the scored result."""
    with RolloutStore.open_for_rejudgment(output) as store:
        pending = store.admit_rejudgment(failure)
        result = {k: v for k, v in failure.items() if not k.startswith("_ng_failure")} | {
            "failure_kind": None,
            "mask_sample": False,
            "reward": reward,
            "_ng_rejudged_failure_id": failure["_ng_failure_id"],
            "_ng_rejudge_operation_id": pending.operation_id,
            "_ng_rejudge_admitted_at": pending.admitted_at,
        }
        assert store.record_rejudgment(failure, result)
        store.close_rejudgment(logical_rollout_id(failure), pending.operation_id)


def test_due_judge_failures_are_the_selected_ones_with_failure_attempts_left(tmp_path, monkeypatch):
    output, failures = _run(tmp_path, judge_failures=2)
    assert due_judge_failures(output) == sorted(logical_rollout_id(f) for f in failures)

    monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", "1")
    (tmp_path / "spent").mkdir()
    spent, _ = _run(tmp_path / "spent", judge_failures=1)
    assert due_judge_failures(spent) == []  # its one failure attempt is spent
    assert due_judge_failures(tmp_path / "missing.jsonl") == []


def test_a_spent_judge_failure_is_left_out_of_the_binding_only_when_asked(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", "1")
    output, _ = _run(tmp_path, judge_failures=1)
    with pytest.raises(ConfigError, match="has spent its 1 saved failure attempts"):
        bind_judge_failures(output)
    assert bind_judge_failures(output, skip_spent=True)["failures"] == {}
    with pytest.raises(ValueError, match="only valid together with `settle_into_source`"):
        RolloutReverificationConfig(
            materialized_inputs_jsonl_fpath=str(materialized_path_for(output)),
            rollouts_jsonl_fpath=str(output),
            output_jsonl_fpath=str(tmp_path / "other.jsonl"),
            judge_failed_only=True,
            skip_spent_judge_failures=True,
        )


async def test_a_run_without_due_judge_failures_judges_nothing(tmp_path, monkeypatch):
    output, _ = _run(tmp_path, judge_failures=0)

    async def must_not_run(self, config):
        raise AssertionError("no judge failure is due, so nothing is judged")

    monkeypatch.setattr(rollout_reverification.RolloutReverificationHelper, "run_from_config", must_not_run)
    assert await rejudge_saved_judge_failures({"output_jsonl_fpath": str(output)}, output) == 0


async def test_the_pass_settles_saved_answers_in_place_once_and_counts_what_it_scored(tmp_path, monkeypatch):
    output, failures = _run(tmp_path, judge_failures=2)
    passes: list[RolloutReverificationConfig] = []

    async def judge_saved_answers(self, config):
        passes.append(config)
        _settle(output, failures[0], 1.0)  # the first judge answers; the second stays down

    monkeypatch.setattr(rollout_reverification.RolloutReverificationHelper, "run_from_config", judge_saved_answers)
    run_config = {
        "output_jsonl_fpath": str(output),
        "materialized_inputs_jsonl_fpath": "collection-only.jsonl",
        "resume_from_cache": True,
        "require_complete": True,
        "num_samples_in_parallel": 4,
    }
    assert await rejudge_saved_judge_failures(run_config, output) == 1

    (config,) = passes
    assert (config.judge_failed_only, config.settle_into_source, config.skip_spent_judge_failures) == (True,) * 3
    assert config.disable_aggregation and not config.resume_from_cache and config.num_samples_in_parallel == 4
    assert Path(config.rollouts_jsonl_fpath) == output
    assert Path(config.materialized_inputs_jsonl_fpath) == materialized_path_for(output)
    judged = Path(config.output_jsonl_fpath)
    assert judged.name == output.name and judged.parent.parent == output.parent / "reverification"
    assert judged.parent.name.startswith("auto-")

    store = RolloutStore.read(output, import_legacy=False)
    settled = {logical_rollout_id(r): r for r in store.selected("success")}[logical_rollout_id(failures[0])]
    assert settled["reward"] == 1.0 and settled["response"] == ANSWER  # the saved answer, not a new one
    assert settled["_ng_rejudged_failure_observations"]["_ng_failure_id"] == "judge-0"
    assert store.manifest.pending_rejudge == {}  # the admission is closed
    # The settled rollout is no longer due; the one whose judge stayed down is, while it has attempts left.
    assert due_judge_failures(output) == [logical_rollout_id(failures[1])]


async def test_the_real_reverify_path_settles_a_saved_judge_failure_without_any_policy_call(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    output, failures = _run(tmp_path, judge_failures=1)
    posted: list[tuple[str, str]] = []

    async def post(*, server_name, url_path, json, **kwargs):
        posted.append((server_name, url_path))
        reply = MagicMock()
        reply.verified = json | {"reward": 1.0}  # the resources server grades the saved answer it was sent
        return reply

    client = MagicMock(global_config_dict={}, post=post)
    monkeypatch.setattr("nemo_gym.rollout_reverification.setup_server_client", lambda: client)
    monkeypatch.setattr(
        "nemo_gym.rollout_reverification._build_agent_to_resources_server_mapping", lambda _: {"agent": "judge_rs"}
    )
    monkeypatch.setattr("nemo_gym.rollout_reverification.raise_for_status", AsyncMock())
    monkeypatch.setattr(
        "nemo_gym.rollout_reverification.get_response_json", AsyncMock(side_effect=lambda reply: reply.verified)
    )

    assert await rejudge_saved_judge_failures({"output_jsonl_fpath": str(output)}, output) == 1

    assert posted == [("judge_rs", "/verify")]  # one verify call to the resources server, nothing else
    store = RolloutStore.read(output, import_legacy=False)
    settled = {logical_rollout_id(r): r for r in store.selected("success")}[logical_rollout_id(failures[0])]
    assert settled["reward"] == 1.0 and settled["response"] == ANSWER
    assert settled["_ng_rejudged_failure_id"] == "judge-0" and settled["_ng_rejudge_operation_id"]
    assert store.manifest.pending_rejudge == {}
    assert due_judge_failures(output) == []
    # Running the pass again judges nothing: the settled failure is not due any more.
    assert await rejudge_saved_judge_failures({"output_jsonl_fpath": str(output)}, output) == 0
    assert len(posted) == 1


def _verifier(monkeypatch, reply) -> list[int]:
    """The run's resources server: ``reply(task, json)`` answers each /verify of a saved answer, or raises."""
    from unittest.mock import AsyncMock, MagicMock

    posted: list[int] = []

    async def post(*, server_name, url_path, json, **kwargs):
        posted.append(json["_ng_task_index"])
        response = MagicMock()
        response.verified = await reply(json["_ng_task_index"], json)
        return response

    client = MagicMock(global_config_dict={}, post=post)
    monkeypatch.setattr("nemo_gym.rollout_reverification.setup_server_client", lambda: client)
    monkeypatch.setattr(
        "nemo_gym.rollout_reverification._build_agent_to_resources_server_mapping", lambda _: {"agent": "judge_rs"}
    )
    monkeypatch.setattr("nemo_gym.rollout_reverification.raise_for_status", AsyncMock())
    monkeypatch.setattr(
        "nemo_gym.rollout_reverification.get_response_json", AsyncMock(side_effect=lambda r: r.verified)
    )
    return posted


def _job(tmp_path: Path, judge_failures: int) -> tuple[Path, Path, list[str]]:
    job = tmp_path / "job"
    (job / "artifacts").mkdir(parents=True)
    output, failures = _run(job / "artifacts", judge_failures=judge_failures)
    return job, output, [logical_rollout_id(f) for f in failures]


def _decision(job: Path, live_job: bool | None = False):
    import subprocess
    import sys

    from nemo_gym.orchestration.recovery import (
        COLLECT_SCRIPT,
        CollectedEvidence,
        decide_from_collected,
        select_managed_evidence,
    )

    raw = subprocess.run([sys.executable, "-c", COLLECT_SCRIPT, str(job)], capture_output=True, text=True, check=True)
    collected = select_managed_evidence(CollectedEvidence.model_validate_json(raw.stdout), job)
    return decide_from_collected(collected, max_infra_retries=3, max_walltime_s=3600, live_job=live_job)


def _state(output: Path) -> tuple[dict, list[str], dict]:
    store = RolloutStore.read(output, import_legacy=False)
    scored = {logical_rollout_id(r): r["reward"] for r in store.selected("success")}
    assert store.unresolved_rejudgments() == []
    return scored, sorted(logical_rollout_id(r) for r in store.failures()), store.manifest.pending_rejudge


async def test_a_pass_that_fails_after_admitting_leaves_the_judge_failure_for_reverify(tmp_path, monkeypatch):
    from nemo_gym.orchestration.recovery import RecoveryAction

    job, output, (identity,) = _job(tmp_path, judge_failures=1)

    async def reply(task, json):
        raise RuntimeError("Event loop is closed")  # the pass fails in-process after its admission

    _verifier(monkeypatch, reply)
    with pytest.raises(RuntimeError, match="Event loop is closed"):
        await rejudge_saved_judge_failures({"output_jsonl_fpath": str(output)}, output)

    _, failed, pending = _state(output)
    assert pending == {} and failed == [identity]  # the admission is withdrawn; the row stays judge_failed
    assert due_judge_failures(output) == [identity]  # a later pass may judge it again
    decision = _decision(job)
    assert decision.action == RecoveryAction.REVERIFY, decision.reason


async def test_a_regrade_that_answered_is_settled_when_a_later_one_disconnects(tmp_path, monkeypatch):
    import asyncio

    from aiohttp import ServerDisconnectedError

    from nemo_gym.orchestration.recovery import RecoveryAction

    job, output, (answered, lost) = _job(tmp_path, judge_failures=2)

    async def reply(task, json):
        if task == 1:
            await asyncio.sleep(0.05)  # the first re-grade has answered by then
            raise ServerDisconnectedError()
        return json | {"reward": 1.0}

    _verifier(monkeypatch, reply)
    with pytest.raises(ServerDisconnectedError):
        await rejudge_saved_judge_failures({"output_jsonl_fpath": str(output)}, output)

    scored, failed, pending = _state(output)
    assert scored[answered] == 1.0 and pending == {}  # the known outcome is settled and its admission closed
    assert failed == [lost] and due_judge_failures(output) == [lost]  # the lost one is withdrawn, not unknown
    decision = _decision(job)
    assert decision.action == RecoveryAction.REVERIFY, decision.reason


@pytest.mark.parametrize(
    "masked",
    [
        {"reward": 0.0, "mask_sample": True},
        {"reward": 0.0, "mask_sample": True, "_ng_failure_class": "verifier_error"},
    ],
)
async def test_a_masked_reply_does_not_keep_another_regrade_from_settling(tmp_path, monkeypatch, masked):
    import asyncio

    job, output, (answered, masked_identity) = _job(tmp_path, judge_failures=2)

    async def reply(task, json):
        if task == 1:
            await asyncio.sleep(0.05)
            return json | masked
        return json | {"reward": 1.0}

    posted = _verifier(monkeypatch, reply)
    assert await rejudge_saved_judge_failures({"output_jsonl_fpath": str(output)}, output) == 1

    scored, failed, pending = _state(output)
    assert scored[answered] == 1.0 and pending == {}
    # The verifier answered without a grade: the judge failure stays as it was, and its admission is closed.
    assert failed == [masked_identity] and sorted(posted) == [0, 1]


def test_only_a_judge_failure_with_no_saved_outcome_is_withdrawn(tmp_path):
    output, failures = _run(tmp_path, judge_failures=1)
    identity = logical_rollout_id(failures[0])
    with RolloutStore.open_for_rejudgment(output) as store:
        pending = store.admit_rejudgment(failures[0])
        with pytest.raises(ConfigError, match="no admitted saved-result re-verification"):
            store.withdraw_rejudgment(identity, pending.operation_id)  # its re-judgment saves its failed outcome
        store.withdraw_rejudgment(identity, pending.operation_id, unsaved_judge_failure=True)
        assert store.manifest.pending_rejudge == {}
    assert due_judge_failures(output) == [identity]


# --- an admission whose pass died ------------------------------------------------------------------------------


def _admitted_by_a_dead_pass(tmp_path: Path) -> tuple[Path, Path, str]:
    """A run whose re-judge pass admitted its judge failure and was then killed: no outcome was saved."""
    job, output, (identity,) = _job(tmp_path, judge_failures=1)
    store = RolloutStore.open_for_rejudgment(output)
    store.admit_rejudgment(next(iter(store.failures())))
    assert RolloutStore.read(output, import_legacy=False).unresolved_rejudgments() == [identity]
    return job, output, identity


@pytest.mark.parametrize(
    ("live_job", "action"),
    [(False, "reverify"), (True, "noop"), (None, "stop")],
    ids=["no-live-job", "live-job", "scheduler-unknown"],
)
def test_an_admission_whose_pass_died_is_reverified_once_no_job_is_live(tmp_path, live_job, action):
    job, _, identity = _admitted_by_a_dead_pass(tmp_path)

    decision = _decision(job, live_job=live_job)

    assert decision.action.value == action, decision.reason
    if live_job is False:
        assert [f"{t}-{r}" for t, r in decision.identities["judge_recoverable"]] == [identity]
    if live_job is None:
        assert "admitted re-judgment outcome unknown" in decision.reason


def test_the_recovery_pass_withdraws_an_admission_whose_pass_died(tmp_path):
    _, output, identity = _admitted_by_a_dead_pass(tmp_path)
    with pytest.raises(ConfigError, match="has an admitted re-judgment"):
        bind_judge_failures(output)  # an ordinary pass still refuses it

    binding = bind_judge_failures(output, withdraw_unresolved=True)

    assert list(binding["failures"]) == [identity]  # bound again for its re-judgment
    assert RolloutStore.read(output, import_legacy=False).manifest.pending_rejudge == {}
    with pytest.raises(ValueError, match="only valid together with `settle_into_source`"):
        RolloutReverificationConfig(
            materialized_inputs_jsonl_fpath=str(materialized_path_for(output)),
            rollouts_jsonl_fpath=str(output),
            output_jsonl_fpath=str(tmp_path / "other.jsonl"),
            judge_failed_only=True,
            withdraw_unresolved_admissions=True,
        )
