# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The in-place re-judgment of a run's saved judge failures at the end of its collection."""

from pathlib import Path
from unittest.mock import Mock

import orjson
import pytest

from nemo_gym import rollout_reverification
from nemo_gym.config.config_types import ConfigError
from nemo_gym.eval.rollout_reverification import (
    RolloutReverificationConfig,
    bind_judge_failures,
    due_judge_failures,
    rejudge_saved_judge_failures,
)
from nemo_gym.rollouts.rollout_records import logical_rollout_id, materialized_path_for
from nemo_gym.rollouts.rollout_recovery import RunManifest
from nemo_gym.rollouts.rollout_store import RolloutStore


ANSWER = {"object": "response", "id": "resp_saved", "status": "completed", "output": []}


def _run(
    tmp_path: Path, *, judge_failures: int, rollouts: int = 3, evidence: dict | None = None
) -> tuple[Path, list[dict]]:
    """A managed run of ``rollouts`` rollouts: the first ``judge_failures`` failed to judge, the others scored 1.0."""
    rows = [{"_ng_task_index": i, "_ng_rollout_index": 0, "agent_ref": {"name": "agent"}} for i in range(rollouts)]
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
                failure = (
                    row
                    | {
                        "_ng_failure_class": "judge_failed",
                        "_ng_failure_id": f"judge-{index}",
                        "failure_kind": "judge_failed",
                        "mask_sample": True,
                        "reward": 0.0,
                        "response": ANSWER,
                    }
                    | (evidence or {})
                )
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


def test_a_settled_row_keeps_the_evidence_of_the_judge_failure_it_settles(tmp_path):
    evidence = {
        "ng_perf": {"model_calls": 3, "total_tokens": 1200},
        "_ng_elapsed_s": 12.5,
        "ng_model_call_capture": {"calls": [{"model_call_id": "policy-0"}]},
        "document_delivery": {"pages": 4},
    }
    output, [failure] = _run(tmp_path, judge_failures=1, evidence=evidence)
    with RolloutStore.open_for_rejudgment(output) as store:
        pending = store.admit_rejudgment(failure)
        # What /verify returns: the grade and its own fields, none of the row's evidence.
        result = {
            "_ng_task_index": failure["_ng_task_index"],
            "_ng_rollout_index": failure["_ng_rollout_index"],
            "agent_ref": failure["agent_ref"],
            "_ng_run_id": failure["_ng_run_id"],
            "_ng_attempt_index": failure["_ng_attempt_index"],
            "response": ANSWER,
            "reward": 0.5,
            "mask_sample": False,
            "_ng_rejudged_failure_id": failure["_ng_failure_id"],
            "_ng_rejudge_operation_id": pending.operation_id,
            "_ng_rejudge_admitted_at": pending.admitted_at,
        }
        assert store.record_rejudgment(failure, result)
        store.close_rejudgment(logical_rollout_id(failure), pending.operation_id)
    settled = next(row for row in RolloutStore.read(output).selected("success") if row["reward"] == 0.5)
    assert {key: settled[key] for key in evidence} == evidence
    assert settled["mask_sample"] is False and "failure_kind" not in settled
    assert not any(key.startswith("_ng_failure") for key in settled)
    assert settled["_ng_rejudged_failure_observations"]["ng_perf"] == evidence["ng_perf"]


async def test_a_run_with_nothing_due_runs_no_pass(tmp_path, monkeypatch):
    output, _ = _run(tmp_path, judge_failures=0)
    before = output.read_bytes()
    assert await rejudge_saved_judge_failures({"output_jsonl_fpath": str(output)}, output) == 0
    assert output.read_bytes() == before and not (output.parent / "reverification").exists()


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


async def test_a_collection_limit_does_not_cap_the_rows_a_re_judge_pass_re_verifies(tmp_path, monkeypatch):
    # A collection `limit` counts tasks; in reverify `limit` caps the rows re-verified. The pass must cover every due
    # judge failure, so the run's limit stays with its collection.
    output, _ = _run(tmp_path, judge_failures=2)
    passes: list[RolloutReverificationConfig] = []

    async def judge_saved_answers(self, config):
        passes.append(config)

    monkeypatch.setattr(rollout_reverification.RolloutReverificationHelper, "run_from_config", judge_saved_answers)
    await rejudge_saved_judge_failures({"output_jsonl_fpath": str(output), "limit": 1}, output)

    (config,) = passes
    assert config.limit is None


async def test_a_run_collected_with_a_limit_re_judges_every_due_judge_failure(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    output, _ = _run(tmp_path, judge_failures=10, rollouts=10)
    posted: list[str] = []

    async def post(*, server_name, url_path, json, **kwargs):
        posted.append(url_path)
        reply = MagicMock()
        reply.verified = json | {"reward": 1.0}
        return reply

    monkeypatch.setattr(
        "nemo_gym.eval.rollout_reverification.setup_server_client",
        lambda: MagicMock(global_config_dict={}, post=post),
    )
    monkeypatch.setattr(
        "nemo_gym.eval.rollout_reverification._build_agent_to_resources_server_mapping",
        lambda _: {"agent": "judge_rs"},
    )
    monkeypatch.setattr("nemo_gym.eval.rollout_reverification.raise_for_status", AsyncMock())
    monkeypatch.setattr(
        "nemo_gym.eval.rollout_reverification.get_response_json", AsyncMock(side_effect=lambda reply: reply.verified)
    )

    # The run was collected with `limit: 4` (tasks); the pass still re-judges all 10 judge failures.
    assert await rejudge_saved_judge_failures({"output_jsonl_fpath": str(output), "limit": 4}, output) == 10
    assert posted == ["/verify"] * 10
    assert due_judge_failures(output) == []


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
    monkeypatch.setattr("nemo_gym.eval.rollout_reverification.setup_server_client", lambda: client)
    monkeypatch.setattr(
        "nemo_gym.eval.rollout_reverification._build_agent_to_resources_server_mapping",
        lambda _: {"agent": "judge_rs"},
    )
    monkeypatch.setattr("nemo_gym.eval.rollout_reverification.raise_for_status", AsyncMock())
    monkeypatch.setattr(
        "nemo_gym.eval.rollout_reverification.get_response_json", AsyncMock(side_effect=lambda reply: reply.verified)
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
    monkeypatch.setattr("nemo_gym.eval.rollout_reverification.setup_server_client", lambda: client)
    monkeypatch.setattr(
        "nemo_gym.eval.rollout_reverification._build_agent_to_resources_server_mapping",
        lambda _: {"agent": "judge_rs"},
    )
    monkeypatch.setattr("nemo_gym.eval.rollout_reverification.raise_for_status", AsyncMock())
    monkeypatch.setattr(
        "nemo_gym.eval.rollout_reverification.get_response_json", AsyncMock(side_effect=lambda r: r.verified)
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


async def test_a_failed_pass_records_what_it_settled_and_is_not_counted_as_settling_nothing(tmp_path, monkeypatch):
    import asyncio

    from aiohttp import ServerDisconnectedError

    from nemo_gym.eval.rollout_reverification import (
        RolloutReverificationConfig,
        RolloutReverificationHelper,
        write_settle_receipt,
    )
    from nemo_gym.orchestration.recovery import failed_judge_jobs_blocking

    _, output, (answered, lost) = _job(tmp_path, judge_failures=2)

    async def reply(task, json):
        if task == 1:
            await asyncio.sleep(0.05)  # the first re-grade has answered by then
            raise ServerDisconnectedError()
        return json | {"reward": 1.0}

    _verifier(monkeypatch, reply)
    helper = RolloutReverificationHelper()
    config = RolloutReverificationConfig.model_validate(
        {
            "rollouts_jsonl_fpath": str(output),
            "materialized_inputs_jsonl_fpath": str(materialized_path_for(output)),
            "output_jsonl_fpath": str(output.parent / "reverification" / "failed-pass" / output.name),
            "judge_failed_only": True,
            "settle_into_source": True,
            "skip_spent_judge_failures": True,
            "disable_aggregation": True,
        }
    )
    with pytest.raises(ServerDisconnectedError):
        await helper.run_from_config(config)

    # The failed receipt counts the grade the pass settled and the admission it withdrew.
    receipt = write_settle_receipt(
        tmp_path / "receipt.json",
        state="failed",
        source=str(output),
        job_id="1",
        settlement=helper.settlement,
        started_at="2026-01-01T00:00:00Z",
    )
    assert {name: receipt[name] for name in ("selected", "judged", "settled", "withdrawn")} == {
        "selected": 2,
        "judged": 1,
        "settled": 1,
        "withdrawn": 1,
    }
    # Two failed jobs that each settled a saved answer made progress: recovery is not stopped for settling nothing.
    failed = {"state": "failed", "settled": receipt["settled"]}
    assert failed_judge_jobs_blocking([failed | {"job_id": "1"}, failed | {"job_id": "2"}]) is None
    nothing = {"state": "failed", "settled": 0}
    assert failed_judge_jobs_blocking([nothing | {"job_id": "1"}, nothing | {"job_id": "2"}]) == ["1", "2"]


def _settling_pass(helper, output: Path, tmp_path: Path):
    """A settling judge-only pass over ``output``, and a reader of the counts its failed receipt records."""
    from nemo_gym.eval.rollout_reverification import RolloutReverificationConfig, write_settle_receipt

    config = RolloutReverificationConfig.model_validate(
        {
            "rollouts_jsonl_fpath": str(output),
            "materialized_inputs_jsonl_fpath": str(materialized_path_for(output)),
            "output_jsonl_fpath": str(output.parent / "reverification" / "failed-pass" / output.name),
            "judge_failed_only": True,
            "settle_into_source": True,
            "skip_spent_judge_failures": True,
            "disable_aggregation": True,
        }
    )
    return config, lambda: {
        name: value
        for name, value in write_settle_receipt(
            tmp_path / "receipt.json",
            state="failed",
            source=str(output),
            job_id="1",
            settlement=helper.settlement,
            started_at="2026-01-01T00:00:00Z",
        ).items()
        if name in ("selected", "judged", "settled", "withdrawn")
    }


async def test_a_grade_saved_before_the_coverage_write_failed_counts_as_settled(tmp_path, monkeypatch):
    from nemo_gym.eval.rollout_reverification import RolloutReverificationHelper
    from nemo_gym.orchestration.recovery import failed_judge_jobs_blocking

    _, output, identities = _job(tmp_path, judge_failures=2)

    async def reply(task, json):
        return json | {"reward": 1.0}

    _verifier(monkeypatch, reply)
    # One exit-time coverage write fails, after the pass has fsynced its grades and closed their admissions.
    record_rejudgment, write_coverage = RolloutStore.record_rejudgment, RolloutStore.write_coverage
    state = {"graded": False, "failed": False}

    def recording(self, failure, result):
        state["graded"] = True
        return record_rejudgment(self, failure, result)

    def failing_once(self, *args, **kwargs):
        if state["graded"] and not state["failed"] and self._results_file is None:  # the exit-time write
            state["failed"] = True
            raise OSError("disk full")
        return write_coverage(self, *args, **kwargs)

    monkeypatch.setattr(RolloutStore, "record_rejudgment", recording)
    monkeypatch.setattr(RolloutStore, "write_coverage", failing_once)
    helper = RolloutReverificationHelper()
    config, receipt = _settling_pass(helper, output, tmp_path)
    with pytest.raises(OSError, match="disk full"):
        await helper.run_from_config(config)

    scored, failed, pending = _state(output)
    assert {identity: scored[identity] for identity in identities} == {identity: 1.0 for identity in identities}
    assert failed == [] and pending == {}  # the new grades stay selected and their admissions closed
    assert receipt() == {"selected": 2, "judged": 2, "settled": 2, "withdrawn": 0}
    # Two such failed jobs made progress: recovery is not stopped for settling nothing.
    failed_job = {"state": "failed", "settled": 2}
    assert failed_judge_jobs_blocking([failed_job | {"job_id": "1"}, failed_job | {"job_id": "2"}]) is None


async def test_a_failed_pass_counts_each_withdrawn_admission_once(tmp_path, monkeypatch):
    from nemo_gym.eval.rollout_reverification import RolloutReverificationHelper

    _, output, identities = _job(tmp_path, judge_failures=2)

    async def reply(task, json):
        if task == 0:
            return json | {"reward": 0.0, "mask_sample": True}  # answered without a grade: withdrawn
        return json | {"reward": 1.0, "response": ANSWER | {"id": "resp_other"}}  # grades another answer: refused

    _verifier(monkeypatch, reply)
    helper = RolloutReverificationHelper()
    config, receipt = _settling_pass(helper, output, tmp_path)
    with pytest.raises(ConfigError):
        await helper.run_from_config(config)

    _, failed, pending = _state(output)
    assert failed == sorted(identities) and pending == {}  # both rows stay as they were; both admissions withdrawn
    assert receipt() == {"selected": 2, "judged": 2, "settled": 0, "withdrawn": 2}


@pytest.mark.parametrize(
    ("counts", "says"),
    [
        (
            {"selected": 2, "settled": 1, "withdrawn": 1},
            "before it failed it settled 1 of 2 selected saved answer(s), which counts as progress, and withdrew 1, "
            "left as they were; the others stay as they were",
        ),
        (
            {"selected": 2, "settled": 0, "withdrawn": 2},
            "before it failed it settled 0 of 2 selected saved answer(s) and withdrew 2, left as they were; the "
            "others stay as they were",
        ),
        ({"selected": 2, "settled": 0, "withdrawn": 0}, "the saved answers it was to grade again stay as they were"),
    ],
)
def test_the_note_on_a_failed_judge_job_says_what_it_settled_and_withdrew(counts, says):
    from nemo_gym.orchestration.recovery import judge_job_note

    job = {"job_id": "1", "state": "failed", "ended_at": "2026-01-01T00:00:00Z", "error": "OSError: disk full"}
    assert judge_job_note(job | counts) == f"judge-only job 1 failed: OSError: disk full; {says}"


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
