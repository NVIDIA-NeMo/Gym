# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Saved results whose grading failed for a judge or transport cause, or was never recorded, are graded again from
their saved answers: at completion, settled into the run, and on ``--recover``, decided as REVERIFY. Deterministic
unmeasurable verdicts are a model fact and are never re-verified, and no policy call is ever made."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock

import orjson
import pytest
from aiohttp import ClientPayloadError, ClientResponseError

from nemo_gym.agents.utils.verify_call import masked_verify_failure, verify_route_failure
from nemo_gym.config.config_types import ConfigError
from nemo_gym.core.failure_transport import FailureReportRow
from nemo_gym.core.typed_exceptions import NgFailureEvent, agent_route_failure, failure_report_for
from nemo_gym.eval.rollout_reverification import bind_judge_failures, due_judge_failures, rejudge_saved_judge_failures
from nemo_gym.orchestration.recovery import (
    COLLECT_SCRIPT,
    CollectedEvidence,
    IdentityState,
    RecoveryAction,
    decide_from_collected,
    decide_recovery,
    reverification_cause,
    saved_row_evidence,
)
from nemo_gym.rollouts.rollout_records import logical_rollout_id, rejudge_predecessor_id
from nemo_gym.rollouts.rollout_recovery import RunManifest
from nemo_gym.rollouts.rollout_store import RolloutStore
from tests.unit_tests.orchestration.test_eval_recovery_gate import OK_BUDGET, ident, infra_failure, inventory, scored


ANSWER = {
    "object": "response",
    "id": "resp_saved",
    "status": "completed",
    "output": [{"type": "message", "role": "assistant", "status": "completed", "content": []}],
}


def _route_failure_fields(cause: BaseException) -> dict:
    """The failure fields an agent saves when its verify call failed with ``cause`` (``agent_route_failure``)."""
    failure = agent_route_failure(cause, operation="verify", stage="verification")
    failure.__cause__ = cause
    return FailureReportRow.fields(failure_report_for(failure)) | {
        "failure_kind": failure.failure_kind,
        "failure_reason": failure.failure_reason,
    }


def verify_transport_masked(task: int = 0) -> dict:
    """A result masked because its verify call lost its reply mid-body: a completed answer, reward 0, attempt 0."""
    return ident(
        task,
        reward=0.0,
        mask_sample=True,
        response=ANSWER,
        **_route_failure_fields(ClientPayloadError("Response payload is not completed")),
    )


def verify_http_error_masked(task: int = 0) -> dict:
    """The row an agent returns from /run when the verifier answered 500: the saved answer, masked by the agent's helper."""
    error = ClientResponseError(
        MagicMock(real_url="http://verifier/verify"), (), status=500, message="Internal Server Error"
    )
    failure = verify_route_failure(error, operation_id="op", attempt=1)
    failure.__cause__ = error
    row = masked_verify_failure({"response": ANSWER}, failure)
    return ident(task, **row, **{k: v for k, v in _route_failure_fields(error).items() if k.startswith("_ng")})


def verifier_unmeasurable(task: int = 0) -> dict:
    """A result masked because the verifier itself reported it could not measure the answer."""
    remote = NgFailureEvent(
        boundary="resources_server",
        failure_kind="verifier_error",
        failure_reason="the verifier could not measure the answer",
        operation="verify",
        stage="verification",
    )
    return ident(task, reward=0.0, mask_sample=True, response=ANSWER, **_route_failure_fields(remote))


def ungraded(task: int = 0, **extra) -> dict:
    """A scored result whose verifier was observed but recorded no grading."""
    trajectory = {
        "task_id": f"t{task}",
        "rollout_id": f"r{task}",
        "verifier_invocations": [{"invocation_id": f"v{task}", "status": "completed", "model_calls": []}],
    }
    return ident(task, reward=1.0, response=ANSWER, ng_trajectory=trajectory, **extra)


def verifier_patch_masked(task: int = 0, *, intact: bool = True, collected: bool = True) -> dict:
    """A result masked because its verifier produced no verdict for the workspace patch it collected, saved with the
    patch and its sha256 (``intact`` False: the saved patch does not hash to the saved digest). ``collected`` False: a
    verifier that read the patch from the agent session at verify time, with no patch collection of its own."""
    import hashlib

    patch = "diff --git a/x b/x\n+fix\n"
    digest = hashlib.sha256(patch.encode("utf-8")).hexdigest() if intact else "0" * 64
    reason = "Verifier did not produce /logs/verifier/reward.json"
    event = {
        "event_id": f"verify-{task}",
        "boundary": "resources_server",
        "operation": "verify",
        "stage": "verification",
        "failure_kind": "verifier_error",
        "failure_reason": reason,
    }
    return ident(
        task,
        reward=0.0,
        mask_sample=True,
        evaluation_completed=False,
        response=ANSWER,
        failure_kind="verifier_error",
        failure_reason=reason,
        _ng_failure_events={event["event_id"]: event},
        model_patch=patch,
        model_patch_sha256=digest,
        **({"patch_collection_time_s": 1.5, "verification_time_s": 30.0} if collected else {}),
    )


# --- the selector ---------------------------------------------------------------------------------------------------


def test_the_selector_takes_judge_and_verify_transport_masks_and_ungraded_results_only() -> None:
    assert reverification_cause(saved_row_evidence(verify_transport_masked())) == "verify_transport"
    judge_masked = ident(0, reward=0.0, mask_sample=True, response=ANSWER, failure_kind="judge_unparseable")
    assert reverification_cause(saved_row_evidence(judge_masked)) == "judge"
    assert reverification_cause(saved_row_evidence(ungraded())) == "ungraded"

    # Deterministic "cannot be measured" verdicts stay a model fact.
    assert reverification_cause(saved_row_evidence(verifier_unmeasurable())) is None
    plain_mask = ident(0, reward=0.0, mask_sample=True, response=ANSWER, failure_kind="agent_limit")
    assert reverification_cause(saved_row_evidence(plain_mask)) is None
    # A verify reply with an HTTP status is the verifier's answer, not a lost one.
    http = verify_transport_masked()
    for event in http["_ng_failure_events"].values():
        event["foreign_error"] = {"exception_type": "ClientResponseError", "message": "", "http_status": 500}
    assert reverification_cause(saved_row_evidence(http)) is None
    # No saved answer, or a failures-sidecar row: never selected here.
    assert reverification_cause(saved_row_evidence(verify_transport_masked() | {"response": None})) is None
    assert reverification_cause(saved_row_evidence(verify_transport_masked() | {"_ng_failure_class": "x"})) is None


def test_a_masked_verify_http_error_with_a_saved_answer_is_selected() -> None:
    row = verify_http_error_masked()

    assert row["failure_kind"] == "judge_failed" and row["mask_sample"] is True
    assert reverification_cause(saved_row_evidence(row)) == "judge"


def test_a_saved_grading_record_artifact_or_settlement_is_recorded_grading() -> None:
    assert reverification_cause(saved_row_evidence(ungraded(judge_response="A"))) is None  # grading artifact
    assert reverification_cause(saved_row_evidence(ungraded(invalid_model_response=True))) is None  # model failure
    operations = ungraded()
    operations["ng_trajectory"]["verifier_invocations"][0]["grading_operations"] = [{"outcome": "completed"}]
    assert reverification_cause(saved_row_evidence(operations)) is None
    record = ungraded()
    record["ng_trajectory"]["verifier_invocations"][0]["grading_record"] = {"verdict": "scored", "score": 1.0}
    assert reverification_cause(saved_row_evidence(record)) is None
    # A result a re-verification already settled is not selected again, and a pin without observations proves nothing.
    assert reverification_cause(saved_row_evidence(ungraded(_ng_rejudge_operation_id="0" * 32))) is None
    assert reverification_cause(saved_row_evidence(ident(0, reward=1.0, response=ANSWER))) is None


# --- --recover --------------------------------------------------------------------------------------------------


def test_recover_decides_reverify_for_a_verify_transport_masked_result() -> None:
    d = decide_recovery(
        materialized=inventory(3),
        results=[scored(0), verify_transport_masked(1), scored(2)],
        failures=[],
        budget=OK_BUDGET,
    )
    assert d.action == RecoveryAction.REVERIFY, d.reason
    assert d.identities[IdentityState.JUDGE_RECOVERABLE] == [(1, 0)]
    assert d.evidence[1].failure_kind == "agent_route_failed"


def test_recover_keeps_a_deterministic_unmeasurable_mask_as_before() -> None:
    d = decide_recovery(
        materialized=inventory(2), results=[scored(0), verifier_unmeasurable(1)], failures=[], budget=OK_BUDGET
    )
    assert d.action == RecoveryAction.STOP and d.counts[IdentityState.MASKED] == 1
    assert "resume would run nothing" in d.reason


def test_the_cluster_projection_selects_the_same_masked_rows(tmp_path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    rows = [verify_transport_masked(0), verifier_unmeasurable(1), ungraded(2), ungraded(3, judge_response="A")]
    (artifacts / "rollouts_materialized_inputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in inventory(4)))
    (artifacts / "rollouts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (artifacts / "rollouts_failures.jsonl").write_text("")
    raw = subprocess.run(
        [sys.executable, "-I", "-S", "-c", COLLECT_SCRIPT, str(tmp_path)], capture_output=True, text=True, check=True
    )
    d = decide_from_collected(
        CollectedEvidence.model_validate_json(raw.stdout), max_infra_retries=3, max_walltime_s=None, live_job=False
    )
    assert d.identities[IdentityState.JUDGE_RECOVERABLE] == [(0, 0)]
    assert d.identities[IdentityState.MASKED] == [(1, 0)]
    # A result whose grading was never recorded stays scored for recovery (the trial gate reports it unobserved);
    # the projection still names it, so the completion pass and the gate read the same selection.
    assert d.identities[IdentityState.SCORED] == [(2, 0), (3, 0)]
    projected = {(row["_ng_task_index"]): row for row in CollectedEvidence.model_validate_json(raw.stdout).results}
    assert reverification_cause(projected[2]) == "ungraded" and reverification_cause(projected[3]) is None


def test_a_mixed_run_reverifies_the_saved_answers_first_then_resubmits_the_policy_work() -> None:
    d = decide_recovery(
        materialized=inventory(3),
        results=[scored(0), verify_transport_masked(1)],
        failures=[infra_failure(2, attempt=0)],
        budget=OK_BUDGET,
    )
    assert d.action == RecoveryAction.RESUBMIT, d.reason
    assert d.reverify_first == [(1, 0)]
    assert d.reason.startswith("1 saved answer(s) are re-verified first by the resumed run, then 1 identity(ies)")


def test_a_mixed_run_with_a_judge_failure_is_no_longer_refused() -> None:
    judge = ident(1, _ng_failure_class="judge_failed", failure_kind="judge_failed", reward=0.0, mask_sample=True)
    d = decide_recovery(
        materialized=inventory(3),
        results=[scored(0)],
        failures=[judge | {"response": ANSWER}, infra_failure(2, attempt=0)],
        budget=OK_BUDGET,
    )
    assert d.action == RecoveryAction.RESUBMIT, d.reason
    assert d.reverify_first == [(1, 0)]


def test_only_saved_answers_reverify_and_only_policy_work_resubmits_as_before() -> None:
    judge_only = decide_recovery(
        materialized=inventory(2), results=[scored(0), verify_transport_masked(1)], failures=[], budget=OK_BUDGET
    )
    assert judge_only.action == RecoveryAction.REVERIFY and judge_only.reverify_first == []
    policy_only = decide_recovery(
        materialized=inventory(2), results=[scored(0)], failures=[infra_failure(1, attempt=0)], budget=OK_BUDGET
    )
    assert policy_only.action == RecoveryAction.RESUBMIT and policy_only.reverify_first == []
    assert policy_only.reason.startswith("1 identity(ies) need the policy again")


def test_the_recover_outcome_names_the_reverify_part_of_a_mixed_resubmit() -> None:
    from nemo_gym.orchestration.submit import _reverify_first_note

    mixed = decide_recovery(
        materialized=inventory(2),
        results=[verify_transport_masked(0)],
        failures=[infra_failure(1, attempt=0)],
        budget=OK_BUDGET,
    )
    assert _reverify_first_note(mixed, "123").startswith("1 saved answer(s) are re-verified by resume job 123 first")
    only_policy = decide_recovery(
        materialized=inventory(1), results=[], failures=[infra_failure(0, attempt=0)], budget=OK_BUDGET
    )
    assert _reverify_first_note(only_policy, "123") is None


# --- completion: settled into the source run ---------------------------------------------------------------------


def _run(tmp_path: Path, outcomes: list[dict], *, not_started: int = 0) -> Path:
    """A managed run with one rollout per outcome, each recorded as its attempt 0, and ``not_started`` more."""
    rows = [
        {"_ng_task_index": i, "_ng_rollout_index": 0, "agent_ref": {"name": "agent"}}
        for i in range(len(outcomes) + not_started)
    ]
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))
    prepare = Mock(
        side_effect=lambda: (rows, RunManifest.create(source, rows, {}, {"agent": {"responses_api_agents": {}}}))
    )
    output = tmp_path / "rollouts.jsonl"
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        for row, outcome in zip(store.pending(), outcomes):
            store.record_dispatch(row)
            store.record_outcome(row | {k: v for k, v in outcome.items() if not k.startswith("_ng_task")})
    return output


@pytest.fixture
def verifier(monkeypatch):
    """The run's resources server: grades what it is sent with ``state["grade"]``; reports ``state["mode"]``."""
    state = {"posted": [], "mode": "stateless", "grade": lambda json: json | {"reward": 1.0}}

    async def post(*, server_name, url_path, json, **kwargs):
        state["posted"].append((server_name, url_path))
        reply = MagicMock()
        # A verify execution observes itself; a settled trajectory names the fresh invocation.
        fresh = [{"invocation_id": f"fresh-{len(state['posted'])}", "status": "completed"}]
        reply.verified = state["grade"](json) | {"ng_verifier_invocations": fresh}
        return reply

    async def get(*, server_name, url_path, **kwargs):
        assert url_path == "/reverify_mode"
        reply = MagicMock()
        reply.verified = state["mode"]
        return reply

    client = MagicMock(global_config_dict={}, post=post, get=get)
    monkeypatch.setattr("nemo_gym.eval.rollout_reverification.setup_server_client", lambda: client)
    monkeypatch.setattr(
        "nemo_gym.eval.rollout_reverification._build_agent_to_resources_server_mapping", lambda _: {"agent": "rs"}
    )
    monkeypatch.setattr("nemo_gym.eval.rollout_reverification.raise_for_status", AsyncMock())
    monkeypatch.setattr(
        "nemo_gym.eval.rollout_reverification.get_response_json", AsyncMock(side_effect=lambda reply: reply.verified)
    )
    return state


def _selected(output: Path) -> dict:
    store = RolloutStore.read(output, import_legacy=False)
    return {logical_rollout_id(r): r for r in store.selected("success")}, store


async def test_a_verify_transport_masked_result_is_reverified_and_settled_at_completion(tmp_path, verifier) -> None:
    masked = verify_transport_masked()
    output = _run(tmp_path, [masked, ungraded(1, judge_response="A")])
    (identity,) = due_judge_failures(output)
    before, _ = _selected(output)

    assert await rejudge_saved_judge_failures({"output_jsonl_fpath": str(output)}, output) == 1

    assert verifier["posted"] == [("rs", "/verify")]  # one verify call of the saved answer; no policy call
    after, store = _selected(output)
    settled = after[identity]
    assert settled["reward"] == 1.0 and not settled.get("mask_sample") and settled["response"] == ANSWER
    assert settled["_ng_attempt_index"] == before[identity]["_ng_attempt_index"] == 0  # identity and attempt kept
    assert settled["_ng_rejudged_failure_id"] == rejudge_predecessor_id(before[identity])
    assert settled["_ng_rejudged_failure_observations"]["failure_kind"] == "agent_route_failed"
    assert store.manifest.pending_rejudge == {} and store.coverage()["masked"] == 0
    assert due_judge_failures(output) == []  # settled once; the pass judges nothing again
    assert await rejudge_saved_judge_failures({"output_jsonl_fpath": str(output)}, output) == 0


async def test_a_masked_verify_http_error_is_settled_by_verify_alone(tmp_path, verifier) -> None:
    output = _run(tmp_path, [verify_http_error_masked()])
    (identity,) = due_judge_failures(output)

    assert await rejudge_saved_judge_failures({"output_jsonl_fpath": str(output)}, output) == 1

    assert verifier["posted"] == [("rs", "/verify")]  # the saved answer graded again; no policy call
    settled = _selected(output)[0][identity]
    assert settled["reward"] == 1.0 and not settled.get("mask_sample") and settled["response"] == ANSWER


async def test_an_ungraded_result_is_reverified_and_one_with_a_grading_artifact_is_not(tmp_path, verifier) -> None:
    output = _run(tmp_path, [ungraded(0), ungraded(1, judge_response="A")])
    assert len(due_judge_failures(output)) == 1  # the row with a saved grading artifact is not due
    verifier["grade"] = lambda json: json | {"reward": 0.5}
    assert await rejudge_saved_judge_failures({"output_jsonl_fpath": str(output)}, output) == 1
    rewards = sorted(row["reward"] for row in _selected(output)[0].values())
    assert rewards == [0.5, 1.0] and len(verifier["posted"]) == 1


def _asked_a_model(row: dict, metadata: dict | None, **extra) -> dict:
    """``row``'s verifier asked one model, whose call the row saved with ``metadata`` (None: no call saved)."""
    trajectory = row["ng_trajectory"]
    trajectory["verifier_invocations"][0]["model_calls"] = [{"model_call_id": "judge-0"}]
    if metadata is not None:
        trajectory["model_calls"] = [{"model_call_id": "judge-0", "response_metadata": metadata}]
    return row | extra


ANSWERED = {"status_code": 200, "response_status": "completed"}
TEXT_ANSWER = ANSWER | {
    "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "42"}]}]
}


async def test_a_result_the_judge_answered_without_a_grading_record_is_not_graded_again(tmp_path, verifier) -> None:
    """A verifier that does not observe its judge records no grading however often it is asked again; a grade the
    judge gave is never replaced, so the pass leaves the row as saved: no second judge call, no second row."""
    output = _run(tmp_path, [_asked_a_model(ungraded(0), ANSWERED), ungraded(1)])
    before = output.read_bytes()

    assert len(due_judge_failures(output)) == 1  # only the row whose verifier asked no model
    assert await rejudge_saved_judge_failures({"output_jsonl_fpath": str(output)}, output) == 1
    rows = [json.loads(line) for line in output.read_bytes().splitlines()]
    assert len(rows) == 3 and len({logical_rollout_id(row) for row in rows}) == 2
    assert output.read_bytes().startswith(before) and len(verifier["posted"]) == 1
    assert due_judge_failures(output) == []


@pytest.mark.parametrize(
    "metadata",
    [
        {"status_code": 504, "response_status": None},
        {"status_code": 500, "response_status": "failed"},
        {"status_code": None, "response_status": None},
        {"status_code": 200, "response_status": "incomplete"},
        None,
    ],
    ids=["timeout", "server_error", "no_response", "incomplete", "call_not_saved"],
)
async def test_a_result_scored_after_every_judge_call_failed_is_graded_again(tmp_path, verifier, metadata) -> None:
    output = _run(tmp_path, [_asked_a_model(ungraded(0), metadata)])
    assert len(due_judge_failures(output)) == 1
    assert await rejudge_saved_judge_failures({"output_jsonl_fpath": str(output)}, output) == 1


async def test_one_answered_judge_call_among_failed_ones_keeps_the_grade(tmp_path) -> None:
    row = _asked_a_model(ungraded(0), {"status_code": 504, "response_status": None})
    row["ng_trajectory"]["verifier_invocations"][0]["model_calls"].append({"model_call_id": "judge-1"})
    row["ng_trajectory"]["model_calls"].append({"model_call_id": "judge-1", "response_metadata": ANSWERED})
    assert due_judge_failures(_run(tmp_path, [row])) == []


async def test_a_result_naming_a_judge_error_is_graded_again_even_after_a_200(tmp_path) -> None:
    row = _asked_a_model(ungraded(0), ANSWERED, failure_kind="judge_invalid")
    assert len(due_judge_failures(_run(tmp_path, [row]))) == 1


@pytest.mark.parametrize(
    "finish",
    ["length", "content_filter"],
)
async def test_a_chat_judge_reply_cut_at_its_cap_is_graded_again(tmp_path, finish) -> None:
    """FrontierScience's non-research verifier saves no verdict, reward 0.0 and a clean invalid flag on a cut reply."""
    metadata = {"status_code": 200, "response_status": None, "dialect": "chat", "finish_reason": finish}
    row = _asked_a_model(
        ungraded(0), metadata, verdict=None, reward=0.0, invalid_judge_response=False, response=TEXT_ANSWER
    )
    assert len(due_judge_failures(_run(tmp_path, [row]))) == 1


async def test_a_chat_judge_reply_that_stopped_normally_keeps_the_grade(tmp_path) -> None:
    metadata = {"status_code": 200, "response_status": None, "dialect": "chat", "finish_reason": "stop"}
    row = _asked_a_model(
        ungraded(0), metadata, verdict=None, reward=0.0, invalid_judge_response=False, response=TEXT_ANSWER
    )
    assert due_judge_failures(_run(tmp_path, [row])) == []


async def test_a_partial_reward_with_a_saved_judge_error_message_is_graded_again(tmp_path) -> None:
    """Finance Agent V2: an answered judge call, reward 0.5 and a nonempty judge_error for unresolved criteria."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    row = _asked_a_model(ungraded(0), ANSWERED, reward=0.5, judge_error="criteria 2, 3 unresolved")
    assert len(due_judge_failures(_run(tmp_path / "a", [row]))) == 1
    empty = _asked_a_model(ungraded(0), ANSWERED, reward=0.5, judge_error="")
    assert due_judge_failures(_run(tmp_path / "b", [empty])) == []


async def test_an_invalid_judge_response_inside_the_rubric_facts_is_graded_again(tmp_path) -> None:
    row = _asked_a_model(ungraded(0), ANSWERED, rubric_score=0.0, scoring_threshold=1.0, invalid_judge_response=True)
    assert len(due_judge_failures(_run(tmp_path, [row]))) == 1


async def test_a_reverification_that_fails_again_leaves_the_saved_result_as_it_was(tmp_path, verifier) -> None:
    output = _run(tmp_path, [verify_transport_masked()])
    before = output.read_bytes()
    verifier["grade"] = lambda json: json | {"reward": 0.0, "mask_sample": True}

    assert await rejudge_saved_judge_failures({"output_jsonl_fpath": str(output)}, output) == 0

    assert output.read_bytes() == before  # nothing saved
    selected, store = _selected(output)
    assert list(selected.values())[0]["mask_sample"] is True
    assert store.manifest.pending_rejudge == {}  # the admission is withdrawn, so a later pass may try again
    assert len(due_judge_failures(output)) == 1


async def test_a_result_scored_in_a_separate_rescoring_output_is_still_settled_into_its_run(
    tmp_path, verifier, monkeypatch
) -> None:
    """A rescore of the run into a separate output (``gym eval reverify`` on a managed run) scores the saved answer
    there and leaves the run as it was; the run's own settle still grades that answer and settles it, once."""
    import nemo_gym.eval.rollout_reverification as reverify
    from nemo_gym.rollouts.rollout_records import RUN_ID_KEY, materialized_path_for, read_records

    monkeypatch.setattr(reverify, "get_exporters", list)
    output = _run(tmp_path, [verify_transport_masked()])
    (identity,) = due_judge_failures(output)
    run_bytes = output.read_bytes()
    rescored = tmp_path / "rescored" / "rollouts.jsonl"
    await reverify.RolloutReverificationHelper().run_from_config(
        reverify.RolloutReverificationConfig(
            materialized_inputs_jsonl_fpath=str(materialized_path_for(output)),
            rollouts_jsonl_fpath=str(output),
            output_jsonl_fpath=str(rescored),
            disable_aggregation=True,
            upload_rollouts=False,
        )
    )
    [row] = list(read_records(rescored))
    assert row["reward"] == 1.0 and row["response"] == ANSWER and RUN_ID_KEY not in row
    assert row["_ng_reverification_source_run_id"] == RolloutStore.read(output).manifest.run_id
    # The rescoring output is a separate grading artifact: the run itself is untouched and still due.
    assert output.read_bytes() == run_bytes and due_judge_failures(output) == [due_judge_failures(output)[0]]

    assert await rejudge_saved_judge_failures({"output_jsonl_fpath": str(output)}, output) == 1

    settled = _selected(output)[0][identity]
    assert settled["reward"] == 1.0 and not settled.get("mask_sample") and settled["response"] == ANSWER
    assert "_ng_reverification_source_run_id" not in settled
    assert due_judge_failures(output) == [] and list(read_records(rescored)) == [row]

    # Settlement changed the grade but kept the same attempt and answer. Its separate re-score stays reusable.
    posted = list(verifier["posted"])
    await reverify.RolloutReverificationHelper().run_from_config(
        reverify.RolloutReverificationConfig(
            materialized_inputs_jsonl_fpath=str(materialized_path_for(output)),
            rollouts_jsonl_fpath=str(output),
            output_jsonl_fpath=str(rescored),
            resume_from_cache=True,
            disable_aggregation=True,
            upload_rollouts=False,
        )
    )
    assert verifier["posted"] == posted and list(read_records(rescored)) == [row]


async def test_an_unsupported_verifier_is_never_asked_to_reverify(tmp_path, verifier) -> None:
    output = _run(tmp_path, [verify_transport_masked()])
    verifier["mode"] = "unsupported"
    assert await rejudge_saved_judge_failures({"output_jsonl_fpath": str(output)}, output) == 0
    assert verifier["posted"] == []
    assert list(_selected(output)[0].values())[0]["mask_sample"] is True


def test_a_settled_grade_that_does_not_bind_the_saved_answer_is_refused(tmp_path) -> None:
    output = _run(tmp_path, [verify_transport_masked()])
    (masked,) = bind_judge_failures(output)["failures"].values()
    with RolloutStore.open_for_rejudgment(output) as store:
        pending = store.admit_rejudgment(masked)
        result = {k: v for k, v in masked.items() if k not in ("mask_sample", "failure_kind", "failure_reason")} | {
            "reward": 1.0,
            "_ng_rejudged_failure_id": rejudge_predecessor_id(masked),
            "_ng_rejudge_operation_id": pending.operation_id,
            "_ng_rejudge_admitted_at": pending.admitted_at,
        }
        other_answer = result | {"response": ANSWER | {"id": "resp_other"}}
        with pytest.raises(ConfigError, match="does not settle the saved judge failure"):
            store.record_rejudgment(masked, other_answer)
        other_attempt = result | {"_ng_attempt_index": 1}
        with pytest.raises(ConfigError, match="_ng_attempt_index"):
            store.record_rejudgment(masked, other_attempt)
        assert store.record_rejudgment(masked, result)  # the same saved answer and attempt settle it
        store.close_rejudgment(logical_rollout_id(masked), pending.operation_id)
    # A cold reader accepts the settlement and selects it.
    selected, reread = _selected(output)
    assert list(selected.values())[0]["reward"] == 1.0 and reread.manifest.pending_rejudge == {}


# --- the collection driver: re-verify first, then decide from the settled run --------------------------------------


class _Collection:
    def __init__(self, order: list, error: BaseException | None = None):
        self.order, self.error = order, error

    async def run_from_config(self, config):
        self.order.append("collect")
        if self.error is not None:
            raise self.error


def test_a_resumed_run_reverifies_its_saved_answers_before_its_policy_work(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    from nemo_gym.cli import eval as eval_cli

    output = _run(tmp_path, [verify_transport_masked()], not_started=1)
    order: list = []
    monkeypatch.setattr(
        eval_cli,
        "_rejudge_saved_judge_failures",
        lambda run_config, out, config, aggregate=True: order.append(("reverify", aggregate)),
    )
    resumed = SimpleNamespace(resume_from_cache=True, require_complete=True)
    eval_cli._collect_and_rejudge(_Collection(order), resumed, {}, output)
    assert order == [("reverify", False), "collect", ("reverify", True)]

    order.clear()
    fresh = SimpleNamespace(resume_from_cache=False, require_complete=True)
    eval_cli._collect_and_rejudge(_Collection(order), fresh, {}, output)
    assert order == ["collect", ("reverify", True)]  # a fresh run has nothing saved to grade first


def test_a_strict_rejection_is_decided_from_the_settled_run(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    from nemo_gym.cli import eval as eval_cli
    from nemo_gym.rollouts.rollout_recovery import IncompleteEvaluationError

    output = _run(tmp_path, [verify_transport_masked()], not_started=1)
    monkeypatch.setattr(eval_cli, "_rejudge_saved_judge_failures", lambda *args, **kwargs: None)
    # The collection's check rejected it as not retryable; the settled run still has a retryable rollout.
    stale = IncompleteEvaluationError("stale", retryable=False)
    config = SimpleNamespace(resume_from_cache=False, require_complete=True)
    with pytest.raises(IncompleteEvaluationError) as raised:
        eval_cli._collect_and_rejudge(_Collection([], stale), config, {}, output)
    assert raised.value is not stale and raised.value.exit_code == 75
    assert "after the saved-answer re-verification" in str(raised.value)


def pairwise_partial(task: int = 0) -> dict:
    """A scored pairwise-judge result as saved: one of its two games has no verdict, so its reward is partial."""
    return ungraded(task, games=[_game("A>B"), _game(None)], policy_answer="saved policy answer") | {"reward": 0.5}


def _game(verdict: str | None) -> dict:
    return {"responses_create_params": {"input": "kept on the run host"}, "response": ANSWER, "verdict": verdict}


def _replay(verdicts: tuple, reward: float):
    """A pairwise verifier: plays every game again from the saved answer and returns the whole new row."""

    def grade(json: dict) -> dict:
        # Only the saved answer is sent beside the task input: no old game, verdict or reward goes with it.
        assert json["response"] == ANSWER and not {"games", "reward", "policy_answer"} & json.keys()
        return json | {
            "reward": reward,
            "games": [_game(verdict) for verdict in verdicts],
            "policy_answer": "saved policy answer",
        }

    return grade


async def test_a_pairwise_result_with_a_game_lacking_its_verdict_is_replaced_whole_by_a_full_rejudgment(
    tmp_path, verifier
) -> None:
    output = _run(tmp_path, [pairwise_partial()])
    (identity,) = due_judge_failures(output)
    before, _ = _selected(output)
    verifier["grade"] = _replay(("A>B", "B>A"), 1.0)

    assert await rejudge_saved_judge_failures({"output_jsonl_fpath": str(output)}, output) == 1

    assert verifier["posted"] == [("rs", "/verify")]  # one verify call: every game is played again in it
    after, store = _selected(output)
    settled = after[identity]
    assert settled["response"] == ANSWER and settled["policy_answer"] == "saved policy answer"  # the same saved answer
    assert [game["verdict"] for game in settled["games"]] == ["A>B", "B>A"]  # both games are new, none is kept
    assert settled["reward"] == 1.0 and before[identity]["reward"] == 0.5
    assert settled["_ng_rejudged_failure_id"] == rejudge_predecessor_id(before[identity])
    assert store.manifest.pending_rejudge == {} and due_judge_failures(output) == []
    assert await rejudge_saved_judge_failures({"output_jsonl_fpath": str(output)}, output) == 0


async def test_a_rejudgment_that_again_lacks_a_game_verdict_settles_once_and_the_gate_still_counts_it(
    tmp_path, verifier
) -> None:
    from nemo_gym.orchestration.recovery import _scored_fault, reverification_cause, saved_row_evidence

    output = _run(tmp_path, [pairwise_partial()])
    (identity,) = due_judge_failures(output)
    verifier["grade"] = _replay((None, "B>A"), 0.5)

    assert await rejudge_saved_judge_failures({"output_jsonl_fpath": str(output)}, output) == 1

    settled = _selected(output)[0][identity]
    assert settled["response"] == ANSWER and settled["_ng_rejudge_operation_id"]
    evidence = saved_row_evidence(settled)
    assert _scored_fault(evidence, None) == "judge_unparseable"  # still a counted grading fault
    assert reverification_cause(evidence) is None  # a settled result is never selected again
    assert due_judge_failures(output) == []
    for _ in range(2):  # later passes judge nothing: no loop
        assert await rejudge_saved_judge_failures({"output_jsonl_fpath": str(output)}, output) == 0
    assert len(verifier["posted"]) == 1


# --- a verifier that produced no verdict for a saved patch ----------------------------------------------------------


def test_a_verifier_error_with_its_saved_patch_intact_is_reverified_from_that_patch() -> None:
    assert reverification_cause(saved_row_evidence(verifier_patch_masked())) == "verifier"
    # A saved patch that does not hash to its digest, or none at all, is no verifier input: the mask stays.
    assert reverification_cause(saved_row_evidence(verifier_patch_masked(intact=False))) is None
    assert reverification_cause(saved_row_evidence(verifier_unmeasurable())) is None
    # A judge failure keeps its own cause.
    judge = verifier_patch_masked() | {"failure_kind": "judge_unparseable"}
    assert reverification_cause(saved_row_evidence(judge)) == "judge"


def test_recover_decides_reverify_for_a_verifier_error_with_its_saved_patch() -> None:
    d = decide_recovery(
        materialized=inventory(3),
        results=[scored(0), verifier_patch_masked(1), verifier_patch_masked(2, intact=False)],
        failures=[],
        budget=OK_BUDGET,
    )
    assert d.identities[IdentityState.JUDGE_RECOVERABLE] == [(1, 0)]
    assert d.identities[IdentityState.MASKED] == [(2, 0)]


def test_the_cluster_projection_reverifies_a_saved_patch_without_carrying_it(tmp_path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    rows = [verifier_patch_masked(0), verifier_patch_masked(1, intact=False)]
    (artifacts / "rollouts_materialized_inputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in inventory(2)))
    (artifacts / "rollouts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (artifacts / "rollouts_failures.jsonl").write_text("")
    raw = subprocess.run(
        [sys.executable, "-I", "-S", "-c", COLLECT_SCRIPT, str(tmp_path)], capture_output=True, text=True, check=True
    )
    assert "diff --git" not in raw.stdout
    d = decide_from_collected(
        CollectedEvidence.model_validate_json(raw.stdout), max_infra_retries=3, max_walltime_s=None, live_job=False
    )
    assert d.identities[IdentityState.JUDGE_RECOVERABLE] == [(0, 0)]
    assert d.identities[IdentityState.MASKED] == [(1, 0)]


def test_a_saved_patch_of_a_verifier_that_reads_the_agent_session_is_not_reverified() -> None:
    # Such a verifier would grade an empty patch without its session: the mask stays, as before.
    row = verifier_patch_masked(collected=False)
    assert reverification_cause(saved_row_evidence(row)) is None
    d = decide_recovery(
        materialized=inventory(2), results=[scored(0), row | {"_ng_task_index": 1}], failures=[], budget=OK_BUDGET
    )
    assert d.identities[IdentityState.MASKED] == [(1, 0)]
