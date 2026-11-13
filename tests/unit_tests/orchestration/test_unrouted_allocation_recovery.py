# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unanswered diagnostics become retryable infrastructure failures only after their allocation ends."""

import hashlib
import shlex
import sys
from pathlib import Path

import pytest

from nemo_gym.agents.agent_outcome import OutcomeAction
from nemo_gym.orchestration.api import ResumeConfig
from nemo_gym.orchestration.recovery import RecoveryOutcomePolicyConfig, decide_from_collected
from nemo_gym.rollouts.omitted_rollouts import omitted_rollout_record
from nemo_gym.rollouts.rollout_collection import RolloutCollectionConfig, _failure_compatibility_row, _failure_outcome
from nemo_gym.rollouts.rollout_recovery import (
    ENDED_ALLOCATION_MANIFEST_ENV,
    PendingRejudge,
    manifest_path_for,
    run_lock,
)
from nemo_gym.rollouts.rollout_store import RolloutStore
from tests.unit_tests.orchestration.test_allocation_cut_recovery import _cut_run, _decide, _digests
from tests.unit_tests.orchestration.test_eval_resubmit import _handle, _resumable_run, _sbatch_env
from tests.unit_tests.orchestration.test_eval_resubmit import _local_index as _local_index
from tests.unit_tests.orchestration.test_eval_resubmit import slurm_bin as slurm_bin
from tests.unit_tests.orchestration.test_resume_script import _run_prologue


def _run(job, *, prior_attempts=0, protected=None, prior_outcome=None, chain=True):
    output = _cut_run(job, scored=1, in_flight=1, not_started=0, chain=chain)
    reader = RolloutStore.read(output)
    row = dict(reader._state.expected["1-0"], _ng_run_id=reader.manifest.run_id, _ng_attempt_index=0)
    with pytest.MonkeyPatch.context() as patch, RolloutStore(output, reader._state) as writer:
        patch.setenv("SLURM_JOB_ID", "100")
        for _ in range(prior_attempts):
            writer.record_outcome(row | (prior_outcome or {"_ng_failure_class": "rollout_allocation_ended"}))
            row.pop("_ng_attempt_index")
            writer.record_dispatch(row)
        diagnostic = {"_ng_failure_class": "agent_run_error", "_ng_failure_http_status": 500, **(protected or {})}
        diagnostic.update(_failure_compatibility_row(_failure_outcome(row, diagnostic, "transport")))
        writer.record_unstarted(omitted_rollout_record(row | diagnostic) | {"_ng_unrouted_run_error": True})
    return output


def _resume(output, *, approved=False):
    reader = RolloutStore.read(output)
    with pytest.MonkeyPatch.context() as patch:
        if approved:
            patch.setenv(
                ENDED_ALLOCATION_MANIFEST_ENV, hashlib.sha256(manifest_path_for(output).read_bytes()).hexdigest()
            )
        return RolloutStore.start_or_resume(
            output, lambda: (list(reader._state.expected.values()), reader.manifest), resume=True
        )


@pytest.mark.parametrize("prior_attempts", [0, 2])
def test_ended_diagnostic_retries_within_budget_and_only_a_grade_completes(tmp_path, prior_attempts):
    output = _run(tmp_path, prior_attempts=prior_attempts)
    before = _digests(tmp_path)
    collected, decision = _decide(tmp_path, allocation_cut=True)
    state = "retryable" if prior_attempts == 0 else "exhausted"
    assert decision.counts == {"scored": 1, state: 1}
    assert decision.action.value == ("resubmit" if prior_attempts == 0 else "stop")
    assert decision.evidence[-1].failure_kind == "rollout_allocation_ended"
    assert decision.evidence[-1].attempts == prior_attempts + 1
    assert _digests(tmp_path) == before  # Planning is read-only.
    assert RolloutStore.read(output).pending() == []
    if prior_attempts:
        extended = decide_from_collected(
            collected, max_infra_retries=2, max_walltime_s=None, live_job=False, allocation_cut=True, extend_attempts=1
        )
        assert extended.action.value == "resubmit" and extended.counts == {"scored": 1, "retryable": 1}
    sidecar = output.with_name("rollouts_failures.jsonl")
    original = sidecar.read_bytes()
    manifest = manifest_path_for(output).read_bytes()
    for _ in range(2):
        with run_lock(output), _resume(output, approved=True) as writer:
            coverage = writer.coverage()
            assert coverage["missing"] == coverage["final_zeros"] == 0
            assert coverage["complete"] is False
            assert bool(writer.pending()) is (prior_attempts == 0)
        reader = RolloutStore.read(output)
        assert reader.coverage() == writer.coverage()
        failure = reader.failures()[0]
        assert failure["_ng_failure_terminal"] is False
        assert failure["_ng_attempt_index"] == prior_attempts
        assert "_ng_failure_counts_as_zero" not in failure and "reward" not in failure
        assert manifest_path_for(output).read_bytes() == manifest
        assert sidecar.read_bytes().startswith(original)
        assert len(sidecar.read_bytes().splitlines()) == len(original.splitlines()) + 1
    # The require_complete gate still rejects this run; no failure is projected into the score.
    coverage = reader.coverage()
    config = RolloutCollectionConfig(input_jsonl_fpath="unused", output_jsonl_fpath=str(output), require_complete=True)
    assert (
        config.completion_rejection(
            expected=2,
            results=reader.selected("success"),
            completed=coverage["successful"] + coverage["final_zeros"],
            retryable=bool(reader.pending()),
            made_progress=True,
        )
        is not None
    )
    if prior_attempts:
        output.with_name("rollouts_attempt_extension").write_text("1\n")
    with run_lock(output), _resume(output, approved=True) as writer:
        retry = writer.pending()[0]
        assert retry["_ng_attempt_index"] == prior_attempts + 1
        writer.record_dispatch(retry)
        writer.record_outcome(retry | {"reward": 1.0})
    assert RolloutStore.read(output).coverage()["complete"] is True
    assert _decide(tmp_path, allocation_cut=False)[1].counts == {"scored": 2}


@pytest.mark.parametrize("allocation_cut,live", [(False, False), (True, True)])
def test_live_or_unconfirmed_allocation_does_not_authorize_replay(tmp_path, allocation_cut, live):
    output = _run(tmp_path)
    before = _digests(tmp_path)
    _, decision = _decide(tmp_path, allocation_cut=allocation_cut, live_job=live)
    assert decision.action.value == ("noop" if live else "stop")
    assert decision.counts == {"scored": 1, "unknown": 1}
    assert RolloutStore.read(output).pending() == []
    assert _digests(tmp_path) == before


@pytest.mark.parametrize(
    "protection",
    [
        {"response": {"object": "response", "id": "saved", "status": "completed", "output": []}},
        {"_ng_failure_partial_response": {}},
        {"reward": 1.0, "evaluation_completed": True},
        {"_ng_failure_counts_as_zero": True},
        {"_ng_failure_terminal": True},
        {"failure_outcome_action": "fail"},
        {"failure_outcome_action": "retry", "failure_outcome_attribution": "model"},
        {"_ng_failure_class": "model_request_error"},
        {"_ng_failure_class": "judge_failed"},
        {"_ng_failure_stage": "verification"},
        {"_ng_failure_events_omitted": True},
        {"_ng_failure_id": "missing"},
        {"_ng_failure_id": "outer", "_ng_failure_events": {"outer": {"cause_id": "missing"}}},
        {"_ng_failure_id": "outer", "_ng_failure_events": {"outer": {"cause_id": "outer"}}},
        {"_ng_failure_class": "harness_termination_unconfirmed"},
        {"_ng_failure_class": "provider_credentials_rejected"},
        {"_ng_failure_class": "provider_quota_exhausted"},
    ],
)
def test_protected_diagnostic_stays_unresolved(tmp_path, protection):
    output = _run(tmp_path, protected=protection)
    before = output.with_name("rollouts_failures.jsonl").read_bytes()
    _, decision = _decide(tmp_path, allocation_cut=True)
    assert decision.action.value == "stop"
    with run_lock(output), _resume(output, approved=True) as writer:
        assert writer.coverage()["missing"] == 1 and not writer.pending()
    assert not RolloutStore.read(output).allocation_end_diagnostics()
    assert output.with_name("rollouts_failures.jsonl").read_bytes() == before


@pytest.mark.parametrize(
    "earlier",
    [
        {"reward": 1.0},
        {
            "_ng_failure_class": "judge_failed",
            "response": {
                "object": "response",
                "id": "saved",
                "status": "completed",
                "output": [
                    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "answer"}]}
                ],
            },
        },
        {"_ng_failure_class": "model_context_limit", "_ng_failure_terminal": True, "_ng_failure_counts_as_zero": True},
    ],
)
def test_earlier_answer_or_grade_is_never_replaced(tmp_path, earlier):
    output = _run(tmp_path, prior_attempts=1, prior_outcome=earlier)
    with run_lock(output), _resume(output, approved=True) as writer:
        assert writer.coverage()["missing"] == 1 and not writer.pending()


def test_pending_rejudgment_stays_unresolved(tmp_path):
    output = _run(tmp_path)
    reader = RolloutStore.read(output)
    pending = PendingRejudge(
        operation_id="a" * 32, predecessor_failure_id="saved", predecessor_sha256="b" * 64, admitted_at=1.0
    )
    reader.manifest.model_copy(update={"pending_rejudge": {"1-0": pending}}).write(manifest_path_for(output))
    with run_lock(output), _resume(output, approved=True) as writer:
        assert writer.unresolved_rejudgments() == ["1-0"] and not writer.pending()


def test_allocation_failure_obeys_configured_recovery_policy(tmp_path):
    _run(tmp_path)
    collected, _ = _decide(tmp_path, allocation_cut=True)
    decision = decide_from_collected(
        collected,
        max_infra_retries=2,
        max_walltime_s=None,
        live_job=False,
        allocation_cut=True,
        outcome_policy=RecoveryOutcomePolicyConfig(by_failure_kind={"rollout_allocation_ended": OutcomeAction.FAIL}),
    )
    assert decision.action.value == "stop" and decision.counts == {"scored": 1, "terminal": 1}


@pytest.mark.parametrize("admission", [None, "stale-manifest"])
def test_plain_resume_never_closes_unconfirmed_diagnostics(tmp_path, monkeypatch, admission):
    output = _run(tmp_path)
    if admission is not None:
        monkeypatch.setenv(ENDED_ALLOCATION_MANIFEST_ENV, admission)
    before = output.with_name("rollouts_failures.jsonl").read_bytes()
    with run_lock(output), _resume(output) as writer:
        assert writer.coverage()["missing"] == 1 and not writer.pending()
    assert output.with_name("rollouts_failures.jsonl").read_bytes() == before


@pytest.mark.parametrize(
    "prior",
    [
        {"_ng_failure_class": "provider_unavailable", "_ng_failure_response": {"status": "failed"}},
        {
            "mask_sample": True,
            "reward": 0.0,
            "failure_kind": "env:exec_transport_failed",
            "failure_outcome_action": "retry",
            "failure_outcome_attribution": "infrastructure",
        },
    ],
)
def test_retryable_prior_attempt_does_not_protect_a_failed_response(tmp_path, prior):
    output = _run(tmp_path, prior_attempts=1, prior_outcome=prior)
    with run_lock(output), _resume(output, approved=True) as writer:
        assert len(writer.pending()) == 1
        assert writer.coverage()["counted_failures"] == 2


def test_auto_continuation_passes_the_confirmed_manifest_to_the_writer(tmp_path):
    output = _run(tmp_path)
    code = (
        f"import sys; sys.path.insert(0, {str(Path(__file__).resolve().parents[3])!r}); "
        "from pathlib import Path; "
        "from tests.unit_tests.orchestration.test_unrouted_allocation_recovery import _resume; "
        f"writer = _resume(Path({str(output)!r})); "
        "writer.__enter__(); assert len(writer.pending()) == 1; writer.__exit__(None, None, None)"
    )
    result = _run_prologue(
        tmp_path,
        ResumeConfig(max_retries=2),
        "FAILED",
        exit_code="1:0",
        driver=f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert RolloutStore.read(output).coverage()["counted_failures"] == 1


def test_recover_passes_the_confirmed_manifest_to_the_queued_writer(tmp_path, slurm_bin, monkeypatch):
    from nemo_gym.orchestration.submit import recover

    job, _, _ = _resumable_run(tmp_path)
    for name in ("rollouts.jsonl", "rollouts_materialized_inputs.jsonl"):
        (job / "artifacts" / name).unlink()
    output = _run(job, chain=False)
    (slurm_bin / "accounting").write_text("100|FAILED|7200|0\n101|FAILED|60|0\n")
    result = recover(*_handle(job))
    assert result.dispatched == "resubmit", result.decision.reason
    admission = _sbatch_env(slurm_bin)[ENDED_ALLOCATION_MANIFEST_ENV]
    assert admission == hashlib.sha256(manifest_path_for(output).read_bytes()).hexdigest()
    monkeypatch.setenv(ENDED_ALLOCATION_MANIFEST_ENV, admission)
    with run_lock(output), _resume(output) as writer:
        assert len(writer.pending()) == 1


@pytest.mark.parametrize("stage", ["agent", "verification"])
def test_failed_agent_diagnostic_is_not_a_saved_answer(tmp_path, stage):
    response = {
        "object": "response",
        "id": "failed",
        "status": "failed",
        "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "partial"}]}],
    }
    output = _run(tmp_path, protected={"_ng_failure_stage": stage, "_ng_failure_response": response})
    with run_lock(output), _resume(output, approved=True) as writer:
        assert bool(writer.pending()) is (stage == "agent")
        assert writer.coverage()["counted_failures"] == (stage == "agent")


@pytest.mark.parametrize(
    "boundary,operation,selected,retry",
    [
        ("model", "model_request", True, False),
        ("resources_server", "verify", True, False),
        ("resources_server", "tool", True, True),
        ("model", "model_request", False, True),
    ],
)
def test_allocation_end_respects_selected_cause_not_recovered_observations(
    tmp_path, boundary, operation, selected, retry
):
    diagnostic = {
        "_ng_failure_class": "environment_server_failed",
        "_ng_failure_id": "outer",
        "_ng_failure_events": {
            "outer": {
                "boundary": "environment",
                "operation": "episode.run",
                "failure_kind": "environment_server_failed",
                "cause_id": "cause" if selected else None,
            },
            "cause": {
                "boundary": boundary,
                "operation": operation,
                "failure_kind": "model_request_error" if boundary == "model" else "infrastructure_error",
                "http_status": 400 if boundary == "model" else 500,
                "cause_id": None,
            },
        },
    }
    output = _run(tmp_path, protected=diagnostic)
    before = output.with_name("rollouts_failures.jsonl").read_bytes()
    _, decision = _decide(tmp_path, allocation_cut=True)
    assert (decision.action.value == "resubmit") is retry
    with run_lock(output), _resume(output, approved=True) as writer:
        assert bool(writer.pending()) is retry
        assert writer.coverage()["final_zeros"] == 0
    if not retry:
        assert output.with_name("rollouts_failures.jsonl").read_bytes() == before
