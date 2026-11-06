# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Judge failures of a provider that cannot grade until the operator acts: a quota that resets, a key to fix.

Modelled on ``test_rejudge_saved_judge_failures`` (the saved-answer re-judge from a real Store and a real collected
decision) and on ``test_rollout_recovery`` (the run's stop on rejected credentials).
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock

import orjson

from nemo_gym.core.failure_kinds import quota_reset_text, quota_reset_unix
from nemo_gym.eval.rollout_reverification import due_judge_failures, rejudge_saved_judge_failures
from nemo_gym.orchestration.recovery import RecoveryAction
from nemo_gym.rollouts.rollout_collection import _credentials_rejected_stop, _quota_exhausted_stop
from nemo_gym.rollouts.rollout_records import logical_rollout_id
from nemo_gym.rollouts.rollout_recovery import RunManifest
from nemo_gym.rollouts.rollout_store import RolloutStore
from tests.unit_tests.eval.test_rejudge_saved_judge_failures import ANSWER, _decision, _verifier


def _reset(days: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).strftime("%Y-%m-%d %H:%M UTC")


def _quota_reason(days: float) -> str:
    return f"AA quota exhausted, resets at {_reset(days)}; grade with replay or resubmit after the reset"


_KEY_REASON = "AA denied access (HTTP 403): the account holding the API key is not approved for CritPt grading"


def _job(tmp_path: Path, kind: str, reason: str, *, rollouts: int = 3) -> tuple[Path, Path, list[dict]]:
    """A managed run whose first rollout stopped on ``kind``, as the CritPt server reports it; the rest scored 1.0."""
    job = tmp_path / "job"
    (job / "artifacts").mkdir(parents=True)
    rows = [{"_ng_task_index": i, "_ng_rollout_index": 0, "agent_ref": {"name": "agent"}} for i in range(rollouts)]
    source = job / "artifacts" / "source.jsonl"
    source.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))
    prepare = Mock(
        side_effect=lambda: (rows, RunManifest.create(source, rows, {}, {"agent": {"responses_api_agents": {}}}))
    )
    output = job / "artifacts" / "rollouts.jsonl"
    stopped = []
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        for index, row in enumerate(store.pending()):
            store.record_dispatch(row)
            if index == 0:
                failure = row | {
                    "_ng_failure_class": "judge_failed",
                    "_ng_failure_id": "stop-0",
                    "failure_kind": kind,
                    "failure_reason": reason,
                    "mask_sample": True,
                    "reward": 0.0,
                    "response": ANSWER,
                }
                store.record_outcome(failure)
                stopped.append(failure)
            else:
                store.record_outcome(row | {"reward": 1.0, "response": ANSWER})
    return job, output, stopped


def test_the_reset_time_is_read_from_the_reason():
    assert (
        quota_reset_text("AA quota exhausted, resets at 2026-11-07 21:33 UTC; grade with replay")
        == "2026-11-07 21:33 UTC"
    )
    assert quota_reset_unix("resets at 1970-01-02 00:00 UTC") == 86400
    assert quota_reset_unix("no time here") is None and quota_reset_unix(None) is None


def test_before_the_reset_a_recovery_decision_is_stop_with_the_hint_not_reverify(tmp_path):
    job, _, _ = _job(tmp_path, "provider_quota_exhausted", _quota_reason(1))
    decision = _decision(job)
    assert decision.action == RecoveryAction.STOP
    assert "provider_quota_exhausted" in decision.reason and "resets at" in decision.reason
    assert "gym eval resubmit --recover" in decision.reason and "no model calls" in decision.reason


def test_after_the_reset_a_recovery_decision_is_reverify_from_the_saved_answers(tmp_path):
    job, _, _ = _job(tmp_path, "provider_quota_exhausted", _quota_reason(-1))
    assert _decision(job).action == RecoveryAction.REVERIFY


def test_a_quota_reason_without_a_reset_time_is_graded_again_not_stuck(tmp_path):
    job, _, _ = _job(tmp_path, "provider_quota_exhausted", "AA quota exhausted, resets at an unknown time")
    assert _decision(job).action == RecoveryAction.REVERIFY


def test_rejected_credentials_on_a_grading_row_are_graded_again_once_the_operator_acts(tmp_path):
    job, _, _ = _job(tmp_path, "provider_credentials_rejected", _KEY_REASON)
    assert _decision(job).action == RecoveryAction.REVERIFY


async def test_after_the_reset_a_plain_recover_rejudges_the_saved_answer_without_any_policy_call(
    tmp_path, monkeypatch
):
    job, output, stopped = _job(tmp_path, "provider_quota_exhausted", _quota_reason(-1))

    async def graded(task, json):
        return json | {"reward": 1.0}

    posted = _verifier(monkeypatch, graded)
    assert due_judge_failures(output) == [logical_rollout_id(stopped[0])]

    assert await rejudge_saved_judge_failures({"output_jsonl_fpath": str(output)}, output) == 1

    assert posted == [0]  # one /verify of the saved answer; no policy or model call exists on this path
    store = RolloutStore.read(output, import_legacy=False)
    settled = {logical_rollout_id(r): r for r in store.selected("success")}[logical_rollout_id(stopped[0])]
    assert settled["reward"] == 1.0 and settled["response"] == ANSWER
    assert due_judge_failures(output) == []


def test_the_run_stop_names_the_quota_and_its_reset():
    rows = [
        {"failure_kind": "provider_quota_exhausted", "failure_reason": _quota_reason(1)},
        {"failure_kind": "provider_quota_exhausted", "failure_reason": _quota_reason(1)},
        {"failure_kind": "judge_failed", "failure_reason": "boom"},
    ]
    stop = _quota_exhausted_stop(rows)
    assert stop is not None and stop.failure_kind == "provider_quota_exhausted" and stop.exit_code == 76
    text = str(stop)
    assert text.startswith("EVAL STOPPED: the grading provider's quota is exhausted (provider_quota_exhausted) on 2 ")
    assert "resets at " in text and "nothing can be graded before then" in text
    assert "gym eval resubmit --recover (re-judges the saved answers, no model calls)" in text
    assert _quota_exhausted_stop(rows[2:]) is None


def test_the_credentials_stop_is_not_sandbox_only():
    stop = _credentials_rejected_stop([{"failure_kind": "provider_credentials_rejected"}])
    assert stop is not None and "sandbox" not in str(stop) and "refused the credentials" in str(stop)
