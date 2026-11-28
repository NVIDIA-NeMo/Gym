# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A judge failure that names its own kind, through judge_failsafe, collection and recovery planning.

Collection saves such a failsafe row under its kind (``judge_unparseable``, or a provider stop) and keeps the
producer's ``judge_failed`` only as its legacy class. Recovery must still plan a re-judge of the saved answer.
"""

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock

import orjson
import pytest
from pydantic import BaseModel

from nemo_gym.core.failure_records import FailureBoundary
from nemo_gym.core.typed_exceptions import NgFailureEvent
from nemo_gym.eval.recovery.decision import IdentityState, RecoveryAction
from nemo_gym.eval.rollout_reverification import due_judge_failures, rejudge_saved_judge_failures
from nemo_gym.rollouts import rollout_collection as collection
from nemo_gym.rollouts.rollout_records import logical_rollout_id
from nemo_gym.rollouts.rollout_recovery import RunManifest
from nemo_gym.rollouts.rollout_store import RolloutStore
from nemo_gym.verification.judge import JudgeError, judge_failsafe, judge_provider_stopped
from tests.unit_tests.eval.test_rejudge_saved_judge_failures import ANSWER, _decision, _verifier


class _VerifyRequest(BaseModel):
    response: dict = {}


def _quota_stop(days: float) -> NgFailureEvent:
    reset = (datetime.now(timezone.utc) + timedelta(days=days)).strftime("%Y-%m-%d %H:%M UTC")
    return judge_provider_stopped(
        f"quota exhausted, resets at {reset}", failure_kind="provider_quota_exhausted", operation="judge.call"
    )


FAILURES: dict[str, Callable[[], BaseException]] = {
    "judge_failed": lambda: JudgeError("judge 503"),
    "unparseable_judge_error": lambda: JudgeError("no verdict in 3 attempts", failure_kind="judge_unparseable"),
    "unparseable_event": lambda: NgFailureEvent(
        boundary=FailureBoundary.RESOURCES_SERVER, failure_kind="judge_unparseable", failure_reason="no verdict"
    ),
}


async def _collected_job(tmp_path: Path, failure: Callable[[], BaseException]) -> tuple[Path, Path, dict]:
    """A managed run of two rollouts. The first one's /verify raised ``failure`` inside judge_failsafe, and collection
    saved what the failsafe returned. The second one scored 1.0."""
    job = tmp_path / "job"
    (job / "artifacts").mkdir(parents=True)
    rows = [{"_ng_task_index": i, "_ng_rollout_index": 0, "agent_ref": {"name": "agent"}} for i in range(2)]
    source = job / "artifacts" / "source.jsonl"
    source.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))
    prepare = Mock(
        side_effect=lambda: (rows, RunManifest.create(source, rows, {}, {"agent": {"responses_api_agents": {}}}))
    )
    output = job / "artifacts" / "rollouts.jsonl"

    async def verify(body):
        raise failure()

    saved = {}
    with RolloutStore.start_or_resume(output, prepare, resume=False) as store:
        for index, row in enumerate(store.pending()):
            store.record_dispatch(row)
            if index == 0:
                result = orjson.loads((await judge_failsafe(verify)(_VerifyRequest(response=ANSWER))).body)
                completed = collection._CompletedRollout(row=row, result=result, rollout_latency_ms=1.0)
                saved = collection._collected_outcome(completed, result).record
                saved[collection.RUN_ID_KEY] = store.manifest.run_id
                store.record_outcome(saved)
            else:
                store.record_outcome(row | {"reward": 1.0, "response": ANSWER})
    return job, output, saved


@pytest.mark.parametrize("name", list(FAILURES))
async def test_a_collected_judge_failure_is_re_judged_from_its_saved_answer(tmp_path, name):
    job, _, saved = await _collected_job(tmp_path, FAILURES[name])
    kind = "judge_failed" if name == "judge_failed" else "judge_unparseable"
    # Collection keeps the failure's own kind and the producer's routing class beside it.
    assert (saved["_ng_failure_class"], saved["failure_kind"]) == (kind, kind)
    assert saved.get("_ng_failure_legacy_class") == (None if kind == "judge_failed" else "judge_failed")

    decision = _decision(job)

    assert decision.action == RecoveryAction.REVERIFY
    [evidence] = [e for e in decision.evidence if e.failure_kind is not None]
    assert (evidence.state, evidence.failure_kind) == (IdentityState.JUDGE_RECOVERABLE, kind)


async def test_an_unparseable_judge_failure_is_re_judged_without_a_policy_call(tmp_path, monkeypatch):
    _, output, saved = await _collected_job(tmp_path, FAILURES["unparseable_judge_error"])

    async def graded(task, json):
        return json | {"reward": 1.0}

    posted = _verifier(monkeypatch, graded)
    assert due_judge_failures(output) == [logical_rollout_id(saved)]

    assert await rejudge_saved_judge_failures({"output_jsonl_fpath": str(output)}, output) == 1

    assert posted == [0]  # one /verify of the saved answer, and no policy call
    store = RolloutStore.read(output, import_legacy=False)
    settled = {logical_rollout_id(r): r for r in store.selected("success")}[logical_rollout_id(saved)]
    assert settled["reward"] == 1.0 and settled["response"] == ANSWER


@pytest.mark.parametrize("days,action", [(1, RecoveryAction.STOP), (-1, RecoveryAction.REVERIFY)])
async def test_a_collected_quota_stop_waits_for_its_reset_then_is_re_judged(tmp_path, days, action):
    job, _, saved = await _collected_job(tmp_path, lambda: _quota_stop(days))
    assert (saved["_ng_failure_class"], saved["_ng_failure_legacy_class"]) == (
        "provider_quota_exhausted",
        "judge_failed",
    )

    decision = _decision(job)

    assert decision.action == action
    if action == RecoveryAction.STOP:
        assert "provider_quota_exhausted" in decision.reason and "resets at" in decision.reason
