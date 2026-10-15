# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Live rollout-health metrics, against a real in-memory reader."""

from copy import deepcopy
from pathlib import Path

import pytest

from nemo_gym import rollout_collection
from nemo_gym.rollout_health import evaluate_record, run_health_checks
from nemo_gym.telemetry import gym_metrics
from tests.unit_tests.telemetry.test_sandbox_active import collected_metrics  # noqa: F401 - fixture
from tests.unit_tests.test_rollout_health import _call, _record, _trajectory_call, _write_fixture


pytest.importorskip("opentelemetry.sdk.metrics")

VERDICT = gym_metrics.HEALTH_VERDICT_ATTRIBUTE
CHECK = gym_metrics.HEALTH_CHECK_ATTRIBUTE
OUTCOME = gym_metrics.ROLLOUT_OUTCOME_ATTRIBUTE


def _points(collected, instrument):
    return {tuple(sorted(p.attributes.items())): p.value for p in collected().get(instrument, [])}


def _key(**attrs):
    return tuple(sorted(attrs.items()))


def _stored(record: dict, calls: list[dict]) -> dict:
    """The record as the driver holds it and the report reads it back: calls inlined in ``ng_trajectory``."""
    stored = deepcopy(record)
    stored["ng_trajectory"]["model_calls"] = [_trajectory_call(call) for call in calls]
    return stored


def _with_hollow_turns(record: dict, count: int) -> dict:
    """A rollout whose turns are all empty, so ``agent_turn_hollow`` finds ``count`` of them."""
    record = deepcopy(record)
    turn = record["ng_trajectory"]["turns"][0]
    turn["answer"] = None
    record["ng_trajectory"]["turns"] = [{**turn, "turn_no": number} for number in range(1, count + 1)]
    return record


def test_a_check_counts_rollouts_not_findings(collected_metrics):  # noqa: F811
    record = _stored(_with_hollow_turns(_record(0, 0), 3), [_call()])
    digest = evaluate_record(record, frozenset())
    hollow = [finding for finding in digest.findings if finding.check == "agent_turn_hollow"]
    assert len(hollow) == 3

    gym_metrics.record_rollout_health(
        digest.verdict, outcome="scored", flagged_checks=(finding.check for finding in digest.findings)
    )
    flagged = _points(collected_metrics, gym_metrics.ROLLOUT_HEALTH_FLAGGED_INSTRUMENT)
    assert flagged == {_key(**{CHECK: "agent_turn_hollow", OUTCOME: "scored"}): 1}
    assert _points(collected_metrics, gym_metrics.ROLLOUT_HEALTH_VERDICT_INSTRUMENT) == {
        _key(**{VERDICT: "unhealthy", OUTCOME: "scored"}): 1
    }


def test_driver_helper_counts_verdicts_per_outcome(collected_metrics):  # noqa: F811
    unhealthy = _stored(_with_hollow_turns(_record(0, 0), 2), [_call()])
    no_trajectory = {"_ng_task_index": 1, "_ng_rollout_index": 0, "response": {"output": []}}
    rollout_collection._record_rollout_health(unhealthy, "scored", frozenset())
    rollout_collection._record_rollout_health(unhealthy, "failed", frozenset())
    rollout_collection._record_rollout_health(no_trajectory, "scored", frozenset())

    assert _points(collected_metrics, gym_metrics.ROLLOUT_HEALTH_VERDICT_INSTRUMENT) == {
        _key(**{VERDICT: "unhealthy", OUTCOME: "scored"}): 1,
        _key(**{VERDICT: "unhealthy", OUTCOME: "failed"}): 1,
        _key(**{VERDICT: "unobserved", OUTCOME: "scored"}): 1,
    }
    # A check counts the rollouts it flagged, split by outcome so the scored ones match the report.
    assert _points(collected_metrics, gym_metrics.ROLLOUT_HEALTH_FLAGGED_INSTRUMENT) == {
        _key(**{CHECK: "agent_turn_hollow", OUTCOME: "scored"}): 1,
        _key(**{CHECK: "agent_turn_hollow", OUTCOME: "failed"}): 1,
    }


def test_ignored_checks_are_not_emitted(collected_metrics):  # noqa: F811
    record = _stored(_with_hollow_turns(_record(0, 0), 1), [_call()])
    rollout_collection._record_rollout_health(record, "scored", frozenset({"agent_turn_hollow"}))
    flagged = {dict(key)[CHECK] for key in _points(collected_metrics, gym_metrics.ROLLOUT_HEALTH_FLAGGED_INSTRUMENT)}
    assert "agent_turn_hollow" not in flagged


def test_live_digest_equals_the_post_run_report_digest(tmp_path: Path):
    """One code path: the in-memory record and its persisted line yield the same digest."""
    rows = [
        (_with_hollow_turns(_record(0, 0), 2), [_call()]),
        (_record(1, 0), [_call()]),
    ]
    path = _write_fixture(tmp_path, rows)
    report = run_health_checks(path, workers=1)
    for (record, calls), reported in zip(rows, report.rollouts):
        live = evaluate_record(_stored(record, calls), frozenset())
        assert live.model_dump() == reported.model_dump()
