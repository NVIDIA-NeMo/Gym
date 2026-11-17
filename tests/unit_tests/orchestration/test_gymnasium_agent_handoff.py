# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The row ``GymnasiumAgent.run`` returns reaches the collector and the trial gate as observed.

Each row is the agent's own ``model_dump(mode="json")`` from a run with fake model and environment servers, kept in
``fixtures/gymnasium-agent-rows-fake-servers.json``; the agent's tests check that it still returns exactly these rows,
so this core test needs no server code. The model server's capture of the call is written to a ``CaptureStore`` from
the same response the fake model server returned, and ``enrich_run_result`` joins the agent's observations to it and
projects ``ng_trajectory``, as ``ng_collect_rollouts`` does. The row is then written to a job dir and read by the
real collector and gate.
"""

import json
from pathlib import Path

import pytest

from nemo_gym.config.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.eval.recovery.evidence import CollectedEvidence
from nemo_gym.eval.recovery.gate import trial_verdict
from nemo_gym.eval.recovery.planner import decide_from_collected
from nemo_gym.models.model_call_capture import CaptureStore
from nemo_gym.rollouts.rollout_collection import enrich_run_result
from tests.unit_tests.orchestration.test_eval_resubmit_decision import collect_with_score, write_job_dir
from tests.unit_tests.orchestration.test_verifier_recorded_trial_coverage import _gate


ROLLOUT_ID = "handoff_0"
FIXTURE = Path(__file__).parent / "fixtures/cpu-grading-mcqa-healthy.json"
FAKE_SERVER_ROWS = Path(__file__).parent / "fixtures/gymnasium-agent-rows-fake-servers.json"


async def _collected_run(tmp_path, case: str):
    """The case's agent row, enriched as the collector does, written to a job dir and read back by the collector."""
    fixture = json.loads(FAKE_SERVER_ROWS.read_text())[case]
    response, row = fixture["response"], fixture["row"]
    CaptureStore(tmp_path / "capture").record(
        ROLLOUT_ID,
        {
            "model_call_id": "call-1",
            "dialect": "responses",
            "model_ref": {"type": "responses_api_models", "name": "policy_model"},
            "request": {"input": [{"role": "user", "content": "play"}]},
            "response": response,
            "status_code": 200,
            "model_operations": [
                {
                    "operation_id": "op-1",
                    "outcome": "completed",
                    "final_attempt_id": "op-1-1",
                    "failure_id": None,
                    "http_attempts": 1,
                    "transport_failures": 0,
                }
            ],
            "failure_report": {"failure_id": None, "omitted": False, "events": {}},
        },
    )
    assert row["ng_agent_observations"]["records"][0]["model_calls"][0]["response_id"] == response["id"]
    await enrich_run_result(
        {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "_ng_rollout_id": ROLLOUT_ID},
        row,
        capture_dirs=[tmp_path / "capture"],
        observability_enabled=True,
    )
    row |= {"_ng_task_index": 0, "_ng_rollout_index": 0}
    job = write_job_dir(tmp_path / "job", n=1, results=[json.loads(json.dumps(row))], failures=[])
    collected = collect_with_score(job, "[]")
    # Hand-built: the run-level routing the dispatcher and the resolved config provide (which server is the
    # environment, which agent and resources server it names), taken from a one-trial fixture of such a run.
    base = CollectedEvidence.model_validate_json(FIXTURE.read_text())
    servers = base.quality["provenance"]["servers"]
    servers["simple"]["implementation"] = "gymnasium_agent"
    [real] = collected.results
    dispatched = {
        k: v for k, v in base.results[0].items() if k in ("agent_ref", "_ng_environment_server", "_ng_result_type")
    }
    base.results = [real | dispatched]
    base.materialized[0]["_ng_collected_task_id"] = real["ng_trial_observations"]["task_id"]
    return base


@pytest.mark.asyncio
async def test_the_agents_row_of_a_finished_episode_is_observed_through_its_reward_record(tmp_path):
    collected = await _collected_run(tmp_path, "finished")
    [row] = collected.results
    assert row["_ng_collected_grading_facts"]["environment_reward"] == 1.0
    assert [call["response_metadata"]["response_id"] for call in row["ng_trial_observations"]["model_calls"]] == [
        "resp-1"
    ]
    gaps = {gap["code"] for gap in row["ng_trial_observations"]["gaps"]}
    assert not gaps & {"model_call_reference_unmatched", "conversation_unavailable"}, gaps
    passed, reason, counts, covered = _gate(collected)
    assert covered is True and passed is True, reason
    assert counts["unobserved"] == 0


@pytest.mark.asyncio
async def test_the_agents_row_of_a_cut_off_answer_is_a_counted_model_failure(tmp_path):
    collected = await _collected_run(tmp_path, "cut_off")
    decision = decide_from_collected(collected, max_infra_retries=1, max_walltime_s=None, live_job=False)
    passed, reason, evaluation = trial_verdict(decision, collected)
    assert passed is True, reason
    assert evaluation["exact"] is True
    assert evaluation["evaluation"]["count"] == 0
    assert evaluation["model"]["kinds"] == {"truncated_answer": 1}
