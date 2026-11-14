# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A row a real ``GymnasiumAgent.run`` returns reaches the collector and the trial gate as observed.

The row is the agent's own ``model_dump(mode="json")``. The model server's capture of each call is written to a
``CaptureStore`` from the same response the mocked model server returned (the one hand-built part: no model server
runs here), and ``enrich_run_result`` joins the agent's observations to it and projects ``ng_trajectory``, as
``ng_collect_rollouts`` does. The row is then written to a job dir and read by the real collector and gate.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.models.model_call_capture import CaptureStore
from nemo_gym.orchestration.recovery import CollectedEvidence, decide_from_collected, trial_verdict
from nemo_gym.rollouts.rollout_collection import enrich_run_result
from responses_api_agents.gymnasium_agent.app import GymnasiumAgentRunRequest
from responses_api_agents.gymnasium_agent.tests.test_app import _FakeHttpResp, _make_agent, _model_response
from tests.unit_tests.orchestration.test_eval_resubmit_decision import collect_with_score, write_job_dir
from tests.unit_tests.orchestration.test_verifier_recorded_trial_coverage import _gate


ROLLOUT_ID = "handoff_0"
FIXTURE = Path(__file__).parent / "fixtures/cpu-grading-mcqa-healthy.json"


async def _collected_run(tmp_path, *, step: dict, response: dict):
    """One real agent run, enriched as the collector does, written to a job dir and read back by the collector."""
    agent = _make_agent()
    store = CaptureStore(tmp_path / "capture")
    payloads = {"/reset": [{"observation": "go", "info": {}}], "/step": [step]}

    async def _post(server_name, url_path, json=None, cookies=None, headers=None, **kw):
        if url_path.endswith("/v1/responses"):
            store.record(
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
            return _FakeHttpResp(response)
        return _FakeHttpResp(payloads[url_path].pop(0))

    agent.server_client.post = AsyncMock(side_effect=_post)
    body = GymnasiumAgentRunRequest(
        responses_create_params={"input": [{"role": "user", "content": "play"}]},
        **{TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "_ng_rollout_id": ROLLOUT_ID},
    )
    row = (await agent.run(MagicMock(cookies={}), body)).model_dump(mode="json")
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
async def test_a_finished_episode_of_a_real_run_is_observed_through_its_reward_record(tmp_path):
    step = {"observation": None, "reward": 1.0, "terminated": True, "truncated": False, "info": {}}
    collected = await _collected_run(tmp_path, step=step, response=_model_response("a") | {"id": "resp-1"})
    [row] = collected.results
    assert row["_ng_collected_grading_facts"]["environment_reward"] == 1.0
    assert [call["response_metadata"]["response_id"] for call in row["ng_trial_observations"]["model_calls"]] == [
        "resp-1"
    ]
    assert not [gap for gap in row["ng_trial_observations"]["gaps"] if gap["code"] == "model_call_reference_unmatched"]
    passed, reason, counts, covered = _gate(collected)
    assert covered is True and passed is True, reason
    assert counts["unobserved"] == 0


@pytest.mark.asyncio
async def test_an_answer_cut_off_in_a_real_run_is_a_counted_model_failure(tmp_path):
    step = {"observation": None, "reward": 0.0, "terminated": True, "truncated": False, "info": {}}
    response = _model_response("a") | {
        "id": "resp-1",
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
    }
    collected = await _collected_run(tmp_path, step=step, response=response)
    decision = decide_from_collected(collected, max_infra_retries=1, max_walltime_s=None, live_job=False)
    passed, reason, evaluation = trial_verdict(decision, collected)
    assert passed is True, reason
    assert evaluation["exact"] is True
    assert evaluation["evaluation"]["count"] == 0
    assert evaluation["model"]["kinds"] == {"truncated_answer": 1}
    assert evaluation["model"]["kinds"] == {"truncated_answer": 1}
