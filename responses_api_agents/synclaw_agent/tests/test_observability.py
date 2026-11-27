# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest
from fastapi import Request, Response

from nemo_gym.rollouts import enrich_run_result
from nemo_gym.rollouts.harness_capabilities import inspect_record
from nemo_gym.trajectory import AgentInvocation, TrajectoryRecord, build_model_call_record
from resources_servers.synclaw.schema import SynClawRunRequest
from responses_api_agents.synclaw_agent.app import _GymToolExecutor
from responses_api_agents.synclaw_agent.tests.test_app import (
    _MESSAGE_ITEM,
    _base_run_request,
    _make_agent,
    _mock_response,
    _prepared_call,
    _responses_payload,
)


def request(rollout_id: str = "0-0") -> Request:
    return Request(
        {
            "type": "http",
            "path": f"/ng-rollout/{rollout_id}/v1/responses",
            "headers": [],
            "path_params": {"rollout_id": rollout_id},
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_error", [False, True])
async def test_real_loop_observations_join_captures_and_pass_evidence(tool_error: bool) -> None:
    agent, client = _make_agent(max_model_len=None, max_turns=2, budget_feedback=False, stop_condition="model")
    client.global_config_dict = {"observability_enabled": True}
    captures = []
    calls = []
    tool_result = (
        {"valid": False, "error": "invalid_tool_call", "retryable": True}
        if tool_error
        else {"valid": True, "results": []}
    )
    outputs = [
        [
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "tool-1",
                "name": "search_building_blocks",
                "arguments": '{"query":"CCO"}',
            }
        ],
        [_MESSAGE_ITEM],
    ]

    async def post(**kwargs):
        path = kwargs["url_path"]
        if path == "/seed_session":
            return _mock_response({"input": [{"role": "user", "content": "Synthesize ethanol."}]})
        if kwargs["server_name"] == "synclaw_agent":
            result = await agent.responses(request(), Response(), kwargs["json"])
            return _mock_response(result.model_dump(mode="json"))
        if path == "/verify":
            assert "_ng_trajectory" not in kwargs["json"]["response"]
            return _mock_response(
                kwargs["json"]
                | {
                    "reward": 0.0,
                    "accepted": False,
                    "submitted": False,
                    "exact_reconstruction": False,
                    "similarity": 0.0,
                    "mask_sample": False,
                    "evaluation_completed": True,
                }
            )
        if path == "/search_building_blocks":
            return _mock_response(tool_result)
        assert kwargs["server_name"] == "model"
        assert path == "/ng-rollout/0-0/v1/responses"
        assert kwargs["headers"] == {"x-session-id": "root"}
        calls.append(deepcopy(kwargs["json"]))
        payload = _responses_payload(outputs[len(captures)])
        payload["id"] = f"resp_{len(captures)}"
        started = time.time()
        exchange = {
            "model_call_id": f"model-{len(captures)}",
            "model_ref": {"type": "responses_api_models", "name": "model"},
            "client_session_id": "root",
            "dialect": "responses",
            "status_code": 200,
            "started_at": started,
            "completed_at": started + 0.001,
            "duration_ms": 1.0,
            "request": deepcopy(kwargs["json"]),
            "response": payload,
        }
        captures.append(build_model_call_record(exchange, call_index=len(captures)).model_dump(mode="json"))
        return _mock_response(payload)

    client.post = post
    body = SynClawRunRequest.model_validate(_base_run_request().model_dump() | {"_ng_rollout_id": "0-0"})
    result = (await agent.run(request(), body)).model_dump(mode="json")
    raw = result["ng_trajectory"]
    assert len(raw["turns"]) == 2
    assert raw["turns"][0]["question"] == calls[0]["input"]
    assert raw["turns"][1]["question"] == calls[1]["input"]
    assert raw["turns"][-1]["resolved"] is False
    assert raw["invocations"][0]["verification_skipped"] is False
    tool = raw["tool_calls"][0]
    assert tool["status"] == ("failed" if tool_error else "completed")
    assert tool["duration_ms"] >= 0
    assert tool["completed_at"] >= tool["started_at"]
    shown = next(item["output"] for item in result["response"]["output"] if item["type"] == "function_call_output")
    assert tool["output"] == shown
    result["ng_model_call_capture"] = {"calls": captures}
    original_response = deepcopy(result["response"])
    await enrich_run_result(
        {"task_id": body.task_id, "_ng_rollout_id": "0-0"}, result, capture_dirs=[], observability_enabled=True
    )
    assert result["response"] == original_response
    assert len(result["ng_trajectory"]["model_calls"]) == 2
    checked = inspect_record(result)
    failed = [check for check in checked["checks"] if check["status"] == "fail"]
    assert not failed, failed


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError("transport failed"), asyncio.CancelledError()])
async def test_tool_failure_and_cancellation_retain_measured_execution(error: BaseException) -> None:
    trajectory = TrajectoryRecord(
        task_id="task", rollout_id="0-0", invocations=[AgentInvocation(invocation_id="root")]
    )
    executor = _GymToolExecutor(AsyncMock(), "resources", trajectory=trajectory)
    executor._dispatch_tool_call = AsyncMock(side_effect=error)
    with pytest.raises(type(error)):
        await executor.execute([_prepared_call("tool-1")], target_smiles="CCO", assistant_turn=1)
    tool = trajectory.tool_calls[0]
    assert tool.status == ("failed" if isinstance(error, Exception) else "cancelled")
    assert tool.error_type == type(error).__name__
    assert tool.completed_at >= tool.started_at
    assert tool.duration_ms >= 0


@pytest.mark.asyncio
async def test_failed_model_call_preserves_invocation_for_boundary() -> None:
    agent, client = _make_agent(max_model_len=None)
    client.global_config_dict = {"observability_enabled": True}
    client.post = AsyncMock(side_effect=RuntimeError("provider unavailable"))
    incoming = request()
    with pytest.raises(RuntimeError, match="provider unavailable"):
        await agent.responses(incoming, Response(), _base_run_request())
    trajectory = incoming.state.nemo_gym_agent_trajectory
    assert trajectory.invocations[0].status == "failed"
    assert trajectory.invocations[0].error_type == "RuntimeError"
    assert trajectory.invocations[0].duration_ms >= 0
    assert client.post.call_args.kwargs["headers"] == {"x-session-id": "root"}
