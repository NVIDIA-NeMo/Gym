# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise mini-SWE, Gym's transport capture, projection, and health checks."""

import asyncio
import json
import socket
import threading

import pytest
import uvicorn
from fastapi import Body, FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from nemo_gym.base_responses_api_model import (
    ModelCallCaptureConfig,
    install_model_call_capture,
    merge_model_call_capture_into_record,
)
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_collection import _attach_trajectory_record
from nemo_gym.rollout_health import run_health_checks
from responses_api_agents.miniswe_sandboxed_agent.harness import HarnessContext, MiniSWEConfig


async def test_sandbox_calls_gym_model_capture_url_directly(tmp_path, runner_factory):
    app = FastAPI()

    @app.post("/v1/responses")
    async def model(body: dict = Body()):
        return {
            "id": "direct-response",
            "object": "response",
            "model": "controlled",
            "created_at": 0,
            "parallel_tool_calls": False,
            "tools": [],
            "tool_choice": "auto",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "submit",
                    "name": "bash",
                    "arguments": '{"command":"echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}',
                }
            ],
            "usage": {
                "input_tokens": 2,
                "output_tokens": 3,
                "total_tokens": 5,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }

    capture_dir = tmp_path / "model_calls"
    install_model_call_capture(
        app,
        ModelCallCaptureConfig(observability_enabled=True, model_call_capture_dir=capture_dir),
        model_server_name="model",
    )
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    try:
        async with asyncio.timeout(5):
            while not server.started:
                await asyncio.sleep(0.01)

        async def unexpected_query(params):
            raise AssertionError("The sandbox used the host query callback")

        harness = await runner_factory(
            context=HarnessContext(session_id="direct-session", task_id="0", rollout_id="0-0", instruction="submit"),
            config=MiniSWEConfig(step_limit=1),
            params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            query=unexpected_query,
            model_name="model",
            directory=tmp_path / "artifacts",
            observability_enabled=True,
        )
        harness.model_base_url = f"http://127.0.0.1:{listener.getsockname()[1]}/ng-rollout/0-0/v1"
        response, outcome, extra = await harness.execute(15)
        assert outcome.reason == "completed"
        assert response.usage.total_tokens == 5
        record = {"_ng_task_index": 0, "_ng_rollout_index": 0, **extra}
        merge_model_call_capture_into_record(record, [capture_dir], include_payloads=True)
        calls = record["ng_model_call_capture"]["calls"]
        assert len(calls) == 1
        assert calls[0]["response_id"] == "direct-response"
        assert calls[0]["client_session_id"] == "direct-session"
        assert extra["ng_agent_observations"]["records"][0]["model_calls"][0]["response_id"] == "direct-response"
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        listener.close()


@pytest.mark.parametrize(
    "scenario", ["success", "rejection", "http_error", "missing_usage", "missing_details", "tool_error"]
)
async def test_captured_loop_preserves_evidence(tmp_path, runner_factory, scenario):
    app = FastAPI()
    requests = []

    @app.post("/v1/responses")
    async def model(body: dict = Body()):
        requests.append(body)
        index = len(requests)
        if scenario == "http_error" and index == 2:
            return JSONResponse({"error": {"message": "controlled failure"}}, status_code=503)
        command = "submit" if index == 2 else "inspect"
        output = [
            {
                "type": "function_call",
                "call_id": f"tool-{index}",
                "name": "bash",
                "arguments": json.dumps(
                    {
                        "command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT; echo finished"
                        if command == "submit"
                        else "echo inspected; exit 7"
                        if scenario == "tool_error"
                        else "echo inspected"
                    }
                ),
            }
        ]
        if scenario == "rejection" and index == 1:
            output = [
                {
                    "type": "message",
                    "id": "rejected",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "No tool call", "annotations": []}],
                }
            ]
        usage = {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "input_tokens_details": {
                "cached_tokens": None if scenario == "missing_details" else (2 if index == 1 else 0)
            },
            "output_tokens_details": {"reasoning_tokens": 3},
        }
        return {
            "id": f"response-{index}",
            "object": "response",
            "model": "controlled",
            "created_at": 0,
            "status": "completed",
            "parallel_tool_calls": False,
            "tools": [],
            "tool_choice": "auto",
            "output": output,
            "usage": None if scenario == "missing_usage" and index == 1 else usage,
        }

    capture_dir = tmp_path / "model_calls"
    install_model_call_capture(
        app,
        ModelCallCaptureConfig(observability_enabled=True, model_call_capture_dir=capture_dir),
        model_server_name="model",
    )
    client = TestClient(app)

    async def query(params):
        response = await asyncio.to_thread(
            client.post, "/ng-rollout/0-0/v1/responses", json=params, headers={"x-session-id": "invocation"}
        )
        response.raise_for_status()
        return NeMoGymResponse.model_validate(response.json())

    harness = await runner_factory(
        context=HarnessContext(
            session_id="invocation", task_id="0", rollout_id="0-0", instruction="inspect then submit"
        ),
        config=MiniSWEConfig(step_limit=2),
        params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        query=query,
        model_name="model",
        directory=tmp_path,
        observability_enabled=True,
    )
    response, outcome, extra = await harness.execute(15)
    client.close()
    record = {
        "_ng_task_index": 0,
        "_ng_rollout_index": 0,
        "reward": 0.0,
        "response": response.model_dump(mode="json"),
        **extra,
    }
    merge_model_call_capture_into_record(record, [capture_dir], include_payloads=True)
    _attach_trajectory_record(record, record)
    trajectory = record["ng_trajectory"]
    invocations = trajectory["invocations"]
    assert len(invocations) == 1
    captured = record["ng_model_call_capture"]["calls"]
    assert len(invocations[0]["model_calls"]) == len(captured) == len(requests)
    assert {ref["model_call_id"] for ref in invocations[0]["model_calls"]} == {
        call["model_call_id"] for call in captured
    }
    assert all(call["client_session_id"] == "invocation" for call in captured)
    assert all(turn["resolved"] is None for turn in trajectory["turns"])
    assert len(trajectory["model_calls"]) == len(requests)
    assert trajectory["model_calls"][0]["request"]["input"] == requests[0]["input"]
    if scenario == "http_error":
        assert outcome.reason == "infrastructure_error"
        assert captured[-1]["response_id"] is None
        assert captured[-1]["status_code"] == 503
    elif scenario == "missing_usage":
        assert response.usage is None
        assert captured[0]["tokens_in"] is None
    elif scenario == "missing_details":
        assert response.usage.input_tokens_details.cached_tokens is None
        assert all(call["cached_tokens"] is None for call in captured)
    elif scenario == "rejection":
        assert len(trajectory["turns"]) == 2
        assert trajectory["turns"][0]["answer"][0]["id"] == "rejected"
        assert trajectory["turns"][0]["step_count"] == 0
        assert "No tool calls" in requests[1]["input"][-1]["content"]
    else:
        assert response.usage.total_tokens == 30
        assert response.usage.input_tokens_details.cached_tokens == 2
        assert trajectory["tool_calls"][0]["status"] == ("failed" if scenario == "tool_error" else "completed")
        assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in trajectory["tool_calls"][-1]["output"]
    path = tmp_path / "evaluator_rollouts.jsonl"
    path.write_text(json.dumps(record) + "\n")
    result = run_health_checks(path, output_dir=tmp_path, workers=1)
    coverage = result.summary["run"]["artifacts"]["coverage"]
    for key in (
        "model_call_zero_completion_tokens",
        "model_call_missing_token_counts",
        "model_call_failed",
        "model_call_runaway_generation",
        "rollout_missing_agent_turns",
        "agent_turn_hollow",
    ):
        assert coverage[key]["evaluated"] == 1, (key, result.summary)
    if scenario == "http_error":
        assert result.summary["run"]["issues"]["model_call_failed"] == 1
