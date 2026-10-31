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

from nemo_gym.base_responses_api_model import ModelCallCaptureConfig, install_model_call_capture
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.observability.model_calls import merge_model_call_capture_into_record
from nemo_gym.rollout_collection import _attach_trajectory_record
from nemo_gym.rollout_health import run_health_checks
from responses_api_agents.miniswe_sandboxed_agent.harness import (
    HarnessContext,
    MiniSWEConfig,
    project_native_trajectory,
)


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
    # The simulated executor (ProcessSandbox) gets the identity a connected AsyncSandbox reports;
    # "missing_details" keeps it unknown to show nothing is invented without one.
    sandbox_id = None if scenario == "missing_details" else "sbx-agent-1"
    if sandbox_id is not None:
        harness.sandbox.sandbox_id = sandbox_id
    response, outcome, extra = await harness.execute(15)
    client.close()
    records = extra["ng_agent_observations"]["records"]
    sandboxes = [r for r in records if r["kind"] == "sandbox"]
    if sandbox_id is None:
        assert sandboxes == []
    else:
        # Identity only: the agent does not own the sandbox lifecycle, so its outcome stays unknown.
        (sandbox,) = sandboxes
        assert (sandbox["role"], sandbox["sandbox_id"], sandbox["outcome"]) == ("agent", sandbox_id, "unknown")
        assert sandbox["sandbox_id"] != harness.context.session_id
        assert all(sandbox[key] is None for key in ("exit_code", "wall_time_s", "cpu_time_s", "peak_memory_mib"))
    assert {r.get("sandbox_id") for r in records if r["kind"] == "tool_call"} == {sandbox_id}
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
    assert {call.get("sandbox_id") for call in trajectory["tool_calls"]} == {sandbox_id}
    # Commands that ran, the submission included, carry the environment's own start, end and monotonic duration
    # into the canonical record; a tool whose command never ran keeps them unknown.
    executed = [call for call in trajectory["tool_calls"] if call.get("completed_at") is not None]
    assert executed
    for call in executed:
        assert isinstance(call["started_at"], float) and call["started_at"] <= call["completed_at"]
        assert isinstance(call["duration_ms"], float) and call["duration_ms"] >= 0
        assert call["timing_source"] == "artifact"
    for call in trajectory["tool_calls"]:
        if call.get("completed_at") is None:
            assert call.get("started_at") is None and call.get("duration_ms") is None
    timing_gaps = {g["detail"] for g in trajectory["gaps"] if g["code"] == "tool_timing_unavailable"}
    assert timing_gaps == {
        call["tool_call_id"] for call in trajectory["tool_calls"] if call.get("duration_ms") is None
    }
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
        assert "No tool calls" in requests[1]["input"][-1]["content"][0]["text"]
    else:
        assert response.usage.total_tokens == 30
        assert response.usage.input_tokens_details.cached_tokens == 2
        assert trajectory["tool_calls"][0]["status"] == ("failed" if scenario == "tool_error" else "completed")
    if scenario != "http_error":
        # Submission exits before saving the final tool observation; its exit message keeps the command's timing.
        assert trajectory["tool_calls"][-1]["output"] is None
        assert trajectory["tool_calls"][-1]["status"] == "completed"
        assert trajectory["tool_calls"][-1]["duration_ms"] is not None
    path = tmp_path / "evaluator_rollouts.jsonl"
    path.write_text(json.dumps(record) + "\n")
    result = run_health_checks(path, output_dir=tmp_path, workers=1)
    coverage = result.summary["run"]["artifacts"]["coverage"]
    for key in (
        "model_call_zero_completion_tokens",
        "model_call_missing_token_counts",
        "rollout_ended_on_failed_model_call",
        "model_call_runaway_generation",
        "rollout_missing_agent_turns",
        "agent_turn_hollow",
    ):
        assert coverage[key]["evaluated"] == 1, (key, result.summary)
    if scenario == "http_error":
        assert result.summary["run"]["issues"]["rollout_ended_on_failed_model_call"] == 1


@pytest.mark.parametrize("final", ["submitted", "limits_exceeded", "no_native_final"])
@pytest.mark.parametrize("cleanup", ["confirmed", "provider_timeout"])
async def test_unconfirmed_process_cleanup_is_a_typed_harness_failure(tmp_path, runner_factory, cleanup, final):
    """A failed cleanup is the run's selected typed failure unless mini-SWE stopped at a native limit.

    After a native limit the agent's result stands and the cleanup stays an unselected failure event; a
    submission keeps the cleanup as its failure. A nonzero model command is always the agent's result.
    """
    from nemo_gym.failure_transport import FailureHeader, FailureReportRow, RequestFailureLog

    app = FastAPI()
    requests = []

    @app.post("/v1/responses")
    async def model(body: dict = Body()):
        requests.append(body)
        if final == "no_native_final" and len(requests) == 2:
            return JSONResponse({"error": {"message": "controlled failure"}}, status_code=503)
        command = "exit 7" if len(requests) == 1 else "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
        return {
            "id": f"response-{len(requests)}",
            "object": "response",
            "model": "controlled",
            "created_at": 0,
            "status": "completed",
            "parallel_tool_calls": False,
            "tools": [],
            "tool_choice": "auto",
            "output": [
                {
                    "type": "function_call",
                    "call_id": f"tool-{len(requests)}",
                    "name": "bash",
                    "arguments": json.dumps({"command": command}),
                }
            ],
        }

    client = TestClient(app)

    async def query(params):
        response = await asyncio.to_thread(client.post, "/v1/responses", json=params)
        response.raise_for_status()
        return NeMoGymResponse.model_validate(response.json())

    harness = await runner_factory(
        context=HarnessContext(session_id="invocation", task_id="0", rollout_id="0-0", instruction="run then submit"),
        config=MiniSWEConfig(step_limit=1 if final == "limits_exceeded" else 3),
        params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        query=query,
        model_name="model",
        directory=tmp_path,
        observability_enabled=True,
    )
    if cleanup == "provider_timeout":
        execute_command = harness.sandbox.exec

        async def exec_with_cleanup_timeout(command, **kwargs):
            if "kill -TERM" in command:
                raise TimeoutError("provider command timed out after 70s")
            return await execute_command(command, **kwargs)

        harness.sandbox.exec = exec_with_cleanup_timeout
    with RequestFailureLog.capture() as log:
        response, outcome, extra = await harness.execute(15)
    client.close()
    # The verify reply merges harness_metadata into the row; Collection then merges the reply header's report.
    forwarded = FailureHeader.decode(FailureHeader.encode(log.reply_report(None)))
    row = FailureReportRow.merged_into(
        {"_ng_task_index": 0, "_ng_rollout_index": 0, **json.loads(json.dumps(extra))}, *filter(None, [forwarded])
    )
    tools = [r for r in extra["ng_agent_observations"]["records"] if r["kind"] == "tool_call"]
    assert tools[0]["status"] == "failed"
    native = ("nonzero_exit", "LimitsExceeded") if final == "limits_exceeded" else None
    if native is not None:
        # mini-SWE's own limit stands whether or not the cleanup was confirmed.
        assert (outcome.reason, outcome.detail) == native
        assert FailureReportRow.FAILURE_ID not in row
    elif cleanup == "confirmed":
        assert outcome.reason == ("completed" if final == "submitted" else "infrastructure_error")
    else:
        # A submission or a run without a native final status is not settled while its processes may live.
        assert outcome.reason == "infrastructure_error"
    if cleanup == "confirmed":
        assert FailureReportRow.EVENTS not in row
        assert not (outcome.detail or "").startswith("Process cleanup failed")
        return
    (event,) = row[FailureReportRow.EVENTS].values()
    assert (event["boundary"], event["failure_kind"], event["operation"]) == (
        "harness",
        "harness_termination_unconfirmed",
        "harness.terminate",
    )
    assert event["foreign_error"]["exception_type"] == "TimeoutError"
    if native is None:
        assert outcome.detail == "Process cleanup failed: provider command timed out after 70s"
        assert row[FailureReportRow.FAILURE_ID] == event["event_id"]


@pytest.mark.parametrize(
    "exit_extra",
    [
        {"exit_status": "LimitsExceeded", "tool_call_id": "tool-1", "started_at": 1.0, "completed_at": 2.0},
        {"tool_call_id": "tool-1", "started_at": 1.0, "completed_at": 2.0, "duration_ms": 1000.0},
        {"exit_status": "Submitted", "started_at": 1.0, "completed_at": 2.0, "duration_ms": 1000.0},
        {"exit_status": "Submitted", "tool_call_id": "other", "started_at": 1.0, "completed_at": 2.0},
    ],
)
def test_only_a_submission_exit_completes_its_tool(exit_extra):
    """An exit that is not mini-SWE's submission of this tool call leaves the call incomplete with unknown timing."""
    trajectory = {
        "messages": [
            {
                "object": "response",
                "id": "response-1",
                "output": [],
                "extra": {"actions": [{"tool_call_id": "tool-1", "command": "echo hi"}], "timestamp": 0.5},
            },
            {"role": "exit", "content": "", "extra": exit_extra},
        ]
    }
    _, (tool,) = project_native_trajectory(trajectory)
    assert tool["status"] == "incomplete"
    assert "started_at" not in tool and "completed_at" not in tool and "duration_ms" not in tool
