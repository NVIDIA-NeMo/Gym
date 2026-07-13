# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import Body
from fastapi.testclient import TestClient
from pydantic import ConfigDict, ValidationError

from nemo_gym.agent_execution_capture import (
    AgentExecutionCapture,
    AgentExecutionCaptureStore,
    AgentExecutionRecorder,
    AgentInvocationRecord,
    ModelCallLink,
    clear_agent_execution_captures_for_rollouts,
    merge_agent_execution_capture_into_record,
)
from nemo_gym.base_resources_server import BaseRunRequest
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef
from nemo_gym.server_utils import ServerClient


MODEL_REF = ModelServerRef(type="responses_api_models", name="policy")


def test_recorder_captures_tool_span_and_model_link() -> None:
    ticks = iter([10, 40])
    wall_times = iter([1.0, 1.5])
    recorder = AgentExecutionRecorder(
        "2-3",
        "simple_agent",
        clock=lambda: next(ticks),
        wall_clock=lambda: next(wall_times),
    )

    recorder.add_model_call_link(response_id="response-1", model_ref=MODEL_REF)
    with recorder.tool_span(
        tool_call_id="call-1",
        measurement_scope="caller_round_trip",
        model_ref=MODEL_REF,
        model_response_id="response-1",
    ):
        pass

    capture = recorder.capture()
    assert capture.model_call_links == [
        ModelCallLink(agent_invocation_id="root", model_ref=MODEL_REF, response_id="response-1")
    ]
    span = capture.tool_spans[0]
    assert (span.tool_call_id, span.agent_invocation_id, span.model_response_id) == ("call-1", "root", "response-1")
    assert (span.started_ns, span.ended_ns, span.duration_ms, span.status) == (10, 40, 0.00003, "returned")
    assert (span.started_at, span.completed_at, span.clock_id) == (1.0, 1.5, recorder.clock_id)


def test_tool_span_preserves_original_error_when_recording_fails() -> None:
    reads = 0

    def clock() -> int:
        nonlocal reads
        reads += 1
        if reads == 1:
            return 100
        raise RuntimeError("clock failed")

    recorder = AgentExecutionRecorder("0-0", "agent", clock=clock, wall_clock=lambda: 1.0)
    with pytest.raises(ValueError, match="tool failed"):
        with recorder.tool_span(tool_call_id="call", measurement_scope="caller_round_trip"):
            raise ValueError("tool failed")

    assert recorder.capture().tool_spans == []
    assert recorder.capture().warnings == ["tool_span_recording_failed"]


def test_capture_accepts_forest_and_rejects_invalid_parent_graph() -> None:
    forest = AgentExecutionCapture(
        rollout_id="0-0",
        agent_server="agent",
        agent_invocations=[
            AgentInvocationRecord(id="root-a", source="agent"),
            AgentInvocationRecord(id="root-b", source="agent"),
        ],
    )
    assert len(forest.agent_invocations) == 2

    with pytest.raises(ValidationError, match="cycle"):
        AgentExecutionCapture(
            rollout_id="0-0",
            agent_server="agent",
            agent_invocations=[
                AgentInvocationRecord(id="root", source="agent"),
                AgentInvocationRecord(id="a", parent_id="b", source="agent"),
                AgentInvocationRecord(id="b", parent_id="a", source="agent"),
            ],
        )

    with pytest.raises(ValidationError, match="multiple agent invocations"):
        AgentExecutionCapture(
            rollout_id="0-0",
            agent_server="agent",
            agent_invocations=[
                AgentInvocationRecord(id="root", source="agent"),
                AgentInvocationRecord(id="child", parent_id="root", source="agent"),
            ],
            model_call_links=[
                ModelCallLink(agent_invocation_id="root", model_ref=MODEL_REF, response_id="response-1"),
                ModelCallLink(agent_invocation_id="child", model_ref=MODEL_REF, response_id="response-1"),
            ],
        )


def test_store_merge_and_clear(tmp_path: Path) -> None:
    recorder = AgentExecutionRecorder("1-2", "agent")
    recorder.add_model_call_link(response_id="r0", model_ref=MODEL_REF)
    store = AgentExecutionCaptureStore(tmp_path)
    store.write(recorder.capture())

    record = {"_ng_task_index": 1, "_ng_rollout_index": 2, "reward": 1.0}
    merge_agent_execution_capture_into_record(record, [tmp_path])
    assert record["ng_agent_execution_capture"]["model_call_links"][0]["response_id"] == "r0"

    clear_agent_execution_captures_for_rollouts([record], [tmp_path])
    assert store.read("1-2") is None


class _RunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


def _capture_agent(tmp_path: Path, *, enabled: bool = True, run_raises: bool = False):
    class _Agent(SimpleResponsesAPIAgent):
        def captures_agent_execution(self) -> bool:
            return True

        async def responses(self, body=Body()):
            raise NotImplementedError

        async def run(self, body: _RunRequest = Body()):
            assert (self.agent_execution_recorder_for_run(body) is not None) is enabled
            if run_raises:
                raise RuntimeError("run failed")
            return {"ok": True}

    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = {
        "observability_enabled": enabled,
        "model_call_capture_dir": str(tmp_path),
    }
    config = BaseResponsesAPIAgentConfig(host="", port=0, entrypoint="", name="agent")
    return _Agent(config=config, server_client=server_client)


@pytest.mark.parametrize("run_raises", [False, True])
def test_run_route_flushes_capture_without_replacing_result_or_error(tmp_path: Path, run_raises: bool) -> None:
    agent = _capture_agent(tmp_path, run_raises=run_raises)
    request = {
        "responses_create_params": {"input": []},
        "_ng_task_index": 3,
        "_ng_rollout_index": 4,
    }

    if run_raises:
        with pytest.raises(RuntimeError, match="run failed"):
            TestClient(agent.setup_webserver()).post("/run", json=request)
    else:
        response = TestClient(agent.setup_webserver()).post("/run", json=request)
        assert response.json() == {"ok": True}

    capture = AgentExecutionCaptureStore(tmp_path).read("3-4")
    assert capture is not None
    assert capture.agent_invocations[0].status == ("failed" if run_raises else "completed")


def test_capture_disabled_creates_no_recorder_or_file(tmp_path: Path) -> None:
    agent = _capture_agent(tmp_path, enabled=False)
    response = TestClient(agent.setup_webserver()).post(
        "/run",
        json={
            "responses_create_params": {"input": []},
            "_ng_task_index": 3,
            "_ng_rollout_index": 4,
        },
    )

    assert response.status_code == 200
    assert not tmp_path.exists() or not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_cancelled_finalization_waits_for_capture_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _capture_agent(tmp_path)
    recorder = AgentExecutionRecorder("3-4", "agent")
    lock, recorders = agent._agent_execution_state()
    with lock:
        recorders[recorder.rollout_id] = recorder

    write_started = threading.Event()
    release_write = threading.Event()
    original_write = AgentExecutionCaptureStore.write

    def delayed_write(store, capture) -> None:
        write_started.set()
        release_write.wait()
        original_write(store, capture)

    monkeypatch.setattr(AgentExecutionCaptureStore, "write", delayed_write)
    task = asyncio.create_task(agent._finish_agent_execution_capture(recorder))
    await asyncio.to_thread(write_started.wait)
    task.cancel()
    await asyncio.sleep(0)
    try:
        assert not task.done()
    finally:
        release_write.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert AgentExecutionCaptureStore(tmp_path).read("3-4") is not None
    with lock:
        assert "3-4" not in recorders
