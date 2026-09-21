# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The step loop: prepare, run Terminus 2 with the step's own budget, verify, stop on the gate."""

import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from responses_api_agents.terminus_2_multi_step_sandboxed_agent.app import (
    Terminus2MultiStepAgent,
    Terminus2MultiStepAgentConfig,
    _combine_step_metrics,
)
from responses_api_agents.terminus_2_sandboxed_agent.app import Terminus2AgentRunRequest


def make_config() -> Terminus2MultiStepAgentConfig:
    return Terminus2MultiStepAgentConfig(
        host="",
        port=0,
        entrypoint="",
        name="oragentbench_agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="oragentbench"),
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        max_turns=None,
        enable_summarize=True,
        proactive_summarization_threshold=8000,
        tmux_pane_width=160,
        tmux_pane_height=40,
        model_context_limit=1000,
        model_output_limit=None,
        interleaved_thinking=False,
        llm_request_timeout=10,
        sandbox_provider="sandbox",
        sandbox_timeout=999.0,
        remote_tmux_binary_path=None,
        skills_dir="/skills",
    )


class FakeHTTPResponse:
    def __init__(self, payload: Dict[str, Any]):
        self._payload = payload
        self.status = 200
        self.cookies = {}

    async def json(self):
        return self._payload

    async def text(self):
        return json.dumps(self._payload)

    async def read(self):
        return json.dumps(self._payload).encode()

    def raise_for_status(self):
        return None


def usage(n: int) -> NeMoGymResponseUsage:
    return NeMoGymResponseUsage(
        input_tokens=n,
        input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=1),
        output_tokens=2 * n,
        output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
        total_tokens=3 * n,
    )


def response_for(step_index: int) -> NeMoGymResponse:
    return NeMoGymResponse(
        id=f"resp_{step_index}",
        created_at=0,
        model="policy_model",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="", content=[NeMoGymResponseOutputText(annotations=[], text=f"step {step_index} done")]
            )
        ],
        tool_choice="auto",
        tools=[],
        parallel_tool_calls=False,
        usage=usage(10),
    )


def step_metrics(step_index: int) -> Dict[str, Any]:
    return {
        "terminus2_completed": True,
        "command_exec_times": [1.0],
        "model_call_times": [2.0],
        "average_command_exec_time": 1.0,
        "average_model_call_time": 2.0,
        "total_command_exec_time": 1.0,
        "total_model_call_time": 2.0,
        "command_exec_time_pct": 10.0,
        "model_call_time_pct": 20.0,
        "terminus2_time_taken": 5.0,
        "model_calls_gt_10min": 0,
        "num_proactive_compactions": 0,
        "num_compactions": 0,
        "error": None,
        "usages": [usage(10)],
    }


def verify_payload(**extra) -> Dict[str, Any]:
    return {
        "responses_create_params": {"input": [{"role": "user", "content": "step 0 instruction"}]},
        "response": response_for(0).model_dump(),
        "reward": 1.0,
        "status": "scored",
        "harness_failure": 0.0,
        "task_name": "t",
        "difficulty": "easy",
        "feasibility": 1.0,
        "quality_raw": 2.0,
        "quality": 1.0,
        "upstream_scalar_reward": 1.0,
        "num_steps": 2,
        "steps_completed": 2,
        "step_results": [],
        "evaluation_completed": True,
        "verification_time_taken": 0.1,
        "test_output": "",
        **extra,
    }


class Harness:
    """Fakes the resources server, the sandbox and Terminus 2; records the protocol the agent drove."""

    def __init__(self, steps: List[Dict[str, Any]], stop_after: Dict[int, bool] = None, setup_ok=None):
        self.steps = steps
        self.stop_after = stop_after or {}
        self.setup_ok = setup_ok or {}
        self.posts: List[tuple] = []
        self.executions: List[tuple] = []
        self.sandbox = AsyncMock()
        self.sandbox.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout="", stderr=""))

    async def post(self, server_name, url_path, json, cookies=None):
        self.posts.append((url_path, json))
        if url_path == "/seed_session":
            return FakeHTTPResponse({"sandbox_handle": "sb-1", "steps": self.steps})
        if url_path == "/prepare_step":
            i = json["step_index"]
            return FakeHTTPResponse(
                {
                    "step_index": i,
                    "name": self.steps[i]["name"],
                    "instruction": f"step {i} instruction",
                    "agent_timeout_s": self.steps[i]["agent_timeout_s"],
                    "setup_ok": self.setup_ok.get(i, True),
                    "setup_output": "",
                }
            )
        if url_path == "/verify_step":
            i = json["step_index"]
            return FakeHTTPResponse(
                {
                    "step_index": i,
                    "name": None,
                    "status": "scored",
                    "feasibility": 1.0,
                    "quality_raw": 2.0,
                    "stop": self.stop_after.get(i, False),
                }
            )
        if url_path == "/verify":
            return FakeHTTPResponse(verify_payload(response=json["response"]))
        raise AssertionError(url_path)

    async def execute(self, request, body, sandbox, timeout_s=None):
        index = len(self.executions)
        inputs = [m.model_dump(exclude_none=True) if hasattr(m, "model_dump") else m for m in body.input]
        self.executions.append(([{"role": m["role"], "content": m["content"]} for m in inputs], timeout_s))
        return response_for(index), step_metrics(index)


def make_agent(harness: Harness, monkeypatch) -> Terminus2MultiStepAgent:
    client = MagicMock(spec=ServerClient)
    client.post = AsyncMock(side_effect=harness.post)
    agent = Terminus2MultiStepAgent(config=make_config(), server_client=client)
    monkeypatch.setattr(agent, "_connect_sandbox", AsyncMock(return_value=harness.sandbox))
    monkeypatch.setattr(agent, "_execute", harness.execute)
    monkeypatch.setattr("responses_api_agents.terminus_2_multi_step_sandboxed_agent.app.raise_for_status", AsyncMock())
    return agent


def make_request() -> MagicMock:
    request = MagicMock()
    request.cookies = {"c": "1"}
    request.session = {SESSION_ID_KEY: "session-1"}
    return request


def run_body() -> Terminus2AgentRunRequest:
    return Terminus2AgentRunRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "user", "content": "step 0 instruction"}]
        ),
        task_name="t",
    )


TWO_STEPS = [{"name": "initial_plan", "agent_timeout_s": 2700.0}, {"name": "step2", "agent_timeout_s": 1200.0}]


class TestStepLoop:
    async def test_each_step_uses_its_own_instruction_and_budget_then_verifies(self, monkeypatch):
        harness = Harness(TWO_STEPS)
        agent = make_agent(harness, monkeypatch)
        result = await agent.run(make_request(), run_body())

        assert [p[0] for p in harness.posts] == [
            "/seed_session",
            "/prepare_step",
            "/verify_step",
            "/prepare_step",
            "/verify_step",
            "/verify",
        ]
        # Step 0 keeps the row's own prompt; later steps take the server's instruction.
        assert [t for _, t in harness.executions] == [2700.0, 1200.0]
        assert harness.executions[0][0][0]["content"] == "step 0 instruction"
        assert harness.executions[1][0][0]["content"] == "step 1 instruction"
        # A fresh tmux server precedes every step after the first.
        tmux_kills = [c for c in harness.sandbox.exec.await_args_list if "tmux kill-server" in c.args[0]]
        assert len(tmux_kills) == 1
        # The verify body carries every step's output and summed usage.
        verify_json = harness.posts[-1][1]
        assert len(verify_json["response"]["output"]) == 2
        assert verify_json["response"]["usage"]["input_tokens"] == 20
        assert result.terminus2_completed is True and result.steps_run == 2
        assert result.command_exec_times == [1.0, 1.0] and result.total_model_call_time == 4.0
        harness.sandbox.stop.assert_awaited_once()

    async def test_gate_stop_skips_remaining_steps(self, monkeypatch):
        harness = Harness(TWO_STEPS, stop_after={0: True})
        agent = make_agent(harness, monkeypatch)
        await agent.run(make_request(), run_body())
        assert [p[0] for p in harness.posts] == ["/seed_session", "/prepare_step", "/verify_step", "/verify"]
        assert len(harness.executions) == 1

    async def test_failed_setup_runs_no_model_and_still_verifies(self, monkeypatch):
        harness = Harness(TWO_STEPS, setup_ok={1: False})
        agent = make_agent(harness, monkeypatch)
        result = await agent.run(make_request(), run_body())
        assert [p[0] for p in harness.posts][-2:] == ["/prepare_step", "/verify"]
        assert len(harness.executions) == 1 and result.steps_run == 1

    async def test_single_step_task_is_the_one_step_case(self, monkeypatch):
        harness = Harness([{"name": None, "agent_timeout_s": 2700.0}])
        agent = make_agent(harness, monkeypatch)
        result = await agent.run(make_request(), run_body())
        assert [p[0] for p in harness.posts] == ["/seed_session", "/prepare_step", "/verify_step", "/verify"]
        assert harness.executions == [([{"role": "user", "content": "step 0 instruction"}], 2700.0)]
        assert not any("tmux kill-server" in c.args[0] for c in harness.sandbox.exec.await_args_list)
        assert result.reward == 1.0

    async def test_exception_in_the_loop_still_verifies_and_reports_the_error(self, monkeypatch):
        harness = Harness(TWO_STEPS)

        async def boom(request, body, sandbox, timeout_s=None):
            raise RuntimeError("model endpoint down")

        agent = make_agent(harness, monkeypatch)
        monkeypatch.setattr(agent, "_execute", boom)
        result = await agent.run(make_request(), run_body())
        assert harness.posts[-1][0] == "/verify"
        assert result.terminus2_completed is False and "model endpoint down" in result.error
        harness.sandbox.stop.assert_awaited_once()


def test_combine_step_metrics_sums_and_keeps_per_step_detail():
    combined = _combine_step_metrics(
        [step_metrics(0), step_metrics(1) | {"terminus2_completed": False}], total_time=20.0
    )
    assert combined["terminus2_completed"] is False and combined["steps_run"] == 2
    assert combined["total_command_exec_time"] == 2.0 and combined["command_exec_time_pct"] == pytest.approx(10.0)
    assert len(combined["usages"]) == 2 and len(combined["step_metrics"]) == 2
    assert "usages" not in combined["step_metrics"][0]


def test_combine_with_no_executed_steps_is_not_completed():
    combined = _combine_step_metrics([], total_time=1.0)
    assert combined["terminus2_completed"] is False and combined["steps_run"] == 0 and combined["error"] is None
