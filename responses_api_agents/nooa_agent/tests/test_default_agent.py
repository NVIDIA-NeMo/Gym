# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from nooa_bench.bench_agent import TaskResult

from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming, NeMoGymResponseFunctionToolCall
from nemo_gym.trajectory import TrajectoryRecord
from responses_api_agents.nooa_agent.config import NOOAAgentConfig, NOOAInvocationConfig
from responses_api_agents.nooa_agent.result import finalize_run_result
from responses_api_agents.nooa_agent.runner import InProcessNOOARunner, NOOARunRequest
from responses_api_agents.nooa_agent.tests.test_gym_llm import FakeHTTPResponse, model_response


ROOT = Path(__file__).resolve().parents[3]


def default_config() -> NOOAAgentConfig:
    config = yaml.safe_load((ROOT / "responses_api_agents/nooa_agent/configs/nooa_agent.yaml").read_text())
    return NOOAAgentConfig(
        name="nooa",
        host="127.0.0.1",
        port=9000,
        **config["nooa_agent"]["responses_api_agents"]["nooa_agent"],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("execution_mode", ["embedded", "sandboxed"])
async def test_default_benchagent_runs_and_serializes_module_locals(
    tmp_path: Path, monkeypatch, execution_mode: str
) -> None:
    monkeypatch.chdir(tmp_path)
    config = default_config()
    assert config.nooa.execution_mode == "embedded"
    wrong_directory = tmp_path / "wrong-working-directory"
    wrong_directory.mkdir()
    # A sandbox worker also uses the in-process engine. Only the agent server's
    # ability to launch a worker is gated in earlier stack layers.
    invocation = NOOAInvocationConfig.model_validate(
        config.nooa.model_dump()
        | {"execution_mode": execution_mode, "init_kwargs": {"working_dir": str(wrong_directory)}}
    )
    evidence = "module evidence: imported"
    execute = f"import os\nprint({evidence!r})"
    finish = (
        "return_result(TaskResult(solution_description='completed by default BenchAgent', "
        "evidence=os.getcwd(), how_to_verify='inspect captured stdout'))"
    )
    code = execute + "\n" + finish
    calls = []

    async def post(**kwargs):
        calls.append(kwargs)
        index = len(calls) - 1
        output = NeMoGymResponseFunctionToolCall(
            id=f"function-{index}",
            call_id=f"call-{index}",
            name="python_cell",
            arguments=json.dumps({"code": code}),
        )
        return FakeHTTPResponse(model_response(output, response_id=f"response-{index}"))

    runner = InProcessNOOARunner(
        invocation=invocation,
        server_client=SimpleNamespace(post=post),
        model_server_name=config.model_server.name,
        max_policy_calls=1,
    )
    result = await runner.run(
        NOOARunRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="Inspect the working directory"),
            model_url_path="/v1/responses",
            task_id="default-agent-task",
            rollout_id="default-agent-rollout",
        )
    )

    assert isinstance(result.return_value, TaskResult)
    assert result.return_value.evidence == str(tmp_path)
    assert result.return_value.solution_description == "completed by default BenchAgent"
    assert result.termination_reason is None
    assert len(calls) == len(result.trajectory.turns) == 1
    executions = [call for call in result.trajectory.tool_calls if call.tool_name == "execute_python"]
    assert any(evidence in call.output for call in executions), [call.output for call in executions]
    assert all(isinstance(call.output, str) for call in executions)
    assert TrajectoryRecord.model_validate_json(result.trajectory.model_dump_json()) == result.trajectory
    response, observations = finalize_run_result(result)
    assert json.loads(response.model_dump_json())["status"] == "completed"
    assert json.loads(response.output[-1].content[0].text) == result.return_value.model_dump()
    assert json.loads(observations.model_dump_json())["records"]


def test_server_accepts_sandboxed_sessions() -> None:
    config = default_config()
    values = config.model_dump()
    values["nooa"]["execution_mode"] = "sandboxed"
    assert NOOAAgentConfig.model_validate(values).nooa.execution_mode == "sandboxed"
