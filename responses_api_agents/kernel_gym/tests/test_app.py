# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import time
from pathlib import Path

import pytest

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import SandboxExecResult
from nemo_gym.server_utils import ServerClient
from responses_api_agents.kernel_gym import app
from responses_api_agents.kernel_gym.app import KernelGymAgent, KernelGymInstanceConfig, KernelGymRunRequest


def test_instance_uses_instance_id_as_task_name() -> None:
    config = KernelGymInstanceConfig.model_construct(problem_info={"instance_id": "kernelbench::level1::1"})

    assert config.task_name == "kernelbench::level1::1"


@pytest.fixture
def sandbox_run(tmp_path: Path, monkeypatch):
    task = tmp_path / "task"
    (task / "tests").mkdir(parents=True)
    (task / "reference.py").write_text("original reference")
    (task / "solution.py").write_text("initial solution")
    (task / "tests" / "verify.py").write_text("trusted verifier")
    (task / "tests" / "test.sh").write_text("python /tests/verify.py")
    (tmp_path / "instruction.txt").write_text("optimize")
    (tmp_path / "agent_runner.py").write_text("trusted runner")
    (tmp_path / "metrics.json").write_text("{}")
    cfg = KernelGymInstanceConfig(
        name="test",
        host="127.0.0.1",
        port=0,
        entrypoint="app.py",
        agent_server_module="responses_api_agents.claude_code_agent.app",
        agent_server_class="ClaudeCodeAgent",
        agent_config_class="ClaudeCodeAgentConfig",
        model_server_url="http://model.invalid",
        problem_info={"task_dir": str(task)},
        body=NeMoGymResponseCreateParamsNonStreaming(input="optimize"),
        persistent_dir=tmp_path,
        verifier_dir=tmp_path / "verifier",
        metrics_fpath=tmp_path / "metrics.json",
        container="image",
        ray_queue_timestamp=time.time(),
        agent_command_str="run-agent",
        results_dir=tmp_path / "results",
    )
    instances = []
    verifier_outcomes = []
    agent_error = []
    agent_response = NeMoGymResponse(
        id="agent-response",
        created_at=0,
        model="actual-model",
        object="response",
        status="completed",
        output=[{"type": "function_call", "call_id": "call-1", "name": "bash", "arguments": "{}"}],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    ).model_dump(mode="json")

    class Sandbox:
        def __init__(self, provider, spec):
            assert not instances or instances[-1].stopped
            self.spec = spec
            self.files = {}
            self.started = False
            self.stopped = False
            self.index = len(instances)
            instances.append(self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            self.stopped = True

        async def start(self):
            self.files = dict(self.spec.files)
            self.started = True

        async def exec(self, command, **kwargs):
            assert self.started and not self.stopped
            if self.index == 0:
                if agent_error:
                    if isinstance(agent_error[0], SandboxExecResult):
                        return agent_error[0]
                    raise agent_error[0]
                self.files.update(
                    {
                        "/workspace/reference.py": "agent changed reference",
                        "/workspace/solution.py": "submitted solution",
                        "/trajectories_mount/response.json": json.dumps(agent_response),
                        "/tests/verify.py": "agent changed verifier",
                        "/logs/verifier/reward.txt": "1",
                    }
                )
            else:
                assert "env" not in kwargs
                assert self.files["/workspace/reference.py"] == "original reference"
                assert self.files["/workspace/solution.py"] == "submitted solution"
                assert self.files["/tests/verify.py"] == "trusted verifier"
                assert "/logs/verifier/reward.txt" not in self.files
                assert "/trajectories_mount/response.json" not in self.files
                result = (
                    verifier_outcomes.pop(0)
                    if verifier_outcomes
                    else {
                        "compiled": True,
                        "correctness": True,
                        "runtime": 1.0,
                        "ref_runtime": 2.0,
                        "speedup": 2.0,
                        "evaluation_completed": True,
                    }
                )
                if isinstance(result, SandboxExecResult):
                    return result
                self.files.update(
                    {
                        "/logs/verifier/result.json": json.dumps(result),
                        "/logs/verifier/reward.txt": "1" if result.get("correctness") else "0",
                        "/logs/verifier/test-stdout.txt": "verifier output",
                    }
                )
            return SandboxExecResult("", "", 0)

        async def download(self, remote, local):
            local.write_text(self.files[remote])

    monkeypatch.setattr(app, "AsyncSandbox", Sandbox)
    return cfg, instances, verifier_outcomes, agent_error, agent_response


async def test_verifier_uses_fresh_sandbox_and_original_reference(sandbox_run):
    cfg, instances, _, _, _ = sandbox_run
    assert await KernelGymAgent._run_kernel(None, cfg)
    assert len(instances) == 2 and all(s.stopped for s in instances)
    metrics = json.loads(cfg.metrics_fpath.read_text())
    assert metrics["evaluation_completed"] and not metrics["mask_sample"]


@pytest.mark.parametrize(
    "failure",
    [
        SandboxExecResult(None, "timeout", 124, "timeout"),
        SandboxExecResult(None, "transport", 0, "sandbox"),
        SandboxExecResult("", "python failed", 1),
        {"evaluation_completed": False, "error": "CUDA unavailable"},
    ],
)
async def test_inconclusive_verification_retries_in_fresh_sandbox(sandbox_run, failure):
    cfg, instances, outcomes, _, _ = sandbox_run
    outcomes.append(failure)
    assert await KernelGymAgent._run_kernel(None, cfg)
    assert len(instances) == 3 and all(s.stopped for s in instances)
    metrics = json.loads(cfg.metrics_fpath.read_text())
    assert metrics["evaluation_completed"] and not metrics["mask_sample"]
    assert not metrics["container_timed_out"]


async def test_exhausted_verifier_retries_mask_the_sample(sandbox_run):
    cfg, instances, outcomes, _, _ = sandbox_run
    outcomes.extend([{"evaluation_completed": False, "error": "CUDA unavailable"}] * 2)
    assert not await KernelGymAgent._run_kernel(None, cfg)
    assert len(instances) == 3 and all(s.stopped for s in instances)
    metrics = json.loads(cfg.metrics_fpath.read_text())
    assert not metrics["evaluation_completed"] and metrics["mask_sample"]


async def test_incorrect_kernel_is_scored_without_retry_or_mask(sandbox_run):
    cfg, instances, outcomes, _, _ = sandbox_run
    outcomes.append({"compiled": True, "correctness": False, "speedup": 0, "evaluation_completed": True})
    assert not await KernelGymAgent._run_kernel(None, cfg)
    assert len(instances) == 2
    metrics = json.loads(cfg.metrics_fpath.read_text())
    assert metrics["evaluation_completed"] and not metrics["mask_sample"]


@pytest.mark.parametrize(
    "failure",
    [
        SandboxExecResult("", "agent failed", 1),
        SandboxExecResult(None, "timeout", 124, "timeout"),
        SandboxExecResult(None, "transport", 0, "sandbox"),
    ],
)
async def test_failed_agent_does_not_grade_seeded_solution(sandbox_run, failure):
    cfg, instances, _, agent_error, _ = sandbox_run
    agent_error.append(failure)
    assert not await KernelGymAgent._run_kernel(None, cfg)
    assert len(instances) == 1 and instances[0].stopped
    metrics = json.loads(cfg.metrics_fpath.read_text())
    assert not metrics["evaluation_completed"] and metrics["mask_sample"]
    assert metrics["agent_timed_out"] is (failure.error_type == "timeout")


async def test_cancellation_stops_agent_without_starting_verification(sandbox_run):
    cfg, instances, _, agent_error, _ = sandbox_run
    agent_error.append(asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await KernelGymAgent._run_kernel(None, cfg)
    assert len(instances) == 1 and instances[0].stopped


@pytest.mark.parametrize(
    "status,error",
    [
        ("failed", None),
        ("completed", {"code": "server_error", "message": "inference failed"}),
        ("failed", {"code": "server_error", "message": "inference failed"}),
    ],
)
async def test_failed_harness_response_is_masked_and_preserved(sandbox_run, monkeypatch, status, error):
    cfg, instances, _, _, agent_response = sandbox_run
    agent_response.update(status=status, error=error)
    agent = KernelGymAgent(config=cfg, server_client=ServerClient.model_construct(global_config_dict={}))
    monkeypatch.setattr(KernelGymAgent, "_setup_params", lambda self, body, rollout_id=None: cfg)

    result = await agent.run(KernelGymRunRequest(responses_create_params=cfg.body))

    assert result.reward == 0 and result.mask_sample and not result.evaluation_completed
    assert len(instances) == 1 and instances[0].stopped
    assert result.response.status == status
    assert (result.response.error.model_dump() if result.response.error else None) == error
    assert result.response.output == NeMoGymResponse.model_validate(agent_response).output
    assert json.loads((cfg.persistent_dir / "response.json").read_text()) == agent_response
    assert not (cfg.persistent_dir / "solution.py").exists()


async def test_incomplete_harness_can_submit_a_kernel(sandbox_run):
    cfg, instances, _, _, agent_response = sandbox_run
    agent_response.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"})

    assert await KernelGymAgent._run_kernel(None, cfg)

    assert len(instances) == 2 and all(s.stopped for s in instances)
    metrics = json.loads(cfg.metrics_fpath.read_text())
    assert metrics["evaluation_completed"] and not metrics["mask_sample"]
