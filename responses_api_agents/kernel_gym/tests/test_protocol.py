# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from asyncio import Semaphore
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.kernel_gym.app import (
    KernelBenchMetrics,
    KernelGymAgent,
    KernelGymConfig,
    KernelGymRunRequest,
)


@pytest.fixture
def protocol_agent(tmp_path: Path):
    config = KernelGymConfig(
        name="test",
        host="127.0.0.1",
        port=0,
        entrypoint="app.py",
        agent_server_module="responses_api_agents.claude_code_agent.app",
        agent_server_class="ClaudeCodeAgent",
        agent_config_class="ClaudeCodeAgentConfig",
        agent_kwargs={"api_key": "agent-secret", "nested": [{"access-token": "nested-secret", "max_tokens": 100}]},
        sandbox_provider={"docker": {}, "apiKey": "provider-secret"},
        kb_sandbox_ttl=100,
    )
    # Exercise the request methods without starting the web server or resolving a provider.
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    with patch.object(KernelGymAgent, "model_post_init", lambda self, context: None):
        agent = KernelGymAgent(config=config, server_client=client)
    agent._sem = Semaphore(1)
    agent._base_results_dir = tmp_path / "results"
    agent._model_server_url = "http://model.invalid"
    agent._model_name = "served-model"
    agent._checkpoint_participant = None
    agent._restart_only = None

    task = tmp_path / "task"
    (task / "environment").mkdir(parents=True)
    (task / "task.toml").write_text("[agent]\ntimeout_sec = 45\n[verifier]\ntimeout_sec = 30\n")
    (task / "environment" / "Dockerfile").write_text("FROM image\nWORKDIR /workspace\n")
    body = NeMoGymResponseCreateParamsNonStreaming(
        model="requested-model",
        input=[{"role": "user", "content": "optimize this kernel"}],
        temperature=0.7,
        max_output_tokens=400,
        parallel_tool_calls=False,
        metadata={"task_dir": str(task), "task_name": "kernel-1", "instruction": "optimize"},
    )
    return agent, body


@pytest.mark.parametrize("global_timeout,expected_timeout", [(None, 45), (90, 90)])
def test_setup_applies_task_timeouts_and_global_override(protocol_agent, global_timeout, expected_timeout):
    agent, body = protocol_agent
    agent.config.global_agent_timeout = global_timeout
    original = body.model_dump()

    params = agent._setup_params(body, "rollout-1")

    assert params.kb_agent_timeout == expected_timeout
    assert params.kb_eval_timeout == 30
    assert params.kb_sandbox_ttl == expected_timeout + 30 + 600
    assert params.problem_info["workdir"] == "/workspace"
    assert params.model_server_url == "http://model.invalid/ng-rollout/rollout-1"
    assert body.model_dump() == original
    assert (params.persistent_dir / "instruction.txt").read_text() == "optimize"
    assert (params.persistent_dir / "agent_runner.py").is_file()


def test_explicit_model_endpoint_does_not_receive_gym_rollout_path(protocol_agent):
    agent, body = protocol_agent
    agent.config.sandbox_model_base_url = "https://external.invalid"
    agent._model_server_url = agent.config.sandbox_model_base_url
    body.metadata["agent_timeout_sec"] = "70"
    body.metadata["verifier_timeout_sec"] = "50"

    params = agent._setup_params(body, "rollout-1")

    assert params.model_server_url == "https://external.invalid"
    assert params.kb_agent_timeout == 70
    assert params.kb_eval_timeout == 50


async def test_run_preserves_trajectory_and_request_and_redacts_configuration(protocol_agent, monkeypatch):
    agent, body = protocol_agent
    original = body.model_dump()
    saved = NeMoGymResponse(
        id="agent-response",
        created_at=0,
        model="served-model",
        object="response",
        status="incomplete",
        incomplete_details={"reason": "max_output_tokens"},
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[{"type": "function", "name": "bash", "parameters": {"type": "object"}, "strict": False}],
        output=[
            {"type": "reasoning", "id": "reason-1", "summary": [{"type": "summary_text", "text": "Use Triton"}]},
            {"type": "function_call", "id": "call-1", "call_id": "call-1", "name": "bash", "arguments": "{}"},
            {"type": "function_call_output", "id": "output-1", "call_id": "call-1", "output": "written"},
        ],
        usage={
            "input_tokens": 11,
            "output_tokens": 7,
            "total_tokens": 18,
            "input_tokens_details": {"cached_tokens": 3},
            "output_tokens_details": {"reasoning_tokens": 2},
        },
    )
    captured = []

    async def run_kernel(self, params):
        captured.append(params)
        params.metrics_fpath.write_text(
            KernelBenchMetrics(resolved=True, correctness=True, speedup=2, evaluation_completed=True).model_dump_json()
        )
        (params.persistent_dir / "response.json").write_text(saved.model_dump_json())
        return True

    monkeypatch.setattr(KernelGymAgent, "_run_kernel", run_kernel)
    result = await agent.run(KernelGymRunRequest(responses_create_params=body))

    assert result.reward == 1 and result.evaluation_completed and not result.mask_sample
    assert result.response.output == saved.output
    assert result.response.usage == saved.usage
    assert result.response.status == "incomplete"
    assert result.response.incomplete_details == saved.incomplete_details
    assert result.response.tools == saved.tools
    assert result.response.metadata is None
    assert result.responses_create_params.input == body.input
    assert result.responses_create_params.model == "served-model"
    assert result.responses_create_params.temperature == 0.7
    assert result.responses_create_params.max_output_tokens == 400
    assert not result.responses_create_params.parallel_tool_calls
    assert body.model_dump() == original
    params = captured[0]
    assert params.agent_kwargs["api_key"] == "agent-secret"
    persisted = json.loads((params.persistent_dir / "params.json").read_text())
    for config in (persisted, result.instance_config):
        assert config["agent_kwargs"]["api_key"] == "***"
        assert config["agent_kwargs"]["nested"] == [{"access-token": "***", "max_tokens": 100}]
        assert config["sandbox_provider"]["apiKey"] == "***"
        assert not config.get("agent_command_str")


async def test_run_reports_incomplete_evaluation_without_agent_response(protocol_agent, monkeypatch):
    agent, body = protocol_agent

    async def run_kernel(self, params):
        params.metrics_fpath.write_text(
            KernelBenchMetrics(resolved=False, sandbox_failed=True, evaluation_completed=False).model_dump_json()
        )
        return False

    monkeypatch.setattr(KernelGymAgent, "_run_kernel", run_kernel)
    result = await agent.run(KernelGymRunRequest(responses_create_params=body))

    assert result.reward == 0 and result.mask_sample and not result.evaluation_completed
    assert result.sandbox_failed
    assert result.response.output == []
    assert result.response.usage is None
    assert result.responses_create_params.input == body.input


async def test_response_failure_keeps_traceback_and_propagates_error(protocol_agent, monkeypatch):
    agent, body = protocol_agent

    async def run_kernel(self, params):
        raise OSError("cannot save submitted solution")

    monkeypatch.setattr(KernelGymAgent, "_run_kernel", run_kernel)
    with pytest.raises(OSError, match="cannot save submitted solution"):
        await agent.responses(body)
    (traceback,) = agent._base_results_dir.glob("*/traceback.err")
    assert "OSError: cannot save submitted solution" in traceback.read_text()
