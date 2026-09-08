# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import signal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponseFunctionToolCall,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.simple_strands_agent.app import (
    SimpleStrandsAgent,
    SimpleStrandsAgentConfig,
    SimpleStrandsAgentRunRequest,
    _extract_instruction,
    trajectory_to_output_items,
)


def test_extract_instruction() -> None:
    instruction, system = _extract_instruction(
        [
            NeMoGymEasyInputMessage(role="system", content="Be precise"),
            NeMoGymEasyInputMessage(role="developer", content="Use tools"),
            NeMoGymEasyInputMessage(role="user", content="Old task"),
            {"role": "assistant", "content": "Previous answer"},
            NeMoGymEasyInputMessage(role="user", content="Solve this"),
        ]
    )
    assert instruction == "User: Old task\n\nAssistant: Previous answer\n\nUser: Solve this"
    assert system == "Be precise\n\nUse tools"


def test_extract_single_turn_instruction_unchanged() -> None:
    instruction, system = _extract_instruction([NeMoGymEasyInputMessage(role="user", content="Solve this")])
    assert instruction == "Solve this"
    assert system is None


def test_trajectory_preserves_reasoning_and_tools() -> None:
    messages = [
        {"role": "user", "content": [{"text": "Solve"}]},
        {
            "role": "assistant",
            "content": [
                {"reasoningContent": {"reasoningText": {"text": "I should calculate."}}},
                {
                    "toolUse": {
                        "toolUseId": "call-1",
                        "name": "bash",
                        "input": {"command": "printf 42"},
                    }
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "call-1",
                        "status": "success",
                        "content": [{"text": "42"}],
                    }
                }
            ],
        },
        {"role": "assistant", "content": [{"text": "\\boxed{42}"}]},
    ]
    output = trajectory_to_output_items(messages)

    assert [item.type for item in output] == [
        "reasoning",
        "function_call",
        "function_call_output",
        "message",
    ]
    assert isinstance(output[1], NeMoGymResponseFunctionToolCall)
    assert output[2].output == "42"
    assert output[3].content[0].text == "\\boxed{42}"


@pytest.mark.asyncio
async def test_run_skips_verification() -> None:
    config = SimpleStrandsAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="simple_strands_agent",
        model_server=ModelServerRef(type="responses_api_models", name="policy"),
        resources_server=ResourcesServerRef(type="resources_servers", name="reasoning_gym"),
        skip_verification=True,
        skip_verification_reward=0.25,
    )
    client = MagicMock(spec=ServerClient)
    seed_response = AsyncMock(cookies={"session": "seeded"})
    agent_response = AsyncMock(cookies={"session": "agent"})
    agent_response.read.return_value = json.dumps(
        {
            "id": "response_id",
            "created_at": 1,
            "model": "model",
            "object": "response",
            "output": [],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }
    ).encode()
    client.post = AsyncMock(side_effect=[seed_response, agent_response])
    agent = SimpleStrandsAgent(config=config, server_client=client)

    result = await agent.run(
        SimpleNamespace(cookies={}),
        SimpleStrandsAgentRunRequest(responses_create_params={"input": []}),
    )

    assert result.reward == 0.25
    assert result.model_extra["verification_skipped"] is True
    assert [call.kwargs["url_path"] for call in client.post.call_args_list] == [
        "/seed_session",
        "/v1/responses",
    ]


@pytest.mark.asyncio
async def test_run_ssa_raises_on_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class Process:
        pid = 123
        returncode = 1

        async def communicate(self):
            return b"", b"failed"

    async def create_subprocess_exec(*args, **kwargs):
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)
    agent = SimpleNamespace(config=SimpleNamespace(timeout=1))
    with pytest.raises(RuntimeError, match="SSA failed: failed"):
        await SimpleStrandsAgent._run_ssa(agent, {"work_dir": str(tmp_path)})


@pytest.mark.asyncio
async def test_run_ssa_raises_on_missing_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class Process:
        pid = 123
        returncode = 0

        async def communicate(self):
            return b"", b""

    async def create_subprocess_exec(*args, **kwargs):
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)
    agent = SimpleNamespace(config=SimpleNamespace(timeout=1))
    with pytest.raises(RuntimeError, match="SSA did not produce a result"):
        await SimpleStrandsAgent._run_ssa(agent, {"work_dir": str(tmp_path)})


@pytest.mark.asyncio
async def test_run_ssa_terminates_on_cancellation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class Process:
        pid = 123
        returncode = None

        def __init__(self):
            self.started = asyncio.Event()
            self.stopped = asyncio.Event()

        async def communicate(self):
            self.started.set()
            await self.stopped.wait()
            return b"", b""

    process = Process()

    async def create_subprocess_exec(*args, **kwargs):
        return process

    def killpg(pid, sig):
        process.returncode = -sig
        process.stopped.set()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)
    monkeypatch.setattr("responses_api_agents.simple_strands_agent.app.os.killpg", killpg)
    agent = SimpleNamespace(
        config=SimpleNamespace(timeout=60),
        _terminate_process=SimpleStrandsAgent._terminate_process,
    )
    task = asyncio.create_task(SimpleStrandsAgent._run_ssa(agent, {"work_dir": str(tmp_path)}))
    await process.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert process.returncode == -signal.SIGTERM
