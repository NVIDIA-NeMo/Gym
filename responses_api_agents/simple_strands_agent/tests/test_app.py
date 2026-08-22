# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponseFunctionToolCall,
)
from responses_api_agents.simple_strands_agent.app import (
    SimpleStrandsAgent,
    _extract_instruction,
    trajectory_to_output_items,
)
from responses_api_agents.simple_strands_agent.setup_ssa import (
    SSA_REVISION,
    _package_dir,
    _prepare_source,
    _workspace_dir,
)


def test_extract_instruction() -> None:
    instruction, system = _extract_instruction(
        [
            NeMoGymEasyInputMessage(role="system", content="Be precise"),
            NeMoGymEasyInputMessage(role="developer", content="Use tools"),
            NeMoGymEasyInputMessage(role="user", content="Old task"),
            NeMoGymEasyInputMessage(role="user", content="Solve this"),
        ]
    )
    assert instruction == "Solve this"
    assert system == "Be precise\n\nUse tools"


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


def test_package_dir_accepts_monorepo_or_package(tmp_path: Path) -> None:
    package = tmp_path / "simple-strands-agent"
    (package / "src" / "ssa").mkdir(parents=True)
    (package / "pyproject.toml").touch()
    assert _package_dir(tmp_path) == package.resolve()
    assert _package_dir(package) == package.resolve()


def test_package_dir_rejects_missing_package(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="package not found"):
        _package_dir(tmp_path)


def test_workspace_dir_finds_lockfile(tmp_path: Path) -> None:
    package = tmp_path / "simple-strands-agent"
    package.mkdir()
    (tmp_path / "uv.lock").touch()
    assert _workspace_dir(package) == tmp_path


@pytest.mark.asyncio
async def test_run_ssa_masks_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class Process:
        pid = 123
        returncode = 1

        async def communicate(self):
            return b"", b"failed"

    async def create_subprocess_exec(*args, **kwargs):
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess_exec)
    agent = SimpleNamespace(_ssa_python=tmp_path / "python", config=SimpleNamespace(timeout=1))
    result = await SimpleStrandsAgent._run_ssa(agent, {"work_dir": str(tmp_path)})
    assert result == {}


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
        _ssa_python=tmp_path / "python",
        config=SimpleNamespace(timeout=60),
        _terminate_process=SimpleStrandsAgent._terminate_process,
    )
    task = asyncio.create_task(SimpleStrandsAgent._run_ssa(agent, {"work_dir": str(tmp_path)}))
    await process.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert process.returncode == -signal.SIGTERM


def test_prepare_source_replaces_wrong_revision(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "stale").touch()
    calls = []

    def run(args, cwd=None, **kwargs):
        calls.append(args)
        if args[1:3] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout="wrong\n")
        if args[1] == "clone":
            clone = Path(args[-1])
            clone.mkdir()
            (clone / "fresh").touch()
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr("responses_api_agents.simple_strands_agent.setup_ssa.subprocess.run", run)
    _prepare_source(source)
    assert (source / "fresh").is_file()
    assert not (source / "stale").exists()
    assert ["git", "checkout", SSA_REVISION] in calls
