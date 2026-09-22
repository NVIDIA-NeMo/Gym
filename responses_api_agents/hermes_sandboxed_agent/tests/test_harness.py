# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import SandboxExecResult
from responses_api_agents.hermes_sandboxed_agent.harness import HarnessContext, HermesConfig, HermesHarness


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_truns": 2},
        {"quiet_mod": False},
        {"use_streaming": True},
        {"toolsets": ["browser"]},
        {"runtime": {"compression": {"enabled": True}}},
        {"tool_delay": -1},
        {"toolsets": []},
    ],
)
def test_unsupported_configuration_is_rejected(overrides):
    """Unsupported settings fail validation instead of silently changing the run."""
    with pytest.raises(ValidationError):
        HermesConfig.model_validate({"name": "hermes", **overrides})


async def test_configured_prompt_and_toolsets_reach_native_worker(tmp_path):
    """Gym configuration changes the native prompt and offered tools for this episode."""
    config = HermesConfig.model_validate(
        {
            "name": "hermes",
            "max_turns": 2,
            "toolsets": ["file"],
            "ephemeral_system_prompt": "Use the configured project conventions.",
            "tool_delay": 0,
        }
    )
    requests = []

    async def query(params):
        requests.append(params)
        return NeMoGymResponse(
            id="configured",
            created_at=0,
            model="model",
            object="response",
            output=[
                {
                    "type": "message",
                    "id": "done",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "Done", "annotations": []}],
                }
            ],
            tool_choice="auto",
            tools=[],
            parallel_tool_calls=False,
        )

    harness = HermesHarness(
        sandbox=SimpleNamespace(),
        context=HarnessContext(session_id="configured", instruction="Inspect the task"),
        config=config,
        params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        query=query,
        model_name="model",
        directory=tmp_path,
    )
    _, outcome, extra = await harness.execute(20)
    assert outcome.reason == "completed", outcome
    assert {tool["name"] for tool in requests[0]["tools"]} == {"read_file", "write_file", "patch", "search_files"}
    assert "Use the configured project conventions." in json.dumps(requests[0]["input"])
    saved = json.loads((tmp_path / "harness-config.json").read_text())
    assert saved["config"] == config.model_dump(mode="json")
    assert extra["hermes_config"] == saved["config"]


@pytest.mark.parametrize("stop", ["completed", "timeout", "cancelled", "turn_limit", "partial", "tool_error", "retry"])
async def test_real_hermes_loop_preserves_sandbox_results_and_stops_before_return(tmp_path, monkeypatch, stop):
    """Hermes retains native turns and paired tool results, including when model I/O is interrupted."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    requests, commands = [], []
    pending = asyncio.Event()
    closed = asyncio.Event()

    async def query(params):
        if stop == "retry" and not pending.is_set():
            pending.set()
            raise RuntimeError("Transient model transport failure")
        requests.append(params)
        if len(requests) == 2 and stop in {"timeout", "cancelled"}:
            pending.set()
            try:
                await asyncio.Future()
            finally:
                # Yield during cleanup to detect returning before asynchronous I/O has stopped.
                await asyncio.sleep(0.05)
                closed.set()
        items = (
            [
                {"type": "reasoning", "id": "reason", "summary": [{"type": "summary_text", "text": "inspect"}]},
                {
                    "type": "function_call",
                    "call_id": "command",
                    "name": "terminal",
                    "arguments": json.dumps({"command": "printf sandbox-result", "timeout": 2, "workdir": "/task"}),
                },
            ]
            if len(requests) == 1
            else [
                {
                    "type": "message",
                    "id": "final",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "Done", "annotations": []}],
                }
            ]
        )
        return NeMoGymResponse(
            id=f"response-{len(requests)}",
            created_at=0,
            model="model",
            object="response",
            status="incomplete" if stop == "partial" and len(requests) == 2 else "completed",
            output=items,
            tool_choice="auto",
            tools=[],
            parallel_tool_calls=False,
            usage={
                "input_tokens": 10,
                "output_tokens": 2,
                "total_tokens": 12,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        )

    async def execute(command, **kwargs):
        commands.append((command, kwargs))
        if stop == "tool_error" and len(commands) == 2:
            raise RuntimeError("Sandbox disconnected")
        return SandboxExecResult("sandbox-result", "", 0)

    harness = HermesHarness(
        observability_enabled=True,
        sandbox=SimpleNamespace(exec=execute),
        context=HarnessContext(session_id="test", instruction="Inspect the sandbox", workdir="/task"),
        config=HermesConfig(name="hermes", max_turns=1 if stop == "turn_limit" else 3),
        params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        query=query,
        model_name="model",
        directory=tmp_path,
    )
    await harness.setup()
    # Native Hermes waits one second after tool execution before requesting the next model turn.
    task = asyncio.create_task(harness.execute(3 if stop == "timeout" else 20))
    if stop == "cancelled":
        await asyncio.wait_for(pending.wait(), 10)
        task.cancel()
    response, outcome, extra = await asyncio.wait_for(task, 15)
    assert outcome.reason == {
        "retry": "completed",
        "turn_limit": "nonzero_exit",
        "partial": "nonzero_exit",
        "tool_error": "infrastructure_error",
    }.get(stop, stop)
    assert commands[-1][1]["cwd"] == "/task"
    assert commands[-1][1]["timeout_s"] == 2
    from tools.terminal_tool import TERMINAL_SCHEMA

    terminal = next(tool for tool in requests[0]["tools"] if tool["name"] == "terminal")
    assert terminal == {"type": "function", **TERMINAL_SCHEMA, "strict": False}
    assert {tool["name"] for tool in requests[0]["tools"]} == {
        "terminal",
        "process",
        "read_file",
        "write_file",
        "patch",
        "search_files",
    }
    assert commands[-1][0].endswith("printf sandbox-result'")
    call = next(item for item in response.output if item.type == "function_call")
    result = next(item for item in response.output if item.type == "function_call_output")
    assert call.call_id == result.call_id == "command"
    if stop == "tool_error":
        assert json.loads(result.output)["output"] == "RuntimeError: Sandbox disconnected"
        assert json.loads(result.output)["exit_code"] == 1
    else:
        assert json.loads(result.output)["output"] == "sandbox-result"
    assert response.output[0].type == "reasoning"
    assert response.usage.total_tokens == (24 if stop in {"completed", "partial", "turn_limit", "retry"} else 12)
    native = json.loads((tmp_path / "trajectory.json").read_text())["messages"]
    assert any(message["role"] == "tool" and message["tool_call_id"] == "command" for message in native)
    if stop in {"timeout", "cancelled"}:
        assert closed.is_set()

    trajectory = extra["ng_trajectory"]
    invocation = trajectory["invocations"][0]
    expected_calls = 2 if stop in {"completed", "partial", "turn_limit", "retry"} else 1
    assert len(invocation["model_calls"]) == len(trajectory["turns"]) == expected_calls
    assert (
        next(item for item in invocation["conversation"] if item["type"] == "function_call_output")["call_id"]
        == "command"
    )
    tool = trajectory["tool_calls"][0]
    assert tool["tool_call_id"] == "command"
    assert tool["output"] == result.output
    assert tool["status"] == ("failed" if stop == "tool_error" else "completed")


async def test_native_file_and_process_tools_use_supplied_backend(tmp_path):
    """Native file and background-process handlers execute through the supplied backend."""
    import shlex
    from uuid import uuid4

    session_id = "hermes-test-" + uuid4().hex
    observations = []
    calls = []

    async def execute(command, **kwargs):
        if command == "command -v setsid":
            return SandboxExecResult("setsid", "", 0)
        # This test backend runs controlled fixture commands locally, without Linux setsid.
        argv = shlex.split(command)
        process = await asyncio.create_subprocess_exec(
            *argv[2:], cwd=kwargs["cwd"], stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), kwargs["timeout_s"])
        return SandboxExecResult(stdout.decode(), stderr.decode(), process.returncode)

    async def query(params):
        outputs = [item for item in params["input"] if item.get("type") == "function_call_output"]
        if outputs:
            observations.append(json.loads(outputs[-1]["output"]))
        sequence = [
            ("write_file", {"path": str(tmp_path / "native.txt"), "content": "native file contents"}),
            ("read_file", {"path": str(tmp_path / "native.txt")}),
            (
                "terminal",
                {"command": "printf background > background.txt", "background": True, "workdir": str(tmp_path)},
            ),
        ]
        if len(calls) < len(sequence):
            name, args = sequence[len(calls)]
        elif len(calls) == len(sequence):
            name, args = "process", {"action": "wait", "session_id": observations[-1]["session_id"], "timeout": 10}
        else:
            return NeMoGymResponse(
                id="finished",
                created_at=0,
                model="model",
                object="response",
                output=[
                    {
                        "type": "message",
                        "id": "final",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Done", "annotations": []}],
                    }
                ],
                tool_choice="auto",
                tools=[],
                parallel_tool_calls=False,
            )
        calls.append(name)
        return NeMoGymResponse(
            id=f"response-{len(calls)}",
            created_at=0,
            model="model",
            object="response",
            output=[
                {
                    "type": "function_call",
                    "call_id": f"call-{len(calls)}",
                    "name": name,
                    "arguments": json.dumps(args),
                }
            ],
            tool_choice="auto",
            tools=[],
            parallel_tool_calls=False,
        )

    harness = HermesHarness(
        observability_enabled=True,
        sandbox=SimpleNamespace(exec=execute),
        context=HarnessContext(session_id=session_id, instruction="Use the supplied tools", workdir=str(tmp_path)),
        config=HermesConfig(name="hermes", max_turns=6),
        params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        query=query,
        model_name="model",
        directory=tmp_path,
    )
    try:
        await harness.setup()
        response, outcome, extra = await harness.execute(40)
        assert outcome.reason == "completed", outcome
        assert calls == ["write_file", "read_file", "terminal", "process"]
        assert (tmp_path / "native.txt").read_text() == "native file contents"
        assert (tmp_path / "background.txt").read_text() == "background"
        assert len([item for item in response.output if item.type == "function_call_output"]) == 4
        assert len(extra["ng_trajectory"]["tool_calls"]) == 4
    finally:
        from pathlib import Path

        Path(f"/tmp/{session_id}.pids").unlink(missing_ok=True)


@pytest.mark.parametrize("tool_name", ["terminal", "read_file"])
async def test_cancellation_retains_completed_tool_in_batch(tmp_path, tool_name):
    blocked = asyncio.Event()
    closed = asyncio.Event()

    async def execute(command, **kwargs):
        if "printf slow" in command or "/slow.txt" in command:
            blocked.set()
            try:
                await asyncio.Future()
            finally:
                await asyncio.sleep(0.05)
                closed.set()
        return SandboxExecResult("5" if "wc -" in command else "fast", "", 0)

    async def query(params):
        return NeMoGymResponse(
            id="parallel",
            created_at=0,
            model="model",
            object="response",
            output=[
                {
                    "type": "function_call",
                    "call_id": name,
                    "name": tool_name,
                    "arguments": json.dumps(
                        {"command": f"printf {name}", "timeout": 15}
                        if tool_name == "terminal"
                        else {"path": f"/{name}.txt"}
                    ),
                }
                for name in ("fast", "slow")
            ],
            tool_choice="auto",
            tools=[],
            parallel_tool_calls=True,
        )

    harness = HermesHarness(
        observability_enabled=True,
        sandbox=SimpleNamespace(exec=execute),
        context=HarnessContext(session_id="parallel", instruction="Read both files", workdir="/task"),
        config=HermesConfig(name="hermes", max_turns=3),
        params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        query=query,
        model_name="model",
        directory=tmp_path,
    )
    await harness.setup()
    task = asyncio.create_task(harness.execute(20))
    try:
        async with asyncio.timeout(10):
            await blocked.wait()
            while not any(
                message.get("tool_call_id") == "fast"
                for message in json.loads((tmp_path / "trajectory.json").read_text())["messages"]
            ):
                await asyncio.sleep(0.01)
    finally:
        task.cancel()
        response, outcome, extra = await task
    assert outcome.reason == "cancelled"
    assert closed.is_set()
    results = [item for item in response.output if item.type == "function_call_output"]
    assert [item.call_id for item in results] == ["fast"]
    assert "fast" in results[0].output
    assert extra["ng_trajectory"]["tool_calls"][0]["output"] == results[0].output


def test_pinned_hermes_version_has_one_source_of_truth():
    """The pinned Hermes revision must agree across code and both requirement files, or this test fails."""
    from pathlib import Path

    from responses_api_agents.hermes_sandboxed_agent.harness import HERMES_REVISION

    package_dir = Path(__file__).resolve().parents[1]
    repo_root = package_dir.parents[1]
    for requirements_path in (
        package_dir / "requirements.txt",
        repo_root / "resources_servers" / "terminal_bench_4" / "requirements.txt",
    ):
        text = requirements_path.read_text()
        assert f"hermes-agent @ git+https://github.com/cmunley1/hermes-agent@{HERMES_REVISION}" in text, (
            f"{requirements_path} does not pin the same Hermes revision as HERMES_REVISION={HERMES_REVISION!r}"
        )
