# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the real runner process and HTTP boundary without requiring a remote model."""

import asyncio
import json
import os
import signal
import sys
from pathlib import Path

import pytest
from aiohttp import web

from responses_api_agents.langgraph_agent import sandbox_runner


ROUTE = "/ng-rollout/episode/training-token-capture/v1/responses"


@pytest.fixture
def runtime_dependencies() -> None:
    pytest.importorskip("langgraph", reason="Install runtime-requirements.txt to test the sandbox graph runtime")
    pytest.importorskip("langchain_core", reason="Install runtime-requirements.txt to test the sandbox graph runtime")


def _response(text: str) -> dict:
    return {
        "id": "response",
        "created_at": 1770000000.0,
        "model": "test",
        "object": "response",
        "output": [
            {
                "id": "message",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


async def _start_server(handler):
    app = web.Application()
    app.router.add_post(ROUTE, handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    return runner, f"http://127.0.0.1:{port}{ROUTE}"


async def _start_worker(tmp_path: Path, model_url: str, workflow: str = "reflection"):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text(
        json.dumps(
            {
                "workflow": workflow,
                "config": {"num_parallel_paths": 2},
                "request": {"input": "Solve this task.", "temperature": 0.0},
                "model_url": model_url,
            }
        )
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "responses_api_agents.langgraph_agent.sandbox_runner",
        str(input_path),
        str(output_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    return process, output_path


async def _stop_worker(process) -> None:
    if process.returncode is None:
        process.kill()
    await process.communicate()


@pytest.mark.parametrize(
    "workflow,call_count",
    [
        ("reflection", 1),
        ("orchestrator", 3),
        ("parallel_thinking", 3),
        ("rewoo", 2),
    ],
)
async def test_runner_executes_whole_graph_in_child_and_preserves_route(
    tmp_path, runtime_dependencies, workflow, call_count
) -> None:
    calls = []

    async def model(request):
        calls.append((request.path, dict(request.headers), await request.json()))
        return web.json_response(_response("<answer>42</answer>"))

    server, url = await _start_server(model)
    process, output_path = await _start_worker(tmp_path, url, workflow)
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
        assert process.returncode == 0, (
            stdout.decode(errors="replace"),
            stderr.decode(errors="replace"),
            output_path.read_text() if output_path.exists() else "missing output",
        )
        output = json.loads(output_path.read_text())
        assert output["runtime"]["pid"] == process.pid != os.getpid()
        assert output["runtime"]["hostname"]
        assert output["runtime"]["python"]
        assert len(output["response"]["output"]) == call_count
        assert len(calls) == call_count
        assert all(path == ROUTE for path, _, _ in calls)
        assert all("Cookie" not in headers and "Authorization" not in headers for _, headers, _ in calls)
        assert all(body["temperature"] == 0.0 for _, _, body in calls)
        assert not list(tmp_path.glob(".*.tmp"))
    finally:
        await _stop_worker(process)
        await server.cleanup()


async def test_runner_records_failure_without_model_content_or_credentials(tmp_path, runtime_dependencies) -> None:
    secret = "model-secret-that-must-not-leave-the-runner"

    async def model(request):
        return web.json_response({"invalid": secret})

    server, url = await _start_server(model)
    process, output_path = await _start_worker(tmp_path, url)
    try:
        await asyncio.wait_for(process.communicate(), timeout=30)
        assert process.returncode == 1
        output_text = output_path.read_text()
        output = json.loads(output_text)
        assert output["error_type"] == "ValidationError"
        assert output["error"] == "LangGraph workflow failed."
        assert output["runtime"]["pid"] == process.pid
        assert secret not in output_text
        assert url not in output_text
    finally:
        await _stop_worker(process)
        await server.cleanup()


async def test_runner_sigterm_cancels_inflight_graph_and_writes_failure(tmp_path, runtime_dependencies) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def model(request):
        started.set()
        await release.wait()
        return web.json_response(_response("too late"))

    server, url = await _start_server(model)
    process, output_path = await _start_worker(tmp_path, url)
    try:
        await asyncio.wait_for(started.wait(), timeout=30)
        process.send_signal(signal.SIGTERM)
        await asyncio.wait_for(process.communicate(), timeout=10)
        assert process.returncode == 143
        output = json.loads(output_path.read_text())
        assert output["error_type"] == "CancelledError"
        assert output["runtime"]["pid"] == process.pid
        assert "response" not in output
    finally:
        release.set()
        await _stop_worker(process)
        await server.cleanup()


def test_invalid_input_leaves_atomic_failure_receipt(tmp_path) -> None:
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text("not JSON")
    assert sandbox_runner._run_worker(input_path, output_path) == 1
    result = json.loads(output_path.read_text())
    assert result["error_type"] == "JSONDecodeError"
    assert not list(tmp_path.glob(".*.tmp"))


async def test_runner_bounds_cancellation_when_a_task_refuses_to_unwind(tmp_path) -> None:
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text("{}")
    script = """
import asyncio
import sys
from pathlib import Path
from responses_api_agents.langgraph_agent import sandbox_runner
async def stubborn(payload):
    print("ready", flush=True)
    while True:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            pass
sandbox_runner._CANCELLATION_GRACE_SECONDS = 0.2
sandbox_runner._run = stubborn
raise SystemExit(sandbox_runner._run_worker(Path(sys.argv[1]), Path(sys.argv[2])))
"""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        script,
        str(input_path),
        str(output_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        assert await asyncio.wait_for(process.stdout.readline(), timeout=10) == b"ready\n"
        process.send_signal(signal.SIGTERM)
        await asyncio.wait_for(process.communicate(), timeout=5)
        assert process.returncode == 143
        output = json.loads(output_path.read_text())
        assert output["error_type"] == "CancelledError"
        assert output["runtime"]["pid"] == process.pid
    finally:
        await _stop_worker(process)
