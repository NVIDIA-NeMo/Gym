# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise checkpoint recovery after a real worker is killed without finalizing."""

import asyncio
import json
import os
import shutil
import signal
import sys
from pathlib import Path

import pytest
from aiohttp import web

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming, NeMoGymResponseFunctionToolCall
from nemo_gym.sandbox.providers.base import SandboxExecResult
from responses_api_agents.nooa_agent.config import NOOAInvocationConfig
from responses_api_agents.nooa_agent.runner import NOOARunFailure, NOOARunRequest
from responses_api_agents.nooa_agent.sandbox_entrypoint import SandboxCheckpoint, write_checkpoint
from responses_api_agents.nooa_agent.sandbox_runner import SandboxNOOARunner
from responses_api_agents.nooa_agent.termination import NOOATerminationReason
from responses_api_agents.nooa_agent.tests.test_gym_llm import model_response
from responses_api_agents.nooa_agent.tests.test_sandbox_entrypoint import run_result


def worker_environment() -> dict[str, str]:
    environment = dict(os.environ)
    root = Path(__file__).resolve().parents[3]
    environment["PYTHONPATH"] = os.pathsep.join(
        dict.fromkeys([str(root), *(str(Path(path).resolve()) for path in sys.path if path)])
    )
    return environment


class LocalWorkerSandbox:
    """Use local files and a real child; supervisor cleanup has separate Linux tests."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.process: asyncio.subprocess.Process | None = None
        self.reaper: asyncio.Task[int] | None = None
        self.events: list[tuple[str, str]] = []

    async def upload(self, source: Path, destination: str) -> None:
        shutil.copyfile(source, destination)

    async def download(self, source: str, destination: Path) -> None:
        self.events.append(("download", Path(source).name))
        shutil.copyfile(source, destination)

    async def exec(
        self, command: str, *, cwd: str, timeout_s: float | None, preserve_background_services: bool = False
    ) -> SandboxExecResult:
        if "process_supervisor.py" in command:
            # Run the actual entrypoint from this checkout. The installed sandbox's
            # isolated interpreter and Linux descendant reaper are not under test.
            with (self.directory / "output.log").open("wb") as stdout:
                with (self.directory / "stderr.log").open("wb") as stderr:
                    self.process = await asyncio.create_subprocess_exec(
                        sys.executable,
                        "-m",
                        "responses_api_agents.nooa_agent.sandbox_entrypoint",
                        str(self.directory / "input.json"),
                        str(self.directory / "result.json"),
                        str(self.directory / "stop.request"),
                        str(self.directory / "completion.json"),
                        cwd=cwd,
                        env=worker_environment(),
                        stdout=stdout,
                        stderr=stderr,
                    )
            self.reaper = asyncio.create_task(self.reap())
            code = await asyncio.shield(self.reaper)
            return SandboxExecResult(stdout="", stderr="", return_code=code)
        if "rm -rf" in command:
            self.events.append(("remove", ""))
            shutil.rmtree(self.directory)
        else:
            assert command.startswith("mkdir -p "), command
            self.directory.mkdir(parents=True, exist_ok=True)
        return SandboxExecResult(stdout="", stderr="", return_code=0)

    async def reap(self) -> int:
        assert self.process is not None
        code = await self.process.wait()
        # Only confirm cleanup after the child has actually exited and been reaped.
        (self.directory / "cleanup.json").write_text(
            json.dumps({"cleanup_confirmed": True, "return_code": code, "timed_out": False, "error": None})
        )
        return code

    async def disconnect(self) -> None:
        self.events.append(("disconnect", ""))


async def checkpoint_when(sandbox: LocalWorkerSandbox, *, completed_tool: bool) -> SandboxCheckpoint:
    async with asyncio.timeout(45):
        while True:
            if sandbox.process is not None and sandbox.process.returncode is not None:
                pytest.fail((sandbox.directory / "stderr.log").read_text(errors="replace"))
            path = sandbox.directory / "checkpoint.json"
            if path.exists():
                checkpoint = SandboxCheckpoint.model_validate_json(path.read_text())
                has_model = checkpoint.response is not None and any(
                    item.type == "function_call" and item.call_id == "call-checkpoint"
                    for item in checkpoint.response.output
                )
                has_tool = any(
                    record.kind == "tool_call"
                    and record.tool_call_id == "call-checkpoint"
                    and record.status == "completed"
                    for record in checkpoint.observations.records
                )
                if has_model and has_tool is completed_tool:
                    return checkpoint
            await asyncio.sleep(0.02)


@pytest.mark.skipif(os.name != "posix", reason="Requires real SIGKILL process termination")
@pytest.mark.asyncio
async def test_sigkill_recovers_model_and_tool_checkpoints_before_cleanup(tmp_path: Path) -> None:
    release_tool = tmp_path / "release-tool"
    next_model_call = asyncio.Event()
    finish_http = asyncio.Event()
    calls = 0
    code = (
        "import asyncio\nfrom pathlib import Path\n"
        f"while not Path({str(release_tool)!r}).exists():\n    await asyncio.sleep(0.02)\n"
        "print('completed-tool-evidence')"
    )

    async def model(request: web.Request) -> web.Response:
        nonlocal calls
        await request.json()
        calls += 1
        if calls > 1:
            next_model_call.set()
            await finish_http.wait()
            return web.Response(status=503)
        return web.json_response(
            model_response(
                NeMoGymResponseFunctionToolCall(
                    id="fc-checkpoint",
                    call_id="call-checkpoint",
                    name="execute_python",
                    arguments=json.dumps({"code": code}),
                ),
                response_id="response-checkpoint",
            )
        )

    application = web.Application()
    application.router.add_post("/v1/responses", model)
    server = web.AppRunner(application)
    await server.setup()
    site = web.TCPSite(server, "127.0.0.1", 0)
    await site.start()
    port = server.addresses[0][1]
    sandbox = LocalWorkerSandbox(tmp_path / "session")
    runner = SandboxNOOARunner(
        sandbox=sandbox,
        workdir=str(tmp_path),
        python=sys.executable,
        invocation=NOOAInvocationConfig(
            agent_class="responses_api_agents.nooa_agent.task_agent:TaskAgent",
            invocation_adapter="responses_api_agents.nooa_agent.task_agent:invoke_solve_task",
        ),
        model_base_url=f"http://127.0.0.1:{port}",
        model_server_name="policy",
        max_policy_calls=3,
    )
    runner.session.session_dir = str(sandbox.directory)
    execution: asyncio.Task | None = None
    try:
        await runner.prepare()
        execution = asyncio.create_task(
            runner.run(
                NOOARunRequest(
                    responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="Execute the task"),
                    model_url_path="/v1/responses",
                )
            )
        )
        first = await checkpoint_when(sandbox, completed_tool=False)
        assert first.partial is True
        assert first.response is not None
        assert any(
            item.type == "function_call" and item.call_id == "call-checkpoint" for item in first.response.output
        )
        release_tool.touch()
        second = await checkpoint_when(sandbox, completed_tool=True)
        await asyncio.wait_for(next_model_call.wait(), timeout=15)
        assert second.partial is True
        assert second.trajectory is not None
        tool = next(item for item in second.trajectory.tool_calls if item.tool_call_id == "call-checkpoint")
        assert tool.status == "completed"
        assert "completed-tool-evidence" in json.dumps(tool.output)
        assert sandbox.process is not None
        sandbox.process.send_signal(signal.SIGKILL)
        with pytest.raises(NOOARunFailure, match="did not exit successfully") as caught:
            await asyncio.wait_for(execution, timeout=10)
        assert sandbox.process.returncode == -signal.SIGKILL
        assert not (sandbox.directory / "result.json").exists()
        result = caught.value.result
        assert result.termination_reason is NOOATerminationReason.INFRASTRUCTURE_ERROR
        assert result.episode.response.status == "failed"
        assert result.episode.response.error is not None
        assert {"partial_checkpoint", "sandbox_result_unavailable"} <= {
            gap.code for gap in result.episode.observations.gaps
        }
        assert result.trajectory == second.trajectory
        assert runner.artifact is None
        assert runner.checkpoint is not None
        assert runner.session.cleanup["cleanup_confirmed"]
        assert ("remove", "") not in sandbox.events
        assert ("disconnect", "") not in sandbox.events
        await runner.close()
        checkpoint_reads = [i for i, event in enumerate(sandbox.events) if event == ("download", "checkpoint.json")]
        assert checkpoint_reads
        assert max(checkpoint_reads) < sandbox.events.index(("remove", "")) < sandbox.events.index(("disconnect", ""))
        assert not sandbox.directory.exists()
    finally:
        if sandbox.process is not None and sandbox.process.returncode is None:
            sandbox.process.kill()
            await asyncio.wait_for(sandbox.process.wait(), timeout=10)
        if sandbox.reaper is not None:
            await asyncio.gather(sandbox.reaper, return_exceptions=True)
        if execution is not None:
            if not execution.done():
                execution.cancel()
            await asyncio.gather(execution, return_exceptions=True)
        finish_http.set()
        await server.cleanup()


@pytest.mark.skipif(os.name != "posix", reason="Requires real SIGKILL process termination")
@pytest.mark.asyncio
async def test_sigkill_during_atomic_write_preserves_published_checkpoint(tmp_path: Path) -> None:
    sandbox = LocalWorkerSandbox(tmp_path / "session")
    sandbox.directory.mkdir()
    path = sandbox.directory / "checkpoint.json"
    previous = run_result()
    previous.episode.response.id = "published-before-interruption"
    write_checkpoint(path, previous)
    published = path.read_bytes()
    marker = sandbox.directory / "temporary-flushed"
    script = """
import sys
import time
from pathlib import Path
from responses_api_agents.nooa_agent import sandbox_entrypoint as entrypoint

path, marker = map(Path, sys.argv[1:])
result = entrypoint.SandboxCheckpoint.model_validate_json(path.read_text()).run_result()
result.episode.response.id = "unpublished-after-interruption"
fsync = entrypoint.os.fsync

def pause_after_flush(fd):
    fsync(fd)
    marker.touch()
    while True:
        time.sleep(0.05)

entrypoint.os.fsync = pause_after_flush
entrypoint.write_checkpoint(path, result)
"""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        script,
        str(path),
        str(marker),
        cwd=tmp_path,
        env=worker_environment(),
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        async with asyncio.timeout(45):
            while not marker.exists():
                if process.returncode is not None:
                    assert process.stderr is not None
                    pytest.fail((await process.stderr.read()).decode(errors="replace"))
                await asyncio.sleep(0.02)
        process.send_signal(signal.SIGKILL)
        assert await asyncio.wait_for(process.wait(), timeout=10) == -signal.SIGKILL
        assert path.read_bytes() == published
        abandoned = list(sandbox.directory.glob(".checkpoint.json.*.tmp"))
        assert len(abandoned) == 1
        unpublished = SandboxCheckpoint.model_validate_json(abandoned[0].read_text())
        assert unpublished.response.id == "unpublished-after-interruption"
        runner = SandboxNOOARunner(
            sandbox=sandbox,
            workdir=str(tmp_path),
            python=sys.executable,
            invocation=NOOAInvocationConfig(
                agent_class="responses_api_agents.nooa_agent.task_agent:TaskAgent",
                invocation_adapter="responses_api_agents.nooa_agent.task_agent:invoke_solve_task",
            ),
            model_base_url="http://unused:8000",
            model_server_name="policy",
            max_policy_calls=3,
        )
        runner.session.session_dir = str(sandbox.directory)
        # The actual writer process has been reaped above.
        await runner._collect_artifacts()
        assert runner.artifact is None
        assert runner.checkpoint is not None
        assert runner.checkpoint.response.id == "published-before-interruption"
        assert runner.checkpoint.response.status == "failed"
        assert runner.checkpoint.termination_reason is NOOATerminationReason.INFRASTRUCTURE_ERROR
        assert all(
            name not in {item.name for item in abandoned} for action, name in sandbox.events if action == "download"
        )
        await runner.close()
        assert sandbox.events.index(("download", "checkpoint.json")) < sandbox.events.index(("remove", ""))
        assert sandbox.events[-1] == ("disconnect", "")
    finally:
        if process.returncode is None:
            process.kill()
            await asyncio.wait_for(process.wait(), timeout=10)
