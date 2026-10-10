# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from responses_api_agents.nooa_agent.runner import NOOARunFailure
from responses_api_agents.nooa_agent.sandbox_entrypoint import RunnerError, SandboxInput, SandboxResult
from responses_api_agents.nooa_agent.sandbox_runner import SandboxNOOARunner
from responses_api_agents.nooa_agent.tests.test_sandbox_entrypoint import payload, run_result


class MemorySandbox:
    def __init__(self) -> None:
        self.files: dict[str, str] = {}
        self.exec = AsyncMock(return_value=SimpleNamespace(return_code=0))
        self.disconnect = AsyncMock()
        self.stop = AsyncMock()

    async def upload(self, source: Path, destination: str) -> None:
        self.files[destination] = source.read_text()

    async def download(self, source: str, destination: Path) -> None:
        if source not in self.files:
            raise FileNotFoundError(source)
        destination.write_text(self.files[source])


def runner() -> tuple[SandboxNOOARunner, MemorySandbox]:
    p = payload()
    sandbox = MemorySandbox()
    return SandboxNOOARunner(
        sandbox=sandbox,
        workdir="/app",
        python="/opt/nooa env/bin/python",
        invocation=p.invocation,
        model_base_url=str(p.model_base_url),
        model_server_name=p.model_server_name,
        max_policy_calls=3,
        context_window=p.context_window,
    ), sandbox


def artifact(error: RunnerError | None = None) -> SandboxResult:
    result = run_result()
    return SandboxResult(
        response=result.episode.response,
        observations=result.episode.observations,
        model_cookies=result.model_cookies,
        resource_cookies=result.resource_cookies,
        error=error,
    )


def complete_files(r: SandboxNOOARunner, s: MemorySandbox, error: RunnerError | None = None) -> None:
    s.files[r.directory + "/completion.json"] = json.dumps({"task_completed": True})
    s.files[r.directory + "/cleanup.json"] = json.dumps({"cleanup_confirmed": True})
    s.files[r.directory + "/result.json"] = artifact(error).model_dump_json()


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [None, 3])
async def test_launch_quotes_paths_runs_in_task_and_preserves_cookies(limit: int | None) -> None:
    r, s = runner()
    r.max_policy_calls = limit
    await r.prepare()
    assert r.directory.startswith("/tmp/nemo-gym-nooa/")
    assert not s.files  # The isolated runtime already contains the NOOA supervisor module.
    complete_files(r, s)
    request = payload().request
    result = await r.run(request)
    command = s.exec.await_args_list[1].args[0]
    assert "'/opt/nooa env/bin/python'" in command
    assert "-I -m responses_api_agents.nooa_agent.sandbox_supervisor" in command
    assert "launch.claim" in command
    assert "stdout.log" in command and "stderr.log" in command
    assert s.exec.await_args_list[1].kwargs == {"cwd": "/app", "timeout_s": 30, "preserve_background_services": True}
    launch = SandboxInput.model_validate_json(s.files[r.directory + "/input.json"])
    assert launch.max_policy_calls == limit
    assert launch.request.rollout_id == request.rollout_id
    assert launch.request.model_url_path == request.model_url_path
    assert launch.context_window == 262144
    assert request.resource_cookies == result.resource_cookies == {"resource": "new"}
    assert request.model_cookies == {"model": "new"}
    assert not r.stopped
    assert "--timeout" not in command


@pytest.mark.asyncio
async def test_close_requires_confirmed_stop_and_keeps_handle_for_retry() -> None:
    r, s = runner()
    r.launched = True
    s.files[r.directory + "/cleanup.json"] = '{"cleanup_confirmed":false}'
    with pytest.raises(RuntimeError, match="unconfirmed"):
        await r.close()
    assert not r.stopped
    s.disconnect.assert_not_awaited()
    s.stop.assert_not_awaited()
    complete_files(r, s)
    await r.close()
    assert r.stopped
    s.disconnect.assert_awaited_once()
    s.stop.assert_not_awaited()
    assert "rm -rf" in s.exec.await_args.args[0]
    assert "/app" not in s.exec.await_args.args[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("contents", [None, "invalid-json", '{"response":{}}'])
async def test_missing_or_malformed_result_records_gap_not_success(contents: str | None) -> None:
    r, s = runner()
    s.files[r.directory + "/cleanup.json"] = '{"cleanup_confirmed":true}'
    if contents is not None:
        s.files[r.directory + "/result.json"] = contents
    with pytest.raises(RuntimeError, match="missing or malformed"):
        await r.run(payload().request)
    assert r.observations.gaps[0].code == "sandbox_result_unavailable"
    await r.close()
    s.disconnect.assert_awaited_once()
    s.stop.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,exception", [("transient", ConnectionError), ("fatal", RuntimeError)])
async def test_child_error_preserves_evidence_and_failure_class(kind: str, exception: type[Exception]) -> None:
    r, s = runner()
    complete_files(r, s, RunnerError(kind=kind, message="dependency failure"))
    with pytest.raises(NOOARunFailure) as caught:
        await r.run(payload().request)
    assert isinstance(caught.value.__cause__, exception)
    assert caught.value.result.episode.observations.gaps[0].code == "test"
    assert r.stopped


@pytest.mark.asyncio
async def test_stop_with_receipt_does_not_signal_reused_pid() -> None:
    r, s = runner()
    r.launched = True
    complete_files(r, s)
    await r.stop()
    await r.stop()
    s.exec.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("completion", [None, [], {}, {"task_completed": False}, {"task_completed": "true"}])
async def test_result_alone_cannot_confirm_task_completion(completion) -> None:
    r, s = runner()
    complete_files(r, s)
    if completion is None:
        del s.files[r.directory + "/completion.json"]
    else:
        s.files[r.directory + "/completion.json"] = json.dumps(completion)
    with pytest.raises(RuntimeError, match="completion is unconfirmed|did not complete"):
        await r.run(payload().request)
    assert r.stopped


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["launch", "poll", "cancel"])
async def test_interrupted_run_stops_immediately_and_preserves_terminal_error(stage: str) -> None:
    r, s = runner()
    terminal = asyncio.CancelledError() if stage == "cancel" else ConnectionError("provider disconnected")
    r.stop = AsyncMock(side_effect=RuntimeError("cleanup also failed"))
    if stage == "launch":
        s.exec.side_effect = terminal
    else:
        s.exec.side_effect = [SimpleNamespace(return_code=0), terminal]
    with pytest.raises(type(terminal)) as caught:
        await r.run(payload().request)
    assert caught.value is terminal
    r.stop.assert_awaited_once()
    s.disconnect.assert_not_awaited()


@pytest.mark.asyncio
async def test_nonobject_cleanup_receipt_still_attempts_stop() -> None:
    r, s = runner()
    r.launched = True
    s.files[r.directory + "/cleanup.json"] = "[]"

    async def execute(command: str, **kwargs) -> SimpleNamespace:
        assert "runner.stop" in command
        assert kwargs == {"cwd": "/", "timeout_s": 25}
        s.files[r.directory + "/cleanup.json"] = '{"cleanup_confirmed":true}'
        return SimpleNamespace(return_code=0)

    s.exec.side_effect = execute
    await r.stop()
    assert r.stopped
    s.exec.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancelling_active_worker_waits_for_stop_receipt_before_disconnect() -> None:
    r, s = runner()
    polling = asyncio.Event()

    async def execute(command: str, **kwargs) -> SimpleNamespace:
        if command.startswith("test -f"):
            polling.set()
            await asyncio.Event().wait()
        elif command.startswith("touch "):
            assert kwargs == {"cwd": "/", "timeout_s": 25}
            s.files[r.directory + "/cleanup.json"] = '{"cleanup_confirmed":true}'
        return SimpleNamespace(return_code=0)

    s.exec.side_effect = execute
    task = asyncio.create_task(r.run(payload().request))
    await asyncio.wait_for(polling.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)
    assert r.stopped
    s.disconnect.assert_not_awaited()
    await r.close()
    s.disconnect.assert_awaited_once()
    assert r.observations.gaps[0].code == "sandbox_result_unavailable"
