# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
    command = s.exec.await_args.args[0]
    assert "'/opt/nooa env/bin/python'" in command
    assert "-I -m responses_api_agents.nooa_agent.sandbox_supervisor" in command
    assert "launch.claim" in command
    assert "stdout.log" in command and "stderr.log" in command
    assert s.exec.await_args.kwargs == {"cwd": "/app", "timeout_s": None}
    launch = SandboxInput.model_validate_json(s.files[r.directory + "/input.json"])
    assert launch.max_policy_calls == limit
    assert launch.request.rollout_id == request.rollout_id
    assert launch.request.model_url_path == request.model_url_path
    assert request.resource_cookies == result.resource_cookies == {"resource": "new"}
    assert request.model_cookies == {"model": "new"}
    assert r.stopped


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
