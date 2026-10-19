# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from responses_api_agents.nooa_agent.result import finalize_run_result
from responses_api_agents.nooa_agent.runner import NOOARunFailure
from responses_api_agents.nooa_agent.sandbox_entrypoint import (
    RunnerError,
    SandboxCheckpoint,
    SandboxInput,
    SandboxResult,
)
from responses_api_agents.nooa_agent.sandbox_runner import SandboxNOOARunner
from responses_api_agents.nooa_agent.termination import NOOATerminationReason
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
@pytest.mark.parametrize(
    "reason",
    [
        NOOATerminationReason.POLICY_BUDGET_EXCEEDED,
        NOOATerminationReason.MAX_ITERATIONS_EXCEEDED,
        NOOATerminationReason.MAX_OUTPUT_TOKENS_EXCEEDED,
    ],
)
async def test_limit_result_survives_worker_boundary_and_cleanup(reason: NOOATerminationReason) -> None:
    r, s = runner()
    partial = run_result()
    partial.termination_reason = reason
    response, observations = finalize_run_result(partial)
    result_artifact = artifact().model_copy(
        update={"response": response, "observations": observations, "termination_reason": reason}
    )
    complete_files(r, s)
    s.files[r.directory + "/result.json"] = result_artifact.model_dump_json()
    result = await r.run(payload().request)
    assert result.termination_reason is reason
    assert result.episode.response == response
    assert result.episode.response.status == "incomplete"
    assert result.episode.response.error is None
    assert result.episode.response.output[-1].content[0].text == "answer"
    assert reason in {gap.code for gap in result.episode.observations.gaps}
    await r.close()
    assert r.stopped
    s.disconnect.assert_awaited_once()
    s.stop.assert_not_awaited()


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


def checkpoint_artifact() -> SandboxCheckpoint:
    # Completed-looking model output and a gradable limit must not authorize
    # verification after a worker dies before publishing its terminal result.
    return SandboxCheckpoint(
        **artifact().model_dump(exclude={"termination_reason"}),
        partial=True,
        termination_reason=NOOATerminationReason.MAX_OUTPUT_TOKENS_EXCEEDED,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("final", [None, "broken", '{"response":{}}', "empty-terminal"])
async def test_checkpoint_recovery_preserves_evidence_but_always_fails(final: str | None) -> None:
    r, s = runner()
    s.files[r.directory + "/cleanup.json"] = '{"cleanup_confirmed":true}'
    s.files[r.directory + "/checkpoint.json"] = checkpoint_artifact().model_dump_json()
    if final == "empty-terminal":
        final = SandboxResult(
            observations=artifact().observations, model_cookies={}, resource_cookies={}
        ).model_dump_json()
    if final is not None:
        s.files[r.directory + "/result.json"] = final
    with pytest.raises(NOOARunFailure, match="missing or malformed") as caught:
        await r.run(payload().request)
    recovered = caught.value.result
    assert type(caught.value.__cause__) is RuntimeError
    assert r.artifact is None
    assert r.checkpoint.partial is True
    assert recovered.termination_reason is NOOATerminationReason.INFRASTRUCTURE_ERROR
    assert recovered.episode.response.status == "failed"
    assert recovered.episode.response.output == checkpoint_artifact().response.output
    assert recovered.episode.response.error.code == "server_error"
    assert recovered.resource_cookies == {"resource": "new"}
    assert {"test", "partial_checkpoint", "sandbox_result_unavailable"} <= {
        gap.code for gap in recovered.episode.observations.gaps
    }
    s.disconnect.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("final_error", [None, RunnerError(kind="fatal", message="worker setup failed")])
async def test_valid_final_always_takes_precedence_over_checkpoint(final_error: RunnerError | None) -> None:
    r, s = runner()
    r.launched = True
    s.files[r.directory + "/checkpoint.json"] = checkpoint_artifact().model_dump_json()
    await r.collect()
    assert r.checkpoint is not None
    final = artifact(final_error)
    if final_error is not None:
        final.response = None
    s.files[r.directory + "/result.json"] = final.model_dump_json()
    await r.collect()
    assert r.artifact == final
    assert r.checkpoint is None
    assert r.observations == final.observations


@pytest.mark.asyncio
@pytest.mark.parametrize("checkpoint", ["broken", "{}", "unmarked"])
async def test_invalid_checkpoint_and_interrupted_temporary_are_not_recovered(checkpoint: str) -> None:
    r, s = runner()
    r.launched = True
    s.files[r.directory + "/checkpoint.json"] = (
        artifact().model_dump_json() if checkpoint == "unmarked" else checkpoint
    )
    s.files[r.directory + "/checkpoint.tmp"] = checkpoint_artifact().model_dump_json()
    await r.collect()
    assert r.artifact is None
    assert r.checkpoint is None
    assert [gap.code for gap in r.observations.gaps] == ["sandbox_result_unavailable"]


@pytest.mark.asyncio
async def test_failed_recollection_retains_last_valid_checkpoint() -> None:
    r, s = runner()
    r.launched = True
    s.files[r.directory + "/checkpoint.json"] = checkpoint_artifact().model_dump_json()
    await r.collect()
    recovered = r.checkpoint.model_copy(deep=True)
    s.files[r.directory + "/checkpoint.json"] = "interrupted transfer"
    await r.collect()
    assert r.checkpoint == recovered


@pytest.mark.asyncio
async def test_checkpoint_is_retrieved_before_session_files_and_handle_are_released() -> None:
    r, s = runner()
    r.launched = True
    s.files[r.directory + "/cleanup.json"] = '{"cleanup_confirmed":true}'
    s.files[r.directory + "/checkpoint.json"] = checkpoint_artifact().model_dump_json()
    events = []
    original_download = s.download

    async def download(source: str, destination: Path) -> None:
        if source.endswith("/checkpoint.json"):
            assert r.stopped
            events.append("checkpoint")
        await original_download(source, destination)

    async def execute(command: str, **kwargs) -> SimpleNamespace:
        assert "rm -rf" in command
        assert r.checkpoint is not None
        assert events == ["checkpoint"]
        events.append("remove")
        s.files.clear()
        return SimpleNamespace(return_code=0)

    async def disconnect() -> None:
        assert events == ["checkpoint", "remove"]
        events.append("disconnect")

    s.download = download
    s.exec.side_effect = execute
    s.disconnect.side_effect = disconnect
    await r.close()
    assert events == ["checkpoint", "remove", "disconnect"]
    assert r.checkpoint.response.output == checkpoint_artifact().response.output
    assert "partial_checkpoint" in {gap.code for gap in r.observations.gaps}


@pytest.mark.asyncio
@pytest.mark.parametrize("final", [None, "malformed"])
async def test_recovered_checkpoint_skips_environment_verification(monkeypatch, final: str | None) -> None:
    from responses_api_agents.nooa_agent.tests.test_app import agent
    from responses_api_agents.nooa_agent.tests.test_environment import _environment, _paths, _request

    r, s = runner()
    s.files[r.directory + "/cleanup.json"] = '{"cleanup_confirmed":true}'
    s.files[r.directory + "/checkpoint.json"] = checkpoint_artifact().model_dump_json()
    if final is not None:
        s.files[r.directory + "/result.json"] = final
    instance, run = agent(monkeypatch)
    run.side_effect = r.run
    environment, client = _environment(instance)
    response = await environment.run_request(_request())
    assert response.failure.stage == "agent"
    assert response.result is None
    assert "/verify" not in _paths(client)
    assert _paths(client)[-2:] == ["/v1/agent_sessions/close", "/close_session"]
    receipt = next(iter(instance._closed_session_records.values())).close_response
    assert {"partial_checkpoint", "sandbox_result_unavailable", NOOATerminationReason.INFRASTRUCTURE_ERROR} <= {
        gap.code for gap in receipt.agent_observations.gaps
    }
    await r.close()
