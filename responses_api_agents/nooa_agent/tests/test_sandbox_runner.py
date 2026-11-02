# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nemo_gym.agent_utils.supervisor_client import CleanupReceipt, HarnessTerminationError
from nemo_gym.sandbox import SandboxExecResult
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
        self.exec = AsyncMock(return_value=SandboxExecResult(stdout="", stderr="", return_code=0))
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


def cleanup_receipt(**updates) -> CleanupReceipt:
    return {"cleanup_confirmed": True, "return_code": 0, "timed_out": False, "error": None, **updates}


def complete_files(r: SandboxNOOARunner, s: MemorySandbox, error: RunnerError | None = None) -> None:
    s.files[r.directory + "/cleanup.json"] = json.dumps(cleanup_receipt())
    s.files[r.directory + "/result.json"] = artifact(error).model_dump_json()


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [None, 3])
async def test_launch_quotes_paths_runs_in_task_and_preserves_cookies(limit: int | None) -> None:
    r, s = runner()
    r.max_policy_calls = limit
    await r.prepare()
    assert r.directory.startswith("/tmp/nemo-gym-nooa/")
    complete_files(r, s)
    request = payload().request
    result = await r.run(request)
    command = s.exec.await_args_list[1].args[0]
    assert "'/opt/nooa env/bin/python'" in command
    assert "-I -m responses_api_agents.nooa_agent.sandbox_entrypoint" in command
    assert "sandbox_supervisor" not in command
    assert "/process_supervisor.sh" in command
    assert r.directory + "/process_supervisor.sh" in s.files
    launch_command = await r._stage_activation(request)
    assert launch_command.argv[-2:] == [r.directory + "/input.json", r.directory + "/result.json"]
    assert len(launch_command.argv) == 6
    assert "--timeout 21600.0" in command
    assert s.exec.await_args_list[1].kwargs == {"cwd": "/app", "timeout_s": 21660}
    launch = SandboxInput.model_validate_json(s.files[r.directory + "/input.json"])
    assert launch.max_policy_calls == limit
    assert launch.request.rollout_id == request.rollout_id
    assert launch.request.model_url_path == request.model_url_path
    assert request.resource_cookies == result.resource_cookies == {"resource": "new"}
    assert request.model_cookies == {"model": "new"}
    assert r.session.cleanup == cleanup_receipt()
    assert not r.session.owns_sandbox


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metadata,budget",
    [
        (None, 21600),
        ({"agent_timeout_sec": "12.5"}, 12.5),
        ({"agent_timeout_sec": "90000"}, 21600),
        ({"agent_timeout_sec": "nan"}, 21600),
        ({"agent_timeout_sec": "-1"}, 21600),
        ({"agent_timeout_sec": "bad"}, 21600),
    ],
)
async def test_task_metadata_caps_execution_budget(metadata, budget: float) -> None:
    r, s = runner()
    complete_files(r, s)
    request = payload().request
    request.responses_create_params.metadata = metadata
    await r.run(request)
    assert f"--timeout {float(budget)} " in s.exec.await_args.args[0]
    assert s.exec.await_args.kwargs["timeout_s"] == budget + 60


@pytest.mark.asyncio
async def test_terminal_result_waits_for_cleanup_before_capture_and_return() -> None:
    r, s = runner()
    complete_files(r, s)
    del s.files[r.directory + "/cleanup.json"]
    events = []
    original_download = s.download

    async def download(source: str, destination: Path) -> None:
        if source.endswith("/result.json"):
            assert r.session.cleanup == cleanup_receipt()
            events.append("capture")
        await original_download(source, destination)

    async def execute(command: str, **kwargs) -> SandboxExecResult:
        if command.startswith("[ -f "):
            events.append("stop")
            s.files[r.directory + "/cleanup.json"] = json.dumps(cleanup_receipt())
        return SandboxExecResult(stdout="", stderr="", return_code=0)

    s.download = download
    s.exec.side_effect = execute
    result = await r.run(payload().request)
    assert events == ["stop", "capture"]
    assert result.episode.response == artifact().response
    assert result.episode.observations.gaps[0].detail == "evidence"
    s.disconnect.assert_not_awaited()
    s.stop.assert_not_awaited()


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
    s.disconnect.assert_awaited_once()
    s.stop.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "receipt",
    [
        cleanup_receipt(return_code=7),
        cleanup_receipt(return_code=None),
        cleanup_receipt(timed_out=True),
        cleanup_receipt(error="supervisor failed"),
    ],
)
@pytest.mark.parametrize("reason", [None, NOOATerminationReason.MAX_OUTPUT_TOKENS_EXCEEDED])
async def test_independent_execution_failure_overrides_gradable_artifact(receipt, reason) -> None:
    r, s = runner()
    complete_files(r, s)
    final = artifact().model_copy(update={"termination_reason": reason})
    s.files[r.directory + "/result.json"] = final.model_dump_json()
    s.files[r.directory + "/cleanup.json"] = json.dumps(receipt)
    request = payload().request
    with pytest.raises(NOOARunFailure) as caught:
        await r.run(request)
    assert caught.value.result.termination_reason is NOOATerminationReason.INFRASTRUCTURE_ERROR
    assert caught.value.result.episode.response.status == "failed"
    assert caught.value.result.episode.response.output == final.response.output
    assert r.artifact.termination_reason is reason
    assert r.observations == final.observations
    assert request.resource_cookies == {"resource": "new"}
    await r.close()
    assert r.artifact == final


@pytest.mark.asyncio
@pytest.mark.parametrize("contents", [None, "invalid-json", '{"response":{}}'])
async def test_missing_or_malformed_result_records_gap_not_success(contents: str | None) -> None:
    r, s = runner()
    s.files[r.directory + "/cleanup.json"] = json.dumps(cleanup_receipt())
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
    assert caught.value.result.termination_reason is NOOATerminationReason.INFRASTRUCTURE_ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["transport", "provider-timeout", "cancel"])
async def test_interrupted_run_preserves_evidence_and_original_failure(kind: str) -> None:
    r, s = runner()
    complete_files(r, s)
    terminal = (
        asyncio.CancelledError("cancelled request") if kind == "cancel" else ConnectionError("provider disconnected")
    )
    if kind == "cancel":
        # Assert the adapter preserves the cancellation it receives from the session.
        await r._collect_artifacts()
        r.session.execute = AsyncMock(side_effect=terminal)
    elif kind == "provider-timeout":
        s.exec.return_value = SandboxExecResult(stdout="", stderr="timeout", return_code=-1, error_type="timeout")
    else:
        s.exec.side_effect = terminal
    request = payload().request
    if kind == "cancel":
        with pytest.raises(asyncio.CancelledError) as caught:
            await r.run(request)
        assert caught.value is terminal
        result = caught.value.nooa_result
        assert result.termination_reason is NOOATerminationReason.CANCELLED
    else:
        with pytest.raises(NOOARunFailure) as caught:
            await r.run(request)
        assert (
            isinstance(caught.value.__cause__, TimeoutError)
            if kind == "provider-timeout"
            else caught.value.__cause__ is terminal
        )
        result = caught.value.result
        assert result.termination_reason is NOOATerminationReason.INFRASTRUCTURE_ERROR
    assert result.episode.response.output == artifact().response.output
    assert request.model_cookies == {"model": "new"}
    s.disconnect.assert_not_awaited()
    s.exec.side_effect = None
    s.exec.return_value = SandboxExecResult(stdout="", stderr="", return_code=0)
    await r.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("receipt", [None, '{"cleanup_confirmed":false}'])
@pytest.mark.parametrize("first_call", ["run", "close"])
async def test_missing_or_malformed_receipt_retains_handle_for_close_retry(
    receipt: str | None, first_call: str
) -> None:
    r, s = runner()
    if receipt is not None:
        s.files[r.directory + "/cleanup.json"] = receipt
    s.files[r.directory + "/checkpoint.json"] = checkpoint_artifact().model_dump_json()
    download = AsyncMock(wraps=s.download)
    s.download = download
    if first_call == "close":
        r.session.launch_started = True
        r.session._collect = r._collect_artifacts
    with pytest.raises(RuntimeError, match="unknown|not confirmed") as caught:
        if first_call == "run":
            await r.run(payload().request)
        else:
            await r.close()
    assert r.observations is not None
    assert [gap.code for gap in r.observations.gaps] == ["sandbox_capture_failed"]
    assert str(caught.value) in r.observations.gaps[0].detail
    observations = r.observations.model_copy(deep=True)
    with pytest.raises(RuntimeError, match="unknown|not confirmed"):
        await r.close()
    assert r.observations == observations
    assert r.session.capture_error is None
    assert r.session.termination_error is None
    assert r.checkpoint is None
    assert not any(call.args[0].endswith(("/checkpoint.json", "/result.json")) for call in download.await_args_list)
    assert not any("rm -rf" in call.args[0] for call in s.exec.await_args_list)
    assert not r.session.closed
    s.disconnect.assert_not_awaited()
    s.files[r.directory + "/cleanup.json"] = json.dumps(cleanup_receipt())
    await r.close()
    assert r.session.closed
    assert r.checkpoint is not None
    assert r.checkpoint.termination_reason is NOOATerminationReason.INFRASTRUCTURE_ERROR
    assert r.checkpoint.response.status == "failed"
    assert {gap.code for gap in r.observations.gaps} == {"test", "partial_checkpoint", "sandbox_result_unavailable"}
    s.disconnect.assert_awaited_once()
    s.stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_definitive_negative_receipt_skips_capture_and_releases_for_resources_teardown() -> None:
    r, s = runner()
    complete_files(r, s)
    s.files[r.directory + "/cleanup.json"] = json.dumps(
        cleanup_receipt(cleanup_confirmed=False, error="child remains")
    )
    download = AsyncMock(wraps=s.download)
    s.download = download
    with pytest.raises(HarnessTerminationError):
        await r.run(payload().request)
    await r.close()
    assert r.artifact is None
    assert not any(call.args[0].endswith("/result.json") for call in download.await_args_list)
    assert {gap.code for gap in r.observations.gaps} == {"sandbox_capture_failed"}
    s.disconnect.assert_awaited_once()
    s.stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_capture_failure_records_gap_and_does_not_block_release() -> None:
    r, s = runner()
    complete_files(r, s)
    r._collect_artifacts = AsyncMock(side_effect=RuntimeError("download unavailable"))
    with pytest.raises(RuntimeError, match="download unavailable"):
        await r.run(payload().request)
    await asyncio.gather(r.close(), r.close())
    assert "sandbox_capture_failed" in {gap.code for gap in r.observations.gaps}
    r._collect_artifacts.assert_awaited_once()
    s.disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_failed_staging_can_release_borrowed_connection() -> None:
    r, s = runner()
    s.upload = AsyncMock(side_effect=ConnectionError("upload failed"))
    with pytest.raises(ConnectionError, match="upload failed"):
        await r.run(payload().request)
    await r.close()
    assert not r.session.launch_started
    assert r.artifact is None
    assert r.observations is None
    s.disconnect.assert_awaited_once()
    s.stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_close_fences_inflight_staging_without_launch() -> None:
    r, s = runner()
    started = asyncio.Event()

    async def upload(source: Path, destination: str) -> None:
        started.set()
        await asyncio.Event().wait()

    s.upload = upload
    execution = asyncio.create_task(r.run(payload().request))
    await asyncio.wait_for(started.wait(), timeout=1)
    await asyncio.wait_for(asyncio.gather(r.close(), r.close()), timeout=1)
    with pytest.raises(asyncio.CancelledError):
        await execution
    assert not r.session.launch_started
    assert all("process_supervisor.sh" not in call.args[0] for call in s.exec.await_args_list)
    s.disconnect.assert_awaited_once()


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
    s.files[r.directory + "/cleanup.json"] = json.dumps(cleanup_receipt())
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
@pytest.mark.parametrize("final", [None, "malformed"])
@pytest.mark.parametrize("has_checkpoint", [False, True])
@pytest.mark.parametrize("log_kind", ["slow", "error", "missing", "available"])
async def test_optional_diagnostics_preserve_recovered_evidence(
    final: str | None, has_checkpoint: bool, log_kind: str
) -> None:
    r, s = runner()
    r.close_timeout = 0.05
    s.files[r.directory + "/cleanup.json"] = json.dumps(cleanup_receipt())
    if final is not None:
        s.files[r.directory + "/result.json"] = final
    if has_checkpoint:
        s.files[r.directory + "/checkpoint.json"] = checkpoint_artifact().model_dump_json()
    if log_kind == "available":
        s.files[r.directory + "/output.log"] = "diagnostic " * 250
    events = []
    original_download = s.download

    async def download(source: str, destination: Path) -> None:
        events.append(Path(source).name)
        if source.endswith("/output.log"):
            if log_kind == "slow":
                await asyncio.Event().wait()
            if log_kind == "error":
                raise ConnectionError("log download failed")
        await original_download(source, destination)

    async def execute(command: str, **kwargs) -> SandboxExecResult:
        if "rm -rf" in command:
            events.append("remove")
            s.files.clear()
        return SandboxExecResult(stdout="", stderr="", return_code=0)

    s.download = download
    s.exec.side_effect = execute
    request = payload().request
    with pytest.raises(RuntimeError) as caught:
        await asyncio.wait_for(r.run(request), timeout=0.5)
    assert events.index("checkpoint.json") < events.index("output.log")
    assert r.session.capture_error is None
    expected_gaps = {"sandbox_result_unavailable"}
    if has_checkpoint:
        assert isinstance(caught.value, NOOARunFailure)
        assert type(caught.value.__cause__) is RuntimeError
        recovered = caught.value.result
        assert recovered.termination_reason is NOOATerminationReason.INFRASTRUCTURE_ERROR
        assert recovered.episode.response.status == "failed"
        assert recovered.episode.response.output == checkpoint_artifact().response.output
        assert request.model_cookies == recovered.model_cookies == {"model": "new"}
        assert request.resource_cookies == recovered.resource_cookies == {"resource": "new"}
        assert r.checkpoint.partial
        expected_gaps |= {"test", "partial_checkpoint"}
    else:
        assert type(caught.value) is RuntimeError
        assert r.checkpoint is None
    if log_kind == "available":
        expected_gaps.add("sandbox_worker_diagnostics")
    assert {gap.code for gap in r.observations.gaps} == expected_gaps
    assert r.diagnostics == (("diagnostic " * 250)[-2000:] if log_kind == "available" else "")
    observations = r.observations.model_copy(deep=True)
    await r.close()
    assert events.index("checkpoint.json") < events.index("remove")
    assert r.observations == observations
    assert (r.checkpoint is not None) is has_checkpoint
    assert r.session.closed and not s.files
    s.disconnect.assert_awaited_once()
    s.stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancelling_optional_diagnostics_retains_checkpoint_and_cancellation() -> None:
    r, s = runner()
    s.files[r.directory + "/cleanup.json"] = json.dumps(cleanup_receipt())
    s.files[r.directory + "/checkpoint.json"] = checkpoint_artifact().model_dump_json()
    log_started = asyncio.Event()
    original_download = s.download

    async def download(source: str, destination: Path) -> None:
        if source.endswith("/output.log"):
            log_started.set()
            await asyncio.Event().wait()
        await original_download(source, destination)

    s.download = download
    task = asyncio.create_task(r.run(payload().request))
    await asyncio.wait_for(log_started.wait(), timeout=1)
    assert r.checkpoint is not None
    task.cancel("cancel diagnostics")
    with pytest.raises(asyncio.CancelledError) as caught:
        await asyncio.wait_for(task, timeout=1)
    assert caught.value.nooa_result.termination_reason is NOOATerminationReason.CANCELLED
    assert caught.value.nooa_result.episode.response.status == "cancelled"
    assert caught.value.nooa_result.episode.response.output == checkpoint_artifact().response.output
    assert "partial_checkpoint" in {gap.code for gap in r.observations.gaps}
    assert r.session.capture_error is None
    await r.close()
    s.disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_execution_failure_retains_checkpoint_despite_slow_optional_log() -> None:
    r, s = runner()
    r.close_timeout = 0.05
    s.files[r.directory + "/cleanup.json"] = json.dumps(cleanup_receipt())
    s.files[r.directory + "/checkpoint.json"] = checkpoint_artifact().model_dump_json()
    original_error = ConnectionError("provider disconnected")
    original_download = s.download

    async def download(source: str, destination: Path) -> None:
        if source.endswith("/output.log"):
            await asyncio.Event().wait()
        await original_download(source, destination)

    async def execute(command: str, **kwargs) -> SandboxExecResult:
        if "process_supervisor.sh" in command:
            raise original_error
        return SandboxExecResult(stdout="", stderr="", return_code=0)

    s.download = download
    s.exec.side_effect = execute
    with pytest.raises(NOOARunFailure) as caught:
        await asyncio.wait_for(r.run(payload().request), timeout=0.5)
    assert caught.value.__cause__ is original_error
    assert caught.value.result.termination_reason is NOOATerminationReason.INFRASTRUCTURE_ERROR
    assert caught.value.result.episode.response.status == "failed"
    assert caught.value.result.episode.response.output == checkpoint_artifact().response.output
    assert "partial_checkpoint" in {gap.code for gap in r.observations.gaps}
    assert r.session.capture_error is None
    await r.close()
    s.disconnect.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("final_error", [None, RunnerError(kind="fatal", message="worker setup failed")])
async def test_valid_final_always_takes_precedence_over_checkpoint(final_error: RunnerError | None) -> None:
    r, s = runner()
    s.files[r.directory + "/checkpoint.json"] = checkpoint_artifact().model_dump_json()
    await r._collect_artifacts()
    assert r.checkpoint is not None
    final = artifact(final_error)
    if final_error is not None:
        final.response = None
    s.files[r.directory + "/result.json"] = final.model_dump_json()
    await r._collect_artifacts()
    assert r.artifact == final
    assert r.checkpoint is None
    assert r.observations == final.observations


@pytest.mark.asyncio
@pytest.mark.parametrize("checkpoint", ["broken", "{}", "unmarked"])
async def test_invalid_checkpoint_and_interrupted_temporary_are_not_recovered(checkpoint: str) -> None:
    r, s = runner()
    s.files[r.directory + "/checkpoint.json"] = (
        artifact().model_dump_json() if checkpoint == "unmarked" else checkpoint
    )
    s.files[r.directory + "/checkpoint.tmp"] = checkpoint_artifact().model_dump_json()
    await r._collect_artifacts()
    assert r.artifact is None
    assert r.checkpoint is None
    assert [gap.code for gap in r.observations.gaps] == ["sandbox_result_unavailable"]


@pytest.mark.asyncio
async def test_failed_recollection_retains_last_valid_checkpoint() -> None:
    r, s = runner()
    s.files[r.directory + "/checkpoint.json"] = checkpoint_artifact().model_dump_json()
    await r._collect_artifacts()
    recovered = r.checkpoint.model_copy(deep=True)
    s.files[r.directory + "/checkpoint.json"] = "interrupted transfer"
    await r._collect_artifacts()
    assert r.checkpoint == recovered


@pytest.mark.asyncio
async def test_checkpoint_is_retrieved_before_session_files_and_handle_are_released() -> None:
    r, s = runner()
    r.session.launch_started = True
    r.session._collect = r._collect_artifacts
    s.files[r.directory + "/cleanup.json"] = json.dumps(cleanup_receipt())
    s.files[r.directory + "/checkpoint.json"] = checkpoint_artifact().model_dump_json()
    events = []
    original_download = s.download

    async def download(source: str, destination: Path) -> None:
        if source.endswith("/checkpoint.json"):
            assert r.session.cleanup is not None
            events.append("checkpoint")
        await original_download(source, destination)

    async def execute(command: str, **kwargs) -> SandboxExecResult:
        assert "rm -rf" in command
        assert r.checkpoint is not None
        assert events == ["checkpoint"]
        events.append("remove")
        s.files.clear()
        return SandboxExecResult(stdout="", stderr="", return_code=0)

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
    s.files[r.directory + "/cleanup.json"] = json.dumps(cleanup_receipt())
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


@pytest.mark.asyncio
async def test_terminal_error_without_response_cannot_recover_a_gradable_checkpoint() -> None:
    r, s = runner()
    final = artifact(RunnerError(kind="fatal", message="worker setup failed")).model_copy(update={"response": None})
    complete_files(r, s)
    s.files[r.directory + "/result.json"] = final.model_dump_json()
    s.files[r.directory + "/checkpoint.json"] = checkpoint_artifact().model_dump_json()
    request = payload().request
    with pytest.raises(RuntimeError, match="worker setup failed") as caught:
        await r.run(request)
    assert not isinstance(caught.value, NOOARunFailure)
    assert r.artifact == final
    assert r.checkpoint is None
    assert request.resource_cookies == final.resource_cookies


@pytest.mark.asyncio
async def test_cancelling_active_worker_captures_checkpoint_before_release() -> None:
    r, s = runner()
    active = asyncio.Event()
    execution_stopped = asyncio.Event()

    async def execute(command: str, **kwargs) -> SandboxExecResult:
        if "process_supervisor.sh" in command:
            active.set()
            try:
                await asyncio.Event().wait()
            finally:
                assert r.session.cleanup == cleanup_receipt()
                assert r.checkpoint is not None
                execution_stopped.set()
        elif command.startswith("[ -f "):
            s.files[r.directory + "/cleanup.json"] = json.dumps(cleanup_receipt())
            s.files[r.directory + "/checkpoint.json"] = checkpoint_artifact().model_dump_json()
        return SandboxExecResult(stdout="", stderr="", return_code=0)

    s.exec.side_effect = execute
    task = asyncio.create_task(r.run(payload().request))
    await asyncio.wait_for(active.wait(), timeout=1)
    task.cancel("request cancelled")
    with pytest.raises(asyncio.CancelledError) as caught:
        await asyncio.wait_for(task, timeout=1)
    assert str(caught.value) == "request cancelled"
    assert caught.value.nooa_result.termination_reason is NOOATerminationReason.CANCELLED
    assert caught.value.nooa_result.episode.response.status == "cancelled"
    assert "partial_checkpoint" in {gap.code for gap in r.observations.gaps}
    assert execution_stopped.is_set()
    s.disconnect.assert_not_awaited()
    await r.close()
    s.disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_during_capture_cannot_turn_final_evidence_into_success() -> None:
    r, s = runner()
    complete_files(r, s)
    capturing = asyncio.Event()
    release = asyncio.Event()
    collect = r._collect_artifacts

    async def collect_artifacts() -> None:
        capturing.set()
        await release.wait()
        await collect()

    r._collect_artifacts = collect_artifacts
    task = asyncio.create_task(r.run(payload().request))
    await asyncio.wait_for(capturing.wait(), timeout=1)
    closing = asyncio.create_task(r.close())
    async with asyncio.timeout(1):
        while not r.session.closing:
            await asyncio.sleep(0)
    release.set()
    with pytest.raises(asyncio.CancelledError, match="session closed") as caught:
        await asyncio.wait_for(task, timeout=1)
    await asyncio.wait_for(closing, timeout=1)
    assert caught.value.nooa_result.termination_reason is NOOATerminationReason.CANCELLED
    assert caught.value.nooa_result.episode.response.status == "cancelled"
    assert r.artifact == artifact()
    assert r.session.closed
    s.disconnect.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result", [SandboxExecResult("", "denied", 1), SandboxExecResult("", "timeout", 0, "timeout")]
)
async def test_prepare_rejects_failed_directory_creation(result: SandboxExecResult) -> None:
    r, s = runner()
    s.exec.return_value = result
    with pytest.raises(RuntimeError, match="Could not prepare NOOA session directory"):
        await r.prepare()
    assert not r.session.launch_started
    s.exec.return_value = SandboxExecResult("", "", 0)
    await r.close()
    s.disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_transient_terminal_download_retries_once_and_preserves_completed_result() -> None:
    r, s = runner()
    complete_files(r, s)
    download = s.download
    attempts = 0

    async def flaky_download(source: str, destination: Path) -> None:
        nonlocal attempts
        if source.endswith("/result.json"):
            attempts += 1
            if attempts == 1:
                raise ConnectionError("temporary download failure")
        await download(source, destination)

    s.download = flaky_download
    result = await r.run(payload().request)
    assert attempts == 2
    assert result.episode.response == artifact().response
    assert r.session.capture_error is None
    assert r.checkpoint is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_type,attempts", [(ConnectionError, 2), (PermissionError, 1), (ValueError, 1), (RuntimeError, 1)]
)
async def test_terminal_download_failure_keeps_exception_class_and_checkpoint(error_type, attempts) -> None:
    from responses_api_agents.nooa_agent.result import is_transient_infrastructure_error

    r, s = runner()
    complete_files(r, s)
    s.files[r.directory + "/checkpoint.json"] = checkpoint_artifact().model_dump_json()
    download = s.download
    count = 0

    async def failed_download(source: str, destination: Path) -> None:
        nonlocal count
        if source.endswith("/result.json"):
            count += 1
            raise error_type("cannot download result")
        await download(source, destination)

    s.download = failed_download
    with pytest.raises(NOOARunFailure) as caught:
        await r.run(payload().request)
    assert count == attempts
    assert isinstance(caught.value.__cause__, error_type)
    assert is_transient_infrastructure_error(caught.value) == (error_type is ConnectionError)
    assert caught.value.result.episode.response.output == checkpoint_artifact().response.output
    assert error_type.__name__ in next(
        gap.detail for gap in r.observations.gaps if gap.code == "sandbox_result_unavailable"
    )
    await r.close()
    s.disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_retry_cannot_exceed_artifact_capture_budget() -> None:
    r, s = runner()
    complete_files(r, s)
    r.close_timeout = 0.02
    download = s.download
    attempts = 0

    async def slow_retry(source: str, destination: Path) -> None:
        nonlocal attempts
        if source.endswith("/result.json"):
            attempts += 1
            if attempts == 1:
                raise ConnectionError("temporary download failure")
            await asyncio.Event().wait()
        await download(source, destination)

    s.download = slow_retry
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(r.run(payload().request), timeout=0.5)
    assert attempts == 2
    assert isinstance(r.session.capture_error, TimeoutError)
    await r.close()
    s.disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_worker_deadline_preserves_checkpoint_without_retryable_dependency_error() -> None:
    from responses_api_agents.nooa_agent.result import is_transient_infrastructure_error

    r, s = runner()
    s.files[r.directory + "/cleanup.json"] = json.dumps(cleanup_receipt(timed_out=True))
    s.files[r.directory + "/checkpoint.json"] = checkpoint_artifact().model_dump_json()
    with pytest.raises(NOOARunFailure, match="worker exceeded its execution deadline") as caught:
        await r.run(payload().request)
    assert not is_transient_infrastructure_error(caught.value)
    assert caught.value.result.termination_reason is NOOATerminationReason.INFRASTRUCTURE_ERROR
    assert caught.value.result.episode.response.status == "failed"
    assert caught.value.result.episode.response.output == checkpoint_artifact().response.output
    assert "partial_checkpoint" in {gap.code for gap in r.observations.gaps}
    await r.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("receipt", [cleanup_receipt(return_code=137), cleanup_receipt(error="startup failed")])
async def test_failed_worker_retains_bounded_diagnostics_gap(receipt: CleanupReceipt) -> None:
    r, s = runner()
    complete_files(r, s)
    s.files[r.directory + "/cleanup.json"] = json.dumps(receipt)
    s.files[r.directory + "/output.log"] = "x" * 2500 + "\nImportError: missing module\x00"
    with pytest.raises(NOOARunFailure) as caught:
        await r.run(payload().request)
    gap = next(
        gap for gap in caught.value.result.episode.observations.gaps if gap.code == "sandbox_worker_diagnostics"
    )
    assert len(gap.detail) <= 2000
    assert gap.detail.endswith("\nImportError: missing module")
    assert "\x00" not in gap.detail
    await r.close()


@pytest.mark.asyncio
async def test_cancel_without_any_artifact_preserves_bare_cancellation() -> None:
    r, s = runner()
    active = asyncio.Event()

    async def execute(command: str, **kwargs) -> SandboxExecResult:
        if "process_supervisor.sh" in command:
            active.set()
            await asyncio.Event().wait()
        elif command.startswith("[ -f "):
            s.files[r.directory + "/cleanup.json"] = json.dumps(cleanup_receipt())
        return SandboxExecResult("", "", 0)

    s.exec.side_effect = execute
    task = asyncio.create_task(r.run(payload().request))
    await asyncio.wait_for(active.wait(), timeout=1)
    task.cancel("request cancelled")
    with pytest.raises(asyncio.CancelledError, match="request cancelled") as caught:
        await asyncio.wait_for(task, timeout=1)
    assert not hasattr(caught.value, "nooa_result")
    assert r.artifact is None and r.checkpoint is None
    assert {gap.code for gap in r.observations.gaps} == {"sandbox_result_unavailable"}
    await r.close()
    s.disconnect.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "receipt,message",
    [
        (cleanup_receipt(return_code=137), "worker did not exit successfully (exit 137)"),
        (cleanup_receipt(timed_out=True), "worker exceeded its execution deadline"),
        (cleanup_receipt(error="startup failed"), "supervisor failed: startup failed"),
    ],
)
@pytest.mark.parametrize("capture_error_type", [RuntimeError, ConnectionError])
@pytest.mark.parametrize("provider_return_code", [0, 1])
@pytest.mark.parametrize("has_checkpoint", [False, True])
async def test_capture_error_does_not_hide_confirmed_worker_failure(
    receipt: CleanupReceipt,
    message: str,
    capture_error_type: type[Exception],
    provider_return_code: int,
    has_checkpoint: bool,
) -> None:
    from responses_api_agents.nooa_agent.result import is_transient_infrastructure_error

    r, s = runner()
    s.files[r.directory + "/cleanup.json"] = json.dumps(receipt)
    s.files[r.directory + "/output.log"] = "ImportError: worker could not start"
    if has_checkpoint:
        s.files[r.directory + "/checkpoint.json"] = checkpoint_artifact().model_dump_json()
    download = s.download
    capture_error = capture_error_type("docker cp download failed: artifact unavailable")

    async def download_artifact(source: str, destination: Path) -> None:
        if source.endswith("/result.json") or (not has_checkpoint and source.endswith("/checkpoint.json")):
            raise capture_error
        await download(source, destination)

    s.download = download_artifact
    s.exec.return_value = SandboxExecResult("", "", provider_return_code)
    with pytest.raises(RuntimeError) as caught:
        await r.run(payload().request)
    assert message in str(caught.value)
    assert not is_transient_infrastructure_error(caught.value)
    assert r.session.capture_error is capture_error
    assert r.diagnostics == "ImportError: worker could not start"
    gaps = {gap.code: gap.detail for gap in r.observations.gaps}
    assert gaps["sandbox_worker_diagnostics"] == r.diagnostics
    assert str(capture_error) in gaps["sandbox_capture_failed"]
    if has_checkpoint:
        assert isinstance(caught.value, NOOARunFailure)
        assert caught.value.result.episode.response.output == checkpoint_artifact().response.output
        assert caught.value.result.termination_reason is NOOATerminationReason.INFRASTRUCTURE_ERROR
    s.exec.return_value = SandboxExecResult("", "", 0)
    await r.close()
    s.disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_provider_execution_failure_is_not_replaced_by_a_capture_error() -> None:
    r, s = runner()
    s.files[r.directory + "/cleanup.json"] = json.dumps(cleanup_receipt(timed_out=True))
    s.files[r.directory + "/checkpoint.json"] = checkpoint_artifact().model_dump_json()
    download = s.download
    provider_error = ConnectionError("provider execution transport disconnected")

    async def download_artifact(source: str, destination: Path) -> None:
        if source.endswith("/result.json"):
            raise RuntimeError("docker cp download failed")
        await download(source, destination)

    async def execute(command: str, **kwargs) -> SandboxExecResult:
        if "process_supervisor.sh" in command:
            raise provider_error
        return SandboxExecResult("", "", 0)

    s.download = download_artifact
    s.exec.side_effect = execute
    with pytest.raises(NOOARunFailure) as caught:
        await r.run(payload().request)
    assert caught.value.__cause__ is provider_error
    assert "sandbox_capture_failed" in {gap.code for gap in r.observations.gaps}
    await r.close()
