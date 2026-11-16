# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Failure masking and resources-owned sandbox cleanup for GDPVal training."""

import asyncio
import json
import shlex
import shutil
import sys
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException, Request

from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.core import NeMoGymResponse
from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult
from nemo_gym.server_runtime import BaseRunRequest
from resources_servers.gdpval_disaggregated import app as resources_app
from resources_servers.gdpval_disaggregated.app import (
    GDPValResourcesConfig,
    GDPValResourcesServer,
    GDPValVerifyRequest,
    TaskSession,
)
from resources_servers.gdpval_disaggregated.gym_bridge import JudgeConfig, judge


@pytest.fixture
def server(tmp_path: Path, monkeypatch) -> GDPValResourcesServer:
    monkeypatch.setattr(resources_app, "ensure_libreoffice", MagicMock(return_value=True))
    data = tmp_path / "data"
    (data / "references" / "sample").mkdir(parents=True)
    (data / "manifest.json").write_text('{"dataset": "synthetic-test"}')
    dataset = data / "tasks.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "task_id": "sample",
                "prompt": "Write a greeting.",
                "reference_root_key": "sample",
                "reference_files": [],
                "rubric_pretty": "A greeting is present.",
            }
        )
        + "\n"
    )
    config = GDPValResourcesConfig(
        name="gdpval",
        host="127.0.0.1",
        port=8080,
        entrypoint="app.py",
        dataset_path=str(dataset),
        data_root=str(data),
        artifact_root=str(tmp_path / "artifacts"),
        sandbox_image="test-image",
        judge_model_server={"type": "responses_api_models", "name": "judge"},
        judge_retry_delay_s=0,
    )
    return GDPValResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def request() -> Request:
    return Request({"type": "http", "session": {SESSION_ID_KEY: "session"}})


def body(**updates: object) -> GDPValVerifyRequest:
    return GDPValVerifyRequest(
        task_id="sample",
        responses_create_params={"input": "Write a greeting."},
        response=NeMoGymResponse(
            id="response",
            created_at=0,
            model="policy",
            object="response",
            output=[],
            tools=[],
            tool_choice="auto",
            parallel_tool_calls=True,
        ),
        **updates,
    )


def live_session(server: GDPValResourcesServer, root: Path) -> TaskSession:
    task = root / "task"
    (task / "agent" / "deliverables").mkdir(parents=True)
    sandbox = MagicMock(spec=AsyncSandbox)
    state = TaskSession("sample", sandbox, root, task, ready=True)
    server._sessions["session"] = state
    return state


async def test_incomplete_attempt_is_masked_and_keeps_compaction_segments(server, tmp_path, monkeypatch):
    state = live_session(server, tmp_path)
    monkeypatch.setattr(GDPValResourcesServer, "_download_outputs", AsyncMock())
    judge_saved = AsyncMock()
    monkeypatch.setattr(GDPValResourcesServer, "_judge_saved_artifacts", judge_saved)
    segments = [{"id": "segment", "metadata": {"training_mode": "compaction"}}]
    result = await server.verify(
        request(),
        body(
            agent_finished=False,
            incomplete_reason="agent_exec_timeout",
            responses=segments,
            agent_error="controller deadline expired",
            sandbox_error="sandbox exec timed out",
        ),
    )
    assert result.reward == 0
    assert result.mask_sample is True
    assert result.reward_valid is False
    assert result.evaluation_completed is False
    assert result.agent_finished is False
    assert result.responses == segments
    assert result.failure_reason == "agent_exec_timeout"
    assert result.incomplete_reason == "agent_exec_timeout"
    assert result.agent_error == "controller deadline expired"
    assert result.sandbox_error == "sandbox exec timed out"
    judge_saved.assert_not_called()
    state.sandbox.stop.assert_awaited_once()
    # Retry returns the saved result without rerunning the episode or losing segments.
    assert await server.verify(request(), body()) == result


async def test_download_failure_is_not_a_policy_zero(server, tmp_path, monkeypatch):
    live_session(server, tmp_path)
    monkeypatch.setattr(GDPValResourcesServer, "_download_outputs", AsyncMock(side_effect=OSError("sandbox gone")))
    result = await server.verify(request(), body())
    assert result.mask_sample and not result.reward_valid
    assert result.failure_reason == "outputs_unavailable"
    assert result.agent_finished is True


async def test_judge_failure_masks_saved_deliverables(server, tmp_path, monkeypatch):
    state = live_session(server, tmp_path)
    (state.task / "agent/deliverables/greeting.txt").write_text("Hello")
    monkeypatch.setattr(GDPValResourcesServer, "_download_outputs", AsyncMock())
    monkeypatch.setattr(
        GDPValResourcesServer, "_judge_saved_artifacts", AsyncMock(side_effect=RuntimeError("judge down"))
    )
    monkeypatch.setattr("resources_servers.gdpval_disaggregated.app.get_server_url", lambda _: "http://judge")
    result = await server.verify(request(), body())
    assert result.mask_sample and result.failure_reason == "judge_unavailable"
    assert (state.task / "artifacts/greeting.txt").read_text() == "Hello"


@pytest.mark.parametrize("policy_failure", [None, "agent_iteration_limit"])
async def test_completed_empty_or_budget_failure_is_valid_zero(server, tmp_path, monkeypatch, policy_failure):
    state = live_session(server, tmp_path)
    monkeypatch.setattr(GDPValResourcesServer, "_download_outputs", AsyncMock())
    monkeypatch.setattr("resources_servers.gdpval_disaggregated.app.get_server_url", lambda _: "http://judge")

    async def run_judge(self, state, path, config):
        await judge(state.task, JudgeConfig.model_validate(config))

    monkeypatch.setattr(GDPValResourcesServer, "_judge_saved_artifacts", run_judge)
    result = await server.verify(request(), body(policy_failure_reason=policy_failure))
    assert result.reward == 0 and not result.mask_sample
    assert result.reward_valid and result.agent_finished and result.evaluation_completed
    assert result.reward_reason == (policy_failure or "no_nonempty_deliverables")
    if policy_failure:
        assert result.policy_failure_reason == policy_failure
    else:
        assert "policy_failure_reason" not in result.model_dump()
    assert result.judge_skipped
    assert json.loads((state.task / "reward.json").read_text())["valid"] is True


async def test_configured_policy_failure_mask_is_not_graded(server, tmp_path, monkeypatch):
    live_session(server, tmp_path)
    server.config.mask_policy_failures = ["agent_context_limit"]
    monkeypatch.setattr(GDPValResourcesServer, "_download_outputs", AsyncMock())
    result = await server.verify(request(), body(policy_failure_reason="agent_context_limit"))
    assert result.mask_sample and result.failure_reason == "agent_context_limit"
    assert result.policy_failure_reason == "agent_context_limit"


async def test_failed_close_keeps_owner_for_retry(server, tmp_path):
    state = live_session(server, tmp_path)
    state.verify_started = True
    state.sandbox.stop.side_effect = [OSError("provider unavailable"), None]
    with pytest.raises(OSError, match="provider unavailable"):
        await server.cleanup_session(request())
    assert server._sessions["session"] is state
    assert await server.cleanup_session(request()) == {"cleaned": True}
    assert not server._sessions
    assert state.sandbox.stop.await_count == 2


async def test_failed_setup_cleanup_remains_retryable(server, monkeypatch):
    sandbox = MagicMock(spec=AsyncSandbox)
    sandbox.start_with_setup.side_effect = RuntimeError("upload failed")
    sandbox.stop.side_effect = [OSError("provider unavailable"), None]
    monkeypatch.setattr(GDPValResourcesServer, "_new_sandbox", lambda _: sandbox)
    run = BaseRunRequest(responses_create_params={"input": "", "metadata": {"task_id": "sample"}})
    with pytest.raises(RuntimeError, match="upload failed"):
        await server.seed_session(request(), run)
    assert server._sessions["session"].setup_failed
    with pytest.raises(HTTPException) as rejected:
        await server.seed_session(request(), run)
    assert rejected.value.status_code == 409
    await server.cleanup_session(request())
    assert not server._sessions


async def test_shutdown_retries_unreleased_sandboxes(server, tmp_path):
    state = live_session(server, tmp_path)
    state.sandbox.stop.side_effect = [OSError("provider unavailable"), None]
    await server.shutdown()
    assert server._sessions["session"] is state
    await server.shutdown()
    assert not server._sessions


def test_multiple_workers_are_rejected(server):
    config = server.config.model_copy(update={"num_workers": 2})
    with pytest.raises(ValueError, match="num_workers=1"):
        GDPValResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


async def test_artifact_processing_failure_is_distinct_from_judge_outage(server, tmp_path, monkeypatch):
    from resources_servers.gdpval_disaggregated.artifact_io import ArtifactProcessingError

    state = live_session(server, tmp_path)
    (state.task / "agent/deliverables/report.pdf").write_bytes(b"broken")
    monkeypatch.setattr(GDPValResourcesServer, "_download_outputs", AsyncMock())
    monkeypatch.setattr(
        GDPValResourcesServer, "_judge_saved_artifacts", AsyncMock(side_effect=ArtifactProcessingError("bad PDF"))
    )
    monkeypatch.setattr("resources_servers.gdpval_disaggregated.app.get_server_url", lambda _: "http://judge")
    result = await server.verify(request(), body())
    assert result.mask_sample and result.failure_reason == "artifact_processing_failed"
    assert (state.task / "artifacts/report.pdf").read_bytes() == b"broken"


async def test_partial_extraction_cannot_be_graded_on_retry(server, tmp_path, monkeypatch):
    state = live_session(server, tmp_path)

    async def partial_download(self, state):
        (state.task / "agent/deliverables/partial.txt").write_text("Partial output")
        raise ValueError("bad archive member")

    monkeypatch.setattr(GDPValResourcesServer, "_download_outputs", partial_download)
    run_judge = AsyncMock()
    monkeypatch.setattr(GDPValResourcesServer, "_judge_saved_artifacts", run_judge)
    first = await server.verify(request(), body())
    second = await server.verify(request(), body())
    assert first.mask_sample and first.failure_reason == "outputs_unavailable"
    assert second == first
    run_judge.assert_not_awaited()
    assert not (state.task / "artifacts").exists()


def test_missing_incomplete_reason_is_rejected_before_execution():
    with pytest.raises(ValueError, match="explicit reason"):
        body(agent_finished=False)


@pytest.mark.parametrize(
    "skipped_path,reason,has_readable_file",
    [
        ("/workspace/deliverables", "symlink or missing root", False),
        ("/workspace/deliverables/unreadable.txt", "PermissionError: access denied", True),
        ("/workspace/deliverables/link.txt", "symlink", True),
    ],
)
async def test_skipped_deliverable_cannot_be_graded(
    server, tmp_path, monkeypatch, skipped_path, reason, has_readable_file
):
    state = live_session(server, tmp_path)
    archive = tmp_path / "packaged.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        if has_readable_file:
            bundle.writestr("workspace/deliverables/readable.txt", "Keep this for inspection")
    skipped = [{"path": skipped_path, "reason": reason}]
    state.sandbox.exec.return_value = SandboxExecResult(
        return_code=0,
        error_type=None,
        stdout=json.dumps({"skipped": skipped}),
        stderr="",
    )

    async def download(remote, local):
        shutil.copyfile(archive, local)

    state.sandbox.download.side_effect = download
    run_judge = AsyncMock()
    monkeypatch.setattr(GDPValResourcesServer, "_judge_saved_artifacts", run_judge)
    result = await server.verify(request(), body())
    assert result.mask_sample and not result.reward_valid and not result.evaluation_completed
    assert result.failure_reason == "outputs_unavailable"
    run_judge.assert_not_awaited()
    assert json.loads((state.task / "packaging-skipped.json").read_text()) == skipped
    if has_readable_file:
        assert (state.task / "agent/deliverables/readable.txt").read_text() == "Keep this for inspection"
    state.sandbox.stop.assert_awaited_once()


async def test_skipped_trace_does_not_invalidate_deliverables(server, tmp_path):
    state = live_session(server, tmp_path)
    archive = tmp_path / "packaged.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("workspace/deliverables/report.txt", "Complete submission")
    skipped = [{"path": "/trace/diagnostic", "reason": "symlink"}]
    state.sandbox.exec.return_value = SandboxExecResult(
        return_code=0,
        error_type=None,
        stdout=json.dumps({"skipped": skipped}),
        stderr="",
    )

    async def download(remote, local):
        shutil.copyfile(archive, local)

    state.sandbox.download.side_effect = download
    await server._download_outputs(state)
    assert (state.task / "agent/deliverables/report.txt").read_text() == "Complete submission"
    assert json.loads((state.task / "packaging-skipped.json").read_text()) == skipped


async def test_packaging_script_rejects_deliverables_root_replaced_by_file(server, tmp_path, monkeypatch):
    state = live_session(server, tmp_path)
    sandbox_root = tmp_path / "sandbox"
    (sandbox_root / "workspace").mkdir(parents=True)
    (sandbox_root / "workspace/deliverables").write_text("Not a directory")
    (sandbox_root / "trace").mkdir()
    archive = sandbox_root / "output.zip"

    async def execute_packaging(command, **kwargs):
        # Execute the production packaging script against real temporary files.
        # Map only the sandbox's absolute paths to their test-host equivalents.
        script = shlex.split(command)[2]
        script = script.replace("/tmp/gdpval-output.zip", str(archive))
        script = script.replace("/workspace/deliverables", str(sandbox_root / "workspace/deliverables"))
        script = script.replace("/trace", str(sandbox_root / "trace"))
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        report = json.loads(stdout)
        for entry in report["skipped"]:
            entry["path"] = entry["path"].removeprefix(str(sandbox_root))
        return SandboxExecResult(
            return_code=process.returncode,
            error_type=None,
            stdout=json.dumps(report),
            stderr=stderr.decode(errors="replace"),
        )

    async def download(remote, local):
        shutil.copyfile(archive, local)

    state.sandbox.exec.side_effect = execute_packaging
    state.sandbox.download.side_effect = download

    async def grade_saved(state, path, config):
        await judge(state.task, JudgeConfig.model_validate(config))

    run_judge = AsyncMock(side_effect=grade_saved)
    monkeypatch.setattr(GDPValResourcesServer, "_judge_saved_artifacts", run_judge)
    monkeypatch.setattr("resources_servers.gdpval_disaggregated.app.get_server_url", lambda _: "http://judge")
    result = await server.verify(request(), body())
    assert result.mask_sample and not result.reward_valid
    assert result.failure_reason == "outputs_unavailable"
    run_judge.assert_not_awaited()
    skipped = json.loads((state.task / "packaging-skipped.json").read_text())
    assert skipped == [{"path": "/workspace/deliverables", "reason": "symlink or non-directory root"}]


@pytest.mark.parametrize(
    "report",
    ["", "truncated JSON {", "[]", "{}", '{"skipped": {}}', '{"skipped": [{}]}', '{"skipped": [{"path": 3}]}'],
)
async def test_invalid_packaging_report_masks_measurement(server, tmp_path, monkeypatch, report):
    state = live_session(server, tmp_path)
    state.sandbox.exec.return_value = SandboxExecResult(
        return_code=0,
        error_type=None,
        stdout=report,
        stderr="",
    )
    run_judge = AsyncMock()
    monkeypatch.setattr(GDPValResourcesServer, "_judge_saved_artifacts", run_judge)
    result = await server.verify(request(), body())
    assert result.mask_sample and not result.reward_valid and not result.evaluation_completed
    assert result.failure_reason == "outputs_unavailable"
    run_judge.assert_not_awaited()
    state.sandbox.download.assert_not_awaited()
    state.sandbox.stop.assert_awaited_once()


def test_startup_ensures_default_office_runtime(server):
    resources_app.ensure_libreoffice.assert_called_once_with()
    assert server._office_rendering_available


def test_invalid_office_wrapper_fails_startup(server, monkeypatch):
    ensure = MagicMock(return_value=False)
    monkeypatch.setattr(resources_app, "ensure_libreoffice", ensure)
    command = ["office-wrapper", "libreoffice"]
    config = server.config.model_copy(update={"libreoffice_command": command})
    with pytest.raises(RuntimeError, match="startup conversion check"):
        GDPValResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    ensure.assert_called_once_with(command, probe_dirs=[Path(config.artifact_root)])


async def test_unavailable_office_runtime_masks_office_submission(server, tmp_path, monkeypatch):
    from docx import Document

    monkeypatch.setattr(resources_app, "ensure_libreoffice", MagicMock(return_value=False))
    config = server.config.model_copy(update={"judge_timeout": 15, "judge_max_attempts": 1})
    unavailable = GDPValResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    assert not unavailable._office_rendering_available
    state = live_session(unavailable, tmp_path)
    submission = state.task / "agent/deliverables/report.docx"
    document = Document()
    document.add_paragraph("A readable Office submission")
    document.save(submission)
    (state.task / "judge.json").write_text(
        json.dumps(
            {
                "task_id": "sample",
                "prompt": "Write a report",
                "rubric_pretty": "The report is readable",
            }
        )
    )
    monkeypatch.setattr(GDPValResourcesServer, "_download_outputs", AsyncMock())
    monkeypatch.setattr(resources_app, "get_server_url", lambda _: "http://unused-judge")
    # The real judge subprocess rejects the document before any model request.
    result = await unavailable.verify(request(), body())
    assert result.mask_sample and not result.reward_valid
    assert result.failure_reason == "artifact_processing_failed"
    assert json.loads((state.task / "judge-failure.json").read_text())["kind"] == "artifact_processing"
    assert (state.task / "artifacts/report.docx").read_bytes() == submission.read_bytes()


async def test_verify_of_a_lost_session_is_masked_not_refused(server, monkeypatch):
    run_judge = AsyncMock()
    monkeypatch.setattr(GDPValResourcesServer, "_judge_saved_artifacts", run_judge)
    result = await server.verify(request(), body(responses=[{"id": "segment"}]))
    assert result.mask_sample and not result.reward_valid and not result.evaluation_completed
    assert result.failure_kind == "session_lost" and result.reward == 0.0 and result.judge_skipped
    assert result.responses == [{"id": "segment"}]
    run_judge.assert_not_awaited()


async def test_verify_for_another_task_is_still_a_conflict(server, tmp_path):
    live_session(server, tmp_path).task_id = "other"
    with pytest.raises(HTTPException) as rejected:
        await server.verify(request(), body())
    assert rejected.value.status_code == 409


async def test_interrupted_verify_is_masked_on_retry(server, tmp_path, monkeypatch):
    state = live_session(server, tmp_path)
    started = asyncio.Event()

    async def hang(self, state):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(GDPValResourcesServer, "_download_outputs", hang)
    first = asyncio.create_task(server.verify(request(), body()))
    await started.wait()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    retried = await server.verify(request(), body())
    assert retried.mask_sample and retried.failure_kind == "session_lost"
    state.sandbox.stop.assert_awaited_once()


async def test_seed_after_cleanup_is_refused(server, monkeypatch):
    sandbox = MagicMock(spec=AsyncSandbox)
    sandbox.serialize.return_value = {"sandbox_id": "box", "workdir": "/workspace"}
    monkeypatch.setattr(GDPValResourcesServer, "_new_sandbox", lambda _: sandbox)
    run = BaseRunRequest(responses_create_params={"input": "", "metadata": {"task_id": "sample"}})
    seeded = await server.seed_session(request(), run)
    assert seeded.sandbox_descriptor == {"sandbox_id": "box", "workdir": "/workspace"}
    sandbox.serialize.assert_awaited_once_with(scope="operate")
    await server.cleanup_session(request())
    with pytest.raises(HTTPException) as rejected:
        await server.seed_session(request(), run)
    assert rejected.value.status_code == 409
    assert not server._sessions
    sandbox.start_with_setup.assert_awaited_once()


async def test_seed_waiting_on_cleanup_does_not_return_a_stopped_sandbox(server, tmp_path):
    state = live_session(server, tmp_path)
    state.verify_started = True
    release = asyncio.Event()

    async def slow_stop():
        await release.wait()

    state.sandbox.stop.side_effect = slow_stop
    cleanup = asyncio.create_task(server.cleanup_session(request()))
    await asyncio.sleep(0)
    run = BaseRunRequest(responses_create_params={"input": "", "metadata": {"task_id": "sample"}})
    seed = asyncio.create_task(server.seed_session(request(), run))
    await asyncio.sleep(0)
    release.set()
    await cleanup
    with pytest.raises(HTTPException) as rejected:
        await seed
    assert rejected.value.status_code == 409
    state.sandbox.serialize.assert_not_awaited()


async def test_unexpected_judge_error_is_masked(server, tmp_path, monkeypatch):
    state = live_session(server, tmp_path)
    (state.task / "agent/deliverables/report.txt").write_text("Hello")
    monkeypatch.setattr(GDPValResourcesServer, "_download_outputs", AsyncMock())
    monkeypatch.setattr(
        GDPValResourcesServer, "_judge_saved_artifacts", AsyncMock(side_effect=OSError("too many open files"))
    )
    monkeypatch.setattr("resources_servers.gdpval_disaggregated.app.get_server_url", lambda _: "http://judge")
    result = await server.verify(request(), body())
    assert result.mask_sample and result.failure_reason == "judge_unavailable"
    assert result.judge_skipped is False


async def test_outputs_unavailable_reports_the_judge_as_skipped(server, tmp_path, monkeypatch):
    live_session(server, tmp_path)
    monkeypatch.setattr(GDPValResourcesServer, "_download_outputs", AsyncMock(side_effect=RuntimeError("gone")))
    result = await server.verify(request(), body())
    assert result.failure_reason == "outputs_unavailable" and result.judge_skipped is True


async def test_judge_deadline_starts_after_a_judge_slot_is_free(server, tmp_path, monkeypatch):
    server.config.judge_timeout = 1
    server._judge_sem = asyncio.Semaphore(1)
    state = live_session(server, tmp_path)
    monkeypatch.setattr(GDPValResourcesServer, "_run_judge", AsyncMock())
    await server._judge_sem.acquire()
    queued = asyncio.create_task(server._judge_saved_artifacts(state, tmp_path / "judge-config.json", {}))
    await asyncio.sleep(1.5)  # longer than judge_timeout, spent waiting for the slot
    server._judge_sem.release()
    await queued
    GDPValResourcesServer._run_judge.assert_awaited_once()


async def test_packaging_stops_before_transfer_when_deliverables_exceed_the_limit(server, tmp_path, monkeypatch):
    state = live_session(server, tmp_path)
    monkeypatch.setattr(resources_app, "MAX_OUTPUT_FILES", 2)
    sandbox_root = tmp_path / "sandbox"
    (sandbox_root / "workspace/deliverables").mkdir(parents=True)
    (sandbox_root / "trace").mkdir()
    for name in ("a.txt", "b.txt", "c.txt"):
        (sandbox_root / "workspace/deliverables" / name).write_text(name)

    async def execute_packaging(command, **kwargs):
        script = shlex.split(command)[2]
        script = script.replace("/tmp/gdpval-output.zip", str(sandbox_root / "output.zip"))
        script = script.replace("/workspace/deliverables", str(sandbox_root / "workspace/deliverables"))
        script = script.replace("/trace", str(sandbox_root / "trace"))
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", script, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        return SandboxExecResult(
            return_code=process.returncode,
            error_type=None,
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
        )

    state.sandbox.exec.side_effect = execute_packaging
    run_judge = AsyncMock()
    monkeypatch.setattr(GDPValResourcesServer, "_judge_saved_artifacts", run_judge)
    result = await server.verify(request(), body())
    assert result.mask_sample and result.failure_reason == "outputs_unavailable"
    assert "exceed the output extraction limit" in (state.task / "outputs-unavailable.txt").read_text()
    state.sandbox.download.assert_not_awaited()
    run_judge.assert_not_awaited()


async def test_packaging_skips_trace_files_over_the_limit(server, tmp_path, monkeypatch):
    state = live_session(server, tmp_path)
    monkeypatch.setattr(resources_app, "MAX_OUTPUT_FILES", 2)
    sandbox_root = tmp_path / "sandbox"
    (sandbox_root / "workspace/deliverables").mkdir(parents=True)
    (sandbox_root / "trace").mkdir()
    (sandbox_root / "workspace/deliverables/report.txt").write_text("Complete submission")
    for name in ("one.json", "two.json"):
        (sandbox_root / "trace" / name).write_text("{}")
    archive = sandbox_root / "output.zip"

    async def execute_packaging(command, **kwargs):
        script = shlex.split(command)[2]
        script = script.replace("/tmp/gdpval-output.zip", str(archive))
        script = script.replace("/workspace/deliverables", str(sandbox_root / "workspace/deliverables"))
        script = script.replace("/trace", str(sandbox_root / "trace"))
        # Archive names relative to the stand-in sandbox root, as the real script names them relative to "/".
        script = script.replace("p.relative_to('/')", f"p.relative_to({str(sandbox_root)!r})")
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", script, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        report = json.loads(stdout)
        for entry in report["skipped"]:
            entry["path"] = entry["path"].removeprefix(str(sandbox_root))
        return SandboxExecResult(
            return_code=process.returncode, error_type=None, stdout=json.dumps(report), stderr=stderr.decode()
        )

    async def download(remote, local):
        shutil.copyfile(archive, local)

    state.sandbox.exec.side_effect = execute_packaging
    state.sandbox.download.side_effect = download
    await server._download_outputs(state)
    assert (state.task / "agent/deliverables/report.txt").read_text() == "Complete submission"
    skipped = json.loads((state.task / "packaging-skipped.json").read_text())
    assert [entry["reason"] for entry in skipped] == ["over the output extraction limit"]
