# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Drive the public hooks, real session contract and retained artifact grading."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException, Request

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.eval.rollout_reverification import InputRolloutPair, _build_verify_payload
from nemo_gym.sandbox import DirectSandboxConnection, SandboxAccess, SandboxConfig
from nemo_gym.server_runtime import (
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    check_resources_session_contract,
)
from nemo_gym.verification import JudgeError, judge_failsafe
from resources_servers.mm_coding_training.app import CodingRequest, CodingSeedRequest
from resources_servers.mm_coding_training.session_app import SessionCodingConfig, SessionCodingTrainingServer
from resources_servers.mm_coding_training.tests.test_app import png, request
from resources_servers.mm_coding_training.tests.test_app import server as server


@pytest.fixture
def session_server(server):
    return SessionCodingTrainingServer(
        config=SessionCodingConfig(
            **server.config.model_dump(),
            sandbox_config=SandboxConfig(ttl_s=14400, ready_timeout_s=600, resources={"cpu": 4, "memory_mib": 16384}),
        ),
        server_client=server.server_client,
    )


def http_request(session_id="session-1"):
    return Request({"type": "http", "session": {SESSION_ID_KEY: session_id}, "headers": []})


def installed(server, body, code=0, error=None):
    sandbox = AsyncMock()
    sandbox.exec.return_value = SimpleNamespace(return_code=code, error_type=error)

    async def download(remote, local):
        Path(local).write_text("<svg/>")

    sandbox.download.side_effect = download
    server._lender._sandboxes["session-1"] = sandbox
    server._tasks["session-1"] = CodingSeedRequest(task=body.task)
    return sandbox


def typed_seed(body):
    return ResourcesSeedSessionRequest.model_validate(
        {
            "resources_session_id": "session-1",
            "episode_id": {"rollout_id": "rollout", "attempt": 0},
            "task_id": {"taskset": "mm-coding:opencode", "task_id": f"{body.task.benchmark}:{body.task.task_id}"},
            "task_data": {"task": body.task.model_dump()},
        }
    )


def fake_lend(server):
    async def lend(session_id, task_id, *, files):
        sandbox = AsyncMock()
        server._lender._sandboxes[session_id] = sandbox
        return SandboxAccess(
            connection=DirectSandboxConnection(provider_config_ref="sandbox", descriptor={"id": session_id}),
            workdir="/workspace",
        )

    return AsyncMock(side_effect=lend)


@pytest.mark.asyncio
async def test_final_file_not_prose_is_independently_graded(session_server):
    body = request("This is prose, not the answer file")
    sandbox = installed(session_server, body)

    async def grade(body, source, result):
        assert source == "<svg/>" and result.response == body.response
        result.reward, result.evaluation_completed = 0.8, True
        return result

    with patch.object(type(session_server), "grade_source", side_effect=grade):
        result = await session_server.verify(http_request(), body)
    assert result.reward == 0.8 and not result.mask_sample and result.evaluation_completed
    sandbox.stop.assert_awaited_once()
    assert not session_server.has_session_state("session-1")


@pytest.mark.parametrize(
    "code,error,masked",
    [(44, None, False), (1, None, False), (137, None, True), (-1, None, True), (0, "timeout", True)],
)
@pytest.mark.asyncio
async def test_missing_artifact_is_model_zero_transport_is_masked(session_server, code, error, masked):
    body = request()
    sandbox = installed(session_server, body, code, error)
    result = await session_server.verify(http_request(), body)
    assert result.reward == 0 and result.mask_sample is masked
    assert result.evaluation_completed is (not masked)
    sandbox.stop.assert_awaited_once()
    sandbox.download.assert_not_awaited()


@pytest.mark.asyncio
async def test_other_task_cannot_read_or_grade_session(session_server):
    body = request()
    sandbox = installed(session_server, body)
    other = body.model_copy(deep=True)
    other.task.task_id = "other-task"
    with pytest.raises(HTTPException) as error:
        await session_server.verify(http_request(), other)
    assert error.value.status_code == 409
    sandbox.exec.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_stop_keeps_sandbox_for_close_retry(session_server):
    box = installed(session_server, request())
    box.stop.side_effect = [RuntimeError("provider unavailable"), None]
    with pytest.raises(RuntimeError):
        await session_server.release_session_state("session-1")
    assert session_server.has_session_state("session-1")
    await session_server.release_session_state("session-1")
    assert not session_server.has_session_state("session-1")


@pytest.mark.asyncio
async def test_seed_delivers_only_public_reference_and_renderer(session_server):
    body = request(benchmark="chart2code")
    body.task.ground_truth = "PRIVATE_GOLD"
    lend = fake_lend(session_server)
    with patch.object(session_server._lender, "lend", lend):
        first = await session_server.seed_session(http_request(), typed_seed(body))
        assert first == await session_server.seed_session(http_request(), typed_seed(body))
    lend.assert_awaited_once()
    files = lend.call_args.kwargs["files"]
    assert files["/workspace/reference.png"] == png()
    assert set(files) == {"/workspace/reference.png", "/opt/mmrl/render.py"}
    assert "PRIVATE_GOLD" not in str(files)


@pytest.mark.asyncio
async def test_seed_stages_offline_dependencies_before_lending(session_server, tmp_path):
    runtime = tmp_path / "plugin.tar.gz"
    runtime.write_bytes(b"operator-provisioned pinned dependency bytes")
    destination = "/opt/mm-coding/plugin-dependencies.tar.gz"
    session_server.config.sandbox_runtime_files = {destination: runtime}
    lend = fake_lend(session_server)
    with patch.object(session_server._lender, "lend", lend):
        await session_server.seed_session(http_request(), typed_seed(request()))
    assert lend.call_args.kwargs["files"][destination] == runtime.read_bytes()


@pytest.mark.asyncio
async def test_seed_refuses_runtime_file_overwriting_task_workspace(session_server, tmp_path):
    runtime = tmp_path / "artifact"
    runtime.write_bytes(b"not an answer")
    session_server.config.sandbox_runtime_files = {"/workspace/answer.svg": runtime}
    lend = fake_lend(session_server)
    with patch.object(session_server._lender, "lend", lend):
        with pytest.raises(ResourcesSeedRejectedError):
            await session_server.seed_session(http_request(), typed_seed(request()))
    lend.assert_not_awaited()


@pytest.mark.asyncio
async def test_verify_retry_returns_saved_grade_without_new_capture_or_judge(session_server):
    body = request()
    box = installed(session_server, body)

    async def grade(body, source, result):
        result.reward, result.evaluation_completed = 0.7, True
        return result

    with patch.object(type(session_server), "grade_source", side_effect=grade) as grading:
        first = await session_server.verify(http_request(), body)
        assert first == await session_server.verify(http_request(), body)
    grading.assert_awaited_once()
    box.download.assert_awaited_once()


@pytest.mark.asyncio
async def test_captured_source_survives_verifier_failure_after_task_box_closes(session_server):
    body = request("Final prose only")
    box = installed(session_server, body)
    renderer = AsyncMock()
    renderer.render.side_effect = RuntimeError("provider unavailable")
    with patch.object(type(session_server), "_renderer", return_value=renderer):
        result = await session_server.verify(http_request(), body)
    assert result.mask_sample and not result.evaluation_completed and result.source_artifact == "<svg/>"
    assert result.response == body.response
    assert result.source_sha256 == hashlib.sha256(result.source_artifact.encode()).hexdigest()
    box.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_judge_retry_uses_retained_source_without_another_sandbox_capture(session_server):
    body = request("Original policy prose")
    box = installed(session_server, body)

    async def fail(body, source, result):
        raise JudgeError("Known judge outage")

    with patch.object(type(session_server), "grade_source", side_effect=fail):
        with pytest.raises(JudgeError):
            await session_server.verify(http_request(), body)

    async def succeed(body, source, result):
        assert source == "<svg/>"
        result.reward, result.evaluation_completed = 0.8, True
        return result

    with patch.object(type(session_server), "grade_source", side_effect=succeed):
        result = await session_server.verify(http_request(), body)
    assert result.reward == 0.8 and result.evaluation_completed
    box.download.assert_awaited_once()
    box.stop.assert_awaited_once()


def test_public_resources_session_contract_includes_lost_verify(session_server):
    body = request()
    with patch.object(session_server._lender, "lend", fake_lend(session_server)):
        check_resources_session_contract(
            session_server.setup_webserver(),
            typed_seed(body),
            keeps_state=True,
            holds_state=session_server.has_session_state,
            verify_body=body.model_dump(mode="json"),
        )


@pytest.mark.asyncio
async def test_reverification_refuses_changed_source_or_policy_response(session_server):
    body = request()
    installed(session_server, body)
    renderer = AsyncMock()
    renderer.render.side_effect = RuntimeError("known provider outage")
    with patch.object(type(session_server), "_renderer", return_value=renderer):
        result = await session_server.verify(http_request(), body)
    replay = body.model_copy(
        update={
            "source_artifact": result.source_artifact,
            "source_sha256": result.source_sha256,
            "source_binding_sha256": result.source_binding_sha256,
        },
        deep=True,
    )
    replay.response.id = "changed-policy-response"
    with pytest.raises(HTTPException, match="binding"):
        await session_server.verify(http_request("fresh-reverify"), replay)


@pytest.mark.asyncio
async def test_saved_receipt_can_be_regraded_without_policy_or_task_box(session_server):
    body = request("Original policy reply")
    box = installed(session_server, body)
    renderer = AsyncMock()
    renderer.render.side_effect = RuntimeError("known provider outage")
    with patch.object(type(session_server), "_renderer", return_value=renderer):
        first = await session_server.verify(http_request(), body)
    replay = body.model_copy(
        update={
            "source_artifact": first.source_artifact,
            "source_sha256": first.source_sha256,
            "source_binding_sha256": first.source_binding_sha256,
        },
        deep=True,
    )

    async def grade(body, source, result):
        assert source == "<svg/>" and body.response.id == replay.response.id
        result.reward, result.evaluation_completed = 0.9, True
        return result

    with patch.object(type(session_server), "grade_source", side_effect=grade) as grading:
        result = await session_server.verify(http_request("fresh-regrade"), replay)
    assert result.evaluation_completed and result.reward == 0.9
    grading.assert_awaited_once()
    box.download.assert_awaited_once()
    box.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_gym_reverify_payload_regrades_a_judge_failure(session_server):
    body = request("Original policy reply")
    installed(session_server, body)
    renderer = AsyncMock()
    renderer.render.return_value = png()

    async def fail(*args, **kwargs):
        raise JudgeError("judge outage")

    with (
        patch.object(type(session_server), "_renderer", return_value=renderer),
        patch.object(type(session_server), "_judge", side_effect=fail),
    ):
        reply = await judge_failsafe(session_server.verify)(http_request(), body)
    sidecar = json.loads(reply.body) | {"_ng_task_index": 0, "_ng_rollout_index": 0}
    materialized = {
        "task": body.task.model_dump(),
        "responses_create_params": body.responses_create_params.model_dump(mode="json", exclude_unset=True),
        "_ng_task_index": 0,
        "_ng_rollout_index": 0,
    }
    payload = _build_verify_payload(InputRolloutPair(input=materialized, rollout=sidecar))

    async def grade(body, source, result):
        assert source == "<svg/>"
        result.reward, result.evaluation_completed = 0.9, True
        return result

    with patch.object(type(session_server), "grade_source", side_effect=grade):
        result = await session_server.verify(http_request("fresh-reverify"), CodingRequest.model_validate(payload))
    assert result.reward == 0.9 and not result.mask_sample
