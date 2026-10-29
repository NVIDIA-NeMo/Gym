# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.base_responses_api_model import CaptureStore
from nemo_gym.client import ServerClient
from nemo_gym.core import JUDGE_FAILED, SESSION_LOST, VERIFIER_ERROR, rollout_context
from nemo_gym.core.episode import VERIFIER_INVOCATION_HEADER, current_verifier_invocation_id
from nemo_gym.core.failure_kinds import TRANSPORT_TIMEOUT
from nemo_gym.judge import JudgeError, call_judge, judge_failsafe
from nemo_gym.observability.model_calls import merge_model_call_capture_into_record
from nemo_gym.observability.verifiers import observe_verifier
from nemo_gym.rollout_collection import _build_trajectory_record


class Verdict(BaseModel):
    grade: float


class VerifyInput(BaseModel):
    response: dict = {}


def collected(path: Path, rollout_id: str, result: dict) -> dict:
    record = {**result, "_ng_rollout_id": rollout_id}
    merge_model_call_capture_into_record(record, [path], include_payloads=True)
    return record


@pytest.mark.asyncio
async def test_one_verify_owns_parallel_judge_calls_and_retry_attempts(tmp_path):
    store = CaptureStore(tmp_path)
    observed_ids = []

    async def post(*, server_name, headers, **kwargs):
        invocation_id = headers[VERIFIER_INVOCATION_HEADER]
        observed_ids.append(invocation_id)
        for attempt in range(2):
            call_id = f"{server_name}-{attempt}"
            store.record(
                "one-verify",
                {
                    "model_call_id": call_id,
                    "status_code": 429 if attempt == 0 else 200,
                    "verifier_invocation_id": invocation_id,
                },
            )
            await asyncio.sleep(0)
        response = MagicMock(ok=True, status=200)
        response.read = AsyncMock(return_value=b'{"grade":1.0}')
        return response

    client = MagicMock(post=post)

    async def verify():
        await asyncio.gather(
            *(
                call_judge(client, server_name=name, url_path="/v1/responses", json={}, response_model=Verdict)
                for name in ("rubric-a", "rubric-b")
            )
        )
        # Parsing/scoring after model calls is part of this same execution.
        assert current_verifier_invocation_id() == observed_ids[0]
        return {"reward": 0.0}

    with rollout_context("one-verify"):
        result = await observe_verifier(verify, enabled=True)()
    assert result["reward"] == 0.0
    assert current_verifier_invocation_id() is None
    assert len(set(observed_ids)) == 1
    record = collected(tmp_path, "one-verify", result)
    trajectory = _build_trajectory_record({"_ng_task_index": 0, "_ng_rollout_index": 0}, record)
    assert len(trajectory.verifier_invocations) == 1
    invocation = trajectory.verifier_invocations[0]
    assert invocation.status == "completed"
    assert invocation.failure_kind is None
    assert len(invocation.model_calls) == 4
    assert {ref.model_call_id for ref in invocation.model_calls} == {
        call.model_call_id for call in trajectory.model_calls
    }
    assert trajectory.invocations == []
    assert trajectory.turns == []


@pytest.mark.asyncio
async def test_concurrent_verifications_keep_separate_identities(tmp_path):
    ids = {}

    async def verify(name):
        ids[name] = current_verifier_invocation_id()
        await asyncio.sleep(0)
        assert current_verifier_invocation_id() == ids[name]
        return {"reward": 1.0}

    wrapped = observe_verifier(verify, enabled=True)
    with rollout_context("parallel"):
        results = await asyncio.gather(wrapped("a"), wrapped("b"))
    assert ids["a"] != ids["b"]
    assert current_verifier_invocation_id() is None
    invocations = [inv for result in results for inv in result["ng_verifier_invocations"]]
    assert list(tmp_path.iterdir()) == []
    assert len(invocations) == 2
    assert all(inv["status"] == "completed" and inv["model_calls"] == [] for inv in invocations)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error,status,kind",
    [
        (RuntimeError("scoring broke"), "failed", VERIFIER_ERROR),
        (TimeoutError("grading deadline"), "incomplete", TRANSPORT_TIMEOUT),
    ],
)
async def test_verify_errors_return_evidence_in_http_failure(tmp_path, error, status, kind):
    async def verify():
        raise error

    with rollout_context("error"):
        response = await observe_verifier(verify, enabled=True)()
    assert response.status_code == 500
    (invocation,) = json.loads(response.body)["ng_verifier_invocations"]
    assert list(tmp_path.iterdir()) == []
    assert invocation["status"] == status
    assert invocation["failure_kind"] == kind
    assert current_verifier_invocation_id() is None


@pytest.mark.asyncio
async def test_judge_failsafe_preserves_shared_failure_kind(tmp_path):
    async def verify(body):
        raise JudgeError("judge unavailable")

    with rollout_context("judge-failure"):
        response = await observe_verifier(judge_failsafe(verify), enabled=True)(VerifyInput())
    assert response.status_code == 200
    payload = json.loads(response.body)
    assert payload["_ng_failure_class"] == JUDGE_FAILED
    (invocation,) = payload["ng_verifier_invocations"]
    assert invocation["status"] == "failed"
    assert invocation["failure_kind"] == JUDGE_FAILED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result,status",
    [
        ({"reward": 0.0}, "completed"),
        ({"reward": 0.0, "evaluation_completed": False}, "incomplete"),
        ({"reward": 0.0, "failure_kind": SESSION_LOST, "mask_sample": True}, "completed"),
    ],
)
async def test_returned_grade_does_not_infer_failure_from_reward_or_mask(tmp_path, result, status):
    async def verify():
        return result

    with rollout_context("grade"):
        returned = await observe_verifier(verify, enabled=True)()
    assert {key: value for key, value in returned.items() if key != "ng_verifier_invocations"} == result
    (invocation,) = returned["ng_verifier_invocations"]
    assert invocation["status"] == status
    assert invocation["failure_kind"] == result.get("failure_kind")


def test_resources_route_observes_programmatic_verify_without_benchmark_changes(tmp_path):
    class Resources(SimpleResourcesServer):
        async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
            assert current_verifier_invocation_id() is not None
            return BaseVerifyResponse(**body.model_dump(), reward=0.0)

    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"observability_enabled": True, "model_call_capture_dir": str(tmp_path)}
    server = Resources(
        config=BaseResourcesServerConfig(name="programmatic", host="127.0.0.1", port=12345, entrypoint="app.py"),
        server_client=client,
    )
    response = TestClient(server.setup_webserver()).post(
        "/ng-rollout/programmatic/verify",
        json={
            "responses_create_params": {"input": "task"},
            "response": {
                "id": "response",
                "created_at": 0,
                "model": "test",
                "object": "response",
                "output": [],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            },
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["reward"] == 0.0
    (invocation,) = response.json()["ng_verifier_invocations"]
    assert invocation["status"] == "completed"
    assert invocation["model_calls"] == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_observability_disabled_leaves_execution_unchanged(tmp_path):
    async def verify():
        assert current_verifier_invocation_id() is None
        return {"reward": 1.0}

    assert observe_verifier(verify, enabled=False) is verify
    with rollout_context("disabled"):
        assert await verify() == {"reward": 1.0}
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_direct_model_requests_inside_verify_inherit_owner(tmp_path, monkeypatch):
    from omegaconf import OmegaConf

    transport = AsyncMock()
    monkeypatch.setattr("nemo_gym.client.server_client.request", transport)
    client = ServerClient(
        head_server_config={"host": "localhost", "port": 0},
        global_config_dict=OmegaConf.create(
            {
                "observability_enabled": True,
                "judge": {"responses_api_models": {"openai_model": {"host": "localhost", "port": 9999}}},
            }
        ),
    )

    async def verify():
        owner = current_verifier_invocation_id()
        await client.post(server_name="judge", url_path="/v1/responses", json={}, headers={"x-custom": "kept"})
        assert transport.call_args.kwargs["headers"] == {
            "x-custom": "kept",
            VERIFIER_INVOCATION_HEADER: owner,
        }
        return {"reward": 1.0}

    with rollout_context("direct"):
        await observe_verifier(verify, enabled=True)()
    assert current_verifier_invocation_id() is None
    transport.reset_mock()
    with rollout_context("policy"):
        await client.post(server_name="judge", url_path="/v1/responses", json={})
    assert VERIFIER_INVOCATION_HEADER not in transport.call_args.kwargs.get("headers", {})


@pytest.mark.asyncio
async def test_cancelled_verify_does_not_swallow_cancellation_or_write_files(tmp_path):
    async def verify():
        raise asyncio.CancelledError()

    with rollout_context("cancelled"), pytest.raises(asyncio.CancelledError):
        await observe_verifier(verify, enabled=True)()
    assert current_verifier_invocation_id() is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_http_error_retains_status_headers_and_detail():
    from fastapi import HTTPException

    async def verify():
        raise HTTPException(503, detail="verifier unavailable", headers={"Retry-After": "2"})

    with rollout_context("http-error"):
        response = await observe_verifier(verify, enabled=True)()
    assert response.status_code == 503
    assert response.headers["retry-after"] == "2"
    payload = json.loads(response.body)
    assert payload["detail"] == "verifier unavailable"
    assert payload["ng_verifier_invocations"][0]["status"] == "failed"
