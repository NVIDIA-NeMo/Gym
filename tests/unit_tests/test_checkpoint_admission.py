# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Admission control drains one policy model-server participant.

The gate closes atomically with admission.
Accepted requests remain counted until their ASGI task exits.
Cancelled attempts are fenced from later admission.
"""

import asyncio
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from nemo_gym._checkpoint import (
    CONTROL_URL_PREFIX,
    GATED_MODEL_ROUTE_SUFFIXES,
    MODEL_ADMISSION_URL_PREFIX,
    AdmissionLimiter,
    AdmissionMiddleware,
    AdmissionParkedError,
    AdmissionState,
    StaleAttemptError,
)
from nemo_gym.base_responses_api_model import BaseResponsesAPIModelConfig, SimpleResponsesAPIModel
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_correlation import ATTEMPT_INDEX_HEADER, ROLLOUT_ID_HEADER
from nemo_gym.server_utils import ServerClient


AUTH_HEADERS = {"authorization": "Bearer test-control-token"}


def _identity_headers(rollout_id: str, attempt_index: int = 0) -> dict[str, str]:
    return {
        ROLLOUT_ID_HEADER: rollout_id,
        ATTEMPT_INDEX_HEADER: str(attempt_index),
    }


# --- limiter ---


def test_close_with_nothing_inflight_pauses_immediately() -> None:
    limiter = AdmissionLimiter()
    limiter.close()
    assert limiter.state == AdmissionState.PAUSED


def test_drain_completes_when_last_inflight_releases() -> None:
    limiter = AdmissionLimiter()
    ticket = limiter.admit(rollout_id="4-2", attempt_index=0)
    limiter.close()
    assert limiter.state == AdmissionState.DRAINING
    limiter.release(ticket)
    assert limiter.state == AdmissionState.PAUSED


def test_new_root_operation_parks_while_draining() -> None:
    limiter = AdmissionLimiter()
    held = limiter.admit(rollout_id="4-2", attempt_index=0)
    limiter.close()
    with pytest.raises(AdmissionParkedError):
        limiter.admit(rollout_id="9-9", attempt_index=0)
    limiter.release(held)


def test_resume_reopens_admission() -> None:
    limiter = AdmissionLimiter()
    limiter.close()
    limiter.resume()
    assert limiter.state == AdmissionState.ACCEPTING
    limiter.release(limiter.admit(rollout_id="4-2", attempt_index=0))


def test_abort_inflight_tombstones_but_waits_for_request_exit() -> None:
    limiter = AdmissionLimiter()
    stuck = limiter.admit(rollout_id="7-1", attempt_index=2)
    limiter.close()
    aborted = limiter.abort_inflight("7-1", 2)
    assert aborted == [stuck.ticket_id]
    # Cancellation does not establish quiescence.
    # The request remains counted until its ASGI task exits.
    assert limiter.state == AdmissionState.DRAINING
    limiter.release(stuck)
    assert limiter.counts()["inflight_total"] == 0

    limiter.resume()
    with pytest.raises(StaleAttemptError):
        limiter.admit(rollout_id="7-1", attempt_index=2)
    limiter.release(limiter.admit(rollout_id="7-1", attempt_index=3))
    assert limiter.tombstones() == [("7-1", 2)]


@pytest.mark.asyncio
async def test_abort_cancels_request_task_before_drain_completes() -> None:
    limiter = AdmissionLimiter()
    admitted = asyncio.Event()

    async def operation() -> None:
        ticket = limiter.admit(
            rollout_id="7-1",
            attempt_index=2,
            task=asyncio.current_task(),
        )
        admitted.set()
        try:
            await asyncio.Event().wait()
        finally:
            limiter.release(ticket)

    task = asyncio.create_task(operation())
    await admitted.wait()
    limiter.close()
    limiter.abort_inflight("7-1", 2)
    with pytest.raises(asyncio.CancelledError):
        await task
    assert limiter.state == AdmissionState.PAUSED
    assert limiter.counts()["inflight_total"] == 0


@pytest.mark.asyncio
async def test_wait_for_drained_long_poll() -> None:
    limiter = AdmissionLimiter()
    ticket = limiter.admit(rollout_id="4-2", attempt_index=0)
    limiter.close()

    async def release_soon() -> None:
        await asyncio.sleep(0.01)
        limiter.release(ticket)

    releaser = asyncio.create_task(release_soon())
    assert await limiter.wait_for_drained(timeout_s=1.0)
    await releaser
    assert limiter.state == AdmissionState.PAUSED


@pytest.mark.asyncio
async def test_wait_for_drained_times_out_with_stragglers() -> None:
    limiter = AdmissionLimiter()
    held = limiter.admit(rollout_id="4-2", attempt_index=0)
    limiter.close()
    assert not await limiter.wait_for_drained(timeout_s=0.01)
    limiter.release(held)


@pytest.mark.asyncio
async def test_pause_admission_linearization_race() -> None:
    for index in range(100):
        limiter = AdmissionLimiter()
        start = asyncio.Event()
        admitted = asyncio.Event()
        release = asyncio.Event()
        side_effects = 0

        async def request() -> str:
            nonlocal side_effects
            await start.wait()
            try:
                ticket = limiter.admit(rollout_id=f"rollout-{index}", attempt_index=0)
            except AdmissionParkedError:
                return "parked"
            side_effects += 1
            admitted.set()
            await release.wait()
            limiter.release(ticket)
            return "admitted"

        async def pause() -> None:
            await start.wait()
            limiter.close()

        request_task = asyncio.create_task(request())
        pause_task = asyncio.create_task(pause())
        start.set()
        await pause_task

        if admitted.is_set():
            assert limiter.counts()["inflight_total"] == 1
            assert side_effects == 1
        else:
            assert limiter.counts()["inflight_total"] == 0
            assert side_effects == 0
        release.set()
        outcome = await request_task
        assert (outcome, side_effects) in {("admitted", 1), ("parked", 0)}


# --- middleware ---


def _gated_app(limiter: AdmissionLimiter) -> TestClient:
    app = FastAPI()

    @app.post("/v1/responses")
    async def responses() -> dict:
        return {"inflight": limiter.counts()["inflight_total"]}

    @app.get("/other")
    async def other() -> dict:
        return {"ok": True}

    app.add_middleware(AdmissionMiddleware, limiter=limiter, gated_suffixes=GATED_MODEL_ROUTE_SUFFIXES)
    return TestClient(app)


def test_middleware_counts_request_until_response_completes() -> None:
    limiter = AdmissionLimiter()
    body = _gated_app(limiter).post("/v1/responses").json()
    # The request was in flight while handled and released afterwards.
    assert body["inflight"] == 1
    assert limiter.counts()["inflight_total"] == 0


def test_middleware_parks_new_calls_when_closed() -> None:
    limiter = AdmissionLimiter()
    client = _gated_app(limiter)
    limiter.close()
    response = client.post("/v1/responses", headers=_identity_headers("4-2"))
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "checkpoint_parked"
    assert response.headers["retry-after"] == "1"


def test_middleware_rejects_stale_attempt() -> None:
    limiter = AdmissionLimiter()
    client = _gated_app(limiter)
    limiter.close()
    limiter.abort_inflight("7-1", 0)
    limiter.resume()
    response = client.post("/v1/responses", headers=_identity_headers("7-1"))
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "stale_attempt"


def test_middleware_resolves_blackbox_identity_from_capture_path() -> None:
    limiter = AdmissionLimiter()
    client = _gated_app(limiter)
    limiter.install_tombstone("7-1", 2)
    response = client.post("/ng-rollout/7-1-a2/v1/responses")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "stale_attempt"


def test_middleware_rejects_capture_path_identity_disagreement() -> None:
    limiter = AdmissionLimiter()
    response = _gated_app(limiter).post(
        "/ng-rollout/7-1-a2/v1/responses",
        headers=_identity_headers("7-1", 3),
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "execution_identity_mismatch"


def test_middleware_leaves_ungated_paths_open_while_paused() -> None:
    limiter = AdmissionLimiter()
    client = _gated_app(limiter)
    limiter.close()
    assert client.get("/other").json() == {"ok": True}


# --- model server routes ---


def _model_server(instance_role: str) -> SimpleResponsesAPIModel:
    class _Model(SimpleResponsesAPIModel):
        def checkpoint_control_auth_token(self) -> str:
            return "test-control-token"

        async def chat_completions(self, request):
            raise NotImplementedError

        async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
            return NeMoGymResponse.model_validate(
                {
                    "id": "resp-1",
                    "created_at": 0.0,
                    "model": "m",
                    "object": "response",
                    "output": [],
                    "parallel_tool_calls": True,
                    "tool_choice": "auto",
                    "tools": [],
                }
            )

    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = {}
    return _Model(
        config=BaseResponsesAPIModelConfig(host="", port=0, entrypoint="", name="policy", instance_role=instance_role),
        server_client=server_client,
    )


def test_policy_model_server_pause_drain_resume_cycle() -> None:
    client = TestClient(_model_server("policy").setup_webserver())

    capabilities = client.get(f"{CONTROL_URL_PREFIX}/capabilities").json()
    assert capabilities["instance_role"] == "policy"
    assert capabilities["admission_states"] == ["accepting", "draining", "paused"]

    # Generation works while accepting.
    assert client.post("/v1/responses", json={"input": "hi"}).status_code == 200

    pause_body = {"checkpoint_id": "ckpt-1", "deadline_ts": 4e9}
    pause = client.post(f"{MODEL_ADMISSION_URL_PREFIX}/pause", json=pause_body, headers=AUTH_HEADERS)
    assert pause.status_code == 200
    assert pause.json() == {
        "state": "paused",
        "workers": {"acknowledged": 1, "expected": 1},
        "inflight_total": 0,
        "waiters_total": 0,
    }

    # New generation parks; control routes stay reachable.
    parked = client.post("/v1/responses", json={"input": "hi"})
    assert parked.status_code == 409
    assert parked.json()["error"]["code"] == "checkpoint_parked"
    status = client.get(
        f"{MODEL_ADMISSION_URL_PREFIX}/status",
        params={"checkpoint_id": "ckpt-1", "deadline_ts": 4e9},
        headers=AUTH_HEADERS,
    ).json()
    assert status["state"] == "paused"
    assert status["per_worker"] == {"0": {"state": "paused", "inflight": 0}}

    # A duplicate pause replays; a competing checkpoint conflicts.
    replay = client.post(f"{MODEL_ADMISSION_URL_PREFIX}/pause", json=pause_body, headers=AUTH_HEADERS)
    assert replay.json() == pause.json()
    conflict = client.post(
        f"{MODEL_ADMISSION_URL_PREFIX}/pause",
        json={"checkpoint_id": "ckpt-2", "deadline_ts": 4e9},
        headers=AUTH_HEADERS,
    )
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "checkpoint_conflict"

    resume = client.post(
        f"{MODEL_ADMISSION_URL_PREFIX}/resume",
        json={"checkpoint_id": "ckpt-1", "deadline_ts": 4e9},
        headers=AUTH_HEADERS,
    )
    assert resume.json() == {
        "state": "accepting",
        "workers": {"acknowledged": 1, "expected": 1},
        "released_waiters": 0,
    }
    assert client.post("/v1/responses", json={"input": "hi"}).status_code == 200

    # The finished checkpoint id is stale for new operations; a new checkpoint proceeds.
    stale = client.post(f"{MODEL_ADMISSION_URL_PREFIX}/pause", json=pause_body, headers=AUTH_HEADERS)
    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "stale_checkpoint"
    assert client.post(
        f"{MODEL_ADMISSION_URL_PREFIX}/pause",
        json={"checkpoint_id": "ckpt-3", "deadline_ts": 4e9},
        headers=AUTH_HEADERS,
    ).json()["state"] in {"paused", "draining"}


def test_model_admission_requires_existing_control_bearer() -> None:
    client = TestClient(_model_server("policy").setup_webserver())
    response = client.post(
        f"{MODEL_ADMISSION_URL_PREFIX}/pause",
        json={"checkpoint_id": "ckpt-1", "deadline_ts": 4e9},
    )
    assert response.status_code == 401


def test_abort_inflight_tombstones_via_route() -> None:
    server = _model_server("policy")
    client = TestClient(server.setup_webserver())
    client.post(
        f"{MODEL_ADMISSION_URL_PREFIX}/pause",
        json={"checkpoint_id": "ckpt-1", "deadline_ts": 4e9},
        headers=AUTH_HEADERS,
    )

    abort = client.post(
        f"{MODEL_ADMISSION_URL_PREFIX}/abort_inflight",
        json={
            "checkpoint_id": "ckpt-1",
            "deadline_ts": 4e9,
            "rollout_id": "7-1",
            "attempt_index": 2,
        },
        headers=AUTH_HEADERS,
    )
    assert abort.status_code == 200
    status = client.get(
        f"{MODEL_ADMISSION_URL_PREFIX}/status",
        params={"checkpoint_id": "ckpt-1", "deadline_ts": 4e9},
        headers=AUTH_HEADERS,
    ).json()
    assert status["tombstones"] == [{"rollout_id": "7-1", "attempt_index": 2}]

    client.post(
        f"{MODEL_ADMISSION_URL_PREFIX}/resume",
        json={"checkpoint_id": "ckpt-1", "deadline_ts": 4e9},
        headers=AUTH_HEADERS,
    )
    stale = client.post("/v1/responses", json={"input": "hi"}, headers=_identity_headers("7-1", 2))
    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "stale_attempt"
    fresh = client.post("/v1/responses", json={"input": "hi"}, headers=_identity_headers("7-1", 3))
    assert fresh.status_code == 200


def test_auxiliary_model_server_never_gates_and_rejects_pause() -> None:
    client = TestClient(_model_server("auxiliary").setup_webserver())

    capabilities = client.get(f"{CONTROL_URL_PREFIX}/capabilities").json()
    assert capabilities["instance_role"] == "auxiliary"
    assert capabilities["admission_states"] == ["accepting"]

    pause = client.post(
        f"{MODEL_ADMISSION_URL_PREFIX}/pause",
        json={"checkpoint_id": "ckpt-1", "deadline_ts": 4e9},
        headers=AUTH_HEADERS,
    )
    assert pause.status_code == 409
    assert pause.json()["error"]["code"] == "not_a_policy_instance"

    # Judge traffic keeps flowing: there is no admission middleware to close.
    assert client.post("/v1/responses", json={"input": "grade"}).status_code == 200
