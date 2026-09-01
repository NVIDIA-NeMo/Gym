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
"""Multi-worker admission coordination over a real Unix-domain socket.

The pinned behaviors: a control request answered by the coordinator reflects
every worker, not one arbitrary worker's in-process state; pause returns
only after every live worker acknowledged the closed admission; the
aggregate is paused only when all workers acked AND the summed in-flight
count is zero; a missing worker is an error, never an implicit zero; and a
worker that joins late receives the current admission state at registration
so it can never serve traffic against a stale one.

Each simulated worker runs a real ``AdmissionLimiter`` plus a
``WorkerAdmissionAgent`` connected to the coordinator's socket — the same
objects a uvicorn worker process would run, on one event loop for a fully
CPU-local simulation.
"""

import asyncio
import os
import shutil
import socket
import sys
import tempfile
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from nemo_gym._checkpoint import (
    MODEL_ADMISSION_URL_PREFIX,
    AdmissionCoordinator,
    AdmissionLimiter,
    AdmissionParkedError,
    AdmissionState,
    ControlCapabilities,
    MultiProcessCapability,
    StaleAttemptError,
    WorkerAdmissionAgent,
    build_coordinator_control_app,
)


class _Pool:
    def __init__(
        self, coordinator: AdmissionCoordinator, workers: list[tuple[AdmissionLimiter, WorkerAdmissionAgent]]
    ):
        self.coordinator = coordinator
        self.workers = workers
        self.app = build_coordinator_control_app(
            coordinator,
            capabilities=ControlCapabilities(
                component="responses_api_models",
                name="policy",
                multi_process=MultiProcessCapability(mode="coordinator", num_workers=coordinator.expected_workers),
                instance_role="policy",
            ),
            ack_timeout_s=2.0,
        )

    def limiter(self, index: int) -> AdmissionLimiter:
        return self.workers[index][0]

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://coordinator")

    async def pause(self, client: httpx.AsyncClient, checkpoint_id: str = "ckpt-1") -> httpx.Response:
        return await client.post(
            f"{MODEL_ADMISSION_URL_PREFIX}/pause", json={"checkpoint_id": checkpoint_id, "deadline_ts": 4e9}
        )


@pytest_asyncio.fixture
async def sock_dir():
    # A dedicated short directory: Unix-domain socket paths have a hard
    # length limit (104 bytes on macOS) that pytest tmp_path can exceed.
    path = Path(tempfile.mkdtemp(prefix="ngckpt-"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


async def _start_pool(sock_dir: Path, *, expected: int = 3, connect: int = 3) -> _Pool:
    coordinator = AdmissionCoordinator(sock_dir / "control.sock", expected_workers=expected)
    await coordinator.start()
    workers = []
    for index in range(connect):
        limiter = AdmissionLimiter()
        agent = WorkerAdmissionAgent(coordinator.socket_path, f"w{index}", limiter, pid=1000 + index)
        await agent.start()
        workers.append((limiter, agent))
    await coordinator.wait_until(lambda s: s["workers"]["live"] == connect, timeout_s=2.0)
    return _Pool(coordinator, workers)


async def _stop_pool(pool: _Pool) -> None:
    for _, agent in pool.workers:
        await agent.stop()
    await pool.coordinator.stop()


@pytest.mark.asyncio
async def test_workers_register_and_report(sock_dir) -> None:
    pool = await _start_pool(sock_dir)
    try:
        status = pool.coordinator.status()
        assert status["state"] == "accepting"
        assert status["workers"]["live"] == 3
        assert status["missing_workers"] == 0
        assert status["inflight_total"] == 0
    finally:
        await _stop_pool(pool)


@pytest.mark.asyncio
async def test_pause_closes_every_worker_and_waits_for_acks(sock_dir) -> None:
    pool = await _start_pool(sock_dir)
    try:
        async with pool.client() as client:
            response = await pool.pause(client)
            assert response.status_code == 200
            body = response.json()
            assert body["workers"] == {"acknowledged": 3, "expected": 3}
            assert body["state"] == "paused"
            assert body["inflight_total"] == 0

        # Every worker's own limiter is now closed: new work parks locally
        # without any further coordinator involvement.
        for index in range(3):
            with pytest.raises(AdmissionParkedError):
                pool.limiter(index).admit(rollout_id="9-9", attempt_index=0)
    finally:
        await _stop_pool(pool)


@pytest.mark.asyncio
async def test_drain_aggregates_inflight_across_workers(sock_dir) -> None:
    pool = await _start_pool(sock_dir)
    try:
        held = pool.limiter(2).admit(rollout_id="4-2", attempt_index=0)
        # Let the counter report reach the coordinator.
        await pool.coordinator.wait_until(lambda s: s["inflight_total"] == 1, timeout_s=2.0)

        async with pool.client() as client:
            paused = await pool.pause(client)
            body = paused.json()
            # One worker still holds an accepted operation: the service is
            # draining, not paused, and the aggregate says whose fault it is.
            assert body["state"] == "draining"
            assert body["inflight_total"] == 1

            long_poll = asyncio.create_task(
                client.get(
                    f"{MODEL_ADMISSION_URL_PREFIX}/status",
                    params={"checkpoint_id": "ckpt-1", "wait_state": "paused", "timeout_s": 2.0},
                )
            )
            await asyncio.sleep(0.05)
            pool.limiter(2).release(held)
            status = (await long_poll).json()
            assert status["state"] == "paused"
            assert status["inflight_total"] == 0
    finally:
        await _stop_pool(pool)


@pytest.mark.asyncio
async def test_missing_worker_is_an_error_not_a_zero(sock_dir) -> None:
    pool = await _start_pool(sock_dir, expected=3, connect=2)
    try:
        status = pool.coordinator.status()
        assert status["missing_workers"] == 1

        async with pool.client() as client:
            response = await pool.pause(client)
            assert response.status_code == 409
            assert response.json()["error"]["code"] == "missing_workers"
        reopened = await pool.coordinator.wait_until(
            lambda status: all(worker[0].state == AdmissionState.ACCEPTING for worker in pool.workers),
            timeout_s=2.0,
        )
        assert reopened["state"] == "accepting"
    finally:
        await _stop_pool(pool)


@pytest.mark.asyncio
async def test_worker_disconnect_shows_up_in_status(sock_dir) -> None:
    pool = await _start_pool(sock_dir)
    try:
        await pool.workers[1][1].stop()
        status = await pool.coordinator.wait_until(lambda s: s["workers"]["live"] == 2, timeout_s=2.0)
        assert status["missing_workers"] == 1
    finally:
        await _stop_pool(pool)


@pytest.mark.asyncio
async def test_tombstone_broadcast_fences_attempt_on_every_worker(sock_dir) -> None:
    pool = await _start_pool(sock_dir)
    try:
        async with pool.client() as client:
            await pool.pause(client)
            abort = await client.post(
                f"{MODEL_ADMISSION_URL_PREFIX}/abort_inflight",
                json={
                    "checkpoint_id": "ckpt-1",
                    "deadline_ts": 4e9,
                    "rollout_id": "7-1-a2",
                    "attempt_index": 2,
                },
            )
            assert abort.status_code == 200
            resume = await client.post(
                f"{MODEL_ADMISSION_URL_PREFIX}/resume",
                json={"checkpoint_id": "ckpt-1", "deadline_ts": 4e9},
            )
            assert resume.json()["state"] == "accepting"

        for index in range(3):
            with pytest.raises(StaleAttemptError):
                pool.limiter(index).admit(rollout_id="7-1-a2", attempt_index=2)
            pool.limiter(index).release(pool.limiter(index).admit(rollout_id="7-1-a3", attempt_index=3))
    finally:
        await _stop_pool(pool)


@pytest.mark.asyncio
async def test_resume_reopens_every_worker(sock_dir) -> None:
    pool = await _start_pool(sock_dir)
    try:
        async with pool.client() as client:
            await pool.pause(client)
            resume = await client.post(
                f"{MODEL_ADMISSION_URL_PREFIX}/resume",
                json={"checkpoint_id": "ckpt-1", "deadline_ts": 4e9},
            )
            assert resume.json()["workers"] == {"acknowledged": 3, "expected": 3}

        for index in range(3):
            assert pool.limiter(index).state == AdmissionState.ACCEPTING
    finally:
        await _stop_pool(pool)


@pytest.mark.asyncio
async def test_late_joining_worker_receives_current_state(sock_dir) -> None:
    pool = await _start_pool(sock_dir, expected=2, connect=1)
    try:
        await pool.coordinator.close_admission("ckpt-1")
        await pool.coordinator.wait_until(lambda s: s["workers"]["acknowledged"] == 1, timeout_s=2.0)

        late_limiter = AdmissionLimiter()
        late_agent = WorkerAdmissionAgent(pool.coordinator.socket_path, "late", late_limiter)
        await late_agent.start()
        pool.workers.append((late_limiter, late_agent))
        await pool.coordinator.wait_until(
            lambda s: s["workers"]["live"] == 2 and s["workers"]["acknowledged"] == 2, timeout_s=2.0
        )

        # The late worker installed the closed admission before serving
        # anything: it can never admit traffic against a stale state.
        with pytest.raises(AdmissionParkedError):
            late_limiter.admit(rollout_id="9-9", attempt_index=0)
    finally:
        await _stop_pool(pool)


@pytest.mark.asyncio
async def test_coordinator_capabilities_declare_coordinator_mode(sock_dir) -> None:
    pool = await _start_pool(sock_dir)
    try:
        async with pool.client() as client:
            body = (await client.get("/ng-control/v1/capabilities")).json()
            assert body["multi_process"] == {"mode": "coordinator", "num_workers": 3}
            assert body["instance_role"] == "policy"
    finally:
        await _stop_pool(pool)


def _unused_tcp_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


@pytest.mark.asyncio
async def test_real_two_worker_uvicorn_pool_closes_as_one_service(sock_dir) -> None:
    coordinator = AdmissionCoordinator(sock_dir / "control.sock", expected_workers=2)
    await coordinator.start()
    release_path = sock_dir / "release"
    port = _unused_tcp_port()
    env = {
        **os.environ,
        "NG_CHECKPOINT_COORDINATOR_SOCKET": str(coordinator.socket_path),
        "NG_CHECKPOINT_RELEASE_PATH": str(release_path),
    }
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "uvicorn",
        "tests.unit_tests.checkpoint_uvicorn_worker_app:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--workers",
        "2",
        "--log-level",
        "warning",
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    held_requests: list[asyncio.Task] = []
    try:
        registered = await coordinator.wait_until(lambda status: status["workers"]["live"] == 2, timeout_s=10.0)
        assert registered["missing_workers"] == 0

        async def hold_request() -> httpx.Response:
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=10.0) as client:
                return await client.post("/hold")

        held_requests = [asyncio.create_task(hold_request()) for _ in range(8)]
        inflight = await coordinator.wait_until(lambda status: status["inflight_total"] == 8, timeout_s=5.0)
        assert inflight["inflight_total"] == 8
        assert all(worker["inflight"] > 0 for worker in inflight["per_worker"].values())

        app = build_coordinator_control_app(
            coordinator,
            capabilities=ControlCapabilities(
                component="responses_api_models",
                name="policy",
                multi_process=MultiProcessCapability(mode="coordinator", num_workers=2),
                instance_role="policy",
            ),
            ack_timeout_s=2.0,
        )
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://coordinator") as control:
            pause = await control.post(
                f"{MODEL_ADMISSION_URL_PREFIX}/pause",
                json={"checkpoint_id": "ckpt-real-workers", "deadline_ts": 4e9},
            )
            assert pause.status_code == 200
            assert pause.json()["workers"] == {"acknowledged": 2, "expected": 2}
            assert pause.json()["state"] == "draining"

            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as data:
                parked = [await data.post("/hold") for _ in range(8)]
            assert all(response.status_code == 409 for response in parked)
            assert all(response.json()["error"]["code"] == "checkpoint_parked" for response in parked)

            release_path.touch()
            assert all(response.status_code == 200 for response in await asyncio.gather(*held_requests))
            status = await control.get(
                f"{MODEL_ADMISSION_URL_PREFIX}/status",
                params={"checkpoint_id": "ckpt-real-workers", "wait_state": "paused", "timeout_s": 5.0},
            )
            assert status.json()["state"] == "paused"
            assert status.json()["inflight_total"] == 0
    finally:
        release_path.touch(exist_ok=True)
        if held_requests:
            await asyncio.gather(*held_requests, return_exceptions=True)
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
        await coordinator.stop()
