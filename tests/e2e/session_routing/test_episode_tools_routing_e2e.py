# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real worker processes of a BaseEnvironmentServerWithTools: each tool call reaches the worker that runs its episode.

Every episode calls its own tools through the server's public port, over direct HTTP or MCP, on a new connection
each time, so the kernel hands each request to any of the 4 workers.
Skipped unless NEMO_GYM_MULTIWORKER_E2E=1; under a minute.
"""

import asyncio
import contextlib
import glob
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import aiohttp
import psutil
import pytest
import yaml

from nemo_gym.runtime_dir import RUNTIME_ROOT
from tests.e2e.session_routing.test_session_routing_e2e import _free_port, _port, _server, _wait_healthy


pytestmark = pytest.mark.skipif(
    os.getenv("NEMO_GYM_MULTIWORKER_E2E") != "1", reason="set NEMO_GYM_MULTIWORKER_E2E=1 to run"
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SERVERS = HERE / "episode_tools_servers"
WORKERS = 4
EPISODES = int(os.getenv("NEMO_GYM_MULTIWORKER_E2E_EPISODES", "200"))
CALLS = 8
# An MCP episode also initializes and lists its tools, each a request of its own.
MCP_EPISODES = int(os.getenv("NEMO_GYM_MULTIWORKER_E2E_MCP_EPISODES", "50"))
MCP_CALLS = 4


class Deployment:
    """The tool counter environment server, with 4 workers."""

    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir
        self.log = work_dir / "environment.log"
        self.config: dict[str, Any] = {
            "dry_run": False,
            "use_absolute_ip": False,
            "skip_venv_if_present": True,
            "head_server": {"host": "127.0.0.1", "port": _free_port()},
            "environment": _server(
                "environment_servers",
                "tool_counter",
                num_workers=WORKERS,
                cleanup_timeout_seconds=5,
                tool_transports=["direct_http", "mcp"],
            ),
        }
        self.proc: subprocess.Popen | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{_port(self.config['environment'])}"

    def start(self) -> None:
        env = os.environ | {
            "NEMO_GYM_CONFIG_DICT": yaml.safe_dump(self.config),
            "NEMO_GYM_CONFIG_PATH": "environment",
            # The test server is imported as environment_servers.tool_counter.
            "PYTHONPATH": os.pathsep.join([str(REPO), str(SERVERS)]),
            "RAY_TMPDIR": "/tmp",
        }
        self.proc = subprocess.Popen(
            [sys.executable, "app.py"],
            cwd=SERVERS / "environment_servers" / "tool_counter",
            env=env,
            stdout=self.log.open("a"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _wait_healthy(self.proc, f"{self.url}/health", self.log)
        self.wait_for_workers()

    def workers(self) -> list[psutil.Process]:
        # uvicorn's workers, not the multiprocessing helper it may also start.
        children = psutil.Process(self.proc.pid).children()
        return [child for child in children if "resource_tracker" not in " ".join(child.cmdline())]

    def wait_for_workers(self, *, exclude: int | None = None, timeout: float = 60) -> None:
        """Wait until every live worker's private session socket accepts connections."""
        deadline = time.time() + timeout
        while True:
            live = [worker for worker in self.workers() if worker.pid != exclude]
            accepting = 0
            for path in glob.glob(os.path.join(RUNTIME_ROOT, f"ng-{self.proc.pid}-*", "*.sock")):
                if not re.fullmatch(r"[0-9a-f]{32}\.sock", os.path.basename(path)):
                    continue
                with socket.socket(socket.AF_UNIX) as probe:
                    probe.settimeout(1)
                    with contextlib.suppress(OSError):
                        probe.connect(path)
                        accepting += 1
            if len(live) >= WORKERS and accepting >= WORKERS:
                return
            if time.time() > deadline:
                raise TimeoutError(f"the workers did not start; see {self.log}")
            time.sleep(0.2)

    def stop(self) -> None:
        if self.proc is None:
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(self.proc.pid, signal.SIGTERM)
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait()


@pytest.fixture(scope="module")
def deployment(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Deployment]:
    deployment = Deployment(tmp_path_factory.mktemp("episode_tools_routing"))
    try:
        deployment.start()
        yield deployment
    finally:
        deployment.stop()


def _run_body(index: int, **task: Any) -> dict:
    return {
        "episode_id": {"rollout_id": f"{index}-0"},
        "task": {"task_id": {"taskset": "counter", "task_id": str(index)}, "task_input": task},
    }


def test_every_tool_call_runs_on_the_worker_that_runs_its_episode(deployment: Deployment) -> None:
    async def run_all() -> list[dict]:
        async with aiohttp.ClientSession(cookie_jar=aiohttp.DummyCookieJar()) as client:

            async def one(index: int) -> dict:
                async with client.post(f"{deployment.url}/run", json=_run_body(index, calls=CALLS)) as response:
                    assert response.status == 200, await response.text()
                    return await response.json()

            return await asyncio.gather(*(one(index) for index in range(EPISODES)))

    results = [response["result"] for response in asyncio.run(run_all())]

    assert Counter(status for result in results for status in result["statuses"]) == Counter({200: EPISODES * CALLS})
    assert all(set(result["ran_on"]) == {result["owner_pid"]} for result in results)
    assert all(result["total"] == CALLS for result in results)
    # The episodes ran on several workers, so most of their calls reached another worker first.
    assert len({result["owner_pid"] for result in results}) > 1


def test_every_mcp_request_runs_on_the_worker_that_runs_its_episode(deployment: Deployment) -> None:
    """Initialize, tools/list and tools/call each reach a random worker; each is routed by the token header."""

    async def run_all() -> list[dict]:
        async with aiohttp.ClientSession(cookie_jar=aiohttp.DummyCookieJar()) as client:

            async def one(index: int) -> dict:
                body = _run_body(2 * 10**5 + index, calls=MCP_CALLS, transport="mcp")
                async with client.post(f"{deployment.url}/run", json=body) as response:
                    assert response.status == 200, await response.text()
                    return await response.json()

            return await asyncio.gather(*(one(index) for index in range(MCP_EPISODES)))

    results = [response["result"] for response in asyncio.run(run_all())]

    # A tools/list that reached another worker would list nothing.
    assert all(result["listed"] == ["add"] for result in results)
    assert Counter(status for result in results for status in result["statuses"]) == Counter(
        {200: MCP_EPISODES * MCP_CALLS}
    )
    assert all(set(result["ran_on"]) == {result["owner_pid"]} for result in results)
    assert len({result["owner_pid"] for result in results}) > 1


def test_a_tool_call_for_an_exited_workers_episode_gets_410(deployment: Deployment, tmp_path: Path) -> None:
    grant_file = tmp_path / "grant.json"

    async def scenario() -> tuple[int, dict]:
        async with aiohttp.ClientSession(cookie_jar=aiohttp.DummyCookieJar()) as client:
            # An episode that keeps serving its tools until its worker is killed.
            episode = asyncio.create_task(
                client.post(
                    f"{deployment.url}/run", json=_run_body(10**6, grant_file=str(grant_file), hold_seconds=120)
                )
            )
            while not grant_file.exists():
                await asyncio.sleep(0.05)
            grant = json.loads(grant_file.read_text())
            owner = psutil.Process(grant["pid"])
            owner.send_signal(signal.SIGKILL)
            owner.wait(timeout=60)
            with contextlib.suppress(aiohttp.ClientError):
                await episode
            await asyncio.to_thread(deployment.wait_for_workers, exclude=grant["pid"])
        # A new client, so no pooled connection to the killed worker is reused.
        async with aiohttp.ClientSession(cookie_jar=aiohttp.DummyCookieJar()) as client:
            async with client.post(f"{grant['base_url']}/add", json={"amount": 1}, headers=grant["headers"]) as reply:
                return reply.status, await reply.json()

    status, body = asyncio.run(scenario())

    assert status == 410 and "no longer running" in body["detail"]
