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
"""Real server processes with several uvicorn workers keep each session on the worker that holds it.

The stateful counter resources server runs with 4 workers behind the legacy ``/run`` relay and Simple Agent,
against a fake inference backend.
Skipped unless NEMO_GYM_MULTIWORKER_E2E=1; under a minute.
"""

import asyncio
import contextlib
import glob
import json
import os
import random
import re
import signal
import socket
import subprocess
import sys
import time
from base64 import b64decode
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import aiohttp
import psutil
import pytest
import requests
import yaml

from nemo_gym.runtime_dir import RUNTIME_ROOT


pytestmark = pytest.mark.skipif(
    os.getenv("NEMO_GYM_MULTIWORKER_E2E") != "1", reason="set NEMO_GYM_MULTIWORKER_E2E=1 to run"
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
WORKERS = 4
EPISODES = int(os.getenv("NEMO_GYM_MULTIWORKER_E2E_EPISODES", "200"))
RESOURCES_COOKIE = "StatefulCounterResourcesServer___resources"
COUNTER_TOOL = {
    "type": "function",
    "name": "increment_counter",
    "description": "",
    "parameters": {
        "type": "object",
        "properties": {"count": {"type": "integer", "description": ""}},
        "required": ["count"],
        "additionalProperties": False,
    },
    "strict": True,
}


# A port the OS picks for port 0 comes from its ephemeral range (from 32768 on Linux),
# which every other process draws from too, Ray among them, so one could take it before the server binds it.
# Ports below that range, and above Ray's worker ports (10002 to 19999), are never picked automatically.
_PORTS = range(20000, 32768)
_handed_out: set[int] = set()


def _free_port() -> int:
    """A port no other process will take before a server started by this test binds it."""
    while len(_handed_out) < len(_PORTS):
        port = random.choice(_PORTS)
        if port in _handed_out:
            continue
        _handed_out.add(port)
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
        return port
    raise RuntimeError("every test port has been handed out")


def _server(server_type: str, implementation: str, **config: Any) -> dict:
    return {
        server_type: {implementation: {"entrypoint": "app.py", "host": "127.0.0.1", "port": _free_port(), **config}}
    }


def _port(entry: dict) -> int:
    return next(iter(next(iter(entry.values())).values()))["port"]


def _wait_healthy(proc: subprocess.Popen, url: str, log: Path, timeout: float = 120) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"{url} exited with {proc.returncode}; see {log}")
        with contextlib.suppress(requests.RequestException):
            if requests.get(url, timeout=1).status_code < 500:
                return
        time.sleep(0.2)
    raise TimeoutError(f"{url} did not become healthy; see {log}")


class Deployment:
    """The legacy relay over Simple Agent, the counter resources server with 4 workers, and a fake backend."""

    def __init__(self, work_dir: Path) -> None:
        self.log_dir = work_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.backend_port = _free_port()
        ref = lambda server_type, name: {"type": server_type, "name": name}  # noqa: E731
        self.config: dict[str, Any] = {
            "dry_run": False,
            "use_absolute_ip": False,
            "skip_venv_if_present": True,
            "head_server": {"host": "127.0.0.1", "port": _free_port()},
            "policy_model": _server(
                "responses_api_models",
                "vllm_model",
                base_url=f"http://127.0.0.1:{self.backend_port}/v1",
                api_key="dummy",
                model="fake-model",
                return_token_id_information=False,
                uses_reasoning_parser=False,
            ),
            "resources": _server(
                "resources_servers",
                "example_session_state_mgmt",
                domain="agent",
                verified=False,
                description="counter",
                num_workers=WORKERS,
                expose_tools_over_mcp=True,
            ),
            "agent": _server(
                "responses_api_agents",
                "simple_agent",
                model_server=ref("responses_api_models", "policy_model"),
                resources_server=ref("resources_servers", "resources"),
            ),
            "environment": _server(
                "environment_servers", "legacy_agent", agent_server=ref("responses_api_agents", "agent")
            ),
        }
        self.dirs = {
            "policy_model": REPO / "responses_api_models/vllm_model",
            "resources": REPO / "resources_servers/example_session_state_mgmt",
            "agent": REPO / "responses_api_agents/simple_agent",
            "environment": REPO / "environment_servers/legacy_agent",
        }
        self.procs: dict[str, subprocess.Popen] = {}

    def url(self, name: str) -> str:
        return f"http://127.0.0.1:{_port(self.config[name])}"

    def start(self) -> None:
        log = self.log_dir / "backend.log"
        self.procs["backend"] = subprocess.Popen(
            [sys.executable, str(HERE / "fake_backend.py"), str(self.backend_port)],
            stdout=log.open("a"),
            stderr=subprocess.STDOUT,
        )
        _wait_healthy(self.procs["backend"], f"http://127.0.0.1:{self.backend_port}/v1/models", log)
        for name, cwd in self.dirs.items():
            env = os.environ | {
                "NEMO_GYM_CONFIG_DICT": yaml.safe_dump(self.config),
                "NEMO_GYM_CONFIG_PATH": name,
                "PYTHONPATH": str(REPO),
                "RAY_TMPDIR": "/tmp",
            }
            self.procs[name] = subprocess.Popen(
                [sys.executable, "app.py"],
                cwd=cwd,
                env=env,
                stdout=(self.log_dir / f"{name}.log").open("a"),
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        for name in self.dirs:
            _wait_healthy(self.procs[name], f"{self.url(name)}/health", self.log_dir / f"{name}.log")
        self._wait_for_workers()

    def _wait_for_workers(self, *, exclude: int | None = None, timeout: float = 60) -> None:
        deadline = time.time() + timeout
        while not self._workers_serving(exclude):
            if time.time() > deadline:
                raise TimeoutError("the resources server's workers did not start")
            time.sleep(0.2)

    def _workers_serving(self, exclude: int | None) -> bool:
        """Every live worker has finished startup: its private session socket accepts connections."""
        if len([worker for worker in self.resources_workers() if worker.pid != exclude]) < WORKERS:
            return False
        pattern = os.path.join(RUNTIME_ROOT, f"ng-{self.procs['resources'].pid}-*", "*.sock")
        accepting = 0
        for path in glob.glob(pattern):
            # Only the workers' session sockets, which are named by routing ID.
            if not re.fullmatch(r"[0-9a-f]{32}\.sock", os.path.basename(path)):
                continue
            with socket.socket(socket.AF_UNIX) as probe:
                probe.settimeout(1)
                try:
                    probe.connect(path)
                except OSError:
                    continue  # A killed worker's socket file.
            accepting += 1
        return accepting >= WORKERS

    def resources_workers(self) -> list[psutil.Process]:
        # uvicorn's workers, not the multiprocessing helper it may also start.
        children = psutil.Process(self.procs["resources"].pid).children()
        return [child for child in children if "resource_tracker" not in " ".join(child.cmdline())]

    def resources_worker_of(self, owner: str) -> psutil.Process:
        """The live worker whose private session socket is named by ``owner``, its routing ID."""
        for worker in self.resources_workers():
            for connection in worker.net_connections(kind="unix"):
                if os.path.basename(connection.laddr or "") == f"{owner}.sock":
                    return worker
        raise LookupError(f"no live resources worker owns {owner}")

    def stop(self) -> None:
        for name, proc in self.procs.items():
            with contextlib.suppress(ProcessLookupError):
                if name == "backend":
                    proc.kill()
                else:
                    os.killpg(proc.pid, signal.SIGTERM)
        for name, proc in self.procs.items():
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()


@pytest.fixture(scope="module")
def deployment(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Deployment]:
    deployment = Deployment(tmp_path_factory.mktemp("session_routing"))
    try:
        deployment.start()
        yield deployment
    finally:
        deployment.stop()


def counter_row(index: int) -> dict:
    return {
        "responses_create_params": {
            "input": [{"role": "user", "content": "add 1 then add 2"}],
            "tools": [COUNTER_TOOL],
        },
        "initial_count": index,
        "expected_count": index + 3,
    }


async def _post(client: aiohttp.ClientSession, url: str, body: dict, cookies: dict) -> aiohttp.ClientResponse:
    response = await client.post(url, json=body, cookies=cookies)
    await response.read()
    return response


def _cookies(response: aiohttp.ClientResponse) -> dict[str, str]:
    return {name: morsel.value for name, morsel in response.cookies.items()}


def _owner(cookies: dict[str, str]) -> str:
    """The worker stamped in a resources session cookie (signature not checked)."""
    payload = cookies[RESOURCES_COOKIE].split(".")[0]
    return json.loads(b64decode(payload + "=" * (-len(payload) % 4)))["nemo_gym_worker"]


def test_counter_episodes_through_run_all_score(deployment: Deployment) -> None:
    """Every increment and the verification reach the worker that seeded the episode's counter."""

    async def run_all() -> list[float]:
        async with aiohttp.ClientSession(cookie_jar=aiohttp.DummyCookieJar()) as client:

            async def one(index: int) -> float:
                response = await _post(client, f"{deployment.url('environment')}/run", counter_row(index), {})
                assert response.status == 200, await response.text()
                return (await response.json())["reward"]

            return await asyncio.gather(*(one(index) for index in range(EPISODES)))

    rewards = asyncio.run(run_all())
    assert Counter(rewards) == Counter({1.0: EPISODES})


async def _seed(client: aiohttp.ClientSession, url: str, index: int) -> dict[str, str]:
    response = await _post(client, f"{url}/seed_session", {"initial_count": index}, {})
    assert response.status == 200, await response.text()
    return _cookies(response)


async def _verify(client: aiohttp.ClientSession, url: str, index: int, cookies: dict) -> aiohttp.ClientResponse:
    body = {"responses_create_params": {"input": []}, "response": _EMPTY_RESPONSE, "expected_count": index}
    return await _post(client, f"{url}/verify", body, cookies)


_EMPTY_RESPONSE = {
    "id": "resp",
    "created_at": 0,
    "model": "fake-model",
    "object": "response",
    "output": [],
    "parallel_tool_calls": False,
    "tool_choice": "auto",
    "tools": [],
}


def test_state_stored_at_seed_is_found_at_verify(deployment: Deployment) -> None:
    """The swe_rebench shape: seed stores per-session state, and verify must find it on any worker."""
    url = deployment.url("resources")

    async def run_all() -> list[tuple[float, dict[str, str]]]:
        async with aiohttp.ClientSession(cookie_jar=aiohttp.DummyCookieJar()) as client:

            async def one(index: int) -> tuple[float, dict[str, str]]:
                cookies = await _seed(client, url, index)
                response = await _verify(client, url, index, cookies)
                assert response.status == 200, await response.text()
                return (await response.json())["reward"], cookies

            return await asyncio.gather(*(one(index) for index in range(EPISODES)))

    results = asyncio.run(run_all())
    assert Counter(reward for reward, _ in results) == Counter({1.0: EPISODES})
    owners = {_owner(cookies) for _, cookies in results}
    # Sessions were spread over several workers, so most verifies landed on a worker that did not seed them.
    assert len(owners) > 1


MCP_TOKEN_HEADER = "X-NeMo-Gym-Session-Token"


async def _mcp_call(client: aiohttp.ClientSession, url: str, token: str, name: str, arguments: dict) -> dict:
    response = await client.post(
        f"{url}/mcp",
        headers={"accept": "application/json, text/event-stream", MCP_TOKEN_HEADER: token},
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}},
    )
    assert response.status == 200, await response.text()
    result = (await response.json())["result"]
    assert result.get("isError") is not True, result
    return json.loads(result["content"][0]["text"])


def test_mcp_clients_without_cookies_reach_their_session(deployment: Deployment) -> None:
    """A CLI harness in a sandbox sends only the MCP session token, never the Gym session cookie."""
    url = deployment.url("resources")

    async def run_all() -> list[int]:
        async with aiohttp.ClientSession(cookie_jar=aiohttp.DummyCookieJar()) as client:

            async def one(index: int) -> int:
                seed = await _post(client, f"{url}/seed_session", {"initial_count": index}, {})
                token = (await seed.json())["mcp"]["headers"][MCP_TOKEN_HEADER]
                await _mcp_call(client, url, token, "increment_counter", {"count": 1})
                await _mcp_call(client, url, token, "increment_counter", {"count": 2})
                count = (await _mcp_call(client, url, token, "get_counter_value", {}))["count"]
                return int(count == index + 3)

            return await asyncio.gather(*(one(index) for index in range(EPISODES)))

    correct = asyncio.run(run_all())
    assert Counter(correct) == Counter({1: EPISODES})


def test_session_of_an_exited_worker_gets_an_explicit_error(deployment: Deployment) -> None:
    url = deployment.url("resources")

    async def seed_all() -> list[dict[str, str]]:
        async with aiohttp.ClientSession(cookie_jar=aiohttp.DummyCookieJar()) as client:
            return await asyncio.gather(*(_seed(client, url, index) for index in range(EPISODES)))

    async def verify_all(sessions: list[dict[str, str]]) -> tuple[list[aiohttp.ClientResponse], list[float]]:
        # A new client, so no pooled connection to the killed worker is reused.
        async with aiohttp.ClientSession(cookie_jar=aiohttp.DummyCookieJar()) as client:
            responses = await asyncio.gather(
                *(_verify(client, url, index, cookies) for index, cookies in enumerate(sessions))
            )
            return responses, [(await r.json())["reward"] for r in responses if r.status == 200]

    sessions = asyncio.run(seed_all())
    # The kernel spreads connections over the workers, but not necessarily over all of them,
    # so kill a worker that holds sessions while another worker holds the rest.
    owners = Counter(_owner(cookies) for cookies in sessions)
    assert len(owners) > 1, owners
    # uvicorn replaces the worker; the replacement does not hold the killed worker's sessions.
    victim = deployment.resources_worker_of(owners.most_common(1)[0][0])
    victim.send_signal(signal.SIGKILL)
    victim.wait(timeout=10)
    deployment._wait_for_workers(exclude=victim.pid)
    responses, rewards = asyncio.run(verify_all(sessions))
    lost_owners = {_owner(cookies) for cookies, r in zip(sessions, responses) if r.status == 410}
    assert len(lost_owners) == 1, Counter(r.status for r in responses)
    assert {r.status for r in responses} == {200, 410}
    # Every session of a live worker still verifies correctly.
    assert rewards and set(rewards) == {1.0}
    assert all(_owner(cookies) in lost_owners for cookies, r in zip(sessions, responses) if r.status != 200)
