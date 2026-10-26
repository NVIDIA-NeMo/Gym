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
"""Stress session routing with real server processes, a fake inference backend, and thousands of sessions.

Skipped unless NEMO_GYM_MULTIWORKER_STRESS=1. Run it from the repository root:

    NEMO_GYM_MULTIWORKER_STRESS=1 RAY_TMPDIR=/tmp pytest -s tests/e2e/session_routing/test_session_routing_stress.py

The deployment is the NOOA counter resources server with 8 workers (plus a 4-worker copy that serves its tools
over MCP, and a 1-worker copy to compare latency against), Simple Agent with 4 workers,
and the single_agent_turn environment server, both its episode request route and its flat-row wrapper.
The tests run in order, and each one prints its numbers:

- every episode of a few thousand concurrent ones scores 1.0, on each route;
- cookieless seeds and closes (a seed that fails without a cookie, a seed whose reply is lost and retried,
  a delayed seed after a close) never create a session twice or leave one behind;
- MCP tool calls that carry only the session token reach their session;
- routing's added latency against one worker, next to the cost of a claim lookup alone;
- once the run is over and the close window has passed, no claim is left;
- killing a worker gives its sessions a 410, other sessions carry on, and its claims go with the server.

NEMO_GYM_STRESS_EPISODES (default 4000) sets the episodes per route, and NEMO_GYM_STRESS_SESSIONS (default 1000)
the sessions of each other test. NEMO_GYM_STRESS_REPORT names a JSON file for all the numbers.
"""

import asyncio
import contextlib
import glob
import hashlib
import json
import os
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Optional

import aiohttp
import psutil
import pytest
import yaml

from nemo_gym.runtime_dir import RUNTIME_ROOT
from tests.e2e.session_routing.test_session_routing_e2e import COUNTER_TOOL, _free_port, _port, _server, _wait_healthy


pytestmark = pytest.mark.skipif(
    os.getenv("NEMO_GYM_MULTIWORKER_STRESS") != "1", reason="set NEMO_GYM_MULTIWORKER_STRESS=1 to run"
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
EPISODES = int(os.getenv("NEMO_GYM_STRESS_EPISODES", "4000"))
SESSIONS = int(os.getenv("NEMO_GYM_STRESS_SESSIONS", "1000"))
# Short, so the claims of finished sessions end within the test.
CLOSE_WINDOW_SECONDS = 5.0
STRESS_SERVER = HERE / "stress_servers" / "resources_servers" / "session_routing_stress"
TOKEN_HEADER = "X-NeMo-Gym-Session-Token"
# Sent on a request that must not reuse a kept-alive connection, so it can reach any worker.
CLOSE = {"Connection": "close"}
REPORT: dict[str, Any] = {}


def _ref(server_type: str, name: str) -> dict:
    return {"type": server_type, "name": name}


class StressDeployment:
    def __init__(self, work_dir: Path) -> None:
        self.log_dir = work_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.backend_port = _free_port()
        stress = dict(
            domain="agent",
            verified=False,
            description="session routing stress",
            session_close_retry_window_seconds=CLOSE_WINDOW_SECONDS,
        )
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
                num_workers=4,
            ),
            "resources": _server("resources_servers", "session_routing_stress", num_workers=8, **stress),
            "mcp_resources": _server(
                "resources_servers", "session_routing_stress", num_workers=4, expose_tools_over_mcp=True, **stress
            ),
            "single_resources": _server("resources_servers", "session_routing_stress", num_workers=1, **stress),
            "agent": _server(
                "responses_api_agents",
                "simple_agent",
                model_server=_ref("responses_api_models", "policy_model"),
                resources_server=_ref("resources_servers", "resources"),
                num_workers=4,
                session_close_retry_window_seconds=CLOSE_WINDOW_SECONDS,
            ),
            "environment": _server(
                "environment_servers",
                "single_agent_turn",
                resources_server=_ref("resources_servers", "resources"),
                agent_server=_ref("responses_api_agents", "agent"),
                resources_tool_transports=["direct_http"],
                default_episode_timeout_seconds=1200,
                cleanup_timeout_seconds=60,
            ),
            "flat_environment": _server(
                "environment_servers",
                "single_agent_turn_legacy",
                resources_server=_ref("resources_servers", "resources"),
                agent_server=_ref("responses_api_agents", "agent"),
                resources_tool_transports=["direct_http"],
                default_episode_timeout_seconds=1200,
                cleanup_timeout_seconds=60,
            ),
        }
        self.dirs = {
            "policy_model": REPO / "responses_api_models/vllm_model",
            "resources": STRESS_SERVER,
            "mcp_resources": STRESS_SERVER,
            "single_resources": STRESS_SERVER,
            "agent": REPO / "responses_api_agents/simple_agent",
            "environment": REPO / "environment_servers/single_agent_turn",
            "flat_environment": REPO / "environment_servers/single_agent_turn_legacy",
        }
        self.procs: dict[str, subprocess.Popen] = {}
        self.runtime_dirs: dict[str, str] = {}

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
                # The stress server is imported as resources_servers.session_routing_stress.
                "PYTHONPATH": os.pathsep.join([str(REPO), str(HERE / "stress_servers")]),
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
            _wait_healthy(self.procs[name], f"{self.url(name)}/health", self.log_dir / f"{name}.log", timeout=180)
        for name in ("resources", "mcp_resources", "agent"):
            self.wait_for_workers(name)

    def workers(self, name: str) -> int:
        return next(iter(next(iter(self.config[name].values())).values()))["num_workers"]

    def runtime_dir(self, name: str) -> Optional[str]:
        found = glob.glob(os.path.join(RUNTIME_ROOT, f"ng-{self.procs[name].pid}-*"))
        return found[0] if found else None

    def sockets(self, name: str) -> list[str]:
        """The private sockets of the server's live workers."""
        directory = self.runtime_dir(name)
        live = []
        for path in glob.glob(os.path.join(directory or "/nonexistent", "*.sock")):
            with contextlib.suppress(OSError):
                with socket.socket(socket.AF_UNIX) as probe:
                    probe.settimeout(1)
                    probe.connect(path)
                    live.append(path)
        return live

    def wait_for_workers(self, name: str, timeout: float = 120) -> None:
        deadline = time.time() + timeout
        while len(self.sockets(name)) < self.workers(name):
            assert time.time() < deadline, f"the workers of {name} did not start"
            time.sleep(0.2)

    def claims(self, name: str) -> Counter:
        """The claim files in the server's runtime directory, counted by the worker that holds them."""
        directory = self.runtime_dir(name)
        holders: Counter = Counter()
        if directory is None:
            return holders
        for path in glob.glob(os.path.join(directory, "*", "*")):
            if path.endswith(".tmp"):
                continue
            with contextlib.suppress(OSError):
                holders[Path(path).read_text()] += 1
        return holders

    def claimant(self, name: str, session_id: str) -> Optional[str]:
        digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
        for path in glob.glob(os.path.join(self.runtime_dir(name) or "/nonexistent", "*", digest)):
            return Path(path).read_text()
        return None

    async def stats(self, name: str) -> dict[str, dict]:
        """Each live worker's report, by routing ID, asked over its private socket so that worker answers."""
        reports = {}
        for path in self.sockets(name):
            async with aiohttp.ClientSession(connector=aiohttp.UnixConnector(path)) as client:
                async with client.get("http://localhost/stress_stats") as response:
                    report = await response.json()
            reports[os.path.basename(path).removesuffix(".sock")] = report
        return reports

    def stop(self) -> None:
        for name, proc in self.procs.items():
            self.stop_one(name)

    def stop_one(self, name: str) -> None:
        proc = self.procs[name]
        with contextlib.suppress(ProcessLookupError):
            if name == "backend":
                proc.kill()
            else:
                os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


class ClaimSampler:
    """Counts the claims of each routing server every half second, for the claim count over time."""

    def __init__(self, deployment: StressDeployment, names: tuple[str, ...]) -> None:
        self.deployment = deployment
        self.names = names
        self.samples: list[dict[str, Any]] = []
        self.start = time.monotonic()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(0.5):
            self.sample()

    def sample(self) -> dict[str, Any]:
        row = {"t": round(time.monotonic() - self.start, 2)}
        for name in self.names:
            row[name] = sum(self.deployment.claims(name).values())
        self.samples.append(row)
        return row

    def mark(self, label: str) -> None:
        self.samples.append({"t": round(time.monotonic() - self.start, 2), "phase": label})

    def __enter__(self) -> "ClaimSampler":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        self._thread.join()


@pytest.fixture(scope="module")
def deployment() -> Iterator[StressDeployment]:
    work_dir = Path(tempfile.mkdtemp(prefix="ng-stress-"))
    deployment = StressDeployment(work_dir)
    REPORT["log_dir"] = str(deployment.log_dir)
    try:
        deployment.start()
        yield deployment
    finally:
        deployment.stop()
        path = os.getenv("NEMO_GYM_STRESS_REPORT")
        if path:
            Path(path).write_text(json.dumps(REPORT, indent=1))


@pytest.fixture(scope="module")
def sampler(deployment: StressDeployment) -> Iterator[ClaimSampler]:
    with ClaimSampler(deployment, ("resources", "mcp_resources", "agent")) as sampler:
        yield sampler
    REPORT["claims_over_time"] = sampler.samples


def _client(timeout: float = 1800) -> aiohttp.ClientSession:
    return aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=0),
        cookie_jar=aiohttp.DummyCookieJar(),
        timeout=aiohttp.ClientTimeout(total=timeout),
    )


def _cookies(response: aiohttp.ClientResponse) -> dict[str, str]:
    return {name: morsel.value for name, morsel in response.cookies.items()}


async def _post(client: aiohttp.ClientSession, url: str, body: Any, cookies: Optional[dict] = None, **kwargs):
    async with client.post(url, json=body, cookies=cookies or {}, **kwargs) as response:
        payload = await response.read()
        return response.status, (json.loads(payload) if payload[:1] in (b"{", b"[") else payload), _cookies(response)


def _task_data(index: int) -> dict:
    return {"initial_count": index, "expected_count": index + 3}


def _create_params() -> dict:
    return {"input": [{"role": "user", "content": "add 1 then add 2"}], "tools": [COUNTER_TOOL]}


def _episode(index: int) -> dict:
    return {
        "episode_id": {"rollout_id": f"episode-{index}", "attempt": 0},
        "task": {
            "task_id": {"taskset": "session-routing-stress", "task_id": str(index)},
            "task_input": {"responses_create_params": _create_params(), "task_data": _task_data(index)},
        },
    }


def _flat_row(index: int) -> dict:
    return {
        "responses_create_params": _create_params(),
        **_task_data(index),
        "_ng_rollout_id": f"flat-{index}",
        "_ng_task_index": index,
    }


def _typed_seed(session_id: str, index: int, **task_data: Any) -> dict:
    return {
        "resources_session_id": session_id,
        "episode_id": {"rollout_id": session_id},
        "task_id": {"taskset": "session-routing-stress", "task_id": str(index)},
        "task_data": _task_data(index) | task_data,
    }


def _close(session_id: str) -> dict:
    return {"resources_session_id": session_id, "episode_id": {"rollout_id": session_id}}


def _percentiles(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "n": len(ordered),
        "p50_ms": round(ordered[len(ordered) // 2] * 1000, 3),
        "p99_ms": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))] * 1000, 3),
        "max_ms": round(ordered[-1] * 1000, 3),
        "mean_ms": round(statistics.mean(ordered) * 1000, 3),
    }


def _report(name: str, value: Any) -> None:
    REPORT[name] = value
    print(f"\n[stress] {name}: {json.dumps(value)}")


async def _no_session_twice(deployment: StressDeployment, name: str) -> tuple[dict, Counter]:
    stats = await deployment.stats(name)
    created = Counter(session_id for report in stats.values() for session_id in report["created"])
    return stats, Counter({session_id: n for session_id, n in created.items() if n > 1})


@pytest.mark.parametrize("route", ["episode", "flat_row"])
def test_every_episode_scores_at_scale(deployment: StressDeployment, sampler: ClaimSampler, route: str) -> None:
    sampler.mark(f"scale {route} start")
    url = deployment.url("environment" if route == "episode" else "flat_environment")

    async def run_all() -> tuple[list, float]:
        async with _client() as client:

            async def one(index: int) -> Any:
                body = _episode(index) if route == "episode" else _flat_row(index)
                status, payload, _ = await _post(client, f"{url}/run", body)
                if status != 200:
                    return f"http {status}: {str(payload)[:200]}"
                result = payload.get("result") if route == "episode" else payload
                if result is None:
                    return f"failure: {str(payload.get('failure'))[:300]}"
                if "_ng_failure_message" in result:
                    return f"failure: {result['_ng_failure_message'][:300]}"
                return result["reward"]

            start = time.monotonic()
            results = await asyncio.gather(*(one(index) for index in range(EPISODES)))
            return results, time.monotonic() - start

    results, elapsed = asyncio.run(run_all())
    outcomes = Counter(r if isinstance(r, float) else r.split(":")[0] for r in results)
    stats, twice = asyncio.run(_no_session_twice(deployment, "resources"))
    _report(
        f"scale_{route}",
        {
            "episodes": EPISODES,
            "seconds": round(elapsed, 1),
            "outcomes": {str(k): v for k, v in outcomes.items()},
            "first_failures": [r for r in results if not isinstance(r, float)][:3],
            "resources_sessions_by_worker": sorted(len(r["created"]) for r in stats.values()),
            "sessions_created_on_two_workers": len(twice),
        },
    )
    sampler.mark(f"scale {route} end")
    assert outcomes == Counter({1.0: EPISODES})
    assert not twice


def test_cookieless_seeds_and_closes_never_duplicate_or_leak(deployment: StressDeployment, sampler: ClaimSampler):
    sampler.mark("cookieless start")
    url = deployment.url("resources")

    async def episode(client: aiohttp.ClientSession, session_id: str, index: int, cookies: dict) -> bool:
        for count in (1, 2):
            status, _, _ = await _post(client, f"{url}/increment_counter", {"count": count}, cookies)
            assert status == 200
        status, payload, _ = await _post(client, f"{url}/get_counter_value", {}, cookies)
        return status == 200 and payload["count"] == index + 3

    async def failed_seed(client: aiohttp.ClientSession, index: int) -> str:
        """The seed fails without a cookie, the cleanup close is cookieless, and a new attempt succeeds."""
        session_id = f"failed-{index}"
        status, _, cookies = await _post(
            client, f"{url}/seed_session", _typed_seed(session_id, index, fail_seed_once=True)
        )
        if status != 500 or cookies:
            return f"seed {status} {cookies}"
        status, _, _ = await _post(client, f"{url}/close_session", _close(session_id))
        if status != 200:
            return f"cleanup close {status}"
        retry = f"{session_id}-attempt-1"
        status, _, cookies = await _post(client, f"{url}/seed_session", _typed_seed(retry, index))
        if status != 200 or not await episode(client, retry, index, cookies):
            return f"retry {status}"
        status, _, _ = await _post(client, f"{url}/close_session", _close(retry))
        return "ok" if status == 200 else f"close {status}"

    async def lost_seed_reply(client: aiohttp.ClientSession, index: int) -> str:
        """The first replies are lost: one is dropped after it arrives, one is cut off; the retry gets through."""
        session_id = f"lost-{index}"
        await _post(client, f"{url}/seed_session", _typed_seed(session_id, index))
        with contextlib.suppress(asyncio.TimeoutError, aiohttp.ClientError):
            async with client.post(
                f"{url}/seed_session",
                json=_typed_seed(session_id, index),
                timeout=aiohttp.ClientTimeout(total=0.0005 * (index % 8)),
            ) as response:
                await response.read()
        status, _, cookies = await _post(client, f"{url}/seed_session", _typed_seed(session_id, index))
        if status != 200 or not await episode(client, session_id, index, cookies):
            return f"retry {status}"
        status, _, _ = await _post(client, f"{url}/close_session", _close(session_id))
        return "ok" if status == 200 else f"close {status}"

    async def delayed_seed(client: aiohttp.ClientSession, index: int) -> str:
        """The cleanup close reaches the server before the seed it cleans up after."""
        session_id = f"delayed-{index}"
        status, _, _ = await _post(client, f"{url}/close_session", _close(session_id))
        if status != 200:
            return f"close {status}"
        status, _, _ = await _post(client, f"{url}/seed_session", _typed_seed(session_id, index))
        return "ok" if status == 409 else f"delayed seed {status}"

    async def run_all() -> dict[str, Counter]:
        async with _client() as client:
            outcomes = {}
            for kind in (failed_seed, lost_seed_reply, delayed_seed):
                results = await asyncio.gather(*(kind(client, index) for index in range(SESSIONS)))
                outcomes[kind.__name__] = Counter(results)
            return outcomes

    outcomes = asyncio.run(run_all())
    stats, twice = asyncio.run(_no_session_twice(deployment, "resources"))
    live = {routing_id: len(report["live"]) for routing_id, report in stats.items()}
    _report(
        "cookieless",
        {
            "sessions_per_kind": SESSIONS,
            "outcomes": {kind: dict(counts) for kind, counts in outcomes.items()},
            "sessions_created_on_two_workers": len(twice),
            "live_sessions_by_worker": live,
            "live_sessions": sorted(session_id for report in stats.values() for session_id in report["live"])[:20],
            "claims_by_worker": dict(deployment.claims("resources")),
        },
    )
    sampler.mark("cookieless end")
    assert all(counts == Counter({"ok": SESSIONS}) for counts in outcomes.values()), outcomes
    assert not twice
    assert sum(live.values()) == 0


def test_mcp_tool_calls_with_only_the_session_token(deployment: StressDeployment, sampler: ClaimSampler) -> None:
    sampler.mark("mcp start")
    url = deployment.url("mcp_resources")

    async def call(client: aiohttp.ClientSession, token: str, name: str, arguments: dict) -> dict:
        status, payload, _ = await _post(
            client,
            f"{url}/mcp",
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}},
            headers={"accept": "application/json, text/event-stream", TOKEN_HEADER: token},
        )
        assert status == 200, payload
        result = payload["result"]
        assert result.get("isError") is not True, result
        return json.loads(result["content"][0]["text"])

    async def one(client: aiohttp.ClientSession, index: int) -> bool:
        session_id = f"mcp-{index}"
        status, payload, _ = await _post(client, f"{url}/seed_session", _typed_seed(session_id, index))
        assert status == 200, payload
        token = payload["resources_tools"]["headers"][TOKEN_HEADER]
        await call(client, token, "increment_counter", {"count": 1})
        await call(client, token, "increment_counter", {"count": 2})
        count = (await call(client, token, "get_counter_value", {}))["count"]
        status, _, _ = await _post(client, f"{url}/close_session", _close(session_id))
        return count == index + 3 and status == 200

    async def run_all() -> list[bool]:
        async with _client() as client:
            return await asyncio.gather(*(one(client, index) for index in range(SESSIONS)))

    correct = asyncio.run(run_all())
    stats, twice = asyncio.run(_no_session_twice(deployment, "mcp_resources"))
    _report(
        "mcp",
        {
            "sequences": SESSIONS,
            "workers": deployment.workers("mcp_resources"),
            "correct": sum(correct),
            "sessions_by_worker": sorted(len(r["created"]) for r in stats.values()),
            "sessions_created_on_two_workers": len(twice),
        },
    )
    sampler.mark("mcp end")
    assert all(correct) and not twice


def test_untyped_seeds_that_never_read_the_session_keep_their_session_on_one_worker(
    deployment: StressDeployment, sampler: ClaimSampler
) -> None:
    """An agent's own /run seeds without a session ID, and only the later tool calls read the session."""
    sampler.mark("untyped seeds start")
    url = deployment.url("resources")
    calls = 5

    async def one(client: aiohttp.ClientSession) -> bool:
        status, payload, cookies = await _post(client, f"{url}/seed_session", {})
        assert status == 200 and cookies, payload
        for _ in range(calls):
            # A new connection each time, so each call can reach any worker.
            status, payload, _ = await _post(client, f"{url}/plain_add", {"count": 1}, cookies, headers=CLOSE)
            assert status == 200, payload
        status, payload, _ = await _post(client, f"{url}/plain_end", {}, cookies, headers=CLOSE)
        return status == 200 and payload["count"] == calls

    async def run_all() -> list[bool]:
        async with _client() as client:
            return await asyncio.gather(*(one(client) for _ in range(SESSIONS)))

    correct = asyncio.run(run_all())
    stats = asyncio.run(deployment.stats("resources"))
    created = Counter(session_id for report in stats.values() for session_id in report["plain_created"])
    split = [session_id for session_id, n in created.items() if n > 1]
    left = sum(len(report["plain_live"]) for report in stats.values())
    _report(
        "untyped_seeds",
        {
            "sessions": SESSIONS,
            "calls_per_session": calls + 1,
            "correct": sum(correct),
            "sessions_created_on_two_workers": len(split),
            "sessions_left": left,
        },
    )
    sampler.mark("untyped seeds end")
    assert all(correct) and not split and not left


def test_routing_overhead_against_one_worker(deployment: StressDeployment, sampler: ClaimSampler) -> None:
    """Sequential tool calls over one kept-alive connection, so the accepting worker never changes."""
    sampler.mark("overhead start")
    requests = 2000

    async def latencies(name: str) -> dict[str, list[float]]:
        url = deployment.url(name)
        connector = aiohttp.TCPConnector(limit=1)
        async with aiohttp.ClientSession(connector=connector, cookie_jar=aiohttp.DummyCookieJar()) as client:
            async with client.get(f"{url}/stress_stats") as response:
                accepting = (await response.json())["routing_id"]
            sessions = []
            # A new connection for each seed, so the sessions spread over the workers.
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(force_close=True)) as seeder:
                for index in range(64):
                    session_id = f"overhead-{name}-{index}"
                    status, _, cookies = await _post(seeder, f"{url}/seed_session", _typed_seed(session_id, index))
                    assert status == 200
                    sessions.append((session_id, cookies))
            by_path: dict[str, list[float]] = {"local": [], "forwarded": []}
            for request in range(requests):
                session_id, cookies = sessions[request % len(sessions)]
                start = time.perf_counter()
                async with client.post(f"{url}/get_counter_value", json={}, cookies=cookies) as response:
                    await response.read()
                    assert response.status == 200
                elapsed = time.perf_counter() - start
                claimant = deployment.claimant(name, session_id)
                by_path["local" if claimant in (None, accepting) else "forwarded"].append(elapsed)
            for session_id, _ in sessions:
                await _post(client, f"{url}/close_session", _close(session_id))
            return by_path

    one = asyncio.run(latencies("single_resources"))
    eight = asyncio.run(latencies("resources"))
    single = _percentiles(one["local"] + one["forwarded"])
    forwarded = _percentiles(eight["forwarded"])
    local = _percentiles(eight["local"]) if eight["local"] else None
    lookup = _lookup_cost()
    _report(
        "overhead",
        {
            "one_worker": single,
            "eight_workers_claimant_accepted": local,
            "eight_workers_forwarded": forwarded,
            "added_by_forwarding_ms": {
                key: round(forwarded[key] - single[key], 3) for key in ("p50_ms", "p99_ms", "max_ms")
            },
            "claim_lookup_alone_us": lookup,
        },
    )
    sampler.mark("overhead end")


def _lookup_cost() -> dict[str, float]:
    """One claim lookup by a worker that does not hold the claim: open and read a small file."""
    directory = tempfile.mkdtemp(prefix="ng-stress-lookup-", dir="/tmp")
    try:
        paths = []
        for index in range(1000):
            path = os.path.join(directory, hashlib.sha256(str(index).encode()).hexdigest())
            Path(path).write_text("a" * 32)
            paths.append(path)
        samples = []
        for request in range(20000):
            start = time.perf_counter_ns()
            with open(paths[request % len(paths)], encoding="ascii") as claim:
                claim.read()
            samples.append(time.perf_counter_ns() - start)
        samples.sort()
        return {
            "p50": samples[len(samples) // 2] / 1000,
            "p99": samples[int(len(samples) * 0.99)] / 1000,
            "max": samples[-1] / 1000,
        }
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_no_claim_remains_once_the_close_window_has_passed(deployment: StressDeployment, sampler: ClaimSampler):
    sampler.mark("idle start")
    # Every session above was closed, verified or failed, so each claim ends a window after its last request.
    deadline = time.monotonic() + CLOSE_WINDOW_SECONDS + 30
    while time.monotonic() < deadline:
        row = sampler.sample()
        if all(row[name] == 0 for name in sampler.names):
            break
        time.sleep(0.5)
    waited = round(time.monotonic() - (deadline - CLOSE_WINDOW_SECONDS - 30), 1)
    remaining = {name: dict(deployment.claims(name)) for name in sampler.names}
    _report("lifetime", {"seconds_waited_after_run": waited, "claims_remaining": remaining})
    sampler.mark("idle end")
    assert all(not holders for holders in remaining.values()), remaining


def test_a_lost_worker_loses_only_its_sessions(deployment: StressDeployment, sampler: ClaimSampler) -> None:
    sampler.mark("worker loss start")
    url = deployment.url("resources")

    async def seed_all() -> list[tuple[str, dict]]:
        async with _client() as client:

            async def one(index: int) -> tuple[str, dict]:
                session_id = f"loss-{index}"
                status, _, cookies = await _post(client, f"{url}/seed_session", _typed_seed(session_id, index))
                assert status == 200
                status, _, _ = await _post(client, f"{url}/increment_counter", {"count": 1}, cookies)
                assert status == 200
                return session_id, cookies

            return await asyncio.gather(*(one(index) for index in range(SESSIONS)))

    async def finish_all(sessions: list[tuple[str, dict]]) -> list[tuple[str, int, Optional[int], int]]:
        async with _client() as client:

            async def one(session_id: str, cookies: dict) -> tuple[str, int, Optional[int], int]:
                status, _, _ = await _post(client, f"{url}/increment_counter", {"count": 2}, cookies)
                count = None
                if status == 200:
                    _, payload, _ = await _post(client, f"{url}/get_counter_value", {}, cookies)
                    count = payload["count"]
                close_status, _, _ = await _post(client, f"{url}/close_session", _close(session_id))
                return session_id, status, count, close_status

            return await asyncio.gather(*(one(s, c) for s, c in sessions))

    sessions = asyncio.run(seed_all())
    owner = {session_id: deployment.claimant("resources", session_id) for session_id, _ in sessions}
    victim_id, victim_sessions = Counter(owner.values()).most_common(1)[0]
    stats = asyncio.run(deployment.stats("resources"))
    victim = psutil.Process(stats[victim_id]["pid"])
    victim.send_signal(signal.SIGKILL)
    # Uvicorn may be in its 30-second health check when the worker dies; only its parent can reap it.
    victim.wait(timeout=60)
    deployment.wait_for_workers("resources")
    results = asyncio.run(finish_all(sessions))
    lost = [r for r in results if owner[r[0]] == victim_id]
    kept = [r for r in results if owner[r[0]] != victim_id]
    index_of = {session_id: index for index, (session_id, _) in enumerate(sessions)}
    claims_before_shutdown = deployment.claims("resources")
    runtime_dir = deployment.runtime_dir("resources")
    deployment.stop_one("resources")
    _report(
        "worker_loss",
        {
            "sessions": SESSIONS,
            "sessions_of_the_lost_worker": victim_sessions,
            "lost_statuses": dict(Counter(f"{status}/{close}" for _, status, _, close in lost)),
            "kept_statuses": dict(Counter(f"{status}/{close}" for _, status, _, close in kept)),
            "kept_correct": sum(count == index_of[s] + 3 for s, _, count, _ in kept),
            "claims_of_the_lost_worker_before_shutdown": claims_before_shutdown.get(victim_id, 0),
            "runtime_dir_after_shutdown": os.path.exists(runtime_dir) if runtime_dir else None,
        },
    )
    sampler.mark("worker loss end")
    assert {(status, close) for _, status, _, close in lost} == {(410, 410)}
    assert all(status == close == 200 and count == index_of[s] + 3 for s, status, count, close in kept)
    assert runtime_dir is not None and not os.path.exists(runtime_dir)
