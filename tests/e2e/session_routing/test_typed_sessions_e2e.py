# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An environment server's seeds and closes reach the worker that holds the session, without a cookie.

The interactive browser resources server runs with 2 workers and real headless browsers.
An environment server names each session by `resources_session_id` and may seed, retry the seed,
and close it without the session cookie.
Each of those requests must act on the one worker that opened the session's browser.

Skipped unless NEMO_GYM_MULTIWORKER_E2E=1.
Also skipped when the server has no virtual environment of its own,
which `gym env test --resources-server interactive_browser` creates.
"""

import asyncio
import contextlib
import glob
import hashlib
import os
import re
import signal
import socket
import subprocess
import time
import uuid
from typing import Iterator

import aiohttp
import psutil
import pytest
import yaml

from nemo_gym.runtime_dir import RUNTIME_ROOT
from nemo_gym.session_routing import SESSION_CLAIMS_DIR
from tests.e2e.session_routing.test_session_routing_e2e import REPO, _free_port, _wait_healthy


SERVER_DIR = REPO / "resources_servers" / "interactive_browser"
SERVER_PYTHON = SERVER_DIR / ".venv" / "bin" / "python"
WORKERS = 2
SESSIONS = int(os.getenv("NEMO_GYM_MULTIWORKER_E2E_BROWSER_SESSIONS", "16"))

pytestmark = [
    pytest.mark.skipif(os.getenv("NEMO_GYM_MULTIWORKER_E2E") != "1", reason="set NEMO_GYM_MULTIWORKER_E2E=1 to run"),
    pytest.mark.skipif(not SERVER_PYTHON.exists(), reason="interactive_browser has no virtual environment"),
]

_EPISODE = {"rollout_id": "rollout", "attempt": 0}
_OTHER_EPISODE = {"rollout_id": "other", "attempt": 0}
_TASK = {"taskset": "browser", "task_id": "task-1"}


@pytest.fixture(scope="module")
def browser_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[str, str]]:
    log = tmp_path_factory.mktemp("typed_sessions") / "interactive_browser.log"
    port = _free_port()
    config = {
        "dry_run": False,
        "use_absolute_ip": False,
        "skip_venv_if_present": True,
        "head_server": {"host": "127.0.0.1", "port": _free_port()},
        "browser": {
            "resources_servers": {
                "interactive_browser": {
                    "entrypoint": "app.py",
                    "host": "127.0.0.1",
                    "port": port,
                    "domain": "agent",
                    "verified": False,
                    "description": "browser",
                    "num_workers": WORKERS,
                }
            }
        },
    }
    env = os.environ | {
        "NEMO_GYM_CONFIG_DICT": yaml.safe_dump(config),
        "NEMO_GYM_CONFIG_PATH": "browser",
        "PYTHONPATH": str(REPO),
        "RAY_TMPDIR": "/tmp",
    }
    proc = subprocess.Popen(
        [str(SERVER_PYTHON), "app.py"],
        cwd=SERVER_DIR,
        env=env,
        stdout=log.open("a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        url = f"http://127.0.0.1:{port}"
        _wait_healthy(proc, f"{url}/health", log)
        _wait_for_workers(proc.pid)
        [claims_dir] = glob.glob(os.path.join(RUNTIME_ROOT, f"ng-{proc.pid}-*", SESSION_CLAIMS_DIR))
        yield url, claims_dir
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def _wait_for_workers(pid: int, timeout: float = 120) -> None:
    """Wait until every worker's private session socket accepts connections."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        workers = [child for child in psutil.Process(pid).children() if "resource_tracker" not in child.cmdline()]
        accepting = 0
        for path in glob.glob(os.path.join(RUNTIME_ROOT, f"ng-{pid}-*", "*.sock")):
            if not re.fullmatch(r"[0-9a-f]{32}\.sock", os.path.basename(path)):
                continue
            with socket.socket(socket.AF_UNIX) as probe:
                probe.settimeout(1)
                with contextlib.suppress(OSError):
                    probe.connect(path)
                    accepting += 1
        if len(workers) >= WORKERS and accepting >= WORKERS:
            return
        time.sleep(0.2)
    raise TimeoutError("the interactive browser server's workers did not start")


def _seed(session_id: str, episode: dict = _EPISODE) -> dict:
    return {
        "resources_session_id": session_id,
        "episode_id": episode,
        "task_id": _TASK,
        "task_data": {"initial_url": "about:blank"},
    }


def _close(session_id: str) -> dict:
    return {"resources_session_id": session_id, "episode_id": _EPISODE}


def _claimant(claims_dir: str, session_id: str) -> str:
    """The worker that claims the session, read from its claim file."""
    with open(os.path.join(claims_dir, hashlib.sha256(session_id.encode("utf-8")).hexdigest())) as claim:
        return claim.read()


async def _post(client: aiohttp.ClientSession, url: str, body: dict, cookies: dict | None = None) -> tuple[int, str]:
    async with client.post(url, json=body, cookies=cookies or {}) as response:
        return response.status, await response.text()


async def _observe(client: aiohttp.ClientSession, url: str, cookies: dict) -> dict:
    async with client.post(f"{url}/browser_observe", cookies=cookies) as response:
        assert response.status == 200, await response.text()
        return await response.json()


def test_cookieless_seed_retry_and_close_act_on_the_worker_that_holds_the_browser(
    browser_server: tuple[str, str],
) -> None:
    url, claims_dir = browser_server

    owners: list[str] = []

    async def one(client: aiohttp.ClientSession, session_id: str) -> list[str]:
        problems = []
        async with client.post(f"{url}/seed_session", json=_seed(session_id)) as response:
            assert response.status == 200, await response.text()
            cookies = {name: morsel.value for name, morsel in response.cookies.items()}
        owners.append(_claimant(claims_dir, session_id))
        # A retried seed for the same episode returns the session it already has.
        status, text = await _post(client, f"{url}/seed_session", _seed(session_id))
        if status != 200:
            problems.append(f"retried seed answered {status}: {text}")
        # The holder refuses a seed of the same ID for another episode;
        # a worker that never saw the ID would open a second browser instead.
        status, _ = await _post(client, f"{url}/seed_session", _seed(session_id, _OTHER_EPISODE))
        if status == 200:
            problems.append("a seed for another episode opened a second browser")
        if (await _observe(client, url, cookies)).get("error"):
            problems.append("the session's browser was missing before the close")
        status, text = await _post(client, f"{url}/close_session", _close(session_id))
        if status != 200:
            problems.append(f"close answered {status}: {text}")
        # The close released the browser on the worker that holds it.
        if not (await _observe(client, url, cookies)).get("error"):
            problems.append("the close left the browser open")
        # A late seed of the closed ID opens nothing.
        status, _ = await _post(client, f"{url}/seed_session", _seed(session_id))
        if status == 200:
            problems.append("a seed after the close opened a browser")
        return problems

    async def run_all() -> dict[str, list[str]]:
        async with aiohttp.ClientSession(cookie_jar=aiohttp.DummyCookieJar()) as client:
            session_ids = [f"browser-{uuid.uuid4().hex}" for _ in range(SESSIONS)]
            results = await asyncio.gather(*(one(client, session_id) for session_id in session_ids))
            return {session_id: problems for session_id, problems in zip(session_ids, results) if problems}

    assert asyncio.run(run_all()) == {}
    # The seeds were spread over both workers, so the cookieless requests often reached the other one first.
    assert len(set(owners)) == WORKERS
