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
import asyncio
import multiprocessing
import os
import socket
import stat
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock

import aiohttp
import pytest
from aiohttp import web

import nemo_gym.client.http_client
from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymChatCompletionCreateParamsNonStreaming
from responses_api_models.openai_model.app import SimpleModelServer, SimpleModelServerConfig
from responses_api_models.openai_model.call_budget import (
    UpstreamCallBudget,
    UpstreamCallBudgetExhaustedError,
    _record,
    reserve_call,
    reserved_calls,
)


def _server(tmp_path: Path, max_calls: int, **kwargs) -> SimpleModelServer:
    kwargs.setdefault("upstream_max_num_tries", 1)
    kwargs.setdefault("openai_base_url", "https://api.openai.com/v1")
    config = SimpleModelServerConfig(
        host="0.0.0.0",
        port=8081,
        openai_api_key="dummy_key",  # pragma: allowlist secret
        openai_model="dummy_model",
        entrypoint="",
        name="judge_model",
        upstream_call_budget={"ledger_path": str(tmp_path / "calls.ledger"), "max_calls": max_calls},
        **kwargs,
    )
    return SimpleModelServer(config=config, server_client=MagicMock(spec=ServerClient, global_config_dict={}))


def _reserve_many(ledger_path: str, max_calls: int, tries: int, results) -> None:
    budget = UpstreamCallBudget(ledger_path=ledger_path, max_calls=max_calls)
    granted = 0
    for _ in range(tries):
        try:
            reserve_call(budget)
            granted += 1
        except UpstreamCallBudgetExhaustedError:
            pass
    results.put(granted)


def test_a_new_ledger_is_created_durably_and_reservations_survive_a_restart(tmp_path: Path) -> None:
    ledger = tmp_path / "calls.ledger"
    budget = UpstreamCallBudget(ledger_path=str(ledger), max_calls=3)
    assert [reserve_call(budget), reserve_call(budget)] == [1, 2]
    assert (tmp_path / "calls.ledger.created").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["calls.ledger", "calls.ledger.created"]

    # A restarted process builds its budget again from config and continues from the ledger.
    restarted = UpstreamCallBudget(ledger_path=str(ledger), max_calls=3)
    assert reserve_call(restarted) == 3
    with pytest.raises(UpstreamCallBudgetExhaustedError, match="3 of 3 calls already reserved"):
        reserve_call(restarted)
    assert reserved_calls(str(ledger)) == 3


def test_concurrent_processes_never_reserve_more_than_the_cap(tmp_path: Path) -> None:
    ledger = str(tmp_path / "calls.ledger")
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    workers = [context.Process(target=_reserve_many, args=(ledger, 25, 20, results)) for _ in range(4)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=60)
        assert worker.exitcode == 0
    assert sum(results.get(timeout=5) for _ in workers) == 25
    assert reserved_calls(ledger) == 25


@pytest.mark.parametrize(
    "content,match",
    [
        (b"", "no valid record"),
        (b"\0" * 1024, "no valid record"),
        (b'{"max_calls": 5, "reserved": 0}', "no valid record"),
        (_record(99, 0), "a cap other than max_calls=5"),
    ],
    ids=["empty", "zeroed", "foreign-format", "other-cap"],
)
def test_an_existing_unusable_ledger_refuses_without_changing_it(tmp_path: Path, content: bytes, match: str) -> None:
    ledger = tmp_path / "calls.ledger"
    ledger.write_bytes(content)
    with pytest.raises(UpstreamCallBudgetExhaustedError, match=match):
        reserve_call(UpstreamCallBudget(ledger_path=str(ledger), max_calls=5))
    assert ledger.read_bytes() == content


def test_a_ledger_lost_after_creation_is_not_created_again(tmp_path: Path) -> None:
    ledger = tmp_path / "calls.ledger"
    budget = UpstreamCallBudget(ledger_path=str(ledger), max_calls=5)
    reserve_call(budget)
    ledger.unlink()
    with pytest.raises(UpstreamCallBudgetExhaustedError, match="is missing but .* shows it was created"):
        reserve_call(budget)
    assert not ledger.exists()


def test_a_ledger_that_cannot_be_created_refuses_the_call(tmp_path: Path) -> None:
    with pytest.raises(UpstreamCallBudgetExhaustedError, match="cannot open or create ledger"):
        reserve_call(UpstreamCallBudget(ledger_path=str(tmp_path / "missing" / "calls.ledger"), max_calls=5))


def _crash_at(ledger_path: str, max_calls: int, boundary: str) -> None:
    """Child process: run one reservation and die (os._exit) at one write boundary, as a node loss would."""
    real_pwrite, real_fsync, real_link = os.pwrite, os.fsync, os.link

    def torn_pwrite(fd, data, offset):
        if boundary == "torn-record-write":
            real_pwrite(fd, data[: len(data) // 2], offset)
            os._exit(9)
        return real_pwrite(fd, data, offset)

    def fsync(fd):
        if boundary == "before-record-fsync" and os.fstat(fd).st_size >= 512:
            os._exit(9)
        real_fsync(fd)
        if boundary == "after-record-fsync" and os.fstat(fd).st_size >= 512:
            os._exit(9)

    def link(src, dst):
        if boundary == "before-ledger-link":
            os._exit(9)
        real_link(src, dst)
        if boundary == "after-ledger-link":
            os._exit(9)

    os.pwrite, os.fsync, os.link = torn_pwrite, fsync, link
    reserve_call(UpstreamCallBudget(ledger_path=ledger_path, max_calls=max_calls))
    os._exit(0)  # the reservation succeeded: the request would have been sent


@pytest.mark.parametrize(
    "boundary,prior,after_crash",
    [
        ("before-ledger-link", 0, None),  # no ledger yet: created again at 0 on restart
        ("after-ledger-link", 0, 0),  # ledger exists at 0, marker not yet written
        ("torn-record-write", 3, 3),
        ("before-record-fsync", 3, {3, 4}),
        ("after-record-fsync", 3, 4),  # reserved durably but never sent: counted, which is the safe side
    ],
)
def test_a_crash_at_any_write_boundary_never_lowers_the_count(
    tmp_path: Path, boundary: str, prior: int, after_crash: int | set[int] | None
) -> None:
    ledger = str(tmp_path / "calls.ledger")
    budget = UpstreamCallBudget(ledger_path=ledger, max_calls=6)
    sent = 0
    for _ in range(prior):
        reserve_call(budget)
        sent += 1

    child = multiprocessing.get_context("spawn").Process(target=_crash_at, args=(ledger, 6, boundary))
    child.start()
    child.join(timeout=60)
    assert child.exitcode == 9  # it died before its reservation returned, so it sent nothing

    if after_crash is None:
        assert not os.path.exists(ledger)
    else:
        assert reserved_calls(ledger) in (after_crash if isinstance(after_crash, set) else {after_crash})
    # Restart: reservations continue from the durable count and stop at the cap.
    while True:
        try:
            reserve_call(budget)
        except UpstreamCallBudgetExhaustedError:
            break
        sent += 1
    assert sent <= 6
    assert reserved_calls(ledger) == 6


def test_the_budget_requires_one_http_attempt_per_call(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="upstream_max_num_tries=1"):
        _server(tmp_path, max_calls=1, upstream_max_num_tries=None)


async def test_concurrent_requests_reach_the_provider_at_most_max_calls_times(tmp_path: Path) -> None:
    server = _server(tmp_path, max_calls=5)
    sent = 0

    async def operation() -> str:
        nonlocal sent
        sent += 1
        await asyncio.sleep(0.01)
        return "ok"

    outcomes = await asyncio.gather(*(server._call_upstream(operation) for _ in range(12)), return_exceptions=True)
    assert sent == 5
    assert sum(outcome == "ok" for outcome in outcomes) == 5
    assert all(isinstance(o, UpstreamCallBudgetExhaustedError) for o in outcomes if o != "ok")


async def test_a_retry_policy_counts_every_attempt_and_stops_on_refusal(tmp_path: Path) -> None:
    server = _server(tmp_path, max_calls=2, upstream_retry_policy={"max_attempts": 5, "backoff_initial_seconds": 0})
    sent = 0

    async def failing_operation() -> str:
        nonlocal sent
        sent += 1
        raise RuntimeError("provider error")

    with pytest.raises(UpstreamCallBudgetExhaustedError):
        await server._call_upstream(failing_operation)
    assert sent == 2


async def test_over_real_http_failing_and_dropped_attempts_never_exceed_the_reserved_calls(
    tmp_path: Path, monkeypatch
) -> None:
    hits: list[int] = []

    async def handler(request: web.Request) -> web.StreamResponse:
        await request.read()
        hits.append(1)
        if len(hits) % 3 == 0:
            request.transport.close()  # a dropped connection after the provider received the request
            return web.Response(status=500)
        return web.Response(status=503 if len(hits) % 2 else 429, body=b'{"error":"busy"}')

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    await web.TCPSite(runner, "127.0.0.1", port).start()
    session = aiohttp.ClientSession()
    monkeypatch.setattr(nemo_gym.client.http_client, "get_global_aiohttp_client", lambda: session)
    try:
        server = _server(
            tmp_path,
            max_calls=7,
            openai_base_url=f"http://127.0.0.1:{port}/v1",
            upstream_retry_policy={"max_attempts": 4, "backoff_initial_seconds": 0},
        )
        body = NeMoGymChatCompletionCreateParamsNonStreaming(messages=[{"role": "user", "content": "hi"}])
        outcomes = await asyncio.wait_for(
            asyncio.gather(*(server.chat_completions(body) for _ in range(5)), return_exceptions=True), timeout=30
        )
    finally:
        await session.close()
        await runner.cleanup()

    assert all(isinstance(outcome, Exception) for outcome in outcomes)
    assert len(hits) == 7
    assert reserved_calls(str(tmp_path / "calls.ledger")) == 7


def test_threads_racing_to_create_a_new_ledger_reserve_exactly_the_cap(tmp_path: Path) -> None:
    budget = UpstreamCallBudget(ledger_path=str(tmp_path / "calls.ledger"), max_calls=10)

    def attempt() -> bool:
        try:
            reserve_call(budget)
            return True
        except UpstreamCallBudgetExhaustedError:
            return False

    with ThreadPoolExecutor(max_workers=32) as pool:
        granted = sum(pool.map(lambda _: attempt(), range(64)))
    assert granted == 10
    assert reserved_calls(budget.ledger_path) == 10
    assert sorted(p.name for p in tmp_path.iterdir()) == ["calls.ledger", "calls.ledger.created"]


class _CountingProvider(ThreadingHTTPServer):
    """A provider stub that counts every request it receives; every third one answers 503."""

    def __init__(self) -> None:
        self.hits = 0
        self.lock = threading.Lock()
        provider = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                with provider.lock:
                    provider.hits += 1
                    status = 503 if provider.hits % 3 == 0 else 200
                self.send_response(status)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *args) -> None:
                pass

        super().__init__(("127.0.0.1", 0), Handler)


def _worker(
    ledger_path: str, max_calls: int, url: str, attempts: int, crash: tuple[str, int] | None, log: str
) -> None:
    """Child process: reserve, then send one HTTP attempt; log each outcome; optionally die inside reservation k."""
    budget = UpstreamCallBudget(ledger_path=ledger_path, max_calls=max_calls)
    with open(log, "a") as out:
        for attempt in range(1, attempts + 1):
            if crash is not None and attempt == crash[1]:
                real_pwrite, real_fsync = os.pwrite, os.fsync

                def torn_pwrite(fd, data, offset):
                    real_pwrite(fd, data[: len(data) // 2], offset)
                    os._exit(9)

                def fsync_then_die(fd):
                    real_fsync(fd)
                    if stat.S_ISREG(os.fstat(fd).st_mode):
                        os._exit(9)

                if crash[0] == "torn-record-write":
                    os.pwrite = torn_pwrite
                else:
                    os.fsync = fsync_then_die
            try:
                reserve_call(budget)
            except UpstreamCallBudgetExhaustedError:
                out.write("refused\n")
                out.flush()
                continue
            request = urllib.request.Request(url, data=b"{}", method="POST")
            try:
                urllib.request.urlopen(request, timeout=10).read()
            except urllib.error.HTTPError:
                pass  # a failed attempt still reached the provider and still counts
            out.write("sent\n")
            out.flush()


def test_crashed_and_restarted_workers_reconcile_with_the_provider_count(tmp_path: Path) -> None:
    provider = _CountingProvider()
    thread = threading.Thread(target=provider.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{provider.server_address[1]}/v1/chat/completions"
    ledger = str(tmp_path / "calls.ledger")
    context = multiprocessing.get_context("spawn")
    # The two surviving workers make at most 30 attempts in the first round, below the cap of 40, so both
    # crashing workers always reach their crash point; the restart round then exhausts the cap.
    cap, attempts = 40, 15
    crashes = {0: ("torn-record-write", 2), 1: ("after-record-fsync", 2)}
    try:
        # Four concurrent workers; two die inside a reservation, at different write boundaries.
        first = [
            context.Process(target=_worker, args=(ledger, cap, url, attempts, crashes.get(i), str(tmp_path / f"w{i}")))
            for i in range(4)
        ]
        for worker in first:
            worker.start()
        for worker in first:
            worker.join(timeout=120)
        assert [worker.exitcode for worker in first] == [9, 9, 0, 0]
        # Restart: three fresh workers on the same ledger.
        second = [
            context.Process(target=_worker, args=(ledger, cap, url, attempts, None, str(tmp_path / f"r{i}")))
            for i in range(3)
        ]
        for worker in second:
            worker.start()
        for worker in second:
            worker.join(timeout=120)
        assert all(worker.exitcode == 0 for worker in second)
    finally:
        provider.shutdown()
        provider.server_close()

    events = [line for log in tmp_path.glob("[wr]*") for line in log.read_text().split()]
    sent, refused = events.count("sent"), events.count("refused")
    reserved = reserved_calls(ledger)
    assert reserved == cap
    # Every request the provider saw was a reserved one; the only reservation never sent is the one whose
    # worker died after its fsync. The torn write reserved nothing.
    assert provider.hits == sent == cap - 1
    # Each surviving worker made all its attempts; each attempt was either sent or refused.
    logged_by_crashed = (crashes[0][1] - 1) + (crashes[1][1] - 1)
    assert sent + refused == logged_by_crashed + 2 * attempts + 3 * attempts
