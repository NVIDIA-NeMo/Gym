# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tool calls to a ``BaseEnvironmentServerWithTools`` reach the worker that runs the episode.

In-process workers share a real socket directory, as the workers of one server do.
"""

import hashlib
import os
import shutil
import tempfile
from collections.abc import Iterator
from typing import Optional
from unittest.mock import MagicMock

import pytest
from fastapi import Body
from fastapi.testclient import TestClient
from httpx import Response
from pydantic import BaseModel

from nemo_gym.base_environment_server import CleanupContext
from nemo_gym.base_environment_server_with_tools import (
    BaseEnvironmentServerWithTools,
    BaseEnvironmentServerWithToolsConfig,
    ServedTools,
)
from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.episode_tokens import EPISODE_TOKEN_HEADER, EpisodeTokens
from nemo_gym.episode_tools import TypedToolset, episode_tool
from nemo_gym.episode_types import BaseEpisodeRequest, BaseEpisodeResponse, EpisodeId, MaterializedTask, TaskId
from nemo_gym.server_utils import ServerClient
from nemo_gym.session_routing import SESSION_CLAIMS_DIR, install_session_routing


class _TaskInput(BaseModel):
    pass


class _Request(BaseEpisodeRequest[_TaskInput]):
    pass


class _Response(BaseEpisodeResponse[str]):
    pass


class _Amount(BaseModel):
    amount: int


class _Counter:
    def __init__(self) -> None:
        self.total = 0

    @episode_tool
    async def add(self, arguments: _Amount) -> dict[str, int]:
        """Add an amount to the episode's total."""
        self.total += arguments.amount
        return {"total": self.total}


class _Server(BaseEnvironmentServerWithTools[_Request, _Response]):
    request_model = _Request
    response_model = _Response

    async def run_episode(self, request: _Request, cleanup: CleanupContext) -> _Response:
        raise NotImplementedError("the tests serve episodes directly")

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        return AggregateMetrics()


def _request(rollout_id: str, attempt: int = 0) -> _Request:
    return _Request(
        episode_id=EpisodeId(rollout_id=rollout_id, attempt=attempt),
        task=MaterializedTask(task_id=TaskId(taskset="t", task_id="0"), task_input=_TaskInput()),
    )


class _Worker:
    """One worker of a ``BaseEnvironmentServerWithTools``, wired as run_webserver wires it."""

    def __init__(self, socket_dir: Optional[str]) -> None:
        config = BaseEnvironmentServerWithToolsConfig(
            name="counter_environment",
            host="127.0.0.1",
            port=1234,
            entrypoint="app.py",
            cleanup_timeout_seconds=1,
            num_workers=2 if socket_dir is not None else 1,
        )
        server_client = MagicMock(spec=ServerClient)
        server_client._resolve_base_url.return_value = "http://127.0.0.1:1234"
        self.server = _Server(config=config, server_client=server_client)
        self.app = self.server.setup_webserver()
        key = self.server.get_session_middleware_key()
        self.routing_id = install_session_routing(
            self.app,
            socket_dir=socket_dir,
            session_cookie=key,
            secret_key=key,
            **self.server.claimed_session_routing(),
        )
        self.client = TestClient(self.app, raise_server_exceptions=False)

    def serve(self, rollout_id: str, attempt: int = 0) -> tuple[ServedTools, CleanupContext, _Counter]:
        """Serve one episode's tools on this worker, in its event loop, as its run() would."""
        request = _request(rollout_id, attempt)
        cleanup = CleanupContext(episode_id=request.episode_id, cleanup_timeout_seconds=1)
        counter = _Counter()
        served = self.client.portal.call(self.server.serve_tools, request, TypedToolset(counter), cleanup)
        return served, cleanup, counter

    def end(self, cleanup: CleanupContext) -> None:
        self.client.portal.call(cleanup.aclose)

    def add(self, token: str, amount: int = 1) -> Response:
        return self.client.post("/add", json={"amount": amount}, headers={EPISODE_TOKEN_HEADER: token})


def _token(served: ServedTools) -> str:
    (grant,) = served.tool_accesses
    return grant.headers[EPISODE_TOKEN_HEADER]


def _claimant(socket_dir: str, session_id: str) -> Optional[str]:
    path = os.path.join(socket_dir, SESSION_CLAIMS_DIR, hashlib.sha256(session_id.encode("utf-8")).hexdigest())
    try:
        with open(path) as claim:
            return claim.read()
    except FileNotFoundError:
        return None


@pytest.fixture
def socket_dir() -> Iterator[str]:
    # Under /tmp, as in production: AF_UNIX paths are limited to about 100 bytes.
    path = tempfile.mkdtemp(prefix="ng-test-", dir="/tmp")
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def workers(socket_dir: str) -> Iterator[tuple[_Worker, _Worker]]:
    a, b = _Worker(socket_dir), _Worker(socket_dir)
    with a.client, b.client:
        yield a, b


def test_a_tool_call_that_reaches_another_worker_runs_on_the_episodes_worker(
    socket_dir: str, workers: tuple[_Worker, _Worker]
) -> None:
    a, b = workers
    served, _, counter = a.serve("3-0")

    first, second = b.add(_token(served), 2), b.add(_token(served), 3)

    assert (first.status_code, first.json(), second.json()) == (200, {"total": 2}, {"total": 5})
    assert counter.total == 5 and [call.name for call in served.calls] == ["add", "add"]
    assert _claimant(socket_dir, "3-0") == a.routing_id


def test_routing_reads_the_episode_and_only_the_episodes_worker_checks_the_nonce(
    workers: tuple[_Worker, _Worker],
) -> None:
    a, b = workers
    served, _, counter = a.serve("3-0")
    forged, _ = EpisodeTokens(a.server.get_session_middleware_key()).mint(EpisodeId(rollout_id="3-0"))

    reply = b.add(forged)

    # Forwarded to the episode's worker, which refuses the nonce of another grant.
    assert reply.status_code == 403 and "not granted for this episode" in reply.json()["detail"]
    assert counter.total == 0 and served.calls == []


def test_a_token_that_names_no_episode_is_refused_where_it_lands(workers: tuple[_Worker, _Worker]) -> None:
    a, b = workers
    a.serve("3-0")

    assert b.add("garbage").status_code == 401
    assert b.client.post("/add", json={"amount": 1}).status_code == 401


def test_each_attempt_of_an_episode_has_its_own_claim(socket_dir: str, workers: tuple[_Worker, _Worker]) -> None:
    a, b = workers
    first_attempt, _, first_counter = a.serve("3-0")
    retry, _, retry_counter = b.serve("3-0", attempt=1)

    a.add(_token(retry), 10)
    b.add(_token(first_attempt), 1)

    assert (first_counter.total, retry_counter.total) == (1, 10)
    assert (_claimant(socket_dir, "3-0"), _claimant(socket_dir, "3-0-a1")) == (a.routing_id, b.routing_id)


def test_an_episode_running_on_another_worker_cannot_serve_its_tools_again(workers: tuple[_Worker, _Worker]) -> None:
    a, b = workers
    a.serve("3-0")

    with pytest.raises(ValueError, match="already serves its tools on another worker"):
        b.serve("3-0")


def test_the_claim_ends_with_the_episode(socket_dir: str, workers: tuple[_Worker, _Worker]) -> None:
    a, b = workers
    served, cleanup, _ = a.serve("3-0")
    a.end(cleanup)

    reply = b.add(_token(served))

    assert _claimant(socket_dir, "3-0") is None
    # Nobody claims the episode, so the worker it reached answers that it does not run it.
    assert reply.status_code == 410 and "not running on this server" in reply.json()["detail"]


def test_a_tool_call_for_an_exited_workers_episode_gets_410(socket_dir: str) -> None:
    a, b = _Worker(socket_dir), _Worker(socket_dir)
    with b.client:
        a.client.__enter__()
        served, _, _ = a.serve("3-0")
        # The worker exits: its private socket closes, and its claim stays.
        a.client.__exit__(None, None, None)

        reply = b.add(_token(served))

    assert _claimant(socket_dir, "3-0") == a.routing_id
    assert reply.status_code == 410 and "no longer running" in reply.json()["detail"]


def test_one_worker_serves_tool_calls_without_claims() -> None:
    worker = _Worker(socket_dir=None)
    with worker.client:
        served, _, counter = worker.serve("3-0")

        reply = worker.add(_token(served), 4)

    assert worker.routing_id is None
    assert reply.json() == {"total": 4} and counter.total == 4
