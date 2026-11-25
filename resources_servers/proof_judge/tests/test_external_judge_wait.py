# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Waiting for an external judge server (JUDGE_SERVER_ARGS) can be given a time limit.

By default the wait has no limit, as before. With `external_judge_wait_timeout_seconds` the first verification fails
with a TimeoutError once the judge server has not accepted a connection in that time.
"""

import time
from unittest.mock import MagicMock

import openai
import pytest

import resources_servers.proof_judge.app as app_module
from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef
from resources_servers.proof_judge.app import (
    ProofWithJudgeResourcesServer,
    ProofWithJudgeResourcesServerConfig,
    _wait_for_server,
)


def _server(**overrides) -> ProofWithJudgeResourcesServer:
    config = ProofWithJudgeResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        **overrides,
    )
    return ProofWithJudgeResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


class _Connections:
    """A stand-in for socket.create_connection that refuses a number of attempts and then accepts."""

    def __init__(self, refusals: int) -> None:
        self.refusals = refusals
        self.attempts = 0

    def __call__(self, address, timeout=None):
        self.attempts += 1
        if self.attempts <= self.refusals:
            raise ConnectionRefusedError("refused")
        return MagicMock(__enter__=lambda self: self, __exit__=lambda *args: False)


def test_a_server_that_does_not_come_up_in_time_fails_the_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    # Accepts after 50 refusals (about 1 s), so a limit that is ignored returns normally and fails the raises below.
    connections = _Connections(refusals=50)
    monkeypatch.setattr(app_module.socket, "create_connection", connections)

    started = time.monotonic()
    with pytest.raises(TimeoutError, match="did not accept connections within 0.1 s"):
        _wait_for_server("judge-node:8000", interval_seconds=0.02, timeout_s=0.1)

    assert time.monotonic() - started < 0.9
    assert connections.attempts < 50


def test_a_server_that_comes_up_only_after_the_deadline_is_not_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    # One refusal, then it accepts. With a 3 s interval the next attempt used to come at 3 s, long after the 0.1 s limit,
    # and succeed; now the sleep is capped at the time left and the deadline is checked before that attempt.
    connections = _Connections(refusals=1)
    monkeypatch.setattr(app_module.socket, "create_connection", connections)

    started = time.monotonic()
    with pytest.raises(TimeoutError, match="did not accept connections within 0.1 s"):
        _wait_for_server("judge-node:8000", interval_seconds=3, timeout_s=0.1)

    assert time.monotonic() - started < 1
    assert connections.attempts == 1


@pytest.mark.parametrize(("finished_at", "accepted"), [(0.05, True), (0.2, False)])
def test_a_connection_that_completes_after_the_deadline_is_not_accepted(
    monkeypatch: pytest.MonkeyPatch, finished_at: float, accepted: bool
) -> None:
    now = [0.0]
    monkeypatch.setattr(app_module.time, "monotonic", lambda: now[0])

    def create_connection(address, timeout=None):
        now[0] = finished_at  # the connection completes this far into the 0.1 s limit
        return MagicMock(__enter__=lambda self: self, __exit__=lambda *args: False)

    monkeypatch.setattr(app_module.socket, "create_connection", create_connection)

    if accepted:
        _wait_for_server("judge-node:8000", timeout_s=0.1)
    else:
        with pytest.raises(TimeoutError, match="did not accept connections within 0.1 s"):
            _wait_for_server("judge-node:8000", timeout_s=0.1)


def test_a_connection_attempt_is_capped_at_the_time_left(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []

    def create_connection(address, timeout=None):
        seen.append(timeout)
        # Fails at once if the deadline stops working, instead of refusing forever and holding the test run.
        assert len(seen) < 100, "the wait kept retrying past its deadline"
        raise ConnectionRefusedError("refused")

    monkeypatch.setattr(app_module.socket, "create_connection", create_connection)

    with pytest.raises(TimeoutError):
        _wait_for_server("judge-node:8000", interval_seconds=0.01, timeout_s=0.2)

    assert all(0 < timeout <= 0.2 for timeout in seen)


def test_a_server_that_comes_up_in_time_ends_the_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    connections = _Connections(refusals=2)
    monkeypatch.setattr(app_module.socket, "create_connection", connections)

    _wait_for_server("judge-node:8000", interval_seconds=0.01, timeout_s=5)

    assert connections.attempts == 3


def test_without_a_limit_the_wait_keeps_going_until_the_server_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    connections = _Connections(refusals=20)
    monkeypatch.setattr(app_module.socket, "create_connection", connections)

    _wait_for_server("judge-node:8000", interval_seconds=0.005)

    assert connections.attempts == 21


@pytest.mark.parametrize(
    ("overrides", "wait_limit", "request_timeout"),
    [
        ({}, None, 60 * 60 * 4),
        ({"external_judge_wait_timeout_seconds": 90, "external_judge_request_timeout_seconds": 300}, 90, 300),
    ],
)
async def test_the_configured_limits_reach_the_wait_and_the_judge_client(
    monkeypatch: pytest.MonkeyPatch, overrides: dict, wait_limit: object, request_timeout: float
) -> None:
    monkeypatch.setattr(app_module, "_get_judge_client_config", lambda: ("judge", "vllm", 8000, ["node-a", "node-b"]))
    waited: list = []
    monkeypatch.setattr(app_module, "_wait_for_server", lambda address, **kwargs: waited.append((address, kwargs)))
    clients: list = []
    monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: clients.append(kwargs) or MagicMock())

    await _server(**overrides)._init_external_clients()

    assert waited == [("node-a:8000", {"timeout_s": wait_limit}), ("node-b:8000", {"timeout_s": wait_limit})]
    assert [client["timeout"] for client in clients] == [request_timeout, request_timeout]


def test_a_limit_of_zero_is_rejected() -> None:
    for field in ("external_judge_wait_timeout_seconds", "external_judge_request_timeout_seconds"):
        with pytest.raises(ValueError):
            _server(**{field: 0})
