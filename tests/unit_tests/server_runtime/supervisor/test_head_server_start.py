# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The head server gets the same taken-port retry as every other server, and its wait is bounded.

A head server whose bind failed ends the wait with a typed start failure that carries EADDRINUSE when its port is
taken, so startup moves it to a fresh port once. A head server that runs but never answers stops the run after
`head_server_spinup_timeout_seconds` (15 minutes by default) with a typed, non-resumable error.
"""

import errno
import socket
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import nemo_gym.server_runtime.supervisor.server_supervisor as env_module
from nemo_gym.cli.utils import exit_cleanly_on_config_error
from nemo_gym.config.global_config import HEAD_SERVER_KEY_NAME
from nemo_gym.core.failure_kinds import SERVER_START_FAILED
from nemo_gym.server_runtime.supervisor.server_supervisor import (
    HeadServerUnresponsiveError,
    RunHelper,
    ServerStartFailedError,
)
from tests.unit_tests.server_runtime.supervisor.test_server_start_failed import taken_port as taken_port


def waiting_runner(monkeypatch, *, alive: bool, statuses: list[str], timeout_s: float | None = None) -> RunHelper:
    runner = RunHelper()
    runner._server_client = MagicMock(poll_for_status=MagicMock(side_effect=statuses))
    runner._head_server_thread = MagicMock(is_alive=MagicMock(return_value=alive))
    if timeout_s is not None:
        runner._head_server_spinup_timeout_seconds = timeout_s
    clock = {"now": 0.0}
    monkeypatch.setattr(env_module, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(env_module, "sleep", lambda seconds: clock.update(now=clock["now"] + seconds))
    return runner


def test_a_head_server_whose_port_is_taken_reports_eaddrinuse(monkeypatch) -> None:
    holder = socket.socket()
    holder.bind(("127.0.0.1", 0))
    holder.listen()
    port = holder.getsockname()[1]
    runner = waiting_runner(monkeypatch, alive=False, statuses=["connection_error"])
    runner._head_server_instance = SimpleNamespace(config=SimpleNamespace(host="127.0.0.1", port=port))
    try:
        with pytest.raises(ServerStartFailedError) as raised:
            runner._wait_for_head_server()
    finally:
        holder.close()

    assert (raised.value.server_name, raised.value.errno) == (HEAD_SERVER_KEY_NAME, errno.EADDRINUSE)
    assert f"port {port} is already in use" in str(raised.value)


def test_a_head_server_that_died_on_a_free_port_reports_no_errno(monkeypatch) -> None:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    runner = waiting_runner(monkeypatch, alive=False, statuses=["connection_error"])
    runner._head_server_instance = SimpleNamespace(config=SimpleNamespace(host="127.0.0.1", port=port))

    with pytest.raises(ServerStartFailedError) as raised:
        runner._wait_for_head_server()
    assert raised.value.errno is None and not isinstance(raised.value, HeadServerUnresponsiveError)


def test_a_taken_head_server_port_is_retried_once_on_a_fresh_port(taken_port: dict) -> None:
    runner = RunHelper()
    runner.shutdown = MagicMock()
    config = env_module.get_global_config_dict()
    head_ports: list[int] = []

    def start(_parser_config: object) -> None:
        runner._server_configs = {"judge_model": taken_port["server_config"]}
        head_ports.append(config[HEAD_SERVER_KEY_NAME].port)
        if len(head_ports) == 1:
            raise ServerStartFailedError(
                "server_start_failed", server_name=HEAD_SERVER_KEY_NAME, errno=errno.EADDRINUSE
            )

    runner._start = MagicMock(side_effect=start)

    runner.start(None)

    assert head_ports == [11000, 20007]
    assert set(taken_port["picked"][0]["disallowed_ports"]) == {21001, 11000}
    assert taken_port["server_config"].port == 21001  # the other servers keep their ports


def test_a_live_head_server_that_never_answers_stops_after_the_bound(monkeypatch) -> None:
    runner = waiting_runner(monkeypatch, alive=True, statuses=["connection_error"] * 1000)

    with pytest.raises(HeadServerUnresponsiveError) as raised:
        runner._wait_for_head_server()

    error = raised.value
    assert error.failure_kind == SERVER_START_FAILED and error.stop_exit_code == 76
    assert "within 900s" in str(error) and "head_server_spinup_timeout_seconds" in str(error)
    assert env_module.monotonic() >= 900  # it waited the default 15 minutes, not longer than one poll beyond


def test_a_zero_bound_waits_until_the_head_server_answers(monkeypatch) -> None:
    runner = waiting_runner(monkeypatch, alive=True, statuses=["connection_error"] * 400 + ["success"], timeout_s=0)

    runner._wait_for_head_server()

    assert env_module.monotonic() > 900


def test_the_cli_exits_with_the_stop_code_not_a_traceback(capsys) -> None:
    @exit_cleanly_on_config_error
    def run() -> None:
        raise HeadServerUnresponsiveError("head server silent", server_name=HEAD_SERVER_KEY_NAME, errno=None)

    with pytest.raises(SystemExit) as raised:
        run()
    assert raised.value.code == 76 and "head server silent" in capsys.readouterr().out

    @exit_cleanly_on_config_error
    def crash() -> None:
        raise ServerStartFailedError("plain start failure", server_name="x", errno=None)

    with pytest.raises(ServerStartFailedError):  # no stop code: raised as before
        crash()
