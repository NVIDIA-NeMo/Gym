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
    ServerStartFailedError,
    ServerSupervisor,
)
from tests.unit_tests.server_runtime.supervisor.test_server_start_failed import taken_port as taken_port


def waiting_supervisor(
    monkeypatch, *, alive: bool, statuses: list[str], timeout_s: float | None = None
) -> ServerSupervisor:
    supervisor = ServerSupervisor()
    supervisor._server_client = MagicMock(poll_for_status=MagicMock(side_effect=statuses))
    supervisor._head_server_thread = MagicMock(is_alive=MagicMock(return_value=alive))
    if timeout_s is not None:
        supervisor._head_server_spinup_timeout_seconds = timeout_s
    clock = {"now": 0.0}
    monkeypatch.setattr(env_module, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(env_module, "sleep", lambda seconds: clock.update(now=clock["now"] + seconds))
    return supervisor


def test_a_head_server_whose_port_is_taken_reports_eaddrinuse(monkeypatch) -> None:
    holder = socket.socket()
    holder.bind(("127.0.0.1", 0))
    holder.listen()
    port = holder.getsockname()[1]
    supervisor = waiting_supervisor(monkeypatch, alive=False, statuses=["connection_error"])
    supervisor._head_server_instance = SimpleNamespace(config=SimpleNamespace(host="127.0.0.1", port=port))
    try:
        with pytest.raises(ServerStartFailedError) as raised:
            supervisor._wait_for_head_server()
    finally:
        holder.close()

    assert (raised.value.server_name, raised.value.errno) == (HEAD_SERVER_KEY_NAME, errno.EADDRINUSE)
    assert f"port {port} is already in use" in str(raised.value)


def test_a_head_server_that_died_on_a_free_port_reports_no_errno(monkeypatch) -> None:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    supervisor = waiting_supervisor(monkeypatch, alive=False, statuses=["connection_error"])
    supervisor._head_server_instance = SimpleNamespace(config=SimpleNamespace(host="127.0.0.1", port=port))

    with pytest.raises(ServerStartFailedError) as raised:
        supervisor._wait_for_head_server()
    assert raised.value.errno is None and not isinstance(raised.value, HeadServerUnresponsiveError)


def test_another_runs_head_server_answering_on_the_port_is_not_ours(monkeypatch) -> None:
    """Another Gym run holds the port: its head server answers there, but ours never bound, so the start fails with
    EADDRINUSE (and is moved to a fresh port) instead of waiting on the other run's server."""
    holder = socket.socket()
    holder.bind(("127.0.0.1", 0))
    holder.listen()
    port = holder.getsockname()[1]
    supervisor = waiting_supervisor(monkeypatch, alive=False, statuses=["success"])
    supervisor._head_server = SimpleNamespace(started=False)
    supervisor._head_server_instance = SimpleNamespace(config=SimpleNamespace(host="127.0.0.1", port=port))
    try:
        with pytest.raises(ServerStartFailedError) as raised:
            supervisor._wait_for_head_server()
    finally:
        holder.close()

    assert (raised.value.server_name, raised.value.errno) == (HEAD_SERVER_KEY_NAME, errno.EADDRINUSE)


def test_an_answer_before_our_head_server_started_is_not_ready(monkeypatch) -> None:
    """An answer on the address while our own head server is still binding doesn't end the wait; ours starting does."""
    supervisor = waiting_supervisor(monkeypatch, alive=True, statuses=["success", "success", "success"])
    head_server = SimpleNamespace(started=False)
    supervisor._head_server = head_server
    polls = supervisor._server_client.poll_for_status

    def poll(name: str) -> str:
        if polls.call_count == 2:
            head_server.started = True
        return "success"

    polls.side_effect = poll
    supervisor._wait_for_head_server()
    assert polls.call_count == 2


def test_a_head_server_that_never_starts_while_another_answers_names_the_other(monkeypatch) -> None:
    """Our head server hangs before binding while another server answers on its port: the timeout says so, rather
    than that our head server "is running" with status `success`."""
    supervisor = waiting_supervisor(monkeypatch, alive=True, statuses=["success"] * 10, timeout_s=6)
    supervisor._head_server = SimpleNamespace(started=False)
    supervisor._head_server_instance = SimpleNamespace(config=SimpleNamespace(host="127.0.0.1", port=11600))

    with pytest.raises(HeadServerUnresponsiveError) as raised:
        supervisor._wait_for_head_server()
    assert "another server answered on its port 11600" in str(raised.value)
    assert "although it is running" not in str(raised.value)


def test_a_real_bind_failure_behind_a_foreign_answering_server_is_eaddrinuse(monkeypatch) -> None:
    """End to end on real sockets: another HTTP server holds the port and answers, our uvicorn head server fails to
    bind on a thread as HeadServer.run_webserver runs it, and the wait reports EADDRINUSE so `start` moves it."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    import requests
    import uvicorn

    class Answer(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    foreign = HTTPServer(("127.0.0.1", 0), Answer)
    port = foreign.server_address[1]
    threading.Thread(target=foreign.serve_forever, daemon=True).start()

    async def app(scope, receive, send):  # never reached: the bind fails
        pass

    ours = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="critical"))
    thread = threading.Thread(target=ours.run, daemon=True)
    thread.start()

    def poll(_name: str) -> str:
        try:
            requests.get(f"http://127.0.0.1:{port}", timeout=2)
            return "success"
        except requests.exceptions.ConnectionError:
            return "connection_error"

    supervisor = ServerSupervisor()
    supervisor._server_client = MagicMock(poll_for_status=MagicMock(side_effect=poll))
    supervisor._head_server, supervisor._head_server_thread = ours, thread
    supervisor._head_server_instance = SimpleNamespace(config=SimpleNamespace(host="127.0.0.1", port=port))
    supervisor._head_server_spinup_timeout_seconds = 30
    monkeypatch.setattr(env_module, "sleep", lambda seconds: thread.join(timeout=seconds))
    try:
        with pytest.raises(ServerStartFailedError) as raised:
            supervisor._wait_for_head_server()
    finally:
        foreign.shutdown()
        foreign.server_close()

    assert (raised.value.server_name, raised.value.errno) == (HEAD_SERVER_KEY_NAME, errno.EADDRINUSE)
    assert not ours.started


def test_a_taken_head_server_port_is_retried_once_on_a_fresh_port(taken_port: dict) -> None:
    supervisor = ServerSupervisor()
    supervisor.shutdown = MagicMock()
    config = env_module.get_global_config_dict()
    head_ports: list[int] = []

    def start(_parser_config: object) -> None:
        supervisor._server_configs = {"judge_model": taken_port["server_config"]}
        head_ports.append(config[HEAD_SERVER_KEY_NAME].port)
        if len(head_ports) == 1:
            raise ServerStartFailedError(
                "server_start_failed", server_name=HEAD_SERVER_KEY_NAME, errno=errno.EADDRINUSE
            )

    supervisor._start = MagicMock(side_effect=start)

    supervisor.start(None)

    assert head_ports == [11000, 20007]
    assert set(taken_port["picked"][0]["disallowed_ports"]) == {21001, 11000}
    assert taken_port["server_config"].port == 21001  # the other servers keep their ports


def test_a_live_head_server_that_never_answers_stops_after_the_bound(monkeypatch) -> None:
    supervisor = waiting_supervisor(monkeypatch, alive=True, statuses=["connection_error"] * 1000)

    with pytest.raises(HeadServerUnresponsiveError) as raised:
        supervisor._wait_for_head_server()

    error = raised.value
    assert error.failure_kind == SERVER_START_FAILED and error.stop_exit_code == 76
    assert "within 900s" in str(error) and "head_server_spinup_timeout_seconds" in str(error)
    assert env_module.monotonic() >= 900  # it waited the default 15 minutes, not longer than one poll beyond


def test_a_zero_bound_waits_until_the_head_server_answers(monkeypatch) -> None:
    supervisor = waiting_supervisor(
        monkeypatch, alive=True, statuses=["connection_error"] * 400 + ["success"], timeout_s=0
    )

    supervisor._wait_for_head_server()

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
