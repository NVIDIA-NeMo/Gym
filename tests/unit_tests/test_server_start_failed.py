# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import errno
import shlex
import socket
import sys
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

import nemo_gym.orchestration.server_supervisor as env_module
from nemo_gym.config.global_config import HEAD_SERVER_KEY_NAME, NEMO_GYM_LOG_DIR_KEY_NAME, UV_CACHE_DIR_KEY_NAME
from nemo_gym.core.failure_kinds import FAILURE_KINDS, SERVER_START_FAILED
from nemo_gym.orchestration.server_supervisor import RunHelper, ServerStartFailedError, _bind_errno


BIND_ERROR = (
    "ERROR:    [Errno 98] error while attempting to bind on address ('127.0.0.1', 21001): address already in use\n"
)


def test_the_bind_errno_is_read_from_the_servers_stderr() -> None:
    assert _bind_errno(BIND_ERROR) == errno.EADDRINUSE
    assert _bind_errno("Traceback (most recent call last):\nValueError: bad config\n") is None
    assert _bind_errno(None) is None


def _exited_server(stderr: str) -> MagicMock:
    process = MagicMock()
    process.poll.return_value = 1
    process.communicate.return_value = (b"", stderr.encode())
    return process


def _polling_runner(*, ready: bool) -> RunHelper:
    runner = RunHelper()
    runner._servers_ready = ready
    runner._head_server_thread = MagicMock(is_alive=MagicMock(return_value=True))
    runner._h2_ping_sidecar = None
    runner._processes = {"judge_model": _exited_server(BIND_ERROR)}
    return runner


def test_a_server_that_exits_before_it_is_ready_is_a_typed_start_failure() -> None:
    with pytest.raises(ServerStartFailedError) as raised:
        _polling_runner(ready=False).poll()

    assert (raised.value.server_name, raised.value.errno) == ("judge_model", errno.EADDRINUSE)
    assert raised.value.failure_kind == SERVER_START_FAILED and SERVER_START_FAILED in FAILURE_KINDS
    assert "server_start_failed: Process `judge_model` finished unexpectedly!" in str(raised.value)
    assert "address already in use" in str(raised.value)


def test_a_server_that_exits_after_every_server_was_ready_is_no_start_failure() -> None:
    with pytest.raises(RuntimeError) as raised:
        _polling_runner(ready=True).poll()

    assert not isinstance(raised.value, ServerStartFailedError)


@pytest.mark.parametrize("log_dir", [False, True], ids=["terminal", "log-dir"])
def test_a_real_server_that_cannot_bind_an_occupied_port_reports_eaddrinuse(tmp_path, capfd, log_dir) -> None:
    """A real child process started the way `_start` starts a server finds its port taken and exits.

    Its bind error reaches the supervisor, so the start failure carries EADDRINUSE and the port retry can run. The
    output is still forwarded live. With a log dir the launch command merges stderr into stdout before `tee`.
    """
    config = {UV_CACHE_DIR_KEY_NAME: str(tmp_path / "uv-cache")}
    if log_dir:
        config[NEMO_GYM_LOG_DIR_KEY_NAME] = str(tmp_path / "logs")
    server = (
        "import uvicorn\nasync def app(scope, receive, send): pass\nuvicorn.run(app, host='127.0.0.1', port={port})"
    )
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = taken.getsockname()[1]
        runner = RunHelper()
        runner._servers_ready = False
        runner._head_server_thread = MagicMock(is_alive=MagicMock(return_value=True))
        runner._h2_ping_sidecar = None
        runner._processes = {}
        runner._output_tails = {}
        command = f"{shlex.quote(sys.executable)} -c {shlex.quote(server.format(port=port))}"
        process = runner._spawn_server(
            "judge_model",
            command,
            tmp_path,
            project_root=None,
            extra_env={},
            global_config_dict=OmegaConf.create(config),
        )
        assert process.wait(timeout=60) != 0

        with pytest.raises(ServerStartFailedError) as raised:
            runner.poll()

    assert (raised.value.server_name, raised.value.errno) == ("judge_model", errno.EADDRINUSE)
    assert "address already in use" in str(raised.value)
    assert "error while attempting to bind" in "".join(capfd.readouterr())


@pytest.fixture
def taken_port(monkeypatch: pytest.MonkeyPatch) -> dict:
    """A config where `judge_model` was given port 21001, and a port picker that hands out 20007."""
    config = OmegaConf.create(
        {
            "judge_model": {"responses_api_models": {"judge": {"port": 21001}}},
            HEAD_SERVER_KEY_NAME: {"port": 11000},
            "port_range_low": 20000,
            "port_range_high": 20100,
        }
    )
    picked: list[dict] = []

    def pick(**kwargs) -> int:
        picked.append(kwargs)
        return 20007

    monkeypatch.setattr(env_module, "get_global_config_dict", lambda **kwargs: config)
    monkeypatch.setattr(env_module, "_find_open_port_using_range", pick)
    return {"server_config": config.judge_model.responses_api_models.judge, "picked": picked}


def _runner_whose_server_binds(taken_port: dict, *, failures: int, error_number: int | None = errno.EADDRINUSE):
    """A RunHelper whose start fails `failures` times as a server that cannot bind its port, then succeeds."""
    runner = RunHelper()
    runner.shutdown = MagicMock()
    ports: list[int] = []

    def start_servers(_parser_config: object) -> None:
        runner._server_configs = {"judge_model": taken_port["server_config"]}
        ports.append(taken_port["server_config"].port)
        if len(ports) <= failures:
            raise ServerStartFailedError("server_start_failed", server_name="judge_model", errno=error_number)

    runner._start = MagicMock(side_effect=start_servers)
    return runner, ports


def test_a_port_taken_at_startup_is_retried_once_on_a_fresh_port(taken_port: dict) -> None:
    runner, ports = _runner_whose_server_binds(taken_port, failures=1)

    runner.start(None)

    assert ports == [21001, 20007]
    runner.shutdown.assert_called_once()
    # The fresh port avoids every port the servers and the head server already use.
    assert set(taken_port["picked"][0]["disallowed_ports"]) == {21001, 11000}


def test_a_second_failed_bind_is_not_retried_again(taken_port: dict) -> None:
    runner, ports = _runner_whose_server_binds(taken_port, failures=2)

    with pytest.raises(ServerStartFailedError):
        runner.start(None)

    assert ports == [21001, 20007]
    assert runner.shutdown.call_count == 2


def test_a_start_failure_other_than_a_taken_port_is_not_retried(taken_port: dict) -> None:
    runner, ports = _runner_whose_server_binds(taken_port, failures=1, error_number=None)

    with pytest.raises(ServerStartFailedError):
        runner.start(None)

    assert ports == [21001]


def test_a_taken_port_is_not_retried_after_the_sidecar_rewrote_the_model_urls(taken_port: dict) -> None:
    runner, ports = _runner_whose_server_binds(taken_port, failures=1)
    start_servers = runner._start.side_effect

    def start_with_sidecar(parser_config: object) -> None:
        runner._h2_ping_sidecar = object()
        start_servers(parser_config)

    runner._start = MagicMock(side_effect=start_with_sidecar)

    with pytest.raises(ServerStartFailedError):
        runner.start(None)

    assert ports == [21001]


def test_a_head_server_that_died_ends_the_wait_with_a_typed_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = RunHelper()
    runner._server_client = MagicMock(poll_for_status=MagicMock(return_value="connection_error"))
    runner._head_server_thread = MagicMock(is_alive=MagicMock(return_value=False))
    monkeypatch.setattr(
        env_module, "sleep", lambda seconds: pytest.fail("the wait went on after the head server died")
    )

    with pytest.raises(ServerStartFailedError) as raised:
        runner._wait_for_head_server()

    assert (raised.value.server_name, raised.value.errno) == (HEAD_SERVER_KEY_NAME, None)
