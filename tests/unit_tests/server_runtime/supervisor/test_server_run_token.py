# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Readiness counts only answers from this run's own servers.

The supervisor gives every server it starts (and its own head server) a run token, which each returns on its liveness
route. An answer on a server's address that carries another token is another run's server holding the port, not this
server being ready. A server that returns no token (its Gym predates run tokens) is accepted as before.
"""

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

import nemo_gym.server_runtime.supervisor.server_supervisor as env_module
from nemo_gym.client.server_client import ServerClient
from nemo_gym.config.global_config import HEAD_SERVER_KEY_NAME
from nemo_gym.server_runtime.base_server import NEMO_GYM_SERVER_RUN_TOKEN_ENV_VAR_NAME, liveness_body
from nemo_gym.server_runtime.supervisor.server_supervisor import ServerStartFailedError, ServerSupervisor


def test_the_liveness_body_carries_the_run_token_when_set(monkeypatch) -> None:
    monkeypatch.delenv(NEMO_GYM_SERVER_RUN_TOKEN_ENV_VAR_NAME, raising=False)
    assert liveness_body() == {"status": "ok"}
    monkeypatch.setenv(NEMO_GYM_SERVER_RUN_TOKEN_ENV_VAR_NAME, "abc")
    assert liveness_body() == {"status": "ok", "run_token": "abc"}


def supervisor_with(tokens: dict[str, object], run_token: str = "ours") -> ServerSupervisor:
    """A supervisor whose servers all answer HTTP, each returning the given token (an exception: the request fails)."""

    def answer_run_token(name: str, **_kwargs) -> object:
        token = tokens[name]
        if isinstance(token, Exception):
            raise token
        return token

    supervisor = ServerSupervisor()
    supervisor._run_token = run_token
    supervisor._server_client = MagicMock(
        poll_for_status=MagicMock(return_value="success"),
        answer_run_token=MagicMock(side_effect=answer_run_token),
    )
    supervisor._server_instance_display_configs = [SimpleNamespace(config_path=name) for name in tokens]
    return supervisor


def test_only_our_servers_answers_count_as_ready() -> None:
    supervisor = supervisor_with(
        {"ours": "ours", "older_gym": None, "other_run": "theirs", "flaky": ConnectionError("reset")}
    )
    assert dict(supervisor.check_http_server_statuses([])) == {
        "ours": "success",
        "older_gym": "success",
        "other_run": "other_server",
        "flaky": "other_server",
    }


def test_without_a_run_token_every_answer_counts() -> None:
    supervisor = supervisor_with({"a": "theirs"})
    del supervisor._run_token
    assert supervisor.check_http_server_statuses([]) == [("a", "success")]


def test_the_head_wait_ignores_another_runs_head_server(monkeypatch) -> None:
    """Our head server thread has died on the bind; the other run's head server answers with its own token."""
    supervisor = supervisor_with({HEAD_SERVER_KEY_NAME: "theirs"})
    supervisor._head_server = SimpleNamespace(started=True)
    supervisor._head_server_thread = MagicMock(is_alive=MagicMock(side_effect=[True, False]))
    supervisor._head_server_instance = SimpleNamespace(config=SimpleNamespace(host="127.0.0.1", port=1))
    monkeypatch.setattr(env_module, "sleep", lambda seconds: None)
    monkeypatch.setattr(env_module, "_port_bind_errno", lambda host, port: None)

    with pytest.raises(ServerStartFailedError):
        supervisor._wait_for_head_server()
    assert supervisor._server_client.answer_run_token.called


@pytest.mark.parametrize(
    ("body", "expected"),
    [(b'{"status": "ok", "run_token": "t1"}', "t1"), (b'{"status": "ok"}', None)],
)
def test_the_client_reads_the_token_from_a_real_liveness_answer(body: bytes, expected: str | None) -> None:
    class Answer(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Answer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = ServerClient.model_construct(
            global_config_dict=OmegaConf.create(
                {HEAD_SERVER_KEY_NAME: {"host": "127.0.0.1", "port": server.server_address[1]}}
            )
        )
        assert client.answer_run_token(HEAD_SERVER_KEY_NAME) == expected
    finally:
        server.shutdown()
        server.server_close()
