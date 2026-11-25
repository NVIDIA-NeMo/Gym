# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tool host's command line, as the Resources Server runs it in a tool sidecar: serve, probe and call."""

import http.server
import json
import socket
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from fastapi.testclient import TestClient

from nemo_gym.core.episode_types import EpisodeId, TaskId
from nemo_gym.environment import tool_sidecar_host
from nemo_gym.server_runtime.base_resources_server import ResourcesSeedSessionRequest
from nemo_gym.server_runtime.tool_sidecar import TOOL_HOST_READY_PATH
from tests.unit_tests.environment.fixtures.tool_server.app import LedgerConfig


class _ToolHostStandIn(http.server.ThreadingHTTPServer):
    """A local server in place of the tool host: it answers the readiness route with ``ready_status`` and any POST
    with 201, two cookies and the request it got, which it also records."""

    ready_status = 200
    requests: list[dict[str, Any]]


class _Handler(http.server.BaseHTTPRequestHandler):
    server: _ToolHostStandIn

    def log_message(self, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        self.send_response(self.server.ready_status if self.path == tool_sidecar_host.READY_PATH else 404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers["Content-Length"]))
        seen = {
            "method": self.command,
            "path": self.path,
            "cookie": self.headers.get("Cookie"),
            "content_type": self.headers.get("Content-Type"),
            "body": json.loads(body),
        }
        self.server.requests.append(seen)
        # A body that is not UTF-8 still comes back as text.
        reply = json.dumps(seen).encode() + b"\xff"
        self.send_response(201)
        self.send_header("Set-Cookie", "session=signed-value; Path=/; HttpOnly")
        self.send_header("Set-Cookie", "other=2")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)


@pytest.fixture
def stand_in() -> Iterator[_ToolHostStandIn]:
    server = _ToolHostStandIn(("127.0.0.1", 0), _Handler)
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def _port(server: _ToolHostStandIn) -> str:
    return str(server.server_address[1])


def _closed_port() -> str:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return str(probe.getsockname()[1])


def test_the_probe_passes_only_once_the_tool_host_answers_ready(stand_in: _ToolHostStandIn) -> None:
    assert tool_sidecar_host.main(["tool_sidecar_host.py", "--ready", _port(stand_in)]) == 0

    stand_in.ready_status = 503
    assert tool_sidecar_host.main(["tool_sidecar_host.py", "--ready", _port(stand_in)]) == 1
    # Nothing listening yet, as before the tool host has started.
    assert tool_sidecar_host.main(["tool_sidecar_host.py", "--ready", _closed_port()]) == 1


def test_a_call_sends_the_request_with_its_cookies_and_writes_the_reply(
    stand_in: _ToolHostStandIn, tmp_path: Path
) -> None:
    request = tmp_path / "call.request.json"
    reply = tmp_path / "call.reply.json"
    request.write_text(
        json.dumps({"method": "POST", "path": "/seed_session", "body": {"task": 1}, "cookies": {"a": "1", "b": "2"}})
    )

    assert tool_sidecar_host.main(["tool_sidecar_host.py", "--call", _port(stand_in), str(request), str(reply)]) == 0

    assert stand_in.requests == [
        {
            "method": "POST",
            "path": "/seed_session",
            "cookie": "a=1; b=2",
            "content_type": "application/json",
            "body": {"task": 1},
        }
    ]
    written = json.loads(reply.read_text())
    assert written["status"] == 201
    assert written["cookies"] == {"session": "signed-value", "other": "2"}
    assert json.loads(written["body"].removesuffix("�")) == stand_in.requests[0]
    # The reply appears whole or not at all.
    assert not Path(f"{reply}.partial").exists()


def test_a_call_without_cookies_sends_none(stand_in: _ToolHostStandIn, tmp_path: Path) -> None:
    request = tmp_path / "call.request.json"
    request.write_text(json.dumps({"method": "POST", "path": "/_tool_sidecar/state", "body": {}, "cookies": {}}))

    tool_sidecar_host.main(["tool_sidecar_host.py", "--call", _port(stand_in), str(request), str(tmp_path / "r.json")])

    assert stand_in.requests[0]["cookie"] is None


def test_serve_runs_the_spec_files_server_on_every_address(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    served: dict[str, Any] = {}

    def run(app: Any, **kwargs: Any) -> None:
        served.update(kwargs, app=app)

    monkeypatch.setattr(uvicorn, "run", run)
    config = LedgerConfig.model_validate(
        {"name": "ledger", "host": "127.0.0.1", "port": 9999, "entrypoint": "app.py"}
    ).model_dump(mode="json")
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "module": "tests.unit_tests.environment.fixtures.tool_server.app",
                "class": "LedgerResourcesServer",
                "config": config,
                "port": 8123,
            }
        )
    )

    assert tool_sidecar_host.main(["tool_sidecar_host.py", str(spec)]) == 0

    assert (served["host"], served["port"]) == ("0.0.0.0", 8123)
    seed = ResourcesSeedSessionRequest(
        resources_session_id="session-1",
        episode_id=EpisodeId(rollout_id="rollout", attempt=1),
        task_id=TaskId(taskset="ledger:test", task_id="t-1"),
        task_data={"opening": "hi"},
    ).model_dump(mode="json")
    # The app it serves is the spec's server, as the tool host builds it.
    with TestClient(served["app"], client=("127.0.0.1", 40000)) as local:
        assert local.get(tool_sidecar_host.READY_PATH).json() == {"ready": True}
        assert local.post("/seed_session", json=seed).status_code == 200
        assert local.post("/pay", json={"value": 4}).text == "pay #1"


def test_the_tool_host_serves_readiness_where_the_environment_server_probes_it() -> None:
    assert tool_sidecar_host.READY_PATH == TOOL_HOST_READY_PATH
