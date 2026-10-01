# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import litellm
import pytest
from minisweagent.models.litellm_response_model import LitellmResponseModel

from responses_api_agents.miniswe_sandboxed_agent import native


def test_native_responses_uses_keepalive_and_preserves_request_timeout(monkeypatch):
    requests = []
    sockets = []
    timeouts = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps(
                {"id": "resp_test", "object": "response", "created_at": 0, "model": "test", "output": []}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    send = httpx.Client.send

    def inspect_send(client, request, **kwargs):
        response = send(client, request, **kwargs)
        sock = response.extensions["network_stream"].get_extra_info("socket")
        sockets.append(sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE))
        idle = getattr(socket, "TCP_KEEPIDLE", getattr(socket, "TCP_KEEPALIVE", None))
        if idle is not None:
            assert sock.getsockopt(socket.IPPROTO_TCP, idle) == 60
        timeouts.append(request.extensions["timeout"]["read"])
        return response

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    original = litellm.responses

    def run_module(name, *, run_name):
        assert (name, run_name) == ("minisweagent.run.mini", "__main__")
        model = LitellmResponseModel(
            model_name="openai/test",
            model_kwargs={
                "api_base": f"http://127.0.0.1:{server.server_port}/v1",
                "api_key": "test",
                "timeout": 1234,
                "temperature": 1.0,
                "top_p": 0.95,
            },
        )
        assert model._query([{"role": "user", "content": "test"}]).id
        raise SystemExit(0)

    monkeypatch.setattr(httpx.Client, "send", inspect_send)
    monkeypatch.setattr(native.runpy, "run_module", run_module)
    try:
        with pytest.raises(SystemExit):
            native.main()
    finally:
        server.shutdown()
        server.server_close()
    assert litellm.responses is original
    assert len(sockets) == 1 and sockets[0] != 0
    assert timeouts == [1234]
    assert requests[0]["temperature"] == 1.0
    assert requests[0]["top_p"] == 0.95
    assert requests[0]["input"] == [{"role": "user", "content": "test"}]
