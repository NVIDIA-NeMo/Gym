# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import json
import logging
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import MagicMock
from urllib.parse import parse_qs, urlsplit

import pytest
import requests

from nemo_gym.client import ServerClient
from resources_servers.google_search import app as google_search_app
from resources_servers.google_search.app import (
    BaseSearchQueryRequest,
    GoogleSearchResourcesServer,
    GoogleSearchResourcesServerConfig,
    box_parser,
)


class TestApp:
    def test_sanity(self) -> None:
        config = GoogleSearchResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            google_api_key="dummy_key",  # pragma: allowlist secret
            google_cx="dummy_cx",
        )
        GoogleSearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def test_box_parser_valid_content(self) -> None:
        """Test box_parser with valid boxed content"""
        # Test basic boxed content
        result = box_parser("The answer is \\boxed{42}")
        assert result == "42"

        # Test with complex content
        result = box_parser("After calculation: \\boxed{x + y = 10}")
        assert result == "x + y = 10"

        # Test with no boxed content
        result = box_parser("No boxed content here")
        assert result is None

        # Test with empty string
        result = box_parser("")
        assert result is None


FAKE_KEY = "AIza-FAKE-CANARY-KEY"  # pragma: allowlist secret


class _FakeGoogleHandler(BaseHTTPRequestHandler):
    """Records each request and answers with the class-level status and body."""

    status = 200
    body = b"{}"
    requests_seen: list[tuple[str, dict[str, str]]] = []

    def do_GET(self) -> None:
        type(self).requests_seen.append((self.path, dict(self.headers)))
        self.send_response(self.status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def fake_google(monkeypatch: pytest.MonkeyPatch):
    """Serve the Custom Search endpoint locally and redirect the server's requests to it."""
    handler = type("Handler", (_FakeGoogleHandler,), {"requests_seen": []})
    server = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    real_get = requests.get

    def get_via_local_server(url: str, **kwargs):
        return real_get(url.replace("https://www.googleapis.com", f"http://127.0.0.1:{server.server_port}"), **kwargs)

    monkeypatch.setattr(google_search_app.requests, "get", get_via_local_server)
    yield handler
    server.shutdown()
    server.server_close()


def _make_server() -> GoogleSearchResourcesServer:
    config = GoogleSearchResourcesServerConfig(
        host="127.0.0.1", port=0, entrypoint="", name="", google_api_key=FAKE_KEY, google_cx="fake-cx"
    )
    return GoogleSearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


class TestSearchKeyHandling:
    async def test_key_sent_in_header_not_url(self, fake_google) -> None:
        fake_google.body = json.dumps({"items": [{"title": "hit"}]}).encode()

        result = await _make_server().search(BaseSearchQueryRequest(query="anything"))

        assert json.loads(result.search_results) == {"items": [{"title": "hit"}]}
        [(path, headers)] = fake_google.requests_seen
        assert FAKE_KEY not in path
        assert parse_qs(urlsplit(path).query) == {"cx": ["fake-cx"], "q": ["anything"]}
        assert headers["X-goog-api-key"] == FAKE_KEY

    @pytest.mark.parametrize("status", [400, 403, 429, 500])
    async def test_http_error_output_has_status_only(
        self, fake_google, status: int, caplog: pytest.LogCaptureFixture
    ) -> None:
        fake_google.status = status

        with caplog.at_level(logging.WARNING, logger=google_search_app.__name__):
            result = await _make_server().search(BaseSearchQueryRequest(query="anything"))

        assert result.search_results == f"Error: search request failed with HTTP {status}"
        assert FAKE_KEY not in caplog.text
        assert str(status) in caplog.text

    async def test_connection_error_output_has_exception_type_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Bind then close a socket so the port is very likely refused.
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        real_get = requests.get
        monkeypatch.setattr(
            google_search_app.requests,
            "get",
            lambda url, **kwargs: real_get(
                url.replace("https://www.googleapis.com", f"http://127.0.0.1:{port}"), **kwargs
            ),
        )

        result = await _make_server().search(BaseSearchQueryRequest(query="anything"))

        assert result.search_results == "Error: search request failed (ConnectionError)"
