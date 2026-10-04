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
import html
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest


def pytest_configure(config):
    """Install the browser before any test launches one."""
    from resources_servers.timewarp.setup_chromium import ensure_chromium

    ensure_chromium()


ARTICLE_PARAGRAPHS = 300


def _page(title: str, body: str) -> str:
    return f"<!doctype html><html><head><title>{title}</title></head><body>{body}</body></html>"


def _render(path: str, query: dict, other_origin: str) -> str | None:
    if path == "/":
        return _page(
            "Home",
            "<h1>Home</h1>"
            '<a href="/article">Biology</a> '
            f'<a href="{other_origin}/">Elsewhere</a> '
            '<a href="/article" target="_blank">Biology in a new tab</a>'
            '<form action="/search"><input name="q" aria-label="Search articles"></form>'
            '<select aria-label="Era"><option value="2001">2001</option><option value="2025">2025</option></select>',
        )
    if path == "/article":
        paragraphs = "".join(f"<p>Paragraph {i} of the Biology article.</p>" for i in range(ARTICLE_PARAGRAPHS))
        return _page("Biology", f"<h1>Biology</h1>{paragraphs}")
    if path == "/search":
        term = html.escape(query.get("q", [""])[0])
        return _page("Search", f"<h1>Results for {term}</h1>")
    return None


def _serve(other_origin: list[str]):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parts = urlsplit(self.path)
            body = _render(parts.path, parse_qs(parts.query), other_origin[0])
            self.send_response(200 if body is not None else 404)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write((body or "not found").encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture(scope="session")
def site():
    """Two origins serving the same pages: ``allowed`` is the task's site, ``other`` is not."""
    other_origin: list[str] = [""]
    allowed, other = _serve(other_origin), _serve(other_origin)
    urls = {
        name: f"http://127.0.0.1:{server.server_address[1]}"
        for name, server in (("allowed", allowed), ("other", other))
    }
    other_origin[0] = urls["other"]
    yield urls
    allowed.shutdown()
    other.shutdown()
