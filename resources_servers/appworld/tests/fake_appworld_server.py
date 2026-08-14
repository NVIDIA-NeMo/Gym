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
"""Stand-in for ``appworld serve environment``, used by the worker-pool tests.

Speaks just enough of the real environment server's contract — ``GET /`` for the
readiness probe and ``POST`` returning ``{"output": ...}`` — that the pool's
spawn / readiness / call / respawn / shutdown paths can be tested without
installing AppWorld. ``POST /boom`` returns a 500 so the error path is reachable.

Invoked exactly the way the pool invokes the real CLI:

    fake_appworld serve environment --port P --root R --no-show-usage
"""

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
        self._send({"message": "Welcome to the fake AppWorld Server!"})

    def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/boom":
            self._send({"detail": "worker exploded"}, status=500)
        elif self.path == "/slow":
            time.sleep(30)
            self._send({"output": "eventually"})
        elif self.path == "/bare":
            self._send({"echo": payload})
        else:
            self._send({"output": payload})

    def log_message(self, *args) -> None:  # silence the default stderr spam
        return


def main() -> None:
    # Threading, like the real (uvicorn-backed) server: aiohttp keeps its
    # connection alive after the readiness probe, and a single-threaded server
    # would then block every later request behind that idle connection.
    args = sys.argv[1:]
    port = int(args[args.index("--port") + 1])
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    server.serve_forever()


if __name__ == "__main__":
    main()
