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
"""Capture actual pinned Codex delegation with a scripted, network-isolated provider.

This fixture establishes trace formats only. It does not score a benchmark or
exercise the host gateway. Run inside the benchmark image with /input and
/evidence mounted; /input/launch.py is the normal Codex launch profile.
"""

import json
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path("/evidence")
LOCK = threading.Lock()
CHILD = threading.Event()
ROOT_THREAD = None
COUNT = 0


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        global COUNT, ROOT_THREAD
        raw = self.rfile.read(int(self.headers["Content-Length"]))
        payload = json.loads(raw)
        with LOCK:
            index = COUNT
            COUNT += 1
            sid = payload["client_metadata"]["thread_id"]
            if ROOT_THREAD is None:
                ROOT_THREAD = sid
            child = sid != ROOT_THREAD
            (ROOT / f"request-{index}.json").write_bytes(raw)
        if index == 0:
            config = json.loads(Path("/input/config.json").read_text())
            item = {
                "type": "function_call",
                "id": "fc-spawn",
                "call_id": "call-spawn",
                "name": "spawn_agent",
                "namespace": "multi_agent_v1",
                "arguments": json.dumps({"message": "Reply fixture.", "fork_context": config["fixture_fork"]}),
            }
        else:
            if child:
                CHILD.set()
            elif not CHILD.wait(20):
                (ROOT / "child-timeout").write_text("Missing child request")
            item = {
                "id": f"msg-{index}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "fixture", "annotations": []}],
            }
        response = {
            "id": f"resp-{index}",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "output": [item],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }
        events = [
            ("response.created", {"response": {**response, "status": "in_progress", "output": []}}),
            ("response.output_item.added", {"output_index": 0, "item": item}),
            ("response.output_item.done", {"output_index": 0, "item": item}),
            ("response.completed", {"response": response}),
        ]
        data = "".join(
            "event: " + name + "\ndata: " + json.dumps({"type": name, **body}) + "\n\n" for name, body in events
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 18999), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        result = subprocess.run(["python3", "/input/launch.py"], input=b"run\n", capture_output=True, timeout=60)
        (ROOT / "process.jsonl").write_bytes(result.stdout)
        (ROOT / "process.stderr").write_bytes(result.stderr)
        sessions = Path("/home/agent/native-codex/config/sessions")
        if sessions.exists():
            shutil.copytree(sessions, ROOT / "sessions")
        if result.returncode or not CHILD.is_set() or (ROOT / "child-timeout").exists():
            raise RuntimeError("Scripted Codex capture failed; inspect its retained evidence")
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
