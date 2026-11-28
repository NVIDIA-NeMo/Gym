# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from textwrap import dedent
from threading import Thread


def test_kernel_transport_reuses_pool_across_short_lived_event_loops() -> None:
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            body = json.dumps(
                {
                    "id": "response",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "policy",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": payload["messages"][0]["content"]},
                        }
                    ],
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    script = dedent("""
        import asyncio
        import sys
        from concurrent.futures import ThreadPoolExecutor
        from omegaconf import OmegaConf
        from nemo_gym import server_utils
        from responses_api_agents.spatialclaw_agent.app import _AiohttpChatCompletions

        # Isolate Gym's process-wide pool just as a separate Jupyter kernel does.
        server_utils.get_global_config_dict = lambda **kwargs: OmegaConf.create({})
        client = _AiohttpChatCompletions(sys.argv[1])

        async def query(text):
            response = await client.create(model="policy", messages=[{"role": "user", "content": text}])
            assert response.choices[0].message.content == text

        def fresh_loop(text):
            asyncio.run(query(text))

        # Native VLMModule closes a caller loop after each synchronous query.
        with ThreadPoolExecutor(max_workers=2) as pool:
            for text in ["first", "second"]:
                pool.submit(fresh_loop, text).result()
            list(pool.map(fresh_loop, ["third", "fourth"]))
    """)
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, f"http://127.0.0.1:{server.server_port}/v1"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert sorted(item["messages"][0]["content"] for item in requests) == ["first", "fourth", "second", "third"]
        assert "Event loop is closed" not in result.stderr
        assert "Unclosed client session" not in result.stderr
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
