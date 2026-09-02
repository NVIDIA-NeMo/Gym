# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nemo_gym.config_types import ResourcesServerRef
from nemo_gym.rollout_observability import AgentInvocation, ContextCompactionObservation
from nemo_gym.server_utils import ServerClient
from responses_api_agents.opencode_agent.app import OpenCodeAgent, OpenCodeAgentConfig


class _OpenAICompatibleHandler(BaseHTTPRequestHandler):
    requests: list[dict] = []
    overflow_sent = False

    def log_message(self, *_args) -> None:
        pass

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_completion(self, text: str) -> None:
        chunks = [
            {
                "id": "chatcmpl-opencode-overflow",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "probe-model",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}],
            },
            {
                "id": "chatcmpl-opencode-overflow",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "probe-model",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            },
        ]
        body = ("".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n").encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self._send_json(200, {"object": "list", "data": [{"id": "probe-model", "object": "model"}]})

    def do_POST(self) -> None:
        size = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(size) or b"{}")
        type(self).requests.append(request)

        if request.get("tools") and not type(self).overflow_sent:
            type(self).overflow_sent = True
            self._send_json(
                400,
                {
                    "error": {
                        "message": (
                            "This model's maximum context length is 128 tokens. However, you requested 256 tokens."
                        ),
                        "type": "BadRequestError",
                        "code": 400,
                    }
                },
            )
        elif type(self).overflow_sent and not request.get("tools"):
            self._send_completion("## Goal\n- Continue the task after compacting context.")
        elif type(self).overflow_sent:
            self._send_completion("Recovered after context compaction.")
        else:
            self._send_completion("OpenCode overflow recovery")


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("opencode") is None, reason="opencode is not installed")
async def test_opencode_compacts_after_openai_compatible_context_error(tmp_path: Path) -> None:
    _OpenAICompatibleHandler.requests = []
    _OpenAICompatibleHandler.overflow_sent = False
    server = ThreadingHTTPServer(("127.0.0.1", 0), _OpenAICompatibleHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        config = OpenCodeAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            resources_server=ResourcesServerRef(type="resources_servers", name=""),
            command=shutil.which("opencode") or "opencode",
            model="probe/probe-model",
            workspace_root=str(tmp_path / "workspaces"),
            timeout=30,
            opencode_config={
                "provider": {
                    "probe": {
                        "npm": "@ai-sdk/openai-compatible",
                        "options": {"baseURL": base_url, "apiKey": "EMPTY"},
                        "models": {
                            "probe-model": {"limit": {"context": 4096, "output": 256}},
                        },
                    }
                }
            },
        )
        with patch("responses_api_agents.opencode_agent.app.OpenCodeAgent.model_post_init"):
            agent = OpenCodeAgent(config=config, server_client=MagicMock(spec=ServerClient))

        output, _, _, observations = await agent._run_opencode("Continue after overflow.", None)
    finally:
        server.shutdown()
        server.server_close()
        await asyncio.to_thread(thread.join, 5)

    output_text = [part.text for item in output for part in getattr(item, "content", [])]
    root = next(record for record in observations.records if isinstance(record, AgentInvocation))
    compactions = [record for record in observations.records if isinstance(record, ContextCompactionObservation)]

    assert _OpenAICompatibleHandler.overflow_sent
    assert output_text[-1] == "Recovered after context compaction."
    assert root.status == "completed"
    assert [(record.trigger, record.outcome) for record in compactions] == [("overflow", "completed")]
    assert len(_OpenAICompatibleHandler.requests) >= 4
