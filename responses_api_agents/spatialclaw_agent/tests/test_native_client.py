# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from textwrap import dedent
from threading import Thread
from unittest.mock import AsyncMock

import pytest

from nemo_gym.openai_utils import NeMoGymAsyncOpenAI
from responses_api_agents.spatialclaw_agent.native_client import GymChatClient, create_native_client


async def test_native_transport_preserves_extra_body_and_usage(monkeypatch) -> None:
    send = AsyncMock(
        return_value={
            "id": "native-response",
            "object": "chat.completion",
            "created": 0,
            "model": "policy",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "B"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        }
    )
    monkeypatch.setattr(NeMoGymAsyncOpenAI, "create_chat_completion", send)
    client = GymChatClient(base_url="http://policy/v1/", api_key="gym")
    messages = [{"role": "user", "content": "Where is the cup?"}]

    response = await client.chat.completions.create(
        model="policy", messages=messages, extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    )

    send.assert_awaited_once_with(model="policy", messages=messages, chat_template_kwargs={"enable_thinking": False})
    assert response.choices[0].message.content == "B"
    assert response.usage.prompt_tokens == 10
    assert response.usage.completion_tokens == 3
    assert client._client.base_url == "http://policy/v1"
    await client.close()


async def test_native_transport_rejects_streaming() -> None:
    client = GymChatClient(base_url="http://policy/v1", api_key="gym")
    with pytest.raises(ValueError, match="non-streaming"):
        await client.chat.completions.create(model="policy", messages=[], stream=True)


@pytest.mark.skipif(not os.environ.get("SPATIALCLAW_ROOT"), reason="requires the original SpatialClaw checkout")
def test_original_client_survives_kernel_serialization(monkeypatch) -> None:
    import cloudpickle

    monkeypatch.syspath_prepend(str(Path(os.environ["SPATIALCLAW_ROOT"]).resolve()))
    from spatial_agent.config import SpatialAgentConfig
    from spatial_agent.llm.client import LLMClient

    config = SpatialAgentConfig(llm_base_url="http://policy/v1", llm_model="policy", llm_api_key="gym")
    client = create_native_client(config)
    assert isinstance(client, LLMClient)
    assert isinstance(client._get_client("http://policy/v1"), GymChatClient)

    restored = cloudpickle.loads(cloudpickle.dumps(client))
    assert restored._client_pool == {}
    assert isinstance(restored._get_client("http://policy/v1"), GymChatClient)
    assert restored._get_client("http://policy/v1")._kernel_transport
    assert not client._get_client("http://policy/v1")._kernel_transport


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
        from responses_api_agents.spatialclaw_agent.native_client import GymChatClient

        # Isolate Gym's process-wide pool just as a separate Jupyter kernel does.
        server_utils.get_global_config_dict = lambda **kwargs: OmegaConf.create({})
        client = GymChatClient(base_url=sys.argv[1], api_key="gym", kernel_transport=True)

        async def query(text):
            response = await client.create(model="policy", messages=[{"role": "user", "content": text}])
            assert response.choices[0].message.content == text

        def fresh_loop(text):
            asyncio.run(query(text))

        # Native VLMModule closes a caller loop after each synchronous query.
        for text in ["first", "second"]:
            fresh_loop(text)
        # Calls can also come from different native VLM worker threads.
        with ThreadPoolExecutor(max_workers=2) as pool:
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
