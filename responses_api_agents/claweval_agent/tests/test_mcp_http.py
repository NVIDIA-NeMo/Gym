# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The native HTTP envelope preserves the existing media tool's schemas and replies."""

import json
from io import BytesIO

import pytest
from mcp import types

from responses_api_agents.claweval_agent.mcp_http import dispatch, make_handler
from responses_api_agents.claweval_agent.mcp_server import MediaServer


def test_initialize_tools_and_notification_use_real_mcp_types():
    media = MediaServer()
    reply = dispatch(
        media, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}}
    )
    assert (
        types.InitializeResult.model_validate(reply["result"]).model_dump(by_alias=True)["protocolVersion"]
        == "2025-03-26"
    )
    request = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
    assert dispatch(media, request) == media.handle(request)
    tools = types.ListToolsResult.model_validate(dispatch(media, request)["result"])
    assert {tool.name for tool in tools.tools} == {"ReadMedia", "BrowserScreenshot"}
    assert dispatch(media, {"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def handler(request, *, path="/mcp", instance="own"):
    kind = make_handler(MediaServer(), instance)
    obj = kind.__new__(kind)
    obj.path = path
    data = json.dumps(request).encode()
    obj.headers = {"Content-Length": str(len(data))}
    obj.rfile = BytesIO(data)
    obj.wfile = BytesIO()
    obj.send_response = lambda status: setattr(obj, "status", status)
    obj.send_header = lambda *args: None
    obj.end_headers = lambda: None
    return obj


def test_http_response_notification_and_readiness_are_scoped():
    obj = handler({"jsonrpc": "2.0", "method": "notifications/initialized"})
    obj.do_POST()
    assert obj.status == 202 and obj.wfile.getvalue() == b""
    obj = handler({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    obj.do_POST()
    assert obj.status == 200
    types.ListToolsResult.model_validate(json.loads(obj.wfile.getvalue())["result"])
    obj = handler({}, path="/health", instance="fresh-child")
    obj.do_GET()
    assert json.loads(obj.wfile.getvalue()) == {"service": "claweval-media", "instance": "fresh-child"}
    obj = handler({})
    obj.do_GET()
    assert obj.status == 405
    obj = handler({})
    obj.headers["Content-Length"] = str(1024 * 1024 + 1)
    obj.do_POST()
    assert obj.status == 413


def test_large_media_reply_preserves_bytes_without_one_whole_response_write() -> None:
    class RecordingWriter(BytesIO):
        largest_write: int = 0

        def write(self, data: bytes) -> int:
            self.largest_write = max(self.largest_write, len(data))
            return super().write(data)

    obj = handler({})
    obj.wfile = RecordingWriter()
    headers = {}
    obj.send_header = lambda name, value: headers.update({name: value})
    content = [{"type": "text", "text": 'Frame sequence: \u8336\n"'}]
    content.extend({"type": "image", "mimeType": "image/png", "data": "A" * 1024**2} for _ in range(8))
    body = {"jsonrpc": "2.0", "id": 1, "result": {"content": content, "isError": False}}
    expected = json.dumps(body, allow_nan=False).encode()

    obj.reply(200, body)

    assert obj.status == 200
    assert obj.wfile.getvalue() == expected
    assert headers["Content-Length"] == str(len(expected))
    assert headers["Connection"] == "close"
    assert obj.wfile.largest_write < len(expected) // 2
    assert obj.close_connection is True


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_invalid_reply_is_rejected_before_sending_headers(value: float) -> None:
    obj = handler({})
    statuses = []
    obj.send_response = statuses.append
    with pytest.raises(ValueError, match="Out of range float values"):
        obj.reply(200, {"content": "A" * 1024**2, "invalid": value})
    assert statuses == []
    assert obj.wfile.getvalue() == b""


@pytest.mark.asyncio
async def test_real_mcp_client_reads_a_task_image_over_local_http(tmp_path, monkeypatch):
    import asyncio
    import threading
    from http.server import HTTPServer

    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    from PIL import Image

    from responses_api_agents.claweval_agent import mcp_server

    monkeypatch.setattr(mcp_server, "WORKSPACE", tmp_path)
    monkeypatch.setattr(mcp_server, "STATE_DIR", tmp_path)
    Image.new("RGB", (4, 3), "blue").save(tmp_path / "test.png")
    httpd = HTTPServer(("127.0.0.1", 0), make_handler(MediaServer(), "unit-only"))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        async with asyncio.timeout(10):
            async with streamable_http_client(f"http://127.0.0.1:{httpd.server_port}/mcp") as (read, write, *_):
                async with ClientSession(read, write) as client:
                    await client.initialize()
                    tools = await client.list_tools()
                    assert {tool.name for tool in tools.tools} == {"ReadMedia", "BrowserScreenshot"}
                    result = await client.call_tool("ReadMedia", {"path": str(tmp_path / "test.png")})
                    reply = result.model_dump(by_alias=True)
                    assert not reply["isError"]
                    assert any(
                        item["type"] == "image" and item["mimeType"] == "image/png" for item in reply["content"]
                    )
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)
