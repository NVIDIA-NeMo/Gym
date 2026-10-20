# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import io
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from responses_api_agents.claweval_agent import mcp_server


@pytest.fixture
def media_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> mcp_server.MediaServer:
    monkeypatch.setattr(mcp_server, "WORKSPACE", tmp_path)
    monkeypatch.setattr(mcp_server, "STATE_DIR", tmp_path / ".claw_tools")
    mcp_server.STATE_DIR.mkdir()
    return mcp_server.MediaServer()


@pytest.mark.parametrize("format", ["PNG", "JPEG", "WEBP", "BMP", "GIF"])
def test_image_payload_preserves_bytes_and_uses_decoded_mime(
    tmp_path: Path, media_server: mcp_server.MediaServer, format: str
) -> None:
    path = tmp_path / "image.dat"
    Image.new("RGB", (3, 2), (16, 32, 64)).save(path, format=format)

    result = media_server.call({"name": "ReadMedia", "arguments": {"path": str(path), "media_type": "image"}})

    assert result["isError"] is False
    payload = result["content"][1]
    assert payload["mimeType"] == Image.MIME[format]
    assert base64.b64decode(payload["data"]) == path.read_bytes()
    with Image.open(io.BytesIO(base64.b64decode(payload["data"]))) as image:
        image.load()
        assert image.size == (3, 2)
    assert media_server.frames == 1


def test_corrupt_image_returns_tool_error_and_preserves_frame_budget(
    tmp_path: Path, media_server: mcp_server.MediaServer
) -> None:
    path = tmp_path / "broken.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\nnot-an-image")

    result = media_server.call({"name": "ReadMedia", "arguments": {"path": str(path), "media_type": "image"}})

    assert result["isError"] is True
    assert "not a decodable image" in result["content"][0]["text"]
    assert all(item["type"] == "text" for item in result["content"])
    assert media_server.frames == 0
    metrics = json.loads((mcp_server.STATE_DIR / "metrics.json").read_text())
    assert metrics == {"media_tool_calls": 1, "media_tool_errors": 1, "media_frames": 0}

    Image.new("RGB", (2, 2)).save(path)
    recovered = media_server.call({"name": "ReadMedia", "arguments": {"path": str(path)}})
    assert recovered["isError"] is False
    assert media_server.frames == 1


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg/ffprobe not installed")
def test_video_in_image_mode_is_recoverable_tool_error(tmp_path: Path, media_server: mcp_server.MediaServer) -> None:
    path = tmp_path / "video.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-threads",
            "1",
            "-filter_threads",
            "1",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=16x16:r=5",
            "-frames:v",
            "2",
            "-c:v",
            "mpeg4",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    arguments = {
        "path": str(path),
        "media_type": "image",
        "start_time": 0,
        "end_time": 0.4,
        "max_frames": 20,
        "fps": 8,
        "include_native_media": False,
    }

    result = media_server.call({"name": "ReadMedia", "arguments": arguments})

    assert result["isError"] is True
    assert "media_type='video'" in result["content"][0]["text"]
    assert all(item["type"] == "text" for item in result["content"])
    assert media_server.frames == 0

    recovered = media_server.call({"name": "ReadMedia", "arguments": {**arguments, "media_type": "video"}})
    assert recovered["isError"] is False
    images = [item for item in recovered["content"] if item["type"] == "image"]
    assert images
    assert media_server.frames == len(images)
    for payload in images:
        assert payload["mimeType"] == "image/png"
        with Image.open(io.BytesIO(base64.b64decode(payload["data"]))) as image:
            image.load()
