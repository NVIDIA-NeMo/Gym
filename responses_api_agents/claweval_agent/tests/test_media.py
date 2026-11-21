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


def test_output_byte_allowance_preserves_default_and_accepts_full_video_budget() -> None:
    assert mcp_server._media_output_budget(None) == 256 * 1024**2
    assert mcp_server._media_output_budget(str(1024**3)) == 1024**3


@pytest.mark.parametrize("value", ["", "False", "0", "-1", "1.5", str(1024**3 + 1)])
def test_invalid_output_byte_allowance_is_rejected(value: str) -> None:
    with pytest.raises(ValueError, match="CLAW_MEDIA_MAX_OUTPUT_BYTES"):
        mcp_server._media_output_budget(value)


def test_tool_description_uses_the_configured_byte_allowance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_server, "MAX_MEDIA_OUTPUT_BYTES", 1024**3)
    reply = mcp_server.MediaServer().handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert "1024 MiB" in reply["result"]["tools"][0]["description"]


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


@pytest.mark.parametrize("size,expected", [("64x32", (51, 32)), ("32x64", (32, 20))])
def test_image_screen_size_resizes_within_requested_bound(
    tmp_path: Path, media_server: mcp_server.MediaServer, size, expected
) -> None:
    path = tmp_path / "image.jpg"
    Image.new("RGB", (128, 80), (16, 32, 64)).save(path)
    result = media_server.call({"name": "ReadMedia", "arguments": {"path": str(path), "screen_size": size}})
    assert result["isError"] is False
    payload = result["content"][1]
    assert payload["mimeType"] == "image/png"
    with Image.open(io.BytesIO(base64.b64decode(payload["data"]))) as image:
        assert image.size == expected
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


def test_read_media_schema_matches_seven_sft_arguments() -> None:
    schema = mcp_server.READ_MEDIA_SCHEMA
    assert set(schema["properties"]) == {
        "path",
        "media_type",
        "max_frames",
        "fps",
        "start_time",
        "end_time",
        "screen_size",
    }
    assert schema["properties"]["screen_size"]["type"] == "string"
    assert schema["properties"]["max_frames"]["default"] == 8
    assert schema["properties"]["fps"]["default"] == 1
    assert schema["properties"]["start_time"]["default"] == 0
    assert schema["properties"]["media_type"]["default"] == "auto"
    assert schema["additionalProperties"] is False


@pytest.fixture
def decoded_video(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Exercise the real tool/PNG serialization; substitute only the decoder process."""
    path = tmp_path / "video.mp4"
    path.write_bytes(b"decoder-input-fixture")
    commands = []
    monkeypatch.setattr(mcp_server, "_probe", lambda _path: {"duration_s": 7200, "width": 2560, "height": 1440})

    def decode(command, **kwargs):
        assert kwargs["check"] is True
        assert kwargs["timeout"] == 180
        commands.append(command)
        count = int(command[command.index("-frames:v") + 1])
        pattern = command[-1]
        for index in range(1, count + 1):
            Image.new("RGB", (2, 2), (index % 256, 0, 0)).save(pattern % index)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(mcp_server.subprocess, "run", decode)
    return path, commands


@pytest.mark.parametrize(
    "frames,fps,start,end,size",
    [
        (190, 0.25, 0, 760, "1280x720"),
        (30, 2, 260, 275, "1280x720"),
        (256, 8, 6560, 7020.947, "2560x1440"),
        (8, 0.1, 0, None, "720x1280"),
    ],
)
def test_sft_video_calls_preserve_fps_clip_size_and_full_count(
    media_server: mcp_server.MediaServer, decoded_video, frames, fps, start, end, size
) -> None:
    path, commands = decoded_video
    result = media_server.call(
        {
            "name": "ReadMedia",
            "arguments": {
                "path": str(path),
                "max_frames": frames,
                "fps": fps,
                "start_time": start,
                "end_time": end,
                "screen_size": size,
            },
        }
    )
    assert result["isError"] is False
    assert len([item for item in result["content"] if item["type"] == "image"]) == frames
    assert media_server.frames == frames
    command = commands[0]
    width, height = size.split("x")
    assert command[command.index("-vf") + 1] == (
        f"fps={float(fps)},scale={width}:{height}:force_original_aspect_ratio=decrease"
    )
    assert float(command[command.index("-ss") + 1]) == start
    assert float(command[command.index("-t") + 1]) == (7200 if end is None else end) - start
    assert int(command[command.index("-frames:v") + 1]) == frames


def test_video_defaults_match_sft_contract(media_server: mcp_server.MediaServer, decoded_video) -> None:
    path, commands = decoded_video
    result = media_server.call({"name": "ReadMedia", "arguments": {"path": str(path)}})
    assert result["isError"] is False
    assert media_server.frames == 8
    assert commands[0][commands[0].index("-vf") + 1] == ("fps=1.0,scale=1280:720:force_original_aspect_ratio=decrease")


@pytest.mark.parametrize("size", ["1280x720", "2560x1440", "720x1280"])
def test_screen_size_is_a_string_not_an_integer(media_server: mcp_server.MediaServer, decoded_video, size) -> None:
    path, _ = decoded_video
    result = media_server.call(
        {"name": "ReadMedia", "arguments": {"path": str(path), "screen_size": size, "max_frames": 1}}
    )
    assert result["isError"] is False


@pytest.mark.parametrize(
    "arguments",
    [
        {"max_frames": 257},
        {"max_frames": True},
        {"max_frames": 1.5},
        {"fps": 0},
        {"fps": 9},
        {"fps": float("nan")},
        {"fps": float("inf")},
        {"start_time": -1},
        {"start_time": True},
        {"end_time": 0},
        {"screen_size": 720},
        {"screen_size": None},
        {"screen_size": "4097x720"},
        {"screen_size": "1280x720x3"},
        {"media_type": "movie"},
        {"include_native_media": True},
        {"pdf_pages": "all"},
        {"dpi": 100},
        {"max_media_bytes": 100},
    ],
)
def test_invalid_or_extra_arguments_return_tool_error_without_decoding(
    media_server: mcp_server.MediaServer, decoded_video, arguments
) -> None:
    path, commands = decoded_video
    result = media_server.call({"name": "ReadMedia", "arguments": {"path": str(path), **arguments}})
    assert result["isError"] is True
    assert media_server.frames == 0
    assert not commands


def test_episode_budget_returns_error_instead_of_shortening_request(
    media_server: mcp_server.MediaServer, decoded_video
) -> None:
    path, commands = decoded_video
    media_server.frames = mcp_server.MAX_EPISODE_FRAMES - 100
    result = media_server.call({"name": "ReadMedia", "arguments": {"path": str(path), "max_frames": 256}})
    assert result["isError"] is True
    assert "remaining episode media budget of 100" in result["content"][0]["text"]
    assert not commands
    assert media_server.frames == mcp_server.MAX_EPISODE_FRAMES - 100


def test_output_budget_returns_error_before_accounting_or_encoding(
    media_server: mcp_server.MediaServer, decoded_video, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = decoded_video
    monkeypatch.setattr(mcp_server, "MAX_MEDIA_OUTPUT_BYTES", 1)
    result = media_server.call({"name": "ReadMedia", "arguments": {"path": str(path), "max_frames": 2}})
    assert result["isError"] is True
    assert "byte tool budget" in result["content"][0]["text"]
    assert media_server.frames == 0


@pytest.mark.parametrize("media_type,suffix", [("audio", ".wav"), ("pdf", ".pdf")])
def test_declared_but_unavailable_media_modes_are_explicit_tool_errors(
    tmp_path: Path, media_server: mcp_server.MediaServer, media_type, suffix
) -> None:
    path = tmp_path / f"unsupported{suffix}"
    path.write_bytes(b"fixture")
    result = media_server.call({"name": "ReadMedia", "arguments": {"path": str(path), "media_type": media_type}})
    assert result["isError"] is True
    assert f"not {media_type}" in result["content"][0]["text"]
    assert media_server.frames == 0


def test_workspace_containment_rejects_outside_and_symlink_paths(
    tmp_path: Path, media_server: mcp_server.MediaServer
) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.png"
    Image.new("RGB", (2, 2)).save(outside)
    link = tmp_path / "escaped.png"
    link.symlink_to(outside)
    for path in [str(outside), str(link), "../outside.png", "filesystem:///etc/passwd"]:
        result = media_server.call({"name": "ReadMedia", "arguments": {"path": path}})
        assert result["isError"] is True
        assert "under /workspace" in result["content"][0]["text"]
    assert media_server.frames == 0
