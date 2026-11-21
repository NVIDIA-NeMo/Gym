# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Episode-local MCP media tools for the ClawEval OpenCode sandbox."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image


WORKSPACE = Path("/workspace").resolve()
STATE_DIR = WORKSPACE / ".claw_tools"
# Explicit sandbox budgets, separate from the SFT tool's seven arguments. A
# valid 256-frame request is never silently shortened to the remaining budget.
MAX_MEDIA_FRAMES = 256
MAX_EPISODE_FRAMES = 4096


def _media_output_budget(value: str | None) -> int:
    """Read the deployment's bounded byte allowance without changing frame pixels."""
    if value is None:
        return 256 * 1024 * 1024
    try:
        budget = int(value)
    except ValueError as exc:
        raise ValueError("CLAW_MEDIA_MAX_OUTPUT_BYTES must be an integer from 1 through 1073741824") from exc
    if not 1 <= budget <= 1024**3:
        raise ValueError("CLAW_MEDIA_MAX_OUTPUT_BYTES must be an integer from 1 through 1073741824")
    return budget


MAX_MEDIA_OUTPUT_BYTES = _media_output_budget(os.environ.get("CLAW_MEDIA_MAX_OUTPUT_BYTES"))
MAX_SCREEN_DIMENSION = 4096

READ_MEDIA_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Path to the image, PDF, audio, or video file."},
        "media_type": {
            "type": "string",
            "enum": ["auto", "image", "audio", "video", "pdf"],
            "default": "auto",
        },
        "max_frames": {"type": "integer", "minimum": 1, "maximum": MAX_MEDIA_FRAMES, "default": 8},
        "fps": {"type": "number", "exclusiveMinimum": 0, "maximum": 8, "default": 1},
        "start_time": {"type": "number", "minimum": 0, "default": 0},
        "end_time": {"type": ["number", "null"]},
        "screen_size": {"type": "string", "description": "Resize output dimension, e.g. '1280x720'."},
    },
    "required": ["path"],
    "additionalProperties": False,
}

SCREENSHOT_SCHEMA = {
    "type": "object",
    "properties": {
        "url": {"type": "string"},
        "wait_seconds": {"type": "number", "minimum": 0, "maximum": 30, "default": 2},
        "frame_count": {"type": "integer", "minimum": 1, "maximum": 8, "default": 4},
        "viewport_width": {"type": "integer", "minimum": 320, "maximum": 1920, "default": 1080},
        "viewport_height": {"type": "integer", "minimum": 240, "maximum": 1080, "default": 720},
    },
    "required": ["url"],
}


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _workspace_path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("path must be a non-empty string")
    path = Path(value)
    path = (WORKSPACE / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(WORKSPACE) or not path.is_file():
        raise ValueError("media path must identify a file under /workspace")
    return path


def _screen_dimensions(value: object) -> tuple[int, int]:
    if not isinstance(value, str):
        raise ValueError("screen_size must be a WIDTHxHEIGHT string")
    try:
        width, height = map(int, value.lower().split("x", 1))
    except ValueError as exc:
        raise ValueError("screen_size must be WIDTHxHEIGHT") from exc
    if not 32 <= width <= MAX_SCREEN_DIMENSION or not 32 <= height <= MAX_SCREEN_DIMENSION:
        raise ValueError(f"screen_size dimensions must be between 32 and {MAX_SCREEN_DIMENSION}")
    return width, height


def _image(path: Path, *, screen_size: tuple[int, int] | None = None) -> dict:
    if path.stat().st_size > MAX_MEDIA_OUTPUT_BYTES:
        raise ValueError(f"Media output exceeds the {MAX_MEDIA_OUTPUT_BYTES}-byte tool budget")
    raw = path.read_bytes()
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            mime = Image.MIME.get(image.format)
            if screen_size is not None:
                image.thumbnail(screen_size, Image.Resampling.LANCZOS)
                output = io.BytesIO()
                image.convert("RGBA" if "A" in image.getbands() else "RGB").save(output, format="PNG")
                raw = output.getvalue()
                mime = "image/png"
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise ValueError(f"{path.name} is not a decodable image. Use media_type='video' for video files.") from exc
    if mime is None:
        raise ValueError(f"{path.name} has an unsupported image format")
    if len(raw) > MAX_MEDIA_OUTPUT_BYTES:
        raise ValueError(f"Media output exceeds the {MAX_MEDIA_OUTPUT_BYTES}-byte tool budget")
    return {
        "type": "image",
        "mimeType": mime,
        "data": base64.b64encode(raw).decode(),
    }


def _probe(path: Path) -> dict:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    record = json.loads(completed.stdout)
    stream = record["streams"][0]
    return {
        "duration_s": float(record["format"]["duration"]),
        "width": stream["width"],
        "height": stream["height"],
    }


class MediaServer:
    def __init__(self) -> None:
        self.calls = 0
        self.errors = 0
        self.frames = 0

    def _write_metrics(self) -> None:
        (STATE_DIR / "metrics.json").write_text(
            json.dumps(
                {
                    "media_tool_calls": self.calls,
                    "media_tool_errors": self.errors,
                    "media_frames": self.frames,
                },
                indent=2,
            )
            + "\n"
        )

    def _remaining(self, requested: int, *, maximum: int = 64) -> int:
        if type(requested) is not int or not 1 <= requested <= maximum:
            raise ValueError(f"frame count must be an integer from 1 to {maximum}")
        remaining = MAX_EPISODE_FRAMES - self.frames
        if requested > remaining:
            raise ValueError(
                f"Requested {requested} frames exceeds the remaining episode media budget of {max(0, remaining)}"
            )
        return requested

    def read_media(self, args: dict) -> dict:
        unknown = set(args) - READ_MEDIA_SCHEMA["properties"].keys()
        if unknown:
            raise ValueError(f"Unsupported ReadMedia arguments: {', '.join(sorted(unknown))}")
        path = _workspace_path(args.get("path"))
        requested_type = args.get("media_type", "auto")
        if requested_type not in READ_MEDIA_SCHEMA["properties"]["media_type"]["enum"]:
            raise ValueError("media_type must be auto, image, audio, video, or pdf")
        suffix = path.suffix.lower()
        media_type = requested_type
        if requested_type == "auto":
            media_type = (
                "image"
                if suffix in {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}
                else "video"
                if suffix in {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
                else "audio"
                if suffix in {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav"}
                else "pdf"
                if suffix == ".pdf"
                else "unknown"
            )
        if media_type == "image":
            self._remaining(1)
            size = _screen_dimensions(args["screen_size"]) if "screen_size" in args else None
            image = _image(path, screen_size=size)
            self.frames += 1
            return {
                "content": [
                    {"type": "text", "text": f"Image: {path.name}"},
                    image,
                ],
                "isError": False,
            }
        if media_type != "video":
            raise ValueError(f"This environment supports image/video media, not {media_type}")

        start = _finite_number(args.get("start_time", 0), "start_time")
        raw_end = args.get("end_time")
        end = None if raw_end is None else _finite_number(raw_end, "end_time")
        fps = _finite_number(args.get("fps", 1), "fps")
        if start < 0 or (end is not None and end <= start) or not 0 < fps <= 8:
            raise ValueError("Require 0 <= start_time < end_time within the video and 0 < fps <= 8")
        count = self._remaining(args.get("max_frames", 8), maximum=MAX_MEDIA_FRAMES)
        width, height = _screen_dimensions(args.get("screen_size", "1280x720"))
        info = _probe(path)
        duration = _finite_number(info["duration_s"], "video duration")
        if not start < duration:
            raise ValueError("start_time must be within the video")
        end = duration if end is None else min(end, duration)
        key = hashlib.sha256(
            json.dumps([str(path), path.stat().st_mtime_ns, start, end, fps, count, width, height]).encode()
        ).hexdigest()
        destination = STATE_DIR / "frames" / key
        destination.mkdir(parents=True, exist_ok=True)
        if not list(destination.glob("*.png")):
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-threads",
                    "1",
                    "-ss",
                    str(start),
                    "-i",
                    str(path),
                    "-t",
                    str(end - start),
                    "-vf",
                    f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=decrease",
                    "-frames:v",
                    str(count),
                    "-filter_threads",
                    "1",
                    str(destination / "%03d.png"),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=180,
            )
        frames = sorted(destination.glob("*.png"))
        if not frames:
            raise ValueError("No frames were decoded from the selected video interval")
        if len(frames) > count:
            raise ValueError("Decoded frame count exceeds max_frames")
        output_bytes = sum(frame.stat().st_size for frame in frames)
        if output_bytes > MAX_MEDIA_OUTPUT_BYTES:
            raise ValueError(f"Media output exceeds the {MAX_MEDIA_OUTPUT_BYTES}-byte tool budget")
        images = [_image(frame) for frame in frames]
        self.frames += len(images)
        return {
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Video: {info['duration_s']:.3f}s, {info['width']}x{info['height']}. "
                        f"Returned {len(frames)} chronological frames from {start:.3f}s to {end:.3f}s."
                    ),
                },
                *images,
            ],
            "isError": False,
        }

    def browser_screenshot(self, args: dict) -> dict:
        url = args.get("url")
        if not isinstance(url, str) or urlparse(url).scheme not in {"http", "https", "file"}:
            raise ValueError("url must use http, https, or file")
        wait = _finite_number(args.get("wait_seconds", 2), "wait_seconds")
        if not 0 <= wait <= 30:
            raise ValueError("wait_seconds must be between 0 and 30")
        count = self._remaining(args.get("frame_count", 4))
        width = args.get("viewport_width", 1080)
        height = args.get("viewport_height", 720)
        if type(width) is not int or type(height) is not int or not 320 <= width <= 1920 or not 240 <= height <= 1080:
            raise ValueError("viewport dimensions are outside the supported range")
        bundled = [
            *Path("/ms-playwright").glob("chromium-*/chrome-linux/chrome"),
            *Path("/ms-playwright").glob("chromium-*/chrome-linux64/chrome"),
            *Path("/ms-playwright").glob(
                "chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell"
            ),
        ]
        chromium = shutil.which("chromium") or shutil.which("chromium-browser")
        if chromium is None and bundled:
            chromium = str(sorted(bundled)[-1])
        if not chromium:
            raise RuntimeError("chromium is not installed")
        output = Path(tempfile.mkdtemp(prefix="browser-", dir=STATE_DIR))
        frames = []
        for index in range(count):
            delay_ms = round(wait * 1000 * (index + 1) / count)
            target = output / f"{index:03d}.png"
            subprocess.run(
                [
                    chromium,
                    "--headless",
                    "--no-sandbox",
                    "--disable-gpu",
                    "--hide-scrollbars",
                    f"--user-data-dir={output / f'profile-{index}'}",
                    f"--window-size={width},{height}",
                    f"--virtual-time-budget={delay_ms}",
                    f"--screenshot={target}",
                    url,
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=max(60, int(wait) + 30),
            )
            frames.append(target)
        self.frames += len(frames)
        return {
            "content": [
                {"type": "text", "text": f"Captured {len(frames)} screenshots over {wait:.3f}s."},
                *[_image(frame) for frame in frames],
            ],
            "isError": False,
        }

    def call(self, params: dict) -> dict:
        self.calls += 1
        try:
            name = params.get("name")
            arguments = params.get("arguments") or {}
            if not isinstance(arguments, dict):
                raise ValueError("tool arguments must be an object")
            if name == "ReadMedia":
                result = self.read_media(arguments)
            elif name == "BrowserScreenshot":
                result = self.browser_screenshot(arguments)
            else:
                raise ValueError(f"Unknown ClawEval media tool: {name}")
        except (ValueError, TypeError, KeyError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
            self.errors += 1
            result = {"content": [{"type": "text", "text": str(exc)[:1000]}], "isError": True}
        self._write_metrics()
        return result

    def handle(self, message: dict) -> dict | None:
        if "id" not in message:
            return None
        method = message.get("method")
        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "claweval-media", "version": "1.0.0"},
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "ReadMedia",
                        "description": (
                            "Read a workspace image or sample a video clip at the requested fps, "
                            "returning at most max_frames visual frames. Supports up to 256 frames "
                            f"per call, a 4096-frame episode budget, and {MAX_MEDIA_OUTPUT_BYTES // (1024**2)} MiB "
                            "of image data per call. "
                            "Audio and PDF decoding are unavailable in this environment."
                        ),
                        "inputSchema": READ_MEDIA_SCHEMA,
                    },
                    {
                        "name": "BrowserScreenshot",
                        "description": "Capture chronological screenshots of a webpage or animation.",
                        "inputSchema": SCREENSHOT_SCHEMA,
                    },
                ]
            }
        elif method == "tools/call":
            result = self.call(message.get("params") or {})
        elif method == "ping":
            result = {}
        else:
            return {
                "jsonrpc": "2.0",
                "id": message["id"],
                "error": {"code": -32601, "message": "Method not found"},
            }
        return {"jsonrpc": "2.0", "id": message["id"], "result": result}


def main() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    server = MediaServer()
    for line in sys.stdin:
        response = server.handle(json.loads(line))
        if response is not None:
            print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
