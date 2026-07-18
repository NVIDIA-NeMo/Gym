#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Probe a Nemotron 3 Nano Omni endpoint with the configured image history."""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import struct
import sys
import urllib.error
import urllib.request
import zlib


def make_solid_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """Create a deterministic RGB PNG with no Pillow dependency."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", binascii.crc32(body) & 0xFFFFFFFF)

    scanline = b"\x00" + bytes(rgb) * width
    raw = scanline * height
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=6))
        + chunk(b"IEND", b"")
    )


def request_json(url: str, api_key: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method="GET" if data is None else "POST")
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="local-vllm")
    parser.add_argument(
        "--model",
        default="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16",
    )
    parser.add_argument("--models-only", action="store_true")
    parser.add_argument(
        "--image-count",
        type=int,
        choices=(1, 3),
        default=3,
        help="Use one current screenshot or the agent-equivalent three-turn/three-image history.",
    )
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    models = request_json(f"{base_url}/models", args.api_key)
    ids = [item.get("id") for item in models.get("data", [])]
    print(json.dumps({"endpoint": base_url, "models": ids}, ensure_ascii=False))
    if args.models_only:
        return 0

    colors = ((40, 180, 80), (50, 100, 210), (220, 130, 40))
    image_urls = [
        "data:image/png;base64,"
        + base64.b64encode(make_solid_png(args.width, args.height, colors[index])).decode("ascii")
        for index in range(args.image_count)
    ]
    messages = [{"role": "system", "content": "You are validating a GUI-agent vision transport."}]
    for index, image_url in enumerate(image_urls):
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {
                        "type": "text",
                        "text": f"Screenshot step {index + 1}. Describe its dominant color briefly.",
                    },
                ],
            }
        )
        if index < len(image_urls) - 1:
            messages.append(
                {
                    "role": "assistant",
                    "content": (
                        f"<think>The screenshot at step {index + 1} is visible.</think>\n"
                        f"## Action:\nObserve step {index + 1}.\n"
                        "## Code:\n```python\npyautogui.click(0.5, 0.5)\n```"
                    ),
                }
            )
    payload = {
        "model": args.model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 128,
    }
    result = request_json(f"{base_url}/chat/completions", args.api_key, payload)
    choice = result.get("choices", [{}])[0]
    message = choice.get("message", {})
    if not message.get("content") and not message.get("reasoning") and not message.get("reasoning_content"):
        raise RuntimeError(f"No content/reasoning in completion: {json.dumps(result)[:1000]}")
    print(
        json.dumps(
            {
                "finish_reason": choice.get("finish_reason"),
                "image_count": args.image_count,
                "image_size": [args.width, args.height],
                "content": message.get("content"),
                "has_reasoning": bool(message.get("reasoning") or message.get("reasoning_content")),
                "usage": result.get("usage"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"probe failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
