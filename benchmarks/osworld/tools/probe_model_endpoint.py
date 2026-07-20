#!/usr/bin/env python3
"""Validate model identity and Nano Omni's multi-image request shape."""

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


def make_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        checksum = binascii.crc32(body) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + body + struct.pack(">I", checksum)

    rows = (b"\x00" + bytes(rgb) * width) * height
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows, level=6))
        + chunk(b"IEND", b"")
    )


def request_json(url: str, api_key: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Authorization": f"Bearer {api_key}"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    method = "GET" if data is None else "POST"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--models-only", action="store_true")
    parser.add_argument("--image-count", type=int, choices=(1, 3), default=3)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    models = request_json(f"{base_url}/models", args.api_key)
    model_ids = [item.get("id") for item in models.get("data", [])]
    if args.model not in model_ids:
        raise RuntimeError(f"expected model {args.model!r}, got {model_ids!r}")
    if args.models_only:
        print(json.dumps({"model_endpoint": "READY", "models": model_ids}))
        return 0

    colors = ((40, 180, 80), (50, 100, 210), (220, 130, 40))
    messages: list[dict] = [{"role": "system", "content": "Validate GUI-agent vision transport."}]
    for index in range(args.image_count):
        encoded = base64.b64encode(make_png(1920, 1080, colors[index])).decode()
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                    {"type": "text", "text": f"Describe screenshot {index + 1} briefly."},
                ],
            }
        )
        if index < args.image_count - 1:
            messages.append({"role": "assistant", "content": "## Action:\nObserve.\n## Code:\n```python\npass\n```"})

    result = request_json(
        f"{base_url}/chat/completions",
        args.api_key,
        {"model": args.model, "messages": messages, "temperature": 0, "max_tokens": 128},
    )
    choice = result.get("choices", [{}])[0]
    message = choice.get("message", {})
    if not any(message.get(key) for key in ("content", "reasoning", "reasoning_content")):
        raise RuntimeError("completion returned no content or reasoning")
    print(json.dumps({"model_endpoint": "READY", "image_count": args.image_count, "usage": result.get("usage")}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"model probe failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
