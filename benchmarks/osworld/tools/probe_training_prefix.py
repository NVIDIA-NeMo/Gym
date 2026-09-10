#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Probe vLLM's multi-turn token-prefix contract used by NeMo-RL training."""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import struct
import urllib.error
import urllib.request
import zlib


def _make_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
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


def _request_json(url: str, api_key: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc


def _image_message(label: str, rgb: tuple[int, int, int]) -> dict:
    encoded = base64.b64encode(_make_png(1920, 1080, rgb)).decode()
    return {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
            {"type": "text", "text": label},
        ],
    }


def _generation_token_ids(completion: dict) -> list[int]:
    content = completion["choices"][0]["logprobs"]["content"]
    tokens = [item["token"] for item in content]
    if not all(isinstance(token, str) and token.startswith("token_id:") for token in tokens):
        raise RuntimeError(f"vLLM did not return token_id:* logprob tokens: {tokens[:5]!r}")
    return [int(token.removeprefix("token_id:")) for token in tokens]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible URL ending in /v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    tokenize_url = base_url.removesuffix("/v1") + "/tokenize"
    messages = [
        {"role": "system", "content": "You are a GUI agent. Return a short next action."},
        _image_message("Screenshot 1: describe one safe next action.", (40, 180, 80)),
    ]
    common = {
        "model": args.model,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "logprobs": True,
        "top_logprobs": 0,
        "return_tokens_as_token_ids": True,
    }
    first = _request_json(
        f"{base_url}/chat/completions",
        args.api_key,
        {**common, "messages": messages},
    )
    first_prompt = _request_json(
        tokenize_url,
        args.api_key,
        {"model": args.model, "messages": messages},
    )["tokens"]
    first_generation = _generation_token_ids(first)
    first_message = first["choices"][0]["message"]
    reasoning = first_message.get("reasoning_content") or first_message.get("reasoning") or ""
    raw_content = (f"<think>{reasoning}</think>" if reasoning else "") + (first_message.get("content") or "")
    # Match the OSWorld Nano config's preserve_reasoning_in_assistant_content
    # path. Replaying vLLM's parsed `reasoning` field does not work for this
    # checkpoint because its chat template only consumes assistant `content`.
    assistant_message = {"role": "assistant", "content": raw_content}
    second_messages = [
        *messages,
        assistant_message,
        _image_message("Screenshot 2: verify and return the next action.", (50, 100, 210)),
    ]
    second_prompt = _request_json(
        tokenize_url,
        args.api_key,
        {"model": args.model, "messages": second_messages},
    )["tokens"]
    seen = [*first_prompt, *first_generation]
    prefix_ok = second_prompt[: len(seen)] == seen
    result = {
        "training_prefix": "READY" if prefix_ok else "FAILED",
        "first_prompt_tokens": len(first_prompt),
        "first_generation_tokens": len(first_generation),
        "second_prompt_tokens": len(second_prompt),
        "expected_prefix_tokens": len(seen),
        "prefix_ok": prefix_ok,
    }
    print(json.dumps(result, sort_keys=True))
    if not prefix_ok:
        mismatch = next(
            (index for index, (actual, expected) in enumerate(zip(second_prompt, seen)) if actual != expected),
            min(len(second_prompt), len(seen)),
        )
        expected_window = seen[max(0, mismatch - 8) : mismatch + 32]
        actual_window = second_prompt[max(0, mismatch - 8) : mismatch + 32]
        expected_text = _request_json(
            base_url.removesuffix("/v1") + "/detokenize",
            args.api_key,
            {"model": args.model, "tokens": expected_window},
        ).get("prompt")
        actual_text = _request_json(
            base_url.removesuffix("/v1") + "/detokenize",
            args.api_key,
            {"model": args.model, "tokens": actual_window},
        ).get("prompt")
        raise RuntimeError(
            "non-contiguous vLLM messages: "
            f"first mismatch at token {mismatch}; "
            f"expected={seen[mismatch : mismatch + 8]!r}, actual={second_prompt[mismatch : mismatch + 8]!r}; "
            f"expected_text={expected_text!r}, actual_text={actual_text!r}; "
            f"message_fields={sorted(first_message)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
