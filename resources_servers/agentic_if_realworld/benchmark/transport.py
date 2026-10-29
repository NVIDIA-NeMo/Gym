# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded, receipted HTTP and lossless SSE decoding. Credentials stay in memory."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from .core import digest, judge, read, write


class SSE:
    """Concatenate historical fields and normalize modern reasoning deltas."""

    def __init__(self):
        self.meta, self.message, self.finish, self.usage = (
            {},
            {"role": "assistant", "content": ""},
            None,
            None,
        )

    def event(self, data: str) -> bool:
        if data == "[DONE]":
            return True
        value = json.loads(data)
        if "error" in value:
            raise ValueError("Error event, not model delivery")
        for key in ["id", "model", "created", "system_fingerprint"]:
            if key in value:
                self.meta[key] = value[key]
        if value.get("usage") is not None:
            self.usage = value["usage"]
        choices = value.get("choices", [])
        if len(choices) > 1:
            raise ValueError("Multiple streamed choices")
        for choice in choices:
            if choice.get("index", 0) != 0:
                raise ValueError("Unexpected streamed choice index")
            delta = choice.get("delta", {})
            if delta.get("role", "assistant") != "assistant":
                raise ValueError("Unexpected streamed role")
            if delta.get("tool_calls") or delta.get("function_call"):
                raise ValueError("Native call in text-only delivery")
            reasoning = delta.get("reasoning")
            if reasoning is not None:
                if not isinstance(reasoning, str):
                    raise ValueError("Unsupported stream field")
                if delta.get("reasoning_content") is not None and delta["reasoning_content"] != reasoning:
                    raise ValueError("Contradictory reasoning fields")
                delta = {**delta, "reasoning_content": reasoning}
            for key in ["content", "reasoning_content", "refusal"]:
                part = delta.get(key)
                if part is not None:
                    if not isinstance(part, str):
                        raise ValueError("Unsupported stream field")
                    self.message[key] = self.message.get(key, "") + part
            if choice.get("finish_reason") is not None:
                if self.finish is not None and self.finish != choice["finish_reason"]:
                    raise ValueError("Contradictory terminal reasons")
                self.finish = choice["finish_reason"]
        return False

    def response(self) -> dict:
        if self.finish is None:
            raise ValueError("No explicit terminal event")
        result = {
            **self.meta,
            "object": "chat.completion",
            "choices": [{"index": 0, "message": self.message, "finish_reason": self.finish}],
        }
        if self.usage is not None:
            result["usage"] = self.usage
        return result


def replay_sse(path: Path) -> dict:
    parser, event = SSE(), []
    for raw in path.read_bytes().splitlines():
        line = raw.decode("utf-8")
        if not line and event:
            if parser.event("\n".join(event)):
                break
            event = []
        elif line.startswith("data:"):
            event.append(line[5:].lstrip(" "))
    if event:
        parser.event("\n".join(event))
    return parser.response()


def delivered(response: dict) -> dict:
    """Reject incomplete, native-action, and reasoning-only deliveries."""
    choices = response["choices"]
    if len(choices) != 1 or choices[0]["finish_reason"] != "stop":
        raise ValueError("Incomplete delivery")
    message = choices[0]["message"]
    fields = judge.response_fields(message)
    if message.get("tool_calls") or message.get("function_call"):
        raise ValueError("Native calls forbidden")
    if not (fields["content"].strip() or fields["refusal"].strip()):
        raise ValueError("Empty or reasoning-only delivery")
    return {"status": "complete", "finish_reason": "stop", "message": message}


async def wire_call(body: dict, *, endpoint: str, key_env: str, path: Path) -> dict:
    """Save before dispatch. Existing started receipts require explicit reconciliation."""
    from aiohttp import ClientTimeout

    from nemo_gym.server_utils import request

    url = urlsplit(endpoint)
    if (
        url.scheme not in {"http", "https"}
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ValueError("Endpoint must be HTTP(S) without embedded credentials or query parameters")
    if path.exists():
        prior = read(path)
        if (
            prior["request_body"] != body
            or prior["request_sha256"] != judge.sha(body)
            or prior["endpoint"] != endpoint
        ):
            raise ValueError("Transport resume fingerprint mismatch")
        if prior["status"] == "started":
            raise ValueError("Ambiguous prior dispatch; never resend automatically")
        for raw in prior.get("raw_attempts", []):
            if digest(Path(raw["path"])) != raw["sha256"]:
                raise ValueError("Saved raw transport changed")
        return prior
    # A missing key is a configuration error: fail before a receipt exists, so no attempt is spent.
    key = os.environ.get(key_env)
    if not key:
        raise ValueError(f"{key_env} is not set; nothing was sent")
    record = {
        "status": "started",
        "request_body": body,
        "request_sha256": judge.sha(body),
        "endpoint": endpoint,
        "raw_attempts": [],
        "started_at": datetime.now(UTC).isoformat(),
    }
    for attempt in range(1, 4):
        # Every retry can be interrupted after dispatch, independently of earlier replies.
        record.update(status="started", http_attempt=attempt)
        write(path, record)
        raw_path = path.with_suffix(f".a{attempt}.http")
        raw_path.touch(mode=0o600, exist_ok=False)
        try:
            response = await request(
                "POST",
                endpoint,
                json=body,
                headers={"Authorization": "Bearer " + key},
                timeout=ClientTimeout(total=7200, connect=60, sock_read=900),
                _max_num_tries=1,
                _resend_after_disconnect=False,
            )
            try:
                record["http_status"] = response.status
                record["content_type"] = response.headers.get("content-type", "")
                with raw_path.open("wb") as stream:
                    async for chunk in response.content.iter_chunked(4096):
                        stream.write(chunk)
                        stream.flush()
                record["raw_attempts"].append({"path": str(raw_path), "sha256": digest(raw_path)})
                if response.status == 429 or response.status >= 500:
                    record.update(status="delivery_error", reason=f"HTTP_{response.status}")
                    write(path, record)
                    if attempt < 3:
                        # Frozen transport uses a rate-limit defer and bounded retries.
                        await asyncio.sleep(65 if response.status == 429 else 2**attempt)
                        continue
                    return record
                if response.status >= 400:
                    record.update(
                        status="delivery_error",
                        reason=f"HTTP_{response.status}",
                        terminal=True,
                    )
                else:
                    value = (
                        replay_sse(raw_path)
                        if "text/event-stream" in response.headers.get("content-type", "")
                        else read(raw_path)
                    )
                    record.update(status="generated", response=value)
                write(path, record)
                return record
            finally:
                response.release()
        except (asyncio.CancelledError, KeyboardInterrupt):
            # The started record remains ambiguous; no automatic redispatch.
            raise
        except Exception as error:  # noqa: BLE001 - deliberate per-cell/transport accounting boundary
            record["raw_attempts"].append({"path": str(raw_path), "sha256": digest(raw_path)})
            record.update(status="delivery_error", reason=type(error).__name__)
            write(path, record)
            return record
    return record


def verified_response(receipt: dict) -> dict:
    """Reconstruct the delivered fields from the saved authoritative HTTP bytes."""
    if receipt["status"] != "generated" or not receipt["raw_attempts"]:
        raise ValueError("No complete transport receipt")
    for raw in receipt["raw_attempts"]:
        if digest(Path(raw["path"])) != raw["sha256"]:
            raise ValueError("Raw transport binding changed")
    raw_path = Path(receipt["raw_attempts"][-1]["path"])
    actual = replay_sse(raw_path) if "text/event-stream" in receipt.get("content_type", "") else read(raw_path)
    if actual != receipt["response"]:
        raise ValueError("Saved response differs from raw delivery")
    return actual
