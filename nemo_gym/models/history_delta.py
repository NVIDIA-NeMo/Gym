# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Experimental single-process conversation transport; capture still sees full requests."""

import hashlib
import json
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


# Match the standalone sandbox transport's request-body limit.
MAX_HISTORY_BODY_BYTES = 32 * 1024 * 1024


class HistoryResync(ValueError):
    """A delta cannot be applied; the client must resend a complete history."""


def _encoded(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _extract_history(
    body: dict[str, Any],
    history_field: str,
) -> tuple[str, str, str, str | None, str, list[dict[str, Any]]]:
    """Validate and extract the transport envelope and history items."""
    envelope = body.get("_gym_history")
    if not isinstance(envelope, dict):
        raise ValueError("_gym_history must be an object")
    if set(envelope) != {"conversation_id", "request_id", "base_version", "mode", "messages_sha256"}:
        raise ValueError("Invalid history transport fields")
    conversation, identity = envelope["conversation_id"], envelope["request_id"]
    if any(not isinstance(v, str) or not v or len(v) > 128 for v in (conversation, identity)):
        raise ValueError("Invalid conversation/request identity")
    mode, base, expected = envelope["mode"], envelope["base_version"], envelope["messages_sha256"]
    if mode not in {"full", "append"} or (base is not None and not isinstance(base, str)):
        raise ValueError("Invalid history mode/version")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("Invalid history digest")
    if history_field not in {"messages", "input"}:
        raise ValueError("Unsupported history field")
    messages = body.get(history_field)
    if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
        raise ValueError(f"{history_field} must be a list of objects")
    return conversation, identity, mode, base, expected, messages


@dataclass
class _History:
    version: str
    messages: list[dict[str, Any]]
    request_hash: str
    touched: float
    size: int


class HistoryBuffer:
    """Bounded, expiring histories owned by one Gym worker, isolated by rollout/conversation."""

    def __init__(self, *, max_bytes: int = 256 * 1024 * 1024, max_entries: int = 64, ttl: float = 1800) -> None:
        self._entries: OrderedDict[tuple[str, str, str], _History] = OrderedDict()
        self._max_bytes, self._max_entries, self._ttl = max_bytes, max_entries, ttl
        self.on_rollout_evicted: Callable[[str], None] | None = None

    def retains_rollout(self, rollout_id: str) -> bool:
        """Whether any conversation history for the rollout remains in this bounded cache."""
        return any(key[0] == rollout_id for key in self._entries)

    def _remove_history(self, key: tuple[str, str, str]) -> None:
        del self._entries[key]
        rollout_id = key[0]
        if self.on_rollout_evicted is not None and not self.retains_rollout(rollout_id):
            self.on_rollout_evicted(rollout_id)

    def cleanup_expired_histories(self, now: float) -> None:
        """Remove histories inactive beyond the TTL, as a fallback for abandoned rollouts."""
        for key in list(self._entries):
            if now - self._entries[key].touched > self._ttl:
                self._remove_history(key)

    def reconstruct(self, rollout_id: str, body: dict[str, Any], *, history_field: str = "messages") -> dict[str, Any]:
        """Apply once before model dispatch; duplicate transport delivery never appends twice.

        This does not cache model responses or promise exactly-once generation. Updates
        are synchronous on the event loop; no await separates version checking and commit.
        """
        conversation, identity, mode, base, expected, messages = _extract_history(body, history_field)
        now = time.monotonic()
        self.cleanup_expired_histories(now)
        key = (rollout_id, conversation, history_field)
        previous = self._entries.get(key)
        request_hash = hashlib.sha256(_encoded(body)).hexdigest()
        # A client may retry after acceptance if it never received the response.
        # Reuse the accepted history so the delta is not appended twice. As on
        # the ordinary full-history path, the retry may generate another response.
        if previous is not None and previous.version == identity:
            if previous.request_hash != request_hash:
                raise ValueError("Request identity reused with a different payload")
            complete = previous.messages
        else:
            if mode == "append":
                if previous is None or previous.version != base:
                    raise HistoryResync("Missing or stale history base")
                complete = previous.messages + messages
            else:
                complete = messages
            encoded = _encoded(complete)
            if hashlib.sha256(encoded).hexdigest() != expected:
                raise HistoryResync("Reconstructed history digest mismatch")
            if len(encoded) > self._max_bytes:
                raise ValueError("History exceeds experiment buffer capacity")
            # Detach mutable input objects from the buffer's owned state.
            complete = json.loads(encoded)
            self._entries[key] = _History(identity, complete, request_hash, now, len(encoded))
        self._entries[key].touched = now
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries or sum(v.size for v in self._entries.values()) > self._max_bytes:
            self._remove_history(next(iter(self._entries)))
        return {k: v for k, v in body.items() if k != "_gym_history"} | {history_field: complete}


class HistoryTransportProtocol:
    """Default shim protocol; integrations may override detection and reconstruction."""

    def matches_request(self, scope: dict[str, Any]) -> bool:
        return any(name.lower() == b"x-gym-history-transport" for name, _ in scope.get("headers", []))

    async def reconstruct_request(
        self,
        buffer: HistoryBuffer | None,
        scope: dict[str, Any],
        receive: Any,
        send: Any,
        rollout_id: str,
        *,
        history_field: str,
    ) -> tuple[dict[str, Any], Any] | None:
        version = dict(scope.get("headers", [])).get(b"x-gym-history-transport")
        if version != b"v1" or buffer is None:
            await _error(send, 400, "gym_history_disabled")
            return None
        return await reconstruct_request(buffer, scope, receive, send, rollout_id, history_field=history_field)


async def reconstruct_request(
    buffer: HistoryBuffer,
    scope: dict[str, Any],
    receive: Any,
    send: Any,
    rollout_id: str,
    *,
    history_field: str = "messages",
) -> tuple[dict[str, Any], Any] | None:
    """Read real wire bytes, then replay full JSON to ordinary validation/capture.

    Only clients declaring this experimental transport enter here. Ordinary
    requests retain their existing validation and capture path.
    """
    chunks = bytearray()
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return None
        if message["type"] != "http.request":
            raise ValueError("Expected an HTTP request body")
        chunks.extend(message.get("body", b""))
        if len(chunks) > MAX_HISTORY_BODY_BYTES:
            await _error(send, 413, "gym_history_invalid")
            return None
        if not message.get("more_body", False):
            break
    try:
        body = json.loads(chunks)
        if not isinstance(body, dict):
            raise ValueError("Expected JSON object")
        body = buffer.reconstruct(rollout_id, body, history_field=history_field)
        full_bytes = json.dumps(body, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    except HistoryResync:
        await _error(send, 409, "gym_history_resync")
        return None
    except (ValueError, TypeError):
        await _error(send, 400, "gym_history_invalid")
        return None
    delivered = False

    async def replay() -> dict[str, Any]:
        nonlocal delivered
        if delivered:
            return await receive()
        delivered = True
        return {"type": "http.request", "body": full_bytes, "more_body": False}

    scope = dict(scope)
    scope["headers"] = [
        (k, v)
        for k, v in scope.get("headers", [])
        if k.lower() not in {b"content-length", b"transfer-encoding", b"x-gym-history-transport"}
    ]
    scope["headers"].append((b"content-length", str(len(full_bytes)).encode()))
    return scope, replay


async def _error(send: Any, status: int, code: str) -> None:
    # Stable code only: never expose the buffered conversation or exception text.
    body = json.dumps({"error": {"message": code, "type": "history_transport", "code": code}}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
        }
    )
    await send({"type": "http.response.body", "body": body})
