# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in delta wrapper for the sandbox's synchronous OpenAI model client."""

import hashlib
import json
from functools import wraps
from typing import Any
from uuid import uuid4

from openai import ConflictError


def _encoded(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class HistoryDeltaClient:
    """One serial conversation across the short-lived clients created by a root agent."""

    def __init__(self) -> None:
        self.conversation_id = uuid4().hex
        self._version: str | None = None
        self._messages: list[bytes] = []

    def call(self, create: Any, *args: Any, **kwargs: Any) -> Any:
        """Transmit an exact prefix delta; reset after edits or an explicit resync response."""
        if args or not isinstance(kwargs.get("messages"), list) or kwargs.get("stream"):
            return create(*args, **kwargs)
        messages = kwargs["messages"]
        encoded = [_encoded(m) for m in messages]
        append = self._version is not None and encoded[: len(self._messages)] == self._messages
        identity = uuid4().hex
        # Reuse the canonical message bytes instead of serializing history again.
        history_bytes = b"[" + b",".join(encoded) + b"]"
        digest = hashlib.sha256(history_bytes).hexdigest()

        def send(mode: str) -> Any:
            envelope = {
                "conversation_id": self.conversation_id,
                "request_id": identity,
                "base_version": self._version if mode == "append" else None,
                "mode": mode,
                "messages_sha256": digest,
            }
            extra = dict(kwargs.get("extra_body") or {})
            if "messages" in extra or "_gym_history" in extra:
                raise ValueError("History transport cannot override messages in extra_body")
            headers = dict(kwargs.get("extra_headers") or {})
            headers["x-gym-history-transport"] = "v1"
            return create(
                **{
                    **kwargs,
                    "messages": messages[len(self._messages) :] if mode == "append" else messages,
                    "extra_body": {**extra, "_gym_history": envelope},
                    "extra_headers": headers,
                }
            )

        try:
            response = send("append" if append else "full")
        except ConflictError as error:
            # Retry only a pre-dispatch protocol rejection. Other model/transport
            # errors retain the existing client's semantics and are never hidden.
            body = error.body
            detail = body.get("error", body) if isinstance(body, dict) else {}
            if not isinstance(detail, dict) or detail.get("code") != "gym_history_resync":
                raise
            identity = uuid4().hex
            response = send("full")
        self._version, self._messages = identity, encoded
        return response


def install_history_delta(agent: Any) -> None:
    """Wrap only this root agent; auxiliary/child paths keep their ordinary full requests."""
    state = HistoryDeltaClient()
    original = agent._create_request_openai_client

    @wraps(original)
    def create_client(*args: Any, **kwargs: Any) -> Any:
        client = original(*args, **kwargs)
        completions = client.chat.completions
        create = completions.create

        @wraps(create)
        def request(*request_args: Any, **request_kwargs: Any) -> Any:
            return state.call(create, *request_args, **request_kwargs)

        completions.create = request
        return client

    agent._create_request_openai_client = create_client
