# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical text projection and the content digests receipts are made of."""

import hashlib
import json
from typing import Any


def content_text(content: object) -> str:
    """Read a text-only input, rejecting unsupported multimodal/history inputs."""
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(
        isinstance(part, dict) and part.get("type") == "input_text" and isinstance(part.get("text"), str)
        for part in content
    ):
        return "\n".join(part["text"] for part in content)
    raise ValueError("fresh-task variants require text-only messages")


def digest(value: object) -> str:
    """Stable content identity independent of mapping insertion order."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def actor_input_digest(messages: list[dict[str, Any]]) -> str:
    """Hash model-visible text, invariant to API message/content-block defaults."""
    return digest([{"role": item["role"], "content": content_text(item["content"])} for item in messages])
