# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Merge all system messages into one at position 0.

Models like Qwen3 reject system messages that appear mid-conversation
(``System message must be at the beginning``).
"""

from __future__ import annotations

import logging

from nemo_gym.adapters.types import AdapterRequest, RequestInterceptor


logger = logging.getLogger(__name__)


def _content_to_str(content) -> str:
    """Extract plain text from either a string or OpenAI list-format content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content) if content else ""


class Interceptor(RequestInterceptor):
    def __init__(self, *, separator: str = "\n\n") -> None:
        self._sep = separator
        self._fix_count = 0

    async def intercept_request(self, req: AdapterRequest) -> AdapterRequest:
        messages: list[dict] = req.body.get("messages", [])
        if not messages:
            return req

        system_indices: list[int] = []
        system_parts: list[str] = []
        non_system: list[dict] = []
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                system_indices.append(i)
                text = _content_to_str(msg.get("content", ""))
                if text:
                    system_parts.append(text)
            else:
                non_system.append(msg)

        if len(system_indices) <= 1 and (not system_indices or system_indices[0] == 0):
            return req

        self._fix_count += 1
        session = req.ctx.extra.get("session_id", "?")

        role_seq = [m.get("role", "?") for m in messages]
        displaced = [i for i in system_indices if i > 0]
        diag_parts: list[str] = []
        for idx in displaced:
            lo = max(0, idx - 2)
            hi = min(len(role_seq), idx + 3)
            window = " ".join(f"[{j}]={role_seq[j]}" for j in range(lo, hi))
            content_preview = _content_to_str(messages[idx].get("content", ""))[:300]
            diag_parts.append(f"system@{idx} window=({window}) content_preview={content_preview!r}")

        logger.warning(
            "consolidate_system: fix #%d session=%s n_msgs=%d system_at=%s n_system=%d roles_head=%s | %s",
            self._fix_count,
            session,
            len(messages),
            system_indices,
            len(system_parts),
            role_seq[:8],
            " | ".join(diag_parts) if diag_parts else "system missing from pos 0",
        )

        merged: list[dict] = []
        if system_parts:
            merged.append({"role": "system", "content": self._sep.join(system_parts)})
        merged.extend(non_system)
        req.body["messages"] = merged
        return req
