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
"""Echo an episode's Responses items (input, model output, tool output) as they happen.

``pretty`` prints one readable block per item: a ``[simple_agent:<step>] <kind>`` header, the text,
and a blank line.
``json`` prints each item exactly as it arrives, one JSON object per line.

TODO: neither format identifies the episode, so concurrent rollouts interleave indistinguishably.
For concurrent rollout support, prefix ``pretty`` headers with a short per-episode id (e.g.
``[<episode id>:<step>]``, from ``uuid.uuid4().hex[:4]`` or the rollout id) and wrap ``json`` items
(e.g. ``{"episode": ..., "step": ..., "item": ...}``).
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Mapping
from typing import Any, Literal, Optional

from pydantic import BaseModel


EchoFormat = Literal["off", "pretty", "json"]
PRETTY_PREFIX = "simple_agent"


def _as_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, BaseModel):
        return item.model_dump(mode="json", exclude_none=True)
    if isinstance(item, Mapping):
        return dict(item)
    return {"type": type(item).__name__, "value": str(item)}


def _text_parts(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text") or part.get("refusal") or f"<{part.get('type')}>")
            for part in content
            if isinstance(part, Mapping)
        )
    return "" if content is None else str(content)


def _describe(item: Mapping[str, Any]) -> tuple[str, str]:
    """(label, text) for one item in the pretty format."""
    kind = item.get("type") or ("message" if "role" in item else "item")
    if kind == "message":
        return str(item.get("role", "message")), _text_parts(item.get("content"))
    if kind == "function_call":
        return f"function_call {item.get('name')} ({item.get('call_id')})", str(item.get("arguments", ""))
    if kind == "function_call_output":
        return f"function_call_output ({item.get('call_id')})", _text_parts(item.get("output"))
    if kind == "reasoning":
        texts = [_text_parts(part.get("text")) for part in item.get("summary") or [] if isinstance(part, Mapping)]
        texts += [_text_parts(part.get("text")) for part in item.get("content") or [] if isinstance(part, Mapping)]
        return "reasoning", "\n".join(text for text in texts if text) or "(no summary)"
    return str(kind), json.dumps(item, ensure_ascii=False)


class ItemEcho:
    """Writes one episode's items to a file (appending) or stdout."""

    def __init__(self, fmt: EchoFormat, *, path: Optional[str] = None, max_chars: int = 2000) -> None:
        self.fmt = fmt
        self.path = path
        self.max_chars = max_chars

    def _write(self, text: str) -> None:
        if self.path is None:
            sys.stdout.write(text)
            sys.stdout.flush()
            return
        # One append per record keeps concurrent episodes from splitting each other's lines.
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(text)

    def _truncate(self, text: str) -> str:
        if self.max_chars <= 0 or len(text) <= self.max_chars:
            return text
        return f"{text[: self.max_chars]}… ({len(text) - self.max_chars} more chars)"

    def items(self, items: Iterable[Any], *, step: int) -> None:
        records = []
        for item in items:
            data = _as_dict(item)
            if self.fmt == "json":
                records.append(json.dumps(data, ensure_ascii=False) + "\n")
                continue
            label, text = _describe(data)
            # Trailing newlines are dropped so exactly one blank line separates items.
            text = self._truncate(text).rstrip("\n")
            body = f"{text}\n" if text else ""
            records.append(f"[{PRETTY_PREFIX}:{step}] {label}\n{body}\n")
        if records:
            self._write("".join(records))

    def done(self, *, status: str, steps: int) -> None:
        if self.fmt == "pretty":
            self._write(f"[{PRETTY_PREFIX}] episode {status} after {steps} step(s)\n\n")
