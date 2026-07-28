# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Which recorded call does this request continue?

The builder can infer that after the fact, by finding the earlier call whose
token sequence is the longest prefix of this call's prompt. That works, but it
cannot do two things:

- **Resolve a retry.** Capture records a call once the response is assembled,
  including one the client never received, so a harness retry leaves two records
  with the same prompt and different generations. Both are equally valid
  children, so inference has to guess between them.
- **Run before the call.** Inference happens after the rollout. Supplying the
  engine the previous call's exact tokens (so the next prompt extends them by
  construction instead of being re-rendered from text) has to happen *before* the
  request is forwarded, when there is nothing yet to infer from.

This module answers the question at request time instead, using only what the
harness already sends. A harness must echo the conversation to continue it, so
the assistant messages in a request are the ones we produced: hashing them in
order identifies the call that produced the last one. Nothing is added to the
wire and nothing depends on the harness preserving a field we invented.

The index is a map keyed by call, not a running cursor, so it is a tree and
forks cost nothing. Two sub-agents branching from one parent both resolve to it
and both get the same prefix -- which is exactly right, and is why the prefix to
supply is ``cum(parent)`` and not ``cum(previous call)``. Entries are added and
never mutated, so concurrent sub-agents cannot corrupt each other's lineage.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Optional


_FINGERPRINT_DOMAIN = b"nemo-gym-lineage"


def canonicalize_tool_arguments(value: Any) -> str:
    """Normalize a tool call's arguments for comparison only.

    Harnesses re-serialize tool-call arguments between turns -- compact one turn,
    pretty-printed the next -- so the same call does not compare equal to itself.
    Comparison is done on sorted-key, separator-normalized JSON; the model's
    original string is what stays in the record and is never rewritten.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return value.strip()
    else:
        parsed = value
    try:
        return json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(parsed)


def _text_of(content: Any) -> str:
    """Flatten a message's *text* content, across the shapes the dialects use.

    Tool calls are deliberately NOT folded in here -- see ``_tools_of``. The two
    dialects carry them in different places, so they have to be normalized
    separately or the same turn hashes differently depending on which side is
    looking at it.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
    return "\n".join(parts)


def _tools_of(message: dict) -> list[tuple[str, str]]:
    """The tool calls in an assistant turn, as (name, canonical args).

    A turn we produced is recorded in OpenAI shape (``tool_calls`` with
    ``function.name`` / ``function.arguments``), but Claude Code echoes it back
    in Anthropic shape (``content`` blocks of ``type: tool_use`` with ``name`` /
    ``input``). Both must reduce to the same list, or a tool-using turn -- the
    only kind that produces multi-turn rollouts -- can never match itself.
    """
    tools: list[tuple[str, str]] = []
    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tools.append((str(block.get("name", "")), canonicalize_tool_arguments(block.get("input"))))
    for call in message.get("tool_calls") or []:
        function = (call or {}).get("function") or {}
        tools.append((str(function.get("name", "")), canonicalize_tool_arguments(function.get("arguments"))))
    return tools


def assistant_fingerprint(messages: list[dict]) -> str:
    """Fingerprint the assistant turns of a request, in order.

    Only assistant turns are hashed: they are what *we* produced, so they
    identify the call that produced the last one. User and tool content varies
    with the environment and is irrelevant to lineage.
    """
    hasher = hashlib.sha256(_FINGERPRINT_DOMAIN)
    count = 0
    for message in messages or []:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        count += 1
        hasher.update(b"\x00")
        hasher.update(_text_of(message.get("content")).encode("utf-8"))
        for name, arguments in _tools_of(message):
            hasher.update(b"\x01")
            hasher.update(name.encode("utf-8"))
            hasher.update(arguments.encode("utf-8"))
    if count == 0:
        return ""
    return hasher.hexdigest()


@dataclass
class LineageNode:
    call_id: str
    cum_tokens: list[int]
    cum_len: int
    digest: str


@dataclass
class RolloutLineage:
    """Per-rollout call index. Append-only, so forks and concurrency are safe."""

    by_fingerprint: dict[str, list[str]] = field(default_factory=dict)
    by_call_id: dict[str, LineageNode] = field(default_factory=dict)

    def resolve(self, messages: list[dict]) -> Optional[LineageNode]:
        """The call this request continues, or ``None``.

        ``None`` for a new root (nothing matches -- a fresh conversation, or one
        the harness rewrote) and, deliberately, for an ambiguous match: if two
        recorded calls produced byte-identical output we cannot tell which one
        this continues, and guessing would attribute tokens to the wrong parent.
        A unique match is required.
        """
        fingerprint = assistant_fingerprint(messages)
        if not fingerprint:
            return None
        call_ids = self.by_fingerprint.get(fingerprint) or []
        if len(call_ids) != 1:
            return None
        return self.by_call_id.get(call_ids[0])

    def record(self, call_id: str, messages: list[dict], cum_tokens: list[int], digest: str) -> None:
        """Index a completed call by the conversation a continuation of it would carry."""
        node = LineageNode(call_id=call_id, cum_tokens=list(cum_tokens), cum_len=len(cum_tokens), digest=digest)
        self.by_call_id[call_id] = node
        fingerprint = assistant_fingerprint(messages)
        if fingerprint:
            self.by_fingerprint.setdefault(fingerprint, []).append(call_id)


class LineageIndex:
    """All live rollouts' lineage, bounded so an abandoned rollout cannot leak.

    A rollout that dies (harness crash, a batch discarded on a NaN-logprob
    retry) is never read again, so eviction cannot rely on consumption. The cap
    evicts whole rollouts in insertion order; losing one costs a fallback to
    re-rendering, never a wrong answer.
    """

    def __init__(self, max_rollouts: int = 512) -> None:
        self._max_rollouts = max_rollouts
        self._rollouts: dict[str, RolloutLineage] = {}

    def for_rollout(self, rollout_id: str) -> RolloutLineage:
        lineage = self._rollouts.get(rollout_id)
        if lineage is None:
            lineage = RolloutLineage()
            self._rollouts[rollout_id] = lineage
            while len(self._rollouts) > self._max_rollouts:
                self._rollouts.pop(next(iter(self._rollouts)))
        return lineage

    def drop(self, rollout_id: str) -> None:
        self._rollouts.pop(rollout_id, None)

    def __len__(self) -> int:
        return len(self._rollouts)


def assistant_turn_from_output_items(output_items: list[dict]) -> dict:
    """Rebuild the assistant turn a harness will echo back, from a served response.

    Indexing a call means fingerprinting the conversation a *continuation* of it
    would carry -- which is this request's messages plus the turn we just
    produced. The next request only matches if that turn is reconstructed in the
    shape the harness echoes: text content, and tool calls with their arguments.

    A mismatch is not a correctness problem, only a missed match: the lookup
    falls back to a new root and the builder infers from token prefixes instead.
    """
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for item in output_items or []:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "function_call":
            tool_calls.append(
                {
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", ""),
                    }
                }
            )
            continue
        content = item.get("content")
        # Responses items carry a list of blocks; a chat message carries a plain string.
        if isinstance(content, str):
            if content:
                text_parts.append(content)
        else:
            for block in content or []:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    text_parts.append(block["text"])
        for block in item.get("summary") or []:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                text_parts.append(block["text"])
        for call in item.get("tool_calls") or []:
            function = (call or {}).get("function") or {}
            tool_calls.append(
                {"function": {"name": function.get("name", ""), "arguments": function.get("arguments", "")}}
            )
    turn: dict = {"role": "assistant", "content": "\n".join(text_parts)}
    if tool_calls:
        turn["tool_calls"] = tool_calls
    return turn
