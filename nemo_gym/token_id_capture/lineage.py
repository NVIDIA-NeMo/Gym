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
the model-authored turns in a request are the ones we produced: hashing them in
order identifies the call that produced the last one. Nothing is added to the
wire and nothing depends on the harness preserving a field we invented, and
nothing depends on which dialect the harness speaks.

The index is a map keyed by call, not a running cursor, so it is a tree and
forks cost nothing. Two sub-agents branching from one parent both resolve to it
and both get the same prefix -- which is exactly right, and is why the prefix to
supply is ``cum(parent)`` and not ``cum(previous call)``. Entries are added and
never mutated, so concurrent sub-agents cannot corrupt each other's lineage.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any


_FINGERPRINT_DOMAIN = b"nemo-gym-lineage"

# --- experimental: canonical text fingerprints (NG_TIC_FP_CANONICAL=1) --------
# Hypothesis under test (call-timing report F5): record indexes the served turn
# with reasoning wrapped as <think>...</think> content, while resolve sees the
# incoming turn after the reasoning parser stripped it — so every reasoning
# turn diverges and lookups always miss. Canonicalizing the *hash input only*
# (records are never rewritten) makes the two sides agree: think blocks are
# dropped and whitespace collapsed. A collision between two genuinely different
# turns that differ only in reasoning/whitespace yields an *ambiguous* match,
# which resolve() already refuses — degrading to fallback, never a wrong parent.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FP_CANONICAL: bool | None = None


def _fp_canonical_enabled() -> bool:
    global _FP_CANONICAL
    if _FP_CANONICAL is None:
        _FP_CANONICAL = os.environ.get("NG_TIC_FP_CANONICAL") == "1"
    return _FP_CANONICAL


def _canonical_turn_text(text: str) -> str:
    return " ".join(_THINK_BLOCK_RE.sub(" ", text).split())


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

    Tool calls are deliberately NOT folded in here -- see ``_tools_of``. The
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
    """The tool calls in a model-authored turn, as (name, canonical args).

    Each dialect puts them somewhere different, and all three have to reduce to
    the same list or a tool-using turn cannot match itself across a round trip.
    That matters because a tool-using turn is the only kind that produces a
    multi-turn rollout.

    - Chat: ``tool_calls`` on the message, with ``function.name`` / ``arguments``.
    - Anthropic: ``content`` blocks of ``type: tool_use``, with ``name`` / ``input``.
    - Responses: a standalone ``function_call`` item whose ``name`` and
      ``arguments`` are at the top level.
    """
    tools: list[tuple[str, str]] = []
    # Responses: the item *is* the tool call, with name and arguments at the top level.
    if message.get("type") == "function_call":
        tools.append((str(message.get("name", "")), canonicalize_tool_arguments(message.get("arguments"))))
    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tools.append((str(block.get("name", "")), canonicalize_tool_arguments(block.get("input"))))
    for call in message.get("tool_calls") or []:
        function = (call or {}).get("function") or {}
        tools.append((str(function.get("name", "")), canonicalize_tool_arguments(function.get("arguments"))))
    return tools


def _is_assistant_authored(message: dict) -> bool:
    """Whether this item is something the model produced.

    Chat and Anthropic mark it with ``role: assistant``. The Responses dialect
    represents a tool call as a sibling ``function_call`` item with no role at
    all, so a role check alone misses every tool call a Responses-speaking
    harness makes.
    """
    if message.get("role") == "assistant":
        return True
    return message.get("type") in ("function_call", "reasoning")


def assistant_fingerprint(messages: list[dict]) -> str:
    """Fingerprint the model-authored turns of a request, in order.

    Only what the model produced is hashed: it identifies the call that produced
    the last one. User and tool content varies with the environment and is
    irrelevant to lineage.

    The three dialects represent the same turn differently, and all three have to
    reduce to the same hash or a turn cannot match itself across a round trip:
    chat puts tool calls on the message, Anthropic nests them in content blocks,
    and Responses emits them as separate items.
    """
    hasher = hashlib.sha256(_FINGERPRINT_DOMAIN)
    count = 0
    for message in messages or []:
        if not isinstance(message, dict) or not _is_assistant_authored(message):
            continue
        count += 1
        _hash_turn(hasher, message)
    if count == 0:
        return ""
    return hasher.hexdigest()


def _hash_turn(hasher: "hashlib._Hash", message: dict) -> None:
    """Feed one model-authored turn into the running fingerprint."""
    text = _text_of(message.get("content"))
    if _fp_canonical_enabled():
        text = _canonical_turn_text(text)
    hasher.update(b"\x00")
    hasher.update(text.encode("utf-8"))
    for name, arguments in _tools_of(message):
        hasher.update(b"\x01")
        hasher.update(name.encode("utf-8"))
        hasher.update(arguments.encode("utf-8"))


def assistant_prefix_fingerprints(messages: list[dict]) -> list[str]:
    """Running fingerprint after each model-authored turn, in order.

    ``result[-1] == assistant_fingerprint(messages)`` (when any turn exists).
    The incremental hasher states are free — this is the observability seam
    for measuring *where* a request first diverges from the recorded lineage
    (call-timing report F5 / longest-prefix matching groundwork).
    """
    hasher = hashlib.sha256(_FINGERPRINT_DOMAIN)
    fps: list[str] = []
    for message in messages or []:
        if not isinstance(message, dict) or not _is_assistant_authored(message):
            continue
        _hash_turn(hasher, message)
        fps.append(hasher.hexdigest())
    return fps


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
    # Running sum of cum_len over recorded nodes, so the index can bound itself on memory without
    # walking every node on every access.
    total_tokens: int = 0
    # Observability: every recorded per-turn prefix fingerprint -> its depth
    # (1-based turn count). ~80 B/turn; dropped with the rollout at seal.
    prefix_depths: dict[str, int] = field(default_factory=dict)

    def resolve(self, messages: list[dict]) -> LineageNode | None:
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
        previous = self.by_call_id.get(call_id)
        if previous is not None:
            self.total_tokens -= previous.cum_len
        self.total_tokens += node.cum_len
        self.by_call_id[call_id] = node
        prefix_fps = assistant_prefix_fingerprints(messages)
        if prefix_fps:
            self.by_fingerprint.setdefault(prefix_fps[-1], []).append(call_id)
        for depth, fp in enumerate(prefix_fps, start=1):
            self.prefix_depths.setdefault(fp, depth)

    def divergence_depth(self, messages: list[dict]) -> tuple[int, int]:
        """(deepest recorded prefix this request matches, its own turn count).

        Observability for the `fallback_no_match` cause: 0 means even the
        first model-authored turn was never recorded in this form (systematic
        per-turn drift); k < n means the request diverges after turn k
        (localized rewrite — the longest-prefix-matching opportunity).
        """
        fps = assistant_prefix_fingerprints(messages)
        matched = 0
        for i, fp in enumerate(fps):
            if fp not in self.prefix_depths:
                break
            matched = i + 1
        return matched, len(fps)


class LineageIndex:
    """All live rollouts' lineage, bounded so an abandoned rollout cannot leak.

    A rollout that dies (harness crash, a batch discarded on a NaN-logprob
    retry) is never read again, and this index lives in the model server process
    while records are consumed elsewhere, so eviction cannot be driven by
    consumption. It is driven by two caps, and losing an entry costs a fallback
    to re-rendering, never a wrong answer.

    **The token cap is the one that matters.** Each node holds the parent's whole
    cumulative sequence, so cost scales with context length, not call count: a
    131k-token call measured ~4.5 MiB as a Python list of ints (~36 bytes per
    token). Bounding on rollouts alone would have let a 512-rollout batch at long
    context hold multiple GiB. The default budget is ~290 MiB.

    Eviction is oldest-first by insertion, and runs on access rather than on
    write, so the budget is exceeded by at most the single call recorded after
    the last check -- one ``cum_len``, ~4.5 MiB at 131k. A rollout still in
    flight can be evicted under pressure; that is intended, and it degrades to
    prefix inference rather than to a wrong parent.
    """

    def __init__(self, max_rollouts: int = 512, max_tokens: int = 8_000_000) -> None:
        self._max_rollouts = max_rollouts
        self._max_tokens = max_tokens
        self._rollouts: dict[str, RolloutLineage] = {}

    def for_rollout(self, rollout_id: str) -> RolloutLineage:
        lineage = self._rollouts.get(rollout_id)
        if lineage is None:
            lineage = RolloutLineage()
            self._rollouts[rollout_id] = lineage
        self._evict()
        return lineage

    def _evict(self) -> None:
        # Checked after every access, not only on insert: a rollout grows as its calls arrive, so
        # the budget can be exceeded without any new rollout appearing.
        while self._rollouts and (len(self._rollouts) > self._max_rollouts or self.total_tokens > self._max_tokens):
            oldest = next(iter(self._rollouts))
            # Never evict the only rollout: at long context a single one can exceed the budget, and
            # dropping it would disable lineage entirely rather than bound it.
            if len(self._rollouts) == 1:
                return
            self._rollouts.pop(oldest)

    @property
    def total_tokens(self) -> int:
        return sum(lineage.total_tokens for lineage in self._rollouts.values())

    def drop(self, rollout_id: str) -> None:
        """Release a rollout's lineage early.

        Unused by Gym's own path -- the model server has no signal that a rollout
        finished. A framework that implements ``TokenSink`` in the same process
        does have one, and should call this when it retires the records.
        """
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
