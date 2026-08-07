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

A rollout is several model calls, and training needs them as one contiguous token
sequence. That means knowing, for each call, which earlier call it continues: its
parent. There are two ways to find one.

The builder can match token prefixes after the rollout, pairing a call with the
earlier call whose token sequence is the longest prefix of this call's prompt. That
needs no bookkeeping at all, and it cannot do two things:

- **Tell two attempts apart.** Capture records a call once its response is
  assembled, including one the client never received, so a harness retry leaves two
  records with the same prompt and different generations. Both are equally good
  prefixes.
- **Answer while a call is in flight.** Prefix matching needs the finished records,
  and those only exist once the rollout is over. Supplying the engine the parent's
  exact tokens, so the next prompt extends them by construction rather than being
  re-rendered from text, has to happen as the request is forwarded.

This module answers the question at request time instead, using only what the
harness already sends. A harness has to echo the conversation back to continue it,
so the model-authored turns arriving in a request are the ones we produced, and
hashing them in order identifies the call that produced the last one::

    call 1 request    [system, user]
    call 1 response   assistant: <tool_call>       stored under hash(assistant turns)

    call 2 request    [system, user,
                       assistant: <tool_call>,     echoed back
                       tool result]                new
                      hash the assistant turns  -> finds call 1, and its tokens

Nothing is added to the wire, nothing depends on the harness preserving a field we
invented, and nothing depends on which dialect it speaks.

Two hashes, for two jobs. ``assistant_fingerprint`` covers model-authored turns
only and is the lookup key: it has to ignore user and tool content, because a tool
result is appended between calls and a key covering it would never match twice.
``conversation_digest`` covers every turn and is the check: precisely because the
fingerprint ignores that content, a harness that rewrites earlier turns while
echoing the same model output produces the same fingerprint, and the digest is what
catches it before the parent's tokens are reused.

The index is a map keyed by call, not a running cursor, so it is a tree and forks
cost nothing. Two sub-agents branching from one parent both resolve to it and both
get the same prefix, which is why the prefix to supply is ``cum(parent)`` and not
``cum(previous call)``. Entries are added and never mutated, so concurrent
sub-agents cannot corrupt each other's lineage.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


_FINGERPRINT_DOMAIN = b"nemo-gym-lineage"
_CONTEXT_DOMAIN = b"nemo-gym-lineage-context"


def canonicalize_tool_arguments(value: Any) -> str:
    """Normalize a tool call's arguments for comparison only.

    Harnesses re-serialize tool-call arguments between turns, compact one turn and
    pretty-printed the next, so the same call does not compare equal to itself.
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

    Tool calls are deliberately not folded in here; see ``_tools_of``. The
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


def _tool_result_text(message: dict) -> str:
    """A tool result's payload, across the shapes the dialects use.

    Each dialect puts it somewhere ``_text_of`` does not look. Responses has a standalone
    ``function_call_output`` item carrying ``output``; Anthropic has ``tool_result`` blocks whose
    payload is under ``content``, not ``text``; chat puts a plain string in ``content``, which
    ``_text_of`` already returns.
    """
    parts: list[str] = []
    if message.get("type") == "function_call_output":
        output = message.get("output")
        parts.append(output if isinstance(output, str) else json.dumps(output, sort_keys=True, default=str))
    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                inner = block.get("content")
                parts.append(inner if isinstance(inner, str) else _text_of(inner))
    return "\n".join(part for part in parts if part)


def _is_assistant_authored(message: dict) -> bool:
    """Whether this item is something the model produced.

    Chat and Anthropic mark it with ``role: assistant``. The Responses dialect
    represents a tool call as a sibling ``function_call`` item with no role at
    all, so a role check alone misses every tool call a Responses-speaking
    harness makes.
    """
    if message.get("role") == "assistant":
        return True
    # Reasoning is deliberately excluded. A harness need not echo standalone reasoning items,
    # and the chat dialect carries reasoning in a field this never reads, so counting it would
    # make the key depend on which dialect the harness speaks and on whether it chose to send
    # the model's thinking back. Two calls differing only in reasoning collide instead, which
    # resolves as ambiguous and falls back.
    return message.get("type") == "function_call"


def conversation_digest(messages: list[dict]) -> str:
    """Hash every turn of a conversation, model-authored or not.

    ``assistant_fingerprint`` deliberately ignores user and tool content so that a request
    still matches the call it continues after a tool result has been appended. That makes it
    a good lookup key and a bad safety check: a harness that rewrites earlier turns while
    echoing the same model output still produces the same fingerprint. This covers the rest
    of the conversation so a match can be verified before its tokens are reused.
    """
    hasher = hashlib.sha256(_CONTEXT_DOMAIN)
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        hasher.update(b"\x00")
        hasher.update(str(message.get("role") or message.get("type") or "").encode("utf-8"))
        hasher.update(b"\x01")
        hasher.update(_text_of(message.get("content")).encode("utf-8"))
        for name, arguments in _tools_of(message):
            hasher.update(b"\x02")
            hasher.update(name.encode("utf-8"))
            hasher.update(arguments.encode("utf-8"))
        # Tool results too. A harness that summarizes, redacts or truncates an earlier result
        # changes what the model is being asked to continue from, and without this the digest is
        # blind to it in the Responses and Anthropic shapes, so a stale parent verifies clean.
        hasher.update(b"\x03")
        hasher.update(_tool_result_text(message).encode("utf-8"))
    return hasher.hexdigest()


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
    # The conversation this call was sent, as sent: its length in messages and a digest over all
    # of them. Deliberately excludes the turn the model produced. A continuation carries these
    # messages, then the model's turn echoed back in whatever shape its dialect uses, then new
    # content. Only the first part has a stable length, because one response can echo back as
    # several items, so that is where the comparison is anchored.
    context_len: int = 0
    context_digest: str = ""


@dataclass
class RolloutLineage:
    """Per-rollout call index. Append-only, so forks and concurrency are safe."""

    by_fingerprint: dict[str, list[str]] = field(default_factory=dict)
    by_call_id: dict[str, LineageNode] = field(default_factory=dict)
    # Running sum of cum_len over recorded nodes, so the index can bound itself on memory without
    # walking every node on every access.
    total_tokens: int = 0

    def resolve(self, messages: list[dict]) -> LineageNode | None:
        """The call this request continues, or ``None``.

        ``None`` for a new root (nothing matches: a fresh conversation, or one
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
        node = self.by_call_id.get(call_ids[0])
        if node is None or not self._continues(node, messages):
            return None
        return node

    @staticmethod
    def _continues(node: LineageNode, messages: list[dict]) -> bool:
        """Whether this request extends the conversation the node was recorded against.

        A continuation carries the conversation this call was sent, unchanged, before anything
        else. So the request's first ``context_len`` messages must hash to what was recorded.
        A harness that rewrote or summarized those turns fails here, which matters because the
        node's tokens encode the conversation as it was, and reusing them would generate from a
        conversation the harness did not send.

        Comparing only the part the model did not produce is what keeps this robust. The
        model's turn comes back in the shape the harness's dialect uses, and one response can
        echo as several items, so any comparison that had to count those messages would break
        on exactly the multi-turn tool-calling case this exists for.
        """
        if not node.context_digest:
            # Unreachable while ``record`` is the only way to build a node, and fails closed if
            # that stops being true: without a digest there is nothing to check the request
            # against, and an unchecked match is the one this guard exists to stop.
            return False
        if len(messages) < node.context_len:
            return False
        return conversation_digest(messages[: node.context_len]) == node.context_digest

    def record(
        self,
        call_id: str,
        messages: list[dict],
        cum_tokens: list[int],
        digest: str,
        context_len: int | None = None,
    ) -> None:
        """Index a completed call by the conversation a continuation of it would carry.

        ``context_len`` is how many leading messages were the request as sent, before the turn
        the model produced. It defaults to everything but the last message, which is right when
        the caller appended a single synthesized turn.
        """
        node = LineageNode(
            call_id=call_id,
            cum_tokens=list(cum_tokens),
            cum_len=len(cum_tokens),
            digest=digest,
            context_len=context_len if context_len is not None else max(len(messages or []) - 1, 0),
            context_digest=conversation_digest(
                (messages or [])[: context_len if context_len is not None else max(len(messages or []) - 1, 0)]
            ),
        )
        previous = self.by_call_id.get(call_id)
        if previous is not None:
            self.total_tokens -= previous.cum_len
        self.total_tokens += node.cum_len
        self.by_call_id[call_id] = node
        fingerprint = assistant_fingerprint(messages)
        if fingerprint:
            self.by_fingerprint.setdefault(fingerprint, []).append(call_id)


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
    the last check, one ``cum_len``, about 4.5 MiB at 131k. A rollout still in
    flight can be evicted under pressure; that is intended, and it degrades to
    prefix matching rather than to a wrong parent.
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

        Unused by Gym's own path, because the model server has no signal that a rollout
        finished. A framework that implements ``TokenSink`` in the same process
        does have one, and should call this when it retires the records.
        """
        self._rollouts.pop(rollout_id, None)

    def __len__(self) -> int:
        return len(self._rollouts)
