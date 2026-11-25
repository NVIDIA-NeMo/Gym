# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, passive severe-repetition checks over retained Responses output.

This synchronous API owns no workers. Execution integrations provide CPU isolation
and wall-clock deadlines. Separate message/reasoning items contribute a blank line;
text parts within an item are concatenated without adding separators.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from nemo_gym.core.responses_api_types import NeMoGymResponseOutputItem


_MAX_CHARS = 2_000_000
_MAX_ITEMS = 8192
_MAX_TOOLS = 1024
_MAX_SEARCH_WORK = 4_000_000
_TEXT_PARTS = {"text", "input_text", "output_text", "summary_text", "reasoning_text", "refusal"}
_NON_TEXT_PARTS = {"input_image", "input_audio", "input_file", "input_video", "image", "audio", "video"}


class LoopingReason(StrEnum):
    """A conclusive severe-repetition witness; reasons are not exhaustive."""

    REPEATED_TEXT = "repeated_text"
    TOOL_CYCLE = "tool_cycle"


@dataclass(frozen=True)
class LoopingAssessment:
    """True proves a severe loop; None means analysis could not complete."""

    is_looping: bool | None
    reasons: tuple[LoopingReason, ...] = ()


@dataclass
class LoopingCounts:
    """Pool sufficient counts before calculating rate and coverage."""

    completed_total: int = 0
    assessed_total: int = 0
    flagged_total: int = 0

    def observe(self, assessment: LoopingAssessment) -> None:
        """Include one completed output, retaining unknowns in the population."""
        self.completed_total += 1
        self.assessed_total += int(type(assessment.is_looping) is bool)
        self.flagged_total += int(assessment.is_looping is True)

    def metrics(self, *, expected: int | None = None) -> dict[str, float]:
        """Return pooled metrics; omit the rate when no output was assessed."""
        population = self.completed_total if expected is None else expected
        if population < self.completed_total:
            raise ValueError("expected looping population is smaller than observed outputs")
        metrics = {"looping/coverage": self.assessed_total / population if population else 0.0}
        if self.assessed_total:
            metrics["looping/rate"] = self.flagged_total / self.assessed_total
        return metrics


class _Limit(ValueError):
    """The relevant input or deterministic search budget was exhausted."""


def _field(item: object, name: str, default: Any = None) -> Any:
    return item.get(name, default) if isinstance(item, Mapping) else getattr(item, name, default)


def _snapshot_output(output: object) -> tuple[tuple, ...] | None:
    """Copy only bounded immutable fields, for typed items or their saved wire form."""
    if not isinstance(output, (list, tuple)) or len(output) > _MAX_ITEMS:
        return None
    characters = 0
    tool_count = 0
    item_count = len(output)
    events = []

    def retain_text(value: object) -> str:
        nonlocal characters
        if not isinstance(value, str):
            raise ValueError("expected text")
        characters += len(value)
        if characters > _MAX_CHARS:
            raise _Limit
        return value

    def retain_parts(value: object, *, result: bool = False) -> tuple[str | None, ...]:
        nonlocal item_count
        if isinstance(value, str):
            return (retain_text(value),)
        if not isinstance(value, (list, tuple)):
            return (None,)
        item_count += len(value)
        if item_count > _MAX_ITEMS:
            raise _Limit
        fragments = []
        for part in value:
            part_type = _field(part, "type")
            if part_type in _TEXT_PARTS:
                text = _field(part, "refusal" if part_type == "refusal" else "text")
                if isinstance(text, str):
                    fragments.append(retain_text(text))
                else:
                    fragments.append(None)
            elif result or part_type not in _NON_TEXT_PARTS:
                fragments.append(None)
        return tuple(fragments)

    try:
        for item in output:
            try:
                item_type = _field(item, "type")
                if item_type == "message":
                    role = _field(item, "role", "assistant")
                    if role in {"user", "system", "developer"}:
                        continue
                    if role != "assistant":
                        events.append(("gap",))
                        continue
                    events.append(("text", retain_parts(_field(item, "content"))))
                elif item_type == "reasoning":
                    events.append(("text", retain_parts(_field(item, "content") or _field(item, "summary", []))))
                elif item_type in {"function_call", "custom_tool_call"}:
                    tool_count += 1
                    if tool_count > _MAX_TOOLS:
                        raise _Limit
                    call_id = retain_text(_field(item, "call_id"))
                    name = retain_text(_field(item, "name"))
                    namespace = _field(item, "namespace")
                    namespace = retain_text(namespace) if namespace is not None else None
                    arguments = retain_text(_field(item, "arguments" if item_type == "function_call" else "input"))
                    events.append((item_type, call_id, namespace, name, arguments))
                elif item_type in {"function_call_output", "custom_tool_call_output"}:
                    call_id = retain_text(_field(item, "call_id"))
                    fragments = retain_parts(_field(item, "output"), result=True)
                    events.append((item_type, call_id, fragments if None not in fragments else None))
                else:
                    events.append(("gap",))
            except _Limit:
                raise
            except (ValueError, TypeError, AttributeError):
                # Keep later evidence, but never bridge a malformed tool interaction.
                events.append(("gap",))
        return tuple(events)
    except _Limit:
        return None


def _canonical_arguments(arguments: str) -> str:
    try:
        parsed = json.loads(arguments)
    except ValueError:
        return "literal:" + arguments
    pending = [(parsed, 0)]
    nodes = 0
    while pending:
        value, depth = pending.pop()
        nodes += 1
        if depth > 32 or nodes > _MAX_ITEMS:
            raise _Limit
        if isinstance(value, dict):
            if nodes + len(pending) + len(value) > _MAX_ITEMS:
                raise _Limit
            pending.extend((child, depth + 1) for child in value.values())
        elif isinstance(value, list):
            if nodes + len(pending) + len(value) > _MAX_ITEMS:
                raise _Limit
            pending.extend((child, depth + 1) for child in value)
    return "json:" + json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _tool_cycle(signatures: list[tuple | None]) -> bool:
    for start, signature in enumerate(signatures):
        if signature is None:
            continue
        for period in range(1, 5):
            required = max(8, 4 * period)
            if start + required > len(signatures):
                continue
            if all(
                signatures[start + offset] is not None
                and signatures[start + offset] == signatures[start + offset % period]
                for offset in range(required)
            ):
                return True
    return False


def _periodic_text(regions: Sequence[str]) -> bool:
    remaining_work = _MAX_SEARCH_WORK
    for text in regions:
        if len(text) < 2048:
            continue
        for position in range(0, len(text) - 64, 64):
            anchor = text[position : position + 64]
            window_end = min(len(text), position + 4096 + 64)
            search_start = position + 1
            while search_start + 64 <= window_end:
                # One search unit covers at most 64 searched characters; extensions cost one per character.
                remaining_work -= max(1, (window_end - search_start + 63) // 64)
                if remaining_work < 0:
                    raise _Limit
                match = text.find(anchor, search_start, window_end)
                if match < 0:
                    break
                period = match - position
                required = max(2048, 16 * period)
                start = position
                end = match + 64
                while start > 0 and end - start < required:
                    remaining_work -= 1
                    if remaining_work < 0:
                        raise _Limit
                    if text[start - 1] != text[start + period - 1]:
                        break
                    start -= 1
                while end < len(text) and end - start < required:
                    remaining_work -= 1
                    if remaining_work < 0:
                        raise _Limit
                    if text[end] != text[end - period]:
                        break
                    end += 1
                if end - start >= required:
                    return True
                search_start = match + 1
    return False


def _assess_snapshot(snapshot: tuple[tuple, ...] | None) -> LoopingAssessment:
    """Scan the private immutable projection; called only in the caller's CPU context."""
    if snapshot is None:
        return LoopingAssessment(is_looping=None)
    complete = True
    text_complete = True
    results = {}
    request_ids = Counter(event[1] for event in snapshot if event[0] in {"function_call", "custom_tool_call"})
    for event in snapshot:
        if event[0] in {"function_call_output", "custom_tool_call_output"}:
            results.setdefault(event[1], []).append(event)
    signatures = []
    text_regions = [[]]
    try:
        for event in snapshot:
            item_type = event[0]
            if item_type == "text":
                fragments = []
                for fragment in event[1]:
                    if fragment is None:
                        text_regions[-1].append("".join(fragments))
                        fragments = []
                        text_regions.append([])
                        signatures.append(None)
                        complete = False
                        text_complete = False
                    else:
                        fragments.append(fragment)
                text_regions[-1].append("".join(fragments))
            elif item_type in {"function_call", "custom_tool_call"}:
                _, call_id, namespace, name, arguments = event
                outcomes = results.get(call_id, [])
                if (
                    not call_id
                    or not name
                    or request_ids[call_id] != 1
                    or len(outcomes) != 1
                    or outcomes[0][0] != item_type + "_output"
                    or outcomes[0][2] is None
                ):
                    complete = False
                    signatures.append(None)
                    continue
                if item_type == "function_call":
                    try:
                        arguments = _canonical_arguments(arguments)
                    except (ValueError, RecursionError, OverflowError):
                        complete = False
                        signatures.append(None)
                        continue
                signatures.append((item_type, namespace, name, arguments, "".join(outcomes[0][2])))
            elif item_type == "gap" or request_ids[event[1]] != 1:
                complete = False
                signatures.append(None)
                if item_type == "gap":
                    text_complete = False
                    text_regions.append([])
        if _tool_cycle(signatures):
            return LoopingAssessment(is_looping=True, reasons=(LoopingReason.TOOL_CYCLE,))
        texts = ["\n\n".join(region) for region in text_regions]
        normalized = [re.sub(r"\s+", " ", text).strip() for text in texts]
        if text_complete:
            paragraphs = Counter(
                re.sub(r"\s+", " ", paragraph).strip() for text in texts for paragraph in re.split(r"\n\s*\n", text)
            )
            redundant = sum(
                (copies - 1) * len(paragraph)
                for paragraph, copies in paragraphs.items()
                if len(paragraph) >= 80 and copies >= 5
            )
            if redundant >= 8192 and redundant >= 0.2 * sum(map(len, normalized)):
                return LoopingAssessment(is_looping=True, reasons=(LoopingReason.REPEATED_TEXT,))
        if _periodic_text(normalized):
            return LoopingAssessment(is_looping=True, reasons=(LoopingReason.REPEATED_TEXT,))
        return LoopingAssessment(is_looping=False if complete else None)
    except (ValueError, TypeError, RecursionError, OverflowError):
        return LoopingAssessment(is_looping=None)


def assess_looping(output: Sequence[NeMoGymResponseOutputItem]) -> LoopingAssessment:
    """Assess severe repetition in retained output, excluding prompts and tool-result text.

    Tool equality uses kind, namespace/name, canonical arguments, and recorded text
    results, independent of IDs and optional transport status. Results may be text
    strings or text-part lists; other result forms interrupt tool cycles. Input
    caps and a deterministic search-work budget return unknown on exhaustion.
    """
    return _assess_snapshot(_snapshot_output(output))
