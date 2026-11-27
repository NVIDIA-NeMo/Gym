# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded severe repetition checks on the assistant text returned by an episode."""

import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Iterable, Mapping

from nemo_gym.core.responses_api_types import (
    NeMoGymResponse,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputRefusal,
    NeMoGymResponseOutputText,
    NeMoGymResponseReasoningItem,
)


@dataclass(frozen=True)
class LoopingLimits:
    """Immutable per-response bounds for relevant text, item/part counts, and search work."""

    max_text_chars: int = 2_000_000
    max_items: int = 8192
    search_work: int = 4_000_000


class LoopingReason(StrEnum):
    """Reasons for a confirmed severe looping assessment."""

    REPEATED_TEXT = "repeated_text"


@dataclass(frozen=True)
class LoopingAssessment:
    """Severe text repetition outcome; None means unassessed and reasons describe confirmed loops."""

    is_looping: bool | None = None
    reasons: tuple[LoopingReason, ...] = ()


def _get(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def collect_response_text(
    response: NeMoGymResponse | Mapping[str, object] | None, limits: LoopingLimits
) -> tuple[str, ...] | None:
    """Read bounded text without copying response dictionaries or token/media metadata.

    Item/part defaults come from Gym's models. Strings and text parts concatenate directly;
    selected output items are separated by a blank line. Status does not discard retained text.
    None denotes unreadable or out-of-profile text; an empty tuple is observed empty output.
    """
    output = _get(response, "output")
    if not isinstance(output, list) or len(output) > limits.max_items:
        return None
    chunks: list[str] = []
    size, items = 0, 0
    for item in output:
        if not isinstance(item, Mapping) and not hasattr(item, "type"):
            return None
        model = NeMoGymResponseReasoningItem if _get(item, "summary") is not None else NeMoGymResponseOutputMessage
        kind = _get(item, "type", model.model_fields["type"].default)
        if kind == NeMoGymResponseOutputMessage.model_fields["type"].default:
            role = _get(item, "role", NeMoGymResponseOutputMessage.model_fields["role"].default)
            if role in {"user", "system", "developer"}:
                continue
            if role != "assistant":
                return None
            content = _get(item, "content")
        elif kind == NeMoGymResponseReasoningItem.model_fields["type"].default:
            content = _get(item, "content")
            if content is None or content == []:
                content = _get(item, "summary")
            elif not isinstance(content, list):
                return None
        elif isinstance(kind, str):
            continue
        else:
            return None
        if isinstance(content, str):
            parts = [content]
        elif isinstance(content, list):
            parts = content
        else:
            return None
        items += len(parts)
        if items > limits.max_items:
            return None
        selected = []
        for part in parts:
            if isinstance(part, str):
                text = part
            else:
                default = (
                    NeMoGymResponseOutputRefusal if _get(part, "refusal") is not None else NeMoGymResponseOutputText
                )
                part_type = _get(part, "type", default.model_fields["type"].default)
                if part_type in {"output_text", "input_text", "text", "reasoning_text", "summary_text"}:
                    text = _get(part, "text")
                elif part_type == "refusal":
                    text = _get(part, "refusal")
                elif part_type in {"input_image", "input_audio", "input_video", "input_file", "output_audio"}:
                    continue
                else:
                    return None
            if not isinstance(text, str):
                return None
            size += len(text)
            if size > limits.max_text_chars:
                return None
            selected.append(text)
        if chunks:
            size += 2
        if size > limits.max_text_chars:
            return None
        chunks.append("".join(selected))
    return tuple(chunks)


def assess_text(chunks: tuple[str, ...], limits: LoopingLimits = LoopingLimits()) -> LoopingAssessment:
    """Apply paragraph and exact-period rules; return unknown if bounded search runs out.

    Search units are indexed 64-character anchors, candidate periods, and individual
    character equality comparisons. Normalization and paragraph counting are linear.
    Index every anchor, but probe candidates at a 64-character stride. Extending a
    candidate left by up to 63 characters preserves coverage at severe-span boundaries.
    The index retains only the previous 4096 positions. Stop at the first severe witness.
    """
    if sum(map(len, chunks)) + max(0, len(chunks) - 1) * 2 > limits.max_text_chars:
        return LoopingAssessment()
    raw_text = "\n\n".join(chunks)
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", raw_text)]
    text = " ".join(raw_text.split())
    if len(text) > limits.max_text_chars:
        return LoopingAssessment()
    redundant = sum((count - 1) * len(p) for p, count in Counter(paragraphs).items() if len(p) >= 80 and count >= 5)
    if redundant >= 8192 and redundant >= 0.2 * len(text):
        return LoopingAssessment(True, (LoopingReason.REPEATED_TEXT,))
    anchors: dict[str, deque[int]] = defaultdict(deque)
    work = limits.search_work
    for pos in range(max(0, len(text) - 63)):
        work -= 1
        if work < 0:
            return LoopingAssessment()
        if pos > 4096:
            expired = text[pos - 4097 : pos - 4097 + 64]
            anchors[expired].popleft()
            if not anchors[expired]:
                del anchors[expired]
        anchor = text[pos : pos + 64]
        if pos % 64 == 0:
            for previous in reversed(anchors.get(anchor, ())):
                work -= 1
                if work < 0:
                    return LoopingAssessment()
                period = pos - previous
                needed = max(2048, 16 * period)
                if previous - 63 + needed > len(text):
                    continue
                start = previous
                while start > max(0, previous - 63):
                    work -= 1
                    if work < 0:
                        return LoopingAssessment()
                    if text[start - 1] != text[start - 1 + period]:
                        break
                    start -= 1
                end = start + needed
                if end > len(text):
                    continue
                for index in range(start + period, end):
                    work -= 1
                    if work < 0:
                        return LoopingAssessment()
                    if text[index] != text[index - period]:
                        break
                else:
                    return LoopingAssessment(True, (LoopingReason.REPEATED_TEXT,))
        anchors[anchor].append(pos)
    return LoopingAssessment(False)


def looping_metrics(rows: Iterable[Mapping[str, object]]) -> dict[str, int | float]:
    """Count selected completed outputs, including masked outputs, before computing ratios.

    Failed and imputed benchmark zeros have no completed output and are excluded. Saved
    completed outputs without an assessment contribute to coverage's denominator only.
    """
    completed = assessed = flagged = 0
    enabled = False
    for row in rows:
        enabled |= "is_looping" in row
        if (
            row.get("_ng_no_persist")
            or row.get("_ng_failure_counts_as_zero")
            or row.get("_ng_failure_class") is not None
        ):
            continue
        completed += 1
        value = row.get("is_looping")
        if type(value) is bool:
            assessed += 1
            flagged += value
    if not enabled:
        return {}
    result = {
        "looping/completed_total": completed,
        "looping/assessed_total": assessed,
        "looping/flagged_total": flagged,
    }
    if completed:
        result["looping/coverage"] = assessed / completed
    if assessed:
        result["looping/rate"] = flagged / assessed
    return result
