# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Passive repetition assessments over fresh, invocation-owned model responses."""

import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from nemo_gym.trajectory import _looping_detector as detector


REFERENCE_DETECTOR_SHA256 = "a5e4eeba1d6cfd2d624ee6406e0837ddb78097f0f3a29ab046fe40821fe8688b"
DETECTOR_SHA256 = "da47c53a0bbc854f16108e458c05cec4628f8cb2b995ce7874ce671c82171765"
EXTRACTOR_VERSION = "invocation-policy-responses-2"
MAX_OUTPUT_CHARS = 2_000_000
MAX_OUTPUT_ITEMS = 8192
MAX_TOOL_CALLS = 1024


class LoopingInputLimit(ValueError):
    """The supplied evidence exceeds a deterministic resource bound."""


@dataclass(frozen=True)
class LoopingAssessment:
    """A compact assessment; an unknown flag is never a measured negative."""

    detector_version: str = detector.VERSION
    detector_sha256: str = DETECTOR_SHA256
    detector_reference_sha256: str = REFERENCE_DETECTOR_SHA256
    extractor_version: str = EXTRACTOR_VERSION
    input_digest: str | None = None
    severe_text: bool | None = None
    severe_tool_cycle: bool | None = None
    severe_union: bool | None = None
    warning_text: bool | None = None
    warning_tool_cycle: bool | None = None
    warning_union: bool | None = None
    gaps: tuple[str, ...] = ()
    text_chars: int = 0
    tool_calls: int = 0
    scan_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable metadata without generated text."""
        return asdict(self)


@dataclass(frozen=True)
class LoopingInput:
    """Ordered generated text and paired tool signatures, with modality coverage."""

    segments: tuple[str, ...]
    tools: tuple[dict[str, str], ...]
    text_known: bool
    tools_known: bool
    gaps: tuple[str, ...]


def unknown_looping_assessment(reason: str) -> LoopingAssessment:
    """Represent unavailable evidence or an analysis failure without inventing a verdict."""
    return LoopingAssessment(gaps=(reason,))


def check_looping_input_size(value: object) -> None:
    """Bound evidence before copying, model validation, normalization, or hashing."""
    stack = [(value, 0)]
    chars = items = 0
    while stack:
        item, depth = stack.pop()
        if depth > 32:
            raise LoopingInputLimit("input_depth_limit")
        if isinstance(item, str):
            chars += len(item)
        elif isinstance(item, Mapping):
            items += len(item)
        elif isinstance(item, (list, tuple)):
            items += len(item)
        if chars > MAX_OUTPUT_CHARS:
            raise LoopingInputLimit("output_character_limit")
        if items > MAX_OUTPUT_ITEMS * 16:
            raise LoopingInputLimit("output_item_limit")
        if isinstance(item, Mapping):
            chars += sum(len(key) for key in item if isinstance(key, str))
            if chars > MAX_OUTPUT_CHARS:
                raise LoopingInputLimit("output_character_limit")
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend((child, depth + 1) for child in item)


def detect_looping(events: LoopingInput) -> LoopingAssessment:
    """Apply the frozen thresholds and three-valued modality union."""
    started = time.perf_counter()
    verdict = detector.detect_transcript(
        [{"text": text} for text in events.segments],
        list(events.tools),
        {"text_known": events.text_known, "tools_known": events.tools_known},
    )
    return LoopingAssessment(
        input_digest=detector.sha(detector.canonical(asdict(events))),
        severe_text=verdict["severe"]["text"],
        severe_tool_cycle=verdict["severe"]["tool_cycle"],
        severe_union=verdict["severe"]["union"],
        warning_text=verdict["warning"]["text"],
        warning_tool_cycle=verdict["warning"]["tool_cycle"],
        warning_union=verdict["warning"]["union"],
        gaps=events.gaps,
        text_chars=sum(map(len, events.segments)),
        tool_calls=len(events.tools),
        scan_ms=(time.perf_counter() - started) * 1000,
    )
