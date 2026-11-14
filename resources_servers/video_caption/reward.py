# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Visual caption recall/precision/F1, length reward, and a degeneration gate."""

import json
from dataclasses import dataclass
from typing import Any, Literal


VisualRewardType = Literal["recall", "precision", "f1"]


@dataclass(frozen=True)
class VideoCaptionRewardResult:
    reward: float
    is_valid_caption: bool
    caption_length_tokens: int
    length_reward: float
    visual_hits: list[int]
    visual_recall_reward: float | None
    precision_visual_hits: list[int]
    visual_precision_reward: float | None
    visual_f1_reward: float | None
    visual_reward_type: VisualRewardType
    visual_reward: float
    reward_before_degeneration: float
    degeneration_checked: bool
    degeneration_detected: bool
    degeneration_reward: float


def normalize_atomic_events(verifier_metadata: dict | None) -> dict[str, list[str]]:
    metadata = verifier_metadata or {}
    raw = metadata.get("atomic_events", metadata.get("event", metadata))
    if not isinstance(raw, dict):
        raise ValueError("verifier_metadata.atomic_events must be an object")
    events = raw.get("visual_events")
    if (
        not isinstance(events, list)
        or not events
        or any(not isinstance(event, str) or not event.strip() for event in events)
    ):
        raise ValueError("visual_events must be a nonempty list of nonempty strings")
    # Deliberately do not inspect audio or audio-visual fields, even if malformed.
    return {"visual_events": [event.strip() for event in events]}


def format_visual_judge_input(caption_text: str, atomic_events: dict) -> str:
    return f"CANDIDATE CAPTION:\n{caption_text}\n\nVISUAL EVENTS:\n" + json.dumps(
        atomic_events["visual_events"], ensure_ascii=False
    )


def visual_hits_response_format(event_count: int) -> dict:
    if event_count < 1:
        raise ValueError("At least one visual event is required")
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "visual_event_coverage",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "visual_hits": {
                        "type": "array",
                        "items": {"type": "integer", "enum": [0, 1]},
                        "minItems": event_count,
                        "maxItems": event_count,
                    }
                },
                "required": ["visual_hits"],
                "additionalProperties": False,
            },
        },
    }


def parse_visual_hits(text: str, event_count: int) -> list[int]:
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"visual_hits"}:
        raise ValueError("Judge response must contain only visual_hits")
    return _validate_visual_hits(value["visual_hits"], event_count)


def _validate_visual_hits(hits: Any, event_count: int) -> list[int]:
    if not isinstance(hits, list) or len(hits) != event_count:
        raise ValueError(f"visual_hits must contain exactly {event_count} entries")
    if any(type(hit) is not int or hit not in (0, 1) for hit in hits):
        raise ValueError("visual_hits must contain only integer 0 or 1")
    return hits


def visual_events_response_format() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "visual_event_extraction",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"visual_events": {"type": "array", "items": {"type": "string"}}},
                "required": ["visual_events"],
                "additionalProperties": False,
            },
        },
    }


def parse_visual_events(text: str) -> list[str]:
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"visual_events"}:
        raise ValueError("Extraction response must contain only visual_events")
    events = value["visual_events"]
    if not isinstance(events, list) or any(not isinstance(event, str) or not event.strip() for event in events):
        raise ValueError("Extracted visual_events must be a list of nonempty strings")
    return [event.strip() for event in events]


def degeneration_response_format() -> dict:
    """A separate text-quality verdict with no recall assessment."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "caption_degeneration",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"verdict": {"type": "string", "enum": ["pass", "fail"]}},
                "required": ["verdict"],
                "additionalProperties": False,
            },
        },
    }


def parse_degeneration_judgment(text: str) -> bool:
    """Validate the judge's pass/fail decision; do not classify defects in code."""
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"verdict"} or value["verdict"] not in ("pass", "fail"):
        raise ValueError("Degeneration judge response must contain only verdict: pass or fail")
    return value["verdict"] == "pass"


def compute_length_reward(
    caption_text: str | None,
    caption_length_tokens: int,
    *,
    min_caption_tokens: int = 100,
    max_caption_tokens: int = 4000,
) -> float:
    if caption_length_tokens < 0 or min_caption_tokens < 0 or max_caption_tokens < min_caption_tokens:
        raise ValueError("Invalid caption token count or bounds")
    return float(
        bool((caption_text or "").strip()) and min_caption_tokens <= caption_length_tokens <= max_caption_tokens
    )


def combine_rewards(
    visual_reward: float,
    length_reward: float,
    *,
    event_recall_weight: float = 1.0,
    length_weight: float = 1.0,
) -> float:
    total = event_recall_weight + length_weight
    if event_recall_weight < 0 or length_weight < 0 or total <= 0:
        raise ValueError("Reward weights must be nonnegative with a positive sum")
    return (event_recall_weight * visual_reward + length_weight * length_reward) / total


def _result(
    caption_text: str | None,
    caption_length_tokens: int,
    hits: list[int] | None,
    *,
    visual_reward_type: VisualRewardType,
    precision_hits: list[int] | None = None,
    degeneration_passed: bool | None = None,
    min_caption_tokens: int = 100,
    max_caption_tokens: int = 4000,
    event_recall_weight: float = 1.0,
    length_weight: float = 1.0,
) -> VideoCaptionRewardResult:
    caption = (caption_text or "").strip()
    recall = (sum(hits) / len(hits) if hits else 0.0) if hits is not None else None
    precision = (
        (sum(precision_hits) / len(precision_hits) if precision_hits else 0.0) if precision_hits is not None else None
    )
    f1 = None
    if recall is not None and precision is not None:
        f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    if visual_reward_type not in ("recall", "precision", "f1"):
        raise ValueError(f"Unknown visual reward type: {visual_reward_type}")
    visual_reward = {"recall": recall, "precision": precision, "f1": f1}[visual_reward_type]
    if visual_reward is None:
        raise ValueError(f"Missing judgments for visual reward type: {visual_reward_type}")
    length = compute_length_reward(
        caption,
        caption_length_tokens,
        min_caption_tokens=min_caption_tokens,
        max_caption_tokens=max_caption_tokens,
    )
    reward = combine_rewards(
        visual_reward,
        length,
        event_recall_weight=event_recall_weight,
        length_weight=length_weight,
    )
    degeneration_detected = degeneration_passed is False
    degeneration_reward = float(not degeneration_detected)
    return VideoCaptionRewardResult(
        reward=reward * degeneration_reward,
        is_valid_caption=bool(caption),
        caption_length_tokens=caption_length_tokens,
        length_reward=length,
        visual_hits=hits if hits is not None else [],
        visual_recall_reward=recall,
        precision_visual_hits=precision_hits if precision_hits is not None else [],
        visual_precision_reward=precision,
        visual_f1_reward=f1,
        visual_reward_type=visual_reward_type,
        visual_reward=visual_reward,
        reward_before_degeneration=reward,
        degeneration_checked=degeneration_passed is not None,
        degeneration_detected=degeneration_detected,
        degeneration_reward=degeneration_reward,
    )


def empty_video_caption_reward(
    caption_text: str | None, caption_length_tokens: int = 0, *, visual_reward_type: VisualRewardType, **kwargs: Any
) -> VideoCaptionRewardResult:
    return _result(
        caption_text,
        caption_length_tokens,
        [] if visual_reward_type in ("recall", "f1") else None,
        visual_reward_type=visual_reward_type,
        precision_hits=[] if visual_reward_type in ("precision", "f1") else None,
        **kwargs,
    )


def compute_video_caption_reward(
    caption_text: str | None,
    caption_length_tokens: int,
    atomic_events: dict,
    judge_output_text: str | None,
    *,
    visual_reward_type: VisualRewardType,
    predicted_visual_events: list[str] | None = None,
    precision_judge_output_text: str | None = None,
    penalize_degeneration: bool,
    degeneration_judge_output_text: str | None = None,
    **kwargs: Any,
) -> VideoCaptionRewardResult:
    caption = (caption_text or "").strip()
    if not caption:
        return empty_video_caption_reward(
            caption, caption_length_tokens, visual_reward_type=visual_reward_type, **kwargs
        )
    hits = None
    precision_hits = None
    degeneration_passed = None
    if visual_reward_type in ("recall", "f1"):
        if judge_output_text is None:
            raise ValueError("Recall requires a judge response")
        hits = parse_visual_hits(judge_output_text, len(atomic_events["visual_events"]))
    if visual_reward_type in ("precision", "f1"):
        if predicted_visual_events is None:
            raise ValueError("Precision requires extracted prediction events")
        if predicted_visual_events:
            if precision_judge_output_text is None:
                raise ValueError("Precision requires a judge response for predicted events")
            precision_hits = parse_visual_hits(precision_judge_output_text, len(predicted_visual_events))
        else:
            precision_hits = []
    if penalize_degeneration:
        if degeneration_judge_output_text is None:
            raise ValueError("Enabled degeneration judging requires a separate judge response")
        degeneration_passed = parse_degeneration_judgment(degeneration_judge_output_text)
    return _result(
        caption_text,
        caption_length_tokens,
        hits,
        visual_reward_type=visual_reward_type,
        precision_hits=precision_hits,
        degeneration_passed=degeneration_passed,
        **kwargs,
    )
