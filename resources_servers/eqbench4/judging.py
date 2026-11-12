# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Judge-prompt filling and verdict parsing for EQ-Bench 4.

The pairwise judge compares two transcripts of the same scenario on several ability dimensions and names, for each
dimension, the winning respondent's code followed by one to five plus signs, or "tie". This module reproduces the
EQ-Bench 4 transcript layout, verdict parsing and margin arithmetic (https://github.com/EQ-bench/eqbench4, revision
93dcb7f5d430433ad49d501e672a98bcff2c9678) as an independent implementation. No upstream prompt text lives here: the
prompts are rendered from the upstream files at preparation time and travel in the dataset rows.
"""

import json
import random
import re
import string
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


MAX_PLUS_PER_DIMENSION = 5
# How the pairwise judge sees a transcript without numeric persona state, as in every EQ-Bench 4 run.
NO_FINAL_STATE = "(No final state available)"

_THINKING_BLOCK = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINKING = re.compile(r"^.*</(?:think|thinking|reasoning)>", re.DOTALL | re.IGNORECASE)
_UNFINISHED_THINKING = re.compile(r"<(?:think|thinking|reasoning)>.*$", re.DOTALL | re.IGNORECASE)
_INTERNAL_TAGGED = re.compile(r"<internal>\s*(.*?)\s*</internal>", re.DOTALL | re.IGNORECASE)
_INTERNAL_BRACKETED = re.compile(r"\[INTERNAL\]\s*(.*?)\s*\[/INTERNAL\]", re.DOTALL)
_STATE_TAGGED = re.compile(r"<state>.*?</state>", re.DOTALL | re.IGNORECASE)
_STATE_BRACKETED = re.compile(r"\[STATE\].*?\[/STATE\]", re.DOTALL)
_JSON_FENCE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)
_SCORES_SECTION = re.compile(r"\[SCORES\](.*?)\[/SCORES\]", re.DOTALL)
_GREEDY_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_FLAT_OBJECT = re.compile(r"\{[^{}]+\}", re.DOTALL)


class PromptParts:
    """A prompt split around named slots: literal, slot, literal, ..., literal."""

    def __init__(self, literals: Sequence[str], slots: Sequence[str]) -> None:
        if len(literals) != len(slots) + 1:
            raise ValueError("a prompt needs exactly one more literal than slots")
        self.literals = list(literals)
        self.slots = list(slots)

    def fill(self, values: Mapping[str, str]) -> str:
        missing = sorted(set(self.slots) - set(values))
        if missing:
            raise KeyError(f"no value for prompt slot(s): {', '.join(missing)}")
        pieces = [self.literals[0]]
        for slot, literal in zip(self.slots, self.literals[1:]):
            pieces.extend((values[slot], literal))
        return "".join(pieces)


def strip_thinking(text: str) -> str:
    """Remove reasoning: closed blocks, anything before a lone closing tag, and a block cut off before it closed."""
    text = _THINKING_BLOCK.sub("", text)
    text = _UNCLOSED_THINKING.sub("", text)
    return _UNFINISHED_THINKING.sub("", text).strip()


def persona_internal(message: Mapping[str, object]) -> str | None:
    """The persona's private monologue for one of its messages, if it wrote one."""
    internal = message.get("internal")
    if isinstance(internal, str):
        return internal
    raw = message.get("raw_response")
    if not isinstance(raw, str):
        return None
    match = _INTERNAL_TAGGED.search(raw) or _INTERNAL_BRACKETED.search(raw)
    return match.group(1).strip() if match else None


def pairwise_transcript(messages: Sequence[Mapping[str, object]]) -> str:
    """Lay out a transcript for the pairwise judge, showing each persona monologue before its dialogue."""
    blocks: list[str] = []
    for message in messages:
        content = message.get("content") or ""
        if message.get("role") == "user":
            internal = persona_internal(message)
            if internal:
                blocks.append(f"---\nUser (internal monologue):\n<internal>\n{internal}\n</internal>")
            blocks.append(f"---\nUser:\n{content}")
        else:
            blocks.append(f"---\nAssistant:\n{content}")
    return "\n".join(blocks)


def trait_transcript(messages: Sequence[Mapping[str, object]], assistant_label: str) -> str:
    """Lay out a transcript for the trait judge: the persona's full reply (minus any state block) each turn."""
    lines: list[str] = []
    for message in messages:
        if message.get("role") == "user":
            raw = message.get("raw_response")
            text = (raw if isinstance(raw, str) else message.get("content") or "").strip()
            text = _STATE_TAGGED.sub("", text)
            text = _STATE_BRACKETED.sub("", text)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            lines.append(f"Person: {text}\n\n")
        else:
            lines.append(f"{assistant_label}: {(message.get('content') or '').strip()}\n\n")
    return "".join(lines)


def respondent_codes(seed: str) -> tuple[str, str]:
    """Two distinct anonymous respondent codes (a capital letter and four digits), fixed by ``seed``."""
    rng = random.Random(seed)

    def draw() -> str:
        return rng.choice(string.ascii_uppercase) + "".join(rng.choices(string.digits, k=4))

    first = draw()
    second = draw()
    while second == first:
        second = draw()
    return first, second


def parse_json_object(text: str) -> dict[str, object]:
    """The judge's JSON object: a fenced block, the whole reply, or its outermost braces; empty when none parses."""
    stripped = text.strip()
    fence = _JSON_FENCE.search(stripped)
    candidates = [fence.group(1).strip() if fence else stripped]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def parse_dimension_verdict(verdict: str, first_code: str, second_code: str) -> tuple[int | None, int]:
    """Which respondent (0 = first, 1 = second, None = tie or unreadable) won one dimension, and by how many '+'."""
    verdict = verdict.strip()
    if verdict.lower() == "tie":
        return None, 0
    code = verdict.rstrip("+").strip()
    if code == first_code:
        return 0, verdict.count("+")
    if code == second_code:
        return 1, verdict.count("+")
    return None, 0


def weighted_margins(
    verdicts: Mapping[str, object],
    first_code: str,
    second_code: str,
    dimension_weights: Mapping[str, float],
) -> tuple[float, float]:
    """Weighted plus totals of the first and second respondent over the scored dimensions."""
    totals = [0.0, 0.0]
    for dimension, weight in dimension_weights.items():
        verdict = verdicts.get(dimension)
        if not isinstance(verdict, str):
            continue
        winner, pluses = parse_dimension_verdict(verdict, first_code, second_code)
        if winner is not None:
            totals[winner] += pluses * weight
    return totals[0], totals[1]


@dataclass(frozen=True)
class ComparisonOutcome:
    """One policy-versus-opponent comparison, averaged over the judge readings that parsed."""

    policy_margin: float
    opponent_margin: float
    fraction: float  # the policy's share of the comparison, in [0, 1]


def combine_readings(
    readings: Sequence[tuple[float, float]], dimension_weights: Mapping[str, float]
) -> ComparisonOutcome:
    """Average (policy, opponent) plus totals over readings and turn the margin into the policy's fraction.

    A win scores 0.5 plus half the average margin as a share of the largest possible margin; a loss mirrors it; equal
    totals score 0.5.
    """
    if not readings:
        raise ValueError("a comparison needs at least one judge reading")
    policy = sum(reading[0] for reading in readings) / len(readings)
    opponent = sum(reading[1] for reading in readings) / len(readings)
    largest = sum(weight * MAX_PLUS_PER_DIMENSION for weight in dimension_weights.values())
    margin = min(abs(policy - opponent) / largest, 1.0) if largest > 0 else 0.0
    if policy > opponent:
        fraction = 0.5 + margin / 2
    elif opponent > policy:
        fraction = 0.5 - margin / 2
    else:
        fraction = 0.5
    return ComparisonOutcome(policy_margin=policy, opponent_margin=opponent, fraction=min(max(fraction, 0.0), 1.0))


def parse_trait_scores(text: str) -> dict[str, float] | None:
    """Numeric trait scores from a trait-judge reply's [SCORES] JSON, or from its first flat JSON object."""
    section = _SCORES_SECTION.search(text)
    if section:
        match = _GREEDY_OBJECT.search(section.group(1).strip())
        if match:
            try:
                scores = json.loads(match.group(0))
                return {key: float(value) for key, value in scores.items()}
            except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
                pass
    match = _FLAT_OBJECT.search(text)
    if match:
        try:
            scores = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        numeric = {
            key: float(value)
            for key, value in scores.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        return numeric or None
    return None
