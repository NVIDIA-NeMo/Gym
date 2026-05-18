# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Utilities for the rubrics-verifier server.

Vendored (with light edits) from
nemo-rlvr/nemo_rl/environments/llm_verification_utils/batched_local_llm_verifier.py
to keep the resources server self-contained.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_ANSWER_RE = re.compile(
    r"<answer>\s*((?:(?!<answer>).)*?)\s*</answer>(?!.*<answer>)", re.DOTALL
)


def extract_answer_from_response(response: str) -> str:
    """Mirror RLVR's rubrics extraction logic.

    - Unclosed ``<think>`` -> empty string (incomplete reasoning).
    - Closed ``<think>...</think>`` -> stripped, then look for ``<answer>...</answer>``.
    - If no ``<answer>`` block, return the post-think text stripped.
    """
    if _THINK_OPEN in response and _THINK_CLOSE not in response:
        return ""
    if _THINK_CLOSE in response:
        response = response.split(_THINK_CLOSE)[-1].strip()
    match = _ANSWER_RE.search(response)
    if match:
        return match.group(1).strip()
    return response.strip()


def extract_rubrics(ground_truth: Any) -> List[Dict[str, Any]]:
    """Pull a list of rubric dicts out of a (possibly nested / JSON-string) ground truth."""
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except json.JSONDecodeError:
            return []

    if not isinstance(ground_truth, dict):
        return []

    rubrics = ground_truth.get("rubrics", [])

    if rubrics and isinstance(rubrics, list) and len(rubrics) > 0:
        if isinstance(rubrics[0], list):
            rubrics = rubrics[0]

    return rubrics or []


def extract_user_question(messages: list) -> Optional[str]:
    """Find the first user message text in NeMoGym input messages."""
    for m in messages or []:
        role = getattr(m, "role", None)
        if role is None and isinstance(m, dict):
            role = m.get("role")
        if role == "user":
            content = getattr(m, "content", None)
            if content is None and isinstance(m, dict):
                content = m.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list) and content:
                first = content[0]
                t = getattr(first, "text", None) if not isinstance(first, dict) else first.get("text")
                if isinstance(t, str):
                    return t
    return None


def format_rubrics_for_prompt(rubrics: List[Dict[str, Any]]) -> str:
    """Format a list of rubrics into the bullet-list block the prompt expects.

    Supports both the new format (``criterion``/``points``) and the legacy
    format (``title``/``description``/``weight``). Unknown rubric shapes fall
    back to ``str(rubric)``.
    """
    lines: List[str] = []
    for i, rubric in enumerate(rubrics):
        if isinstance(rubric, dict):
            if "criterion" in rubric:
                criterion = rubric.get("criterion", "")
                points = rubric.get("points", 1)
                lines.append(
                    f"Rubric-{i}:\n  Criterion: {criterion}\n  Points: {points}"
                )
            else:
                title = rubric.get("title", f"Criterion {i}")
                description = rubric.get("description", "")
                weight = rubric.get("weight", 1)
                lines.append(
                    f"Rubric-{i}:\n  Title: {title}\n  Description: {description}\n  Weight: {weight}"
                )
        else:
            lines.append(f"Rubric-{i}: {rubric}")
    return "\n\n".join(lines)


def parse_rubrics_response(response: str, num_rubrics: int) -> Dict[str, Dict]:
    r"""Parse a judge response into a mapping ``Rubric-i -> {passed: bool, ...}``.

    Tries (in order): the whole response as JSON, the last ```json ... ```
    block, and the last triple-backtick block (greedy). Raises ``Exception``
    if none parse, or if any expected ``Rubric-i`` key is missing. Coerces
    the ``passed`` field to a boolean.
    """
    if _THINK_CLOSE in response:
        response = response.split(_THINK_CLOSE)[-1].strip()

    parsed: Optional[Dict[str, Any]] = None

    try:
        parsed = json.loads(response.strip())
    except json.JSONDecodeError:
        pass

    if parsed is None:
        matches = list(
            re.finditer(
                r"```(?:json)?\s*([\s\S]*?)\s*```",
                response,
                flags=re.IGNORECASE,
            )
        )
        if matches:
            try:
                payload = matches[-1].group(1).strip()
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                pass

    if parsed is None:
        matches = list(re.finditer(r"```([\s\S]*?)```", response))
        if matches:
            for match in reversed(matches):
                try:
                    payload = match.group(1).strip()
                    if payload.startswith("json"):
                        payload = payload[4:].strip()
                    parsed = json.loads(payload)
                    break
                except json.JSONDecodeError:
                    continue

    if parsed is None:
        raise Exception(
            "Response does not contain valid JSON (tried: raw JSON, ```json```, and ```)."
        )

    for i in range(num_rubrics):
        key = f"Rubric-{i}"
        if key not in parsed:
            raise Exception(
                f"No {key} found in the response. Got rubrics: {list(parsed.keys())}"
            )
        if "passed" not in parsed[key]:
            parsed[key]["passed"] = False
        else:
            val = parsed[key]["passed"]
            if isinstance(val, str):
                parsed[key]["passed"] = val.lower() in ("true", "1", "yes", "pass")
            else:
                parsed[key]["passed"] = bool(val)

    return parsed


def compute_weighted_reward(
    rubrics: List[Dict[str, Any]], parsed_response: Dict[str, Dict]
) -> tuple[float, int, int]:
    """Compute a weighted reward from per-rubric pass/fail flags.

    - Positive-weight rubrics contribute their weight to the denominator and
      to the numerator on PASS.
    - Negative-weight rubrics (pitfalls) subtract their weight from the
      numerator on FAIL (pitfall present).
    - The reward is ``max(0.0, earned / total_positive_weight)``. If there is
      no positive weight at all, the reward is 1.0 iff ``earned >= 0``.

    Returns ``(reward, passed_count, total_count)``. ``passed_count`` includes
    pitfalls that were avoided (PASS on a negative-weight rubric).
    """
    total_positive_weight = 0.0
    earned_score = 0.0
    passed_count = 0

    for i, rubric in enumerate(rubrics):
        weight = rubric.get("points", rubric.get("weight", 1))
        rubric_key = f"Rubric-{i}"
        passed = parsed_response.get(rubric_key, {}).get("passed", False)

        if weight >= 0:
            total_positive_weight += weight
            if passed:
                earned_score += weight
                passed_count += 1
        else:
            if not passed:
                earned_score += weight
            else:
                passed_count += 1

    if total_positive_weight > 0:
        reward = max(0.0, earned_score / total_positive_weight)
    else:
        reward = 1.0 if earned_score >= 0 else 0.0

    return reward, passed_count, len(rubrics)
