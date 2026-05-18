# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Utilities for parsing the loose-verifier judge response.

Vendored (with light edits) from
nemo-rlvr/nemo_rl/environments/llm_verification_utils/batched_local_llm_verifier.py
to keep the resources server self-contained.
"""
from __future__ import annotations

import json
import re
from typing import Optional


def fix_invalid_json_escapes(s: str) -> str:
    """Double all backslashes to neutralize invalid JSON escapes (e.g. \\geq, \\leq)."""
    return s.replace("\\", "\\\\")


def parse_individual_response(response: str) -> int:
    """Parse a judge response and return the 0/1 score.

    Tolerates LaTeX backslashes inside the JSON payload and trailing commas.
    Falls back to a regex match for ``"score": <0|1>`` if JSON parsing fails.
    Raises ``Exception`` if no recognizable score is present.
    """
    matches = list(
        re.finditer(
            r"```(?:json)?\s*([\s\S]*?)\s*```",
            response,
            flags=re.IGNORECASE,
        )
    )
    if not matches:
        raise Exception("Response does not contain a valid JSON block.")
    payload = matches[-1].group(1)
    payload = fix_invalid_json_escapes(payload)

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as e:
        fixed_payload = re.sub(r",\s*\}", "}", payload)
        try:
            parsed = json.loads(fixed_payload)
        except json.JSONDecodeError:
            score_match = re.search(r'"score"\s*:\s*([01])', payload)
            if score_match:
                return int(score_match.group(1))
            raise Exception(
                f"Could not parse JSON: {e}\nPayload: {payload[:500]}"
            )

    if "score" in parsed:
        return int(float(parsed["score"]))
    if "Score" in parsed:
        return int(float(parsed["Score"]))
    for key in parsed:
        if isinstance(parsed[key], dict):
            if "score" in parsed[key]:
                return int(float(parsed[key]["score"]))
            if "Score" in parsed[key]:
                return int(float(parsed[key]["Score"]))
    raise Exception(f"Could not find score in response: {parsed}")


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_ANSWER_RE = re.compile(
    r"<answer>\s*((?:(?!<answer>).)*?)\s*</answer>(?!.*<answer>)", re.DOTALL
)


def extract_answer_from_response(response: str) -> str:
    """Mirror RLVR's StemQA answer extraction.

    Returns "" if the response opens <think> without closing it. Otherwise
    strips the <think> block (if any) and returns the content of the LAST
    <answer>...</answer> block, or "" if no <answer> is present.
    """
    if _THINK_OPEN in response and _THINK_CLOSE not in response:
        return ""
    if _THINK_CLOSE in response:
        response = response.split(_THINK_CLOSE)[-1].strip()
    match = _ANSWER_RE.search(response)
    if match:
        return match.group(1).strip()
    return ""


def extract_mc_answer(response: str) -> str:
    """Extract the multi-choice letter from RLVR's regex pattern.

    Strips a closed <think> block before searching. Returns "" if no match.
    """
    if _THINK_CLOSE in response:
        response = response.split(_THINK_CLOSE)[-1].strip()
    matches = re.findall(r"[Aa]nswer\s*(?:is|:)?\s*\(([A-Za-z])\)", response)
    return matches[-1] if matches else ""


def extract_user_question(messages: list) -> Optional[str]:
    """Find the first user message text in a list of NeMoGym input messages."""
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
