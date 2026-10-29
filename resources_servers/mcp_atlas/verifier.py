# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MCP-Atlas claim-coverage scoring, ported from upstream ``services/scoring/score_claims.py``.

Each ground-truth claim is judged on its own against the final answer only (the prompt is not shown
to the judge). A claim scores 1 / 0.5 / 0 for fulfilled / partially fulfilled / not fulfilled, the
task's coverage is the mean rounded to three places, and a task passes at coverage >= 0.75.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from nemo_gym.openai_utils import NeMoGymResponse


# Verbatim from upstream score_claims.py (scaleapi/mcp-atlas 5c11ac44).
CLAIM_PROMPT = """You are evaluating how well a model's response addresses a specific expert-defined claim.
SCORING CRITERIA:
- fulfilled: Claim is completely and accurately addressed. The response covers all key details.
- partially_fulfilled: Claim is partially addressed. The response covers some but not all key details.
- not_fulfilled: Claim is not addressed. The response does not include any key details.
NUMERICAL COMPARISON GUIDELINES:
- For numerical values, use reasonable approximation thresholds:
  * Exact match NOT required for decimals
  * Values within 5% of the claimed number are considered matching
  * For percentages, ±1 percentage points is acceptable
  * Round to appropriate significant figures based on context
- Consider the precision appropriate to the domain:
  * Scientific measurements may need higher precision
  * General statistics/estimates can have looser matching
  * Financial figures should match to reasonable business precision (e.g., millions/billions don't need exact cents)
- If a number is expressed differently but mathematically equivalent (e.g., "0.5" vs "50%" vs "half"), consider it a match
CLAIM TO EVALUATE:
{claim}
MODEL RESPONSE TO ANALYZE:
{response}
INSTRUCTIONS:
1. Determine if the core requirement of the claim is met in the response
2. Check if all key components from the claim appear substantively in the response
   - For numerical values, apply the flexible matching guidelines above
   - Focus on whether the same magnitude and meaning are conveyed
3. Assign the appropriate coverage_outcome
4. Provide specific justification referencing what was/wasn't covered
   - When numbers differ slightly, note if they're within acceptable range
5. Provide a confidence level (0.0-1.0) for your assessment
Be rigorous but fair in your assessment. Focus on whether the response conveys the same information as the claim, not on exact numerical precision unless precision is critical to the claim's meaning."""

CLAIM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "claim_text": {"type": "string"},
        "coverage_outcome": {"type": "string", "enum": ["fulfilled", "partially_fulfilled", "not_fulfilled"]},
        "justification": {"type": "string"},
        "confidence_level": {"type": "number"},
    },
    "required": ["claim_text", "coverage_outcome", "justification", "confidence_level"],
}

OUTCOME_SCORE = {"fulfilled": 1.0, "partially_fulfilled": 0.5, "not_fulfilled": 0.0}
# Upstream truncates the judged response at this length.
MAX_RESPONSE_CHARS = 500_000


def final_answer(response: NeMoGymResponse) -> str:
    """The text of the last assistant message with non-empty content, as the upstream harness reports it."""
    for item in reversed(response.output):
        if getattr(item, "type", None) != "message" or getattr(item, "role", None) != "assistant":
            continue
        content = getattr(item, "content", None)
        if isinstance(content, str):
            text = content
        else:
            text = "".join(t for part in content or [] if isinstance(t := getattr(part, "text", None), str))
        if text.strip():
            return text
    return ""


def judge_request(*, claim: str, answer: str, max_tokens: Optional[int]) -> dict[str, Any]:
    """One Chat Completions request that judges one claim (temperature 0, structured output).

    The judge model server fills in its own model name.
    """
    params: dict[str, Any] = {
        "temperature": 0.0,
        "messages": [
            {"role": "user", "content": CLAIM_PROMPT.format(claim=claim, response=answer[:MAX_RESPONSE_CHARS])}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "claim_evaluation", "schema": CLAIM_SCHEMA},
        },
    }
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    return params


def parse_verdict(claim: str, content: Optional[str]) -> Optional[dict[str, Any]]:
    """Read the judge's JSON verdict, or None when the reply has no valid ``coverage_outcome``."""
    try:
        verdict = json.loads(content or "")
    except json.JSONDecodeError:
        return None
    if not isinstance(verdict, dict) or verdict.get("coverage_outcome") not in OUTCOME_SCORE:
        return None
    outcome = verdict["coverage_outcome"]
    return {
        "claim": claim,
        "coverage_outcome": outcome,
        "score": OUTCOME_SCORE[outcome],
        "justification": str(verdict.get("justification", "")),
    }


def coverage(verdicts: list[dict[str, Any]], n_claims: int) -> float:
    """Mean claim score, rounded to three places as upstream does."""
    if n_claims == 0:
        return 0.0
    return round(sum(v["score"] for v in verdicts) / n_claims, 3)
