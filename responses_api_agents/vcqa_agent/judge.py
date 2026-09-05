# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM-judge-based scoring for VCQA rollouts.

A VCQA judge rubric is a list of items (typically `must_have` and
`good_to_have`). For each must-have item the judge is asked "does the
model's answer satisfy this criterion? Reply YES or NO." in a single chat
completion call. Reward is `must_pass / must_total` in [0, 1]. If
`must_total == 0`, reward defaults to `0.0` and an `error` is set on the
verify response.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import orjson
from pydantic import BaseModel

from nemo_gym.server_utils import get_response_json, request


_JUDGE_SYSTEM_PROMPT = (
    "You grade whether a model's answer to a code-investigation problem "
    "satisfies a single must-have criterion. Read the problem, the model's "
    "answer, and the criterion. Reply with exactly one word on its own line: "
    "YES if the answer satisfies the criterion, NO otherwise. Do not add any "
    "explanation."
)


class JudgeItem(BaseModel):
    category: str
    description: str


class JudgeResult(BaseModel):
    must_pass: int
    must_total: int
    reward: float
    per_item: List[Dict[str, Any]]
    error: Optional[str] = None


_MUST_HAVE_VALUES = {"must_have", "must-have", "must have", "must"}


def extract_must_have_items(rubric: Optional[Dict[str, Any]]) -> List[JudgeItem]:
    """Pull the `judge` channel's must-have items out of a VCQA rubric.

    Tolerant of two schemas:

    - **`appliedcompute/vcqa-v1` schema**: each item has `id`, `title` (the
      rubric statement), `importance` (`"must have"` / `"good to have"`,
      with a space), plus optional `evidence_required` (list of strings the
      answer must cite).
    - **Legacy schema** (still accepted for backward compatibility):
      `{category: "must_have", description: "..."}`.

    Items missing both `title` and `description` are dropped.
    """
    if not rubric:
        return []
    raw_items = rubric.get("judge") if isinstance(rubric, dict) else None
    if not raw_items:
        return []

    items: List[JudgeItem] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue

        importance = str(raw.get("importance", "") or raw.get("category", "")).strip().lower()
        if importance not in _MUST_HAVE_VALUES:
            continue

        statement = str(raw.get("title", "") or raw.get("description", "")).strip()
        if not statement:
            continue

        evidence = raw.get("evidence_required")
        if isinstance(evidence, list) and evidence:
            evidence_lines = "\n".join(f"- {e}" for e in evidence if isinstance(e, str))
            statement = f"{statement}\n\nThe answer must cite the following evidence:\n{evidence_lines}"

        items.append(JudgeItem(category="must_have", description=statement))
    return items


def build_user_prompt(problem_statement: str, model_answer: str, criterion: str) -> str:
    return (
        "PROBLEM:\n"
        f"{problem_statement}\n\n"
        "MODEL ANSWER:\n"
        f"{model_answer}\n\n"
        "CRITERION:\n"
        f"{criterion}\n\n"
        "Reply with exactly one word: YES or NO."
    )


def parse_yes_no(text: str) -> Optional[bool]:
    """Return True for YES, False for NO, None if the response is unparseable."""
    if not text:
        return None
    cleaned = text.strip().splitlines()[0].strip().upper().rstrip(".")
    if cleaned == "YES":
        return True
    if cleaned == "NO":
        return False
    return None


async def _judge_one_item(
    *,
    base_url: str,
    api_key: str,
    model_name: str,
    problem_statement: str,
    model_answer: str,
    criterion: str,
    timeout_s: int,
    cookies: Optional[Dict[str, str]],
    max_completion_tokens: int,
    reasoning_effort: Optional[str],
) -> Dict[str, Any]:
    """Single judge call. Always returns a dict; never raises."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body: Dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(problem_statement, model_answer, criterion),
            },
        ],
        # gpt-5 / o-series reasoning models reject `max_tokens` and ignore
        # `temperature`; `max_completion_tokens` is the cross-model name.
        "max_completion_tokens": max_completion_tokens,
    }
    # Reasoning models hide their thinking inside the completion-tokens
    # budget; without an explicit signal to stop reasoning early, they can
    # burn the whole budget before producing a single visible character.
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort

    try:
        response = await asyncio.wait_for(
            request(
                method="POST",
                url=url,
                headers=headers,
                data=orjson.dumps(body),
                cookies=cookies or {},
            ),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        return {"criterion": criterion, "passed": False, "error": "judge timed out", "raw": None}
    except Exception as e:
        return {
            "criterion": criterion,
            "passed": False,
            "error": f"judge request raised {type(e).__name__}: {e}",
            "raw": None,
        }

    if not response.ok:
        body_text = (await response.content.read()).decode(errors="replace")
        return {
            "criterion": criterion,
            "passed": False,
            "error": f"judge returned HTTP {response.status}: {body_text[:512]}",
            "raw": None,
        }

    try:
        payload = await get_response_json(response)
        text = payload["choices"][0]["message"]["content"]
    except Exception as e:
        return {
            "criterion": criterion,
            "passed": False,
            "error": f"judge response unparseable ({type(e).__name__}: {e})",
            "raw": None,
        }

    parsed = parse_yes_no(text)
    if parsed is None:
        return {
            "criterion": criterion,
            "passed": False,
            "error": f"judge reply not YES/NO: {text!r}",
            "raw": text,
        }
    return {"criterion": criterion, "passed": parsed, "error": None, "raw": text}


async def grade(
    *,
    rubric: Optional[Dict[str, Any]],
    problem_statement: str,
    model_answer: str,
    base_url: str,
    api_key: str,
    model_name: str,
    timeout_s: int = 60,
    cookies: Optional[Dict[str, str]] = None,
    max_completion_tokens: int = 2048,
    reasoning_effort: Optional[str] = None,
) -> JudgeResult:
    """Run the judge against the must-have items in `rubric`."""
    must_have = extract_must_have_items(rubric)
    if not must_have:
        return JudgeResult(
            must_pass=0,
            must_total=0,
            reward=0.0,
            per_item=[],
            error="no must_have rubric items",
        )

    if not model_answer.strip():
        per_item = [
            {"criterion": item.description, "passed": False, "error": "empty answer", "raw": None}
            for item in must_have
        ]
        return JudgeResult(
            must_pass=0,
            must_total=len(must_have),
            reward=0.0,
            per_item=per_item,
            error="empty model answer",
        )

    coros = [
        _judge_one_item(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            problem_statement=problem_statement,
            model_answer=model_answer,
            criterion=item.description,
            timeout_s=timeout_s,
            cookies=cookies,
            max_completion_tokens=max_completion_tokens,
            reasoning_effort=reasoning_effort,
        )
        for item in must_have
    ]
    per_item = await asyncio.gather(*coros)

    must_pass = sum(1 for r in per_item if r.get("passed"))
    must_total = len(per_item)
    reward = must_pass / must_total if must_total else 0.0
    return JudgeResult(
        must_pass=must_pass,
        must_total=must_total,
        reward=reward,
        per_item=per_item,
        error=None,
    )
