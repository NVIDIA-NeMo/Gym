# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM-judge-based scoring for VCQA rollouts.

A VCQA judge rubric is a list of items (typically `must_have` and
`good_to_have`). For each must-have item the judge is asked "does the
model's answer satisfy this criterion? Reply YES or NO." in a single chat
completion call. Reward is `must_pass / must_total` in [0, 1]. If
`must_total == 0`, reward defaults to `0.0` and an `error` is set on the
verify response.

Transient judge failures (timeouts, transport errors, HTTP 408/409/425/429/5xx,
replies that are not YES/NO) are retried with exponential backoff. An item
that still fails after the last attempt counts as NO and is reported in
`JudgeResult.judge_errors`, so the caller can mark the sample as not
validly measured instead of training on a judge outage.
"""

from __future__ import annotations

import asyncio
import re
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

import orjson
from pydantic import BaseModel

from nemo_gym.client import get_response_json, request


_JUDGE_SYSTEM_PROMPT = (
    "You grade whether a model's answer to a code-investigation problem "
    "satisfies a single must-have criterion. Read the problem, the model's "
    "answer, and the criterion. Reply with exactly one word on its own line: "
    "YES if the answer satisfies the criterion, NO otherwise. Do not add any "
    "explanation."
)

# Retried besides every 5xx.
_RETRYABLE_HTTP_STATUSES = frozenset({408, 409, 425, 429})
_THINK_BLOCK = re.compile(r"^\s*<think>.*?</think>", re.DOTALL)
_LEADING_VERDICT = re.compile(r"^(YES|NO)\b")


class JudgeItem(BaseModel):
    category: str
    description: str


class JudgeResult(BaseModel):
    must_pass: int
    must_total: int
    reward: float
    per_item: List[Dict[str, Any]]
    error: Optional[str] = None
    # Items whose last judge attempt still failed (timeout, HTTP error, unparseable reply).
    judge_errors: int = 0


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
    """Return True for YES, False for NO, None if the response is unparseable.

    The verdict is read from the first non-empty line, after dropping a leading
    `<think>...</think>` block (judges served without a reasoning parser) and
    markdown emphasis. A line that starts with the verdict word ("YES." or
    "NO - the answer ...") counts; anything else is unparseable.
    """
    if not text:
        return None
    text = _THINK_BLOCK.sub("", text)
    for line in text.splitlines():
        cleaned = line.strip().strip("*_`'\"").strip().upper()
        if not cleaned:
            continue
        match = _LEADING_VERDICT.match(cleaned)
        if match is None:
            return None
        return match.group(1) == "YES"
    return None


async def _judge_attempt(
    *, url: str, headers: Dict[str, str], body: Dict[str, Any], timeout_s: int, cookies: Optional[Dict[str, str]]
) -> Dict[str, Any]:
    """One judge call. Returns passed/error/raw plus whether a retry may help; never raises.

    `timeout_s` bounds the whole exchange, reading either reply body included, so a reply that stalls cannot keep
    holding a slot of the shared judge semaphore.
    """
    try:
        return await asyncio.wait_for(
            _judge_exchange(url=url, headers=headers, body=body, cookies=cookies), timeout=timeout_s
        )
    except asyncio.TimeoutError:
        return {"passed": False, "error": "judge timed out", "raw": None, "retryable": True}
    except Exception as e:
        return {
            "passed": False,
            "error": f"judge request raised {type(e).__name__}: {e}",
            "raw": None,
            "retryable": True,
        }


async def _judge_exchange(
    *, url: str, headers: Dict[str, str], body: Dict[str, Any], cookies: Optional[Dict[str, str]]
) -> Dict[str, Any]:
    response = await request(method="POST", url=url, headers=headers, data=orjson.dumps(body), cookies=cookies or {})
    try:
        outcome = await _read_judge_reply(response)
    except BaseException:
        # Cancelled at the deadline or failed mid-body: drop the connection rather than return it to the pool.
        response.close()
        raise
    response.release()
    return outcome


async def _read_judge_reply(response: Any) -> Dict[str, Any]:
    if not response.ok:
        body_text = (await response.content.read()).decode(errors="replace")
        return {
            "passed": False,
            "error": f"judge returned HTTP {response.status}: {body_text[:512]}",
            "raw": None,
            "retryable": response.status in _RETRYABLE_HTTP_STATUSES or 500 <= response.status < 600,
        }

    try:
        payload = await get_response_json(response)
        text = payload["choices"][0]["message"]["content"]
    except Exception as e:
        return {
            "passed": False,
            "error": f"judge response unparseable ({type(e).__name__}: {e})",
            "raw": None,
            "retryable": True,
        }

    parsed = parse_yes_no(text or "")
    if parsed is None:
        return {"passed": False, "error": f"judge reply not YES/NO: {text!r}", "raw": text, "retryable": True}
    return {"passed": parsed, "error": None, "raw": text, "retryable": False}


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
    extra_body: Optional[Dict[str, Any]] = None,
    max_attempts: int = 1,
    retry_backoff_s: float = 2.0,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> Dict[str, Any]:
    """Judge one criterion, retrying transient failures. Always returns a dict; never raises."""
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
    # Endpoint-specific knobs, e.g. {"chat_template_kwargs": {"enable_thinking": false}}.
    if extra_body:
        body.update(extra_body)

    attempts = max(1, int(max_attempts))
    outcome: Dict[str, Any] = {}
    attempt = 0
    for attempt in range(1, attempts + 1):
        async with semaphore if semaphore is not None else nullcontext():
            outcome = await _judge_attempt(url=url, headers=headers, body=body, timeout_s=timeout_s, cookies=cookies)
        if not outcome["retryable"] or attempt == attempts:
            break
        await asyncio.sleep(retry_backoff_s * (2 ** (attempt - 1)))

    return {
        "criterion": criterion,
        "passed": outcome["passed"],
        "error": outcome["error"],
        "raw": outcome["raw"],
        "attempts": attempt,
    }


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
    extra_body: Optional[Dict[str, Any]] = None,
    max_attempts: int = 1,
    retry_backoff_s: float = 2.0,
    semaphore: Optional[asyncio.Semaphore] = None,
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
            extra_body=extra_body,
            max_attempts=max_attempts,
            retry_backoff_s=retry_backoff_s,
            semaphore=semaphore,
        )
        for item in must_have
    ]
    per_item = await asyncio.gather(*coros)

    must_pass = sum(1 for r in per_item if r.get("passed"))
    must_total = len(per_item)
    judge_errors = sum(1 for r in per_item if r.get("error"))
    reward = must_pass / must_total if must_total else 0.0
    return JudgeResult(
        must_pass=must_pass,
        must_total=must_total,
        reward=reward,
        per_item=per_item,
        error=f"{judge_errors} of {must_total} judge calls failed" if judge_errors else None,
        judge_errors=judge_errors,
    )
