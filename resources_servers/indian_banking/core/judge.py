# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Bounded LLM judge for the task's nl_assertions (the NL_ASSERTION reward component).

Scores confirmation/consent/tracking behaviour the deterministic components
can't see. Kept bounded against reward hacking: only enters dense via
W_JUDGE*SHAPE (never strict, so it can't rescue a failing trajectory), sees
only the customer-facing transcript (no tool calls, DB state, or gold actions),
and fails open (judge term left out) rather than silently rescoring on error.

The prompt/JSON shape follows tau2-bench's ``evaluator_nl_assertions.py`` so
training and evaluation grade against the same rubric. This module builds the
request and parses the verdict; the transport (a NeMo Gym model server) lives
in app.py.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import OrderedDict
from typing import NamedTuple, Optional


# Bump when the prompt changes so cached verdicts cannot cross versions.
PROMPT_VERSION = "v1"

# The system prompt below paraphrases the NL-assertion evaluator prompt of
# Sierra Research's tau2-bench (https://github.com/sierra-research/tau2-bench,
# ``evaluator_nl_assertions.py``, Copyright (c) 2025 Sierra Research, MIT License).
SYSTEM_PROMPT = """\
TASK
- You will be given a list of expected outcomes and a conversation collected during a test run.
- The conversation is between a bank's virtual agent and a customer.
- Judge whether the AGENT satisfies each expected outcome. Grade each one individually.
- Judge only from the conversation shown. If the conversation does not show an outcome
  being satisfied, it is not met. Do not assume unstated actions.

FORMAT
- Reply with a single JSON object, no prose outside it:
{
  "results": [
    {"expectedOutcome": "<repeat the expectation>", "reasoning": "<short>", "metExpectation": true}
  ]
}"""

# Verdict cache keyed by (judge model, prompt version, task, transcript hash);
# bounded LRU so long RL runs cannot grow it without limit.
_CACHE_MAX = 4096
_cache: "OrderedDict[str, float]" = OrderedDict()


def cache_get(key: str) -> Optional[float]:
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    return None


def cache_put(key: str, score: float) -> None:
    _cache[key] = score
    _cache.move_to_end(key)
    while len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)


def _plain(x):
    """reward_basis may arrive as a tuple/set/array; normalise to list."""
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return list(x)
    tolist = getattr(x, "tolist", None)
    return list(tolist()) if callable(tolist) else [x]


def _parse(content: Optional[str]) -> dict:
    """Parse the verdict JSON, tolerating <think> blocks and code fences."""
    raw = (content or "").strip()
    if not raw:
        raise json.JSONDecodeError("empty judge response", raw, 0)
    raw = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def _transcript(nl_history: list[dict]) -> str:
    """Customer-facing turns only. Tool traffic never reaches the judge."""
    lines = []
    for m in nl_history or []:
        role = m.get("role")
        if role in ("assistant", "user"):
            speaker = "agent" if role == "assistant" else "customer"
            lines.append(f"{speaker}: {(m.get('content') or '').strip()}")
    return "\n".join(lines)


def cache_key(task_id: str, transcript: str) -> str:
    return f"{PROMPT_VERSION}:{task_id}:{hashlib.sha256(transcript.encode()).hexdigest()}"


class JudgeRequest(NamedTuple):
    """Transport-agnostic judge call, built only by :func:`build_judge_request`."""

    system_prompt: str
    user_prompt: str
    num_assertions: int
    cache_key: str


def build_judge_request(task: dict, nl_history: list[dict]) -> tuple[Optional[JudgeRequest], Optional[float]]:
    """Gate the judge and build its request.

    Returns (None, None) if not scored (no assertions, or NL_ASSERTION not in
    reward_basis), (None, 0.0) if the transcript is empty, or (JudgeRequest, None)
    to call the model and pass the response + num_assertions to :func:`parse_verdict`.
    """
    ec = task.get("evaluation_criteria") or {}
    assertions = list(ec.get("nl_assertions") or [])
    if not assertions:
        return None, None

    # Scored only when NL_ASSERTION is in reward_basis.
    basis = set(_plain(ec.get("reward_basis")) or [])
    if "NL_ASSERTION" not in basis:
        return None, None

    transcript = _transcript(nl_history)
    if not transcript.strip():
        return None, 0.0  # nothing said: nothing can be satisfied

    user_prompt = "EXPECTED OUTCOMES:\n" + "\n".join(f"- {a}" for a in assertions) + "\n\nCONVERSATION:\n" + transcript
    key = cache_key(str(task.get("task_id")), transcript)
    return JudgeRequest(SYSTEM_PROMPT, user_prompt, len(assertions), key), None


def parse_verdict(content: Optional[str], num_assertions: int) -> float:
    """Parse a judge response into [0,1]. Raises on malformed output; the caller
    decides retry/fail-open rather than this swallowing into a fake 0.0."""
    results = _parse(content).get("results") or []
    if not results:
        raise ValueError("judge returned no results")
    met = sum(1 for r in results if r.get("metExpectation") is True)
    # Divide by assertions asked, not answered, so a truncated verdict can't inflate the score.
    return min(1.0, met / num_assertions)
