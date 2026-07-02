# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Shared LLM-as-judge failure abstraction.

A failed judge call (auth, rate limit, timeout, endpoint/HTTP error, empty
response) is a distinct outcome, not a wrong answer. Judge-based resources
servers record failures with the same field names and semantics via:

- ``JudgeFailureMixin`` — the two standard verify-response fields.
- ``run_judge`` — await a judge call, recording any error verbatim instead of
  raising or silently defaulting to a low score.
- ``judge_failure_metrics`` — failure count + a judge-success-only reward, for a
  benchmark's ``compute_metrics``.
"""

from typing import Any, Awaitable, Optional

from pydantic import BaseModel


class JudgeFailureMixin(BaseModel):
    """Standard judge-failure fields to mix into a verify-response model.

    ``judge_failed`` is a bool, so it auto-aggregates to ``mean/judge_failed``
    (the per-job and per-task failure rate). ``judge_failure_reason`` is the raw
    error text, carried per-sample into the rollout (dropped from aggregates).
    """

    judge_failed: bool = False
    judge_failure_reason: Optional[str] = None


async def run_judge(coro: Awaitable[Any]) -> tuple[Optional[Any], Optional[str]]:
    """Await a judge call. Return ``(result, None)`` on success or
    ``(None, raw_error_text)`` on any failure. Never raises.

    The error is recorded verbatim — no provider-specific classification.
    """
    try:
        return await coro, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def judge_failure_metrics(tasks: list[list[dict]]) -> dict:
    """Judge-failure metrics from verify responses grouped by task.

    ``mean/reward`` (failures counted as 0.0) and ``mean/judge_failed`` (the
    rate) are emitted automatically by the aggregator. This adds the absolute
    failure count and the judge-success-only mean reward (failures excluded from
    the denominator; ``None`` when every sample failed).
    """
    rows = [r for task in tasks for r in task]
    if not rows:
        return {}
    scored = [r for r in rows if not r.get("judge_failed")]
    return {
        "judge_failures": len(rows) - len(scored),
        "reward[judge_ok_only]": (sum(r.get("reward", 0.0) for r in scored) / len(scored)) if scored else None,
    }
