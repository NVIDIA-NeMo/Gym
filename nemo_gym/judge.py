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
response) is a distinct outcome, not a wrong answer. Rather than scoring it 0.0
in the main results — which would silently deflate accuracy — judge-based
resources servers route the sample to the failures sidecar so the headline
metric is computed over judged samples only.

- ``run_judge`` — await a judge call, recording any error verbatim instead of
  raising or silently defaulting to a low score.
- ``judge_failure`` — stamp a verify-response as a judge failure: reward 0.0 and
  the routing keys that send the row to ``<output>_failures.jsonl`` (via
  ``_ng_failure_class``). The failure is transient — no ``_ng_failure_terminal``
  flag — so resume re-dispatches it (up to ``NEMO_GYM_MAX_ROLLOUT_ATTEMPTS``).
  The verify-response must already carry the model's final output so a later
  judge-only replay can skip regeneration.
"""

from typing import Any, Awaitable, Optional

from pydantic import BaseModel


# Routing keys read by nemo_gym.rollout_collection. ``_ng_failure_class`` present
# (and no ``_ng_no_persist``) routes the row to the failures sidecar. Absent
# ``_ng_failure_terminal`` keeps the failure retryable on resume.
NG_FAILURE_CLASS_KEY = "_ng_failure_class"
NG_JUDGE_FAILED_KEY = "_ng_failure_judge_failed"
NG_JUDGE_ERROR_KEY = "_ng_failure_judge_error"
JUDGE_FAILURE_CLASS = "judge_failed"


async def run_judge(coro: Awaitable[Any]) -> tuple[Optional[Any], Optional[str]]:
    """Await a judge call. Return ``(result, None)`` on success or
    ``(None, raw_error_text)`` on any failure. Never raises.

    The error is recorded verbatim — no provider-specific classification.
    """
    try:
        return await coro, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def judge_failure(response: BaseModel, error: Optional[str]) -> BaseModel:
    """Mark ``response`` as a judge failure and return it for ``verify`` to yield.

    Sets ``reward=0.0`` and the sidecar-routing keys ``_ng_failure_class``,
    ``_ng_failure_judge_failed`` and ``_ng_failure_judge_error``. The row is
    routed to ``<output>_failures.jsonl`` and excluded from the headline metric;
    it is retried on resume. ``response`` should already carry the model's final
    output (e.g. the verify request's ``response``).

    Requires ``model_config = ConfigDict(extra="allow")`` on the response model
    so the underscore-prefixed routing keys survive serialization.
    """
    data = response.model_dump()
    data["reward"] = 0.0
    data[NG_FAILURE_CLASS_KEY] = JUDGE_FAILURE_CLASS
    data[NG_JUDGE_FAILED_KEY] = True
    data[NG_JUDGE_ERROR_KEY] = error or "empty judge response"
    return type(response).model_validate(data)
