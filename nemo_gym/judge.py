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

A failed judge call is a distinct outcome, not a wrong answer. Resources servers
``await run_judge(<judge call>)`` (and ``raise JudgeError`` for an empty/unusable
response); ``judge_failsafe`` wraps every verify endpoint so a JudgeError becomes
a row tagged ``_ng_failure_class="judge_failed"``, which rollout_collection routes
to ``<output>_failures.jsonl`` — excluded from the metric, retryable on resume.
"""

import functools
from typing import Any, Awaitable, Callable

from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse


class JudgeError(Exception):
    """A judge call failed; judge_failsafe routes the row to the failures sidecar."""


async def run_judge(coro: Awaitable[Any]) -> Any:
    """Await a judge call; re-raise any exception as JudgeError (recorded verbatim)."""
    try:
        return await coro
    except JudgeError:
        raise
    except Exception as e:
        raise JudgeError(f"{type(e).__name__}: {e}") from e


def judge_failsafe(verify_fn: Callable) -> Callable:
    """Wrap verify() so a JudgeError returns a sidecar-routed row (reward 0.0, the
    routing keys, the request's ``response`` carried) instead of propagating.
    functools.wraps keeps the request type so FastAPI still parses the body."""

    @functools.wraps(verify_fn)
    async def wrapper(body):
        try:
            return await verify_fn(body)
        except JudgeError as e:
            data = body.model_dump() | {
                "reward": 0.0,
                "_ng_failure_class": "judge_failed",
                "_ng_failure_judge_error": str(e),
            }
            return JSONResponse(content=jsonable_encoder(data))

    return wrapper
