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
``await run_judge(<judge call>)``; ``judge_failsafe`` wraps every verify endpoint
so a JudgeError becomes a row tagged ``_ng_failure_class="judge_failed"``, which
rollout_collection routes to ``<output>_failures.jsonl`` — excluded from the
metric, retryable on resume.

Boundary: a failed *call* (transport/timeout/auth/HTTP) → JudgeError → sidecar; a
*received-but-unparseable* response is a legitimate wrong answer (let the parser
score it, don't raise). Empty output is a per-benchmark call, so servers differ.
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
    functools.wraps keeps verify's signature so FastAPI injects the same params
    (``body``, and ``request`` for servers that take it); ``*args, **kwargs`` pass
    them straight through."""

    @functools.wraps(verify_fn)
    async def wrapper(*args, **kwargs):
        try:
            return await verify_fn(*args, **kwargs)
        except JudgeError as e:
            body = kwargs.get("body") or next(
                (a for a in (*kwargs.values(), *args) if hasattr(a, "model_dump") and hasattr(a, "response")),
                None,
            )
            if body is None:  # verify always has a request body; guard against an opaque 500
                raise RuntimeError("judge_failsafe: could not locate the verify request body") from e
            data = body.model_dump() | {
                "reward": 0.0,
                "_ng_failure_class": "judge_failed",
                "_ng_failure_judge_error": str(e),
            }
            return JSONResponse(content=jsonable_encoder(data))

    return wrapper
