# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded task locations for checkpoint waits, without locals or source text."""

import asyncio
from collections import deque
from pathlib import Path
from types import AsyncGeneratorType, CoroutineType, GeneratorType

from pydantic import JsonValue


def task_wait_diagnostics(task: asyncio.Task | None) -> dict[str, JsonValue]:
    """Describe a task's await chain without evaluating boundaries or task results.

    Only code locations are read: no frame locals, source lines, exception
    messages, task names, or coroutine representations may enter this report.
    """
    if task is None:
        return {"present": False}
    # ASGI middleware can be many frames deep. Keep the innermost locations
    # so a long wrapper chain does not hide the actual agent/HTTP/event wait.
    chain: deque[JsonValue] = deque(maxlen=12)
    current: object | None = task.get_coro()
    seen: set[int] = set()
    for _ in range(64):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        if isinstance(current, CoroutineType):
            frame, following = current.cr_frame, current.cr_await
        elif isinstance(current, GeneratorType):
            frame, following = current.gi_frame, current.gi_yieldfrom
        elif isinstance(current, AsyncGeneratorType):
            frame, following = current.ag_frame, current.ag_await
        else:
            break
        if frame is not None:
            chain.append(
                {
                    "file": Path(frame.f_code.co_filename).name[:100],
                    "function": frame.f_code.co_qualname[:160],
                    "line": frame.f_lineno,
                }
            )
        current = following
    return {
        "present": True,
        "done": task.done(),
        "cancelled": task.cancelled(),
        "cancelling": task.cancelling(),
        "await_chain": list(chain),
        "await_leaf": None if current is None else type(current).__name__[:100],
        "chain_truncated": len(seen) > 12,
        "scan_truncated": len(seen) == 64 and current is not None,
    }
