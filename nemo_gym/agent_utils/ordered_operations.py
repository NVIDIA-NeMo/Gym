# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process exactly-once ordered operations for activations and simulator steps."""

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from fastapi import HTTPException
from pydantic import BaseModel


def request_fingerprint(request: BaseModel) -> str:
    """Hash the complete canonical wire request, including explicitly supplied controls."""
    encoded = json.dumps(request.model_dump(mode="json", exclude_unset=True), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass
class _Operation[Result: BaseModel]:
    fingerprint: str
    task: asyncio.Task[Result]


class OrderedOperationLedger[Result: BaseModel]:
    """Join/replay identical calls, reject conflicts, and fence work before closing.

    Owners must serialize seed/close binding separately and retain this ledger for
    the session lifetime. Disconnecting a waiter leaves execution alive. Failed
    operations are retained and replay their failure; a new physical episode is
    needed to retry with different inputs. This is process-local, not crash recovery.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._operations: list[_Operation[Result]] = []
        self._closing = False

    @property
    def completed(self) -> list[Result]:
        """Return isolated copies of successful results in operation order."""
        return [
            operation.task.result().model_copy(deep=True)
            for operation in self._operations
            if operation.task.done() and not operation.task.cancelled() and operation.task.exception() is None
        ]

    async def execute(self, *, index: int, request: BaseModel, operation: Callable[[], Awaitable[Result]]) -> Result:
        """Run the next operation once or join an exact retry, without holding the lock while running."""
        fingerprint = request_fingerprint(request)
        async with self._lock:
            if self._closing:
                raise HTTPException(409, "Session is closing")
            if index < 0 or index > len(self._operations):
                raise HTTPException(409, "Operation is out of order")
            if index < len(self._operations):
                entry = self._operations[index]
                if fingerprint != entry.fingerprint:
                    raise HTTPException(409, "Operation ID is already bound to different input")
            else:
                if self._operations:
                    previous = self._operations[-1].task
                    if not previous.done():
                        raise HTTPException(409, "Another operation is in flight")
                    if previous.cancelled() or previous.exception() is not None:
                        raise HTTPException(409, "Previous operation failed; close this session")
                # Consume failures even if every HTTP waiter disconnects; the task still retains
                # its exception for an identical retry and for the next-operation guard.
                task = asyncio.create_task(operation())
                task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
                entry = _Operation(fingerprint=fingerprint, task=task)
                self._operations.append(entry)
        return (await asyncio.shield(entry.task)).model_copy(deep=True)

    async def close(self, *, timeout: float) -> None:
        """Fence new work, cancel/await the active operation, then allow owner cleanup.

        A timeout retains the active task and the fence, so cleanup cannot be
        declared successful while an activation might still launch remotely.
        """
        async with self._lock:
            self._closing = True
            active = self._operations[-1].task if self._operations else None
            if active is not None and not active.done():
                active.cancel()
        if active is None:
            return
        try:
            async with asyncio.timeout(timeout):
                await asyncio.shield(active)
        except asyncio.CancelledError:
            if not active.cancelled():
                raise
        except Exception:
            if not active.done():
                raise
            # The adapter cleanup hook separately confirms remote process cleanup.
