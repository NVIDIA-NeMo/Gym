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

import asyncio
import os
import signal
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, TypeVar

import psutil


T = TypeVar("T")


@dataclass(frozen=True)
class NativeCapture:
    stdout: bytes
    stderr: bytes
    returncode: int
    timed_out: bool


async def await_cleanup(task: asyncio.Task[T]) -> T:
    """Finish child reaping despite repeated caller cancellation, then propagate it."""
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    if cancelled:
        if not task.cancelled():
            task.exception()
        raise asyncio.CancelledError
    return task.result()


async def create_native_subprocess(*argv: str, **kwargs: Any) -> asyncio.subprocess.Process:
    """Keep the spawn handle through cancellation so the child can be killed and reaped."""
    creation = asyncio.create_task(asyncio.create_subprocess_exec(*argv, **kwargs))
    try:
        return await asyncio.shield(creation)
    except asyncio.CancelledError:
        try:
            await await_cleanup(creation)
        except (Exception, asyncio.CancelledError):
            pass
        if creation.done() and not creation.cancelled():
            try:
                process = creation.result()
            except Exception:
                raise asyncio.CancelledError from None
            kill_process_tree(process)
            communication = asyncio.create_task(process.communicate())
            try:
                await await_cleanup(communication)
            except (Exception, asyncio.CancelledError):
                pass
        raise


async def capture_native_subprocess(*argv: str, timeout: float, **kwargs: Any) -> NativeCapture:
    """Capture a native CLI without making pipe EOF a condition of child exit.

    A descendant may inherit stdout or stderr after the direct child dies. File
    output lets us reap that child and preserve the bytes it wrote without
    waiting for an escaped descendant to close an inherited pipe.
    """
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        proc = await create_native_subprocess(*argv, stdout=stdout, stderr=stderr, **kwargs)
        waiter = asyncio.create_task(proc.wait())
        timed_out = False
        try:
            await asyncio.wait_for(asyncio.shield(waiter), timeout)
        except asyncio.TimeoutError:
            timed_out = True
            kill_process_tree(proc)
            await await_cleanup(waiter)
        except asyncio.CancelledError:
            kill_process_tree(proc)
            try:
                await await_cleanup(waiter)
            except asyncio.CancelledError:
                pass
            raise
        # The direct CLI may exit normally while leaving same-group children.
        kill_process_tree(proc)
        stdout.seek(0)
        stderr.seek(0)
        return NativeCapture(stdout.read(), stderr.read(), proc.returncode, timed_out)


def kill_process_tree(proc: asyncio.subprocess.Process) -> None:
    """Kill an invocation's observed descendants and its process group.

    The caller must launch ``proc`` with ``start_new_session=True`` so its PID
    is also its private process group ID, then await output collection/reaping.
    Descendant discovery includes children that created separate groups; it is
    a snapshot, not containment against reparenting or concurrent forks.
    """
    descendants = []
    with suppress(psutil.NoSuchProcess):
        descendants = psutil.Process(proc.pid).children(recursive=True)
    for child in reversed(descendants):
        with suppress(psutil.NoSuchProcess):
            child.kill()
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except (AttributeError, OSError):
        # Platforms without process groups still stop the direct child.
        with suppress(ProcessLookupError):
            proc.kill()
