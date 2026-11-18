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
from typing import cast

import anyio
import pytest

from nemo_gym.sandbox import stop_sandbox
from nemo_gym.sandbox.api import AsyncSandbox


class _Sandbox:
    """A sandbox whose stop has to await before it finishes, so a stop the caller cancels never finishes."""

    def __init__(self, *, stop_takes_s: float = 0.01, error: Exception | None = None) -> None:
        self.stop_takes_s = stop_takes_s
        self.error = error
        self.stopped = False

    async def stop(self) -> None:
        await anyio.sleep(self.stop_takes_s)
        if self.error is not None:
            raise self.error
        self.stopped = True


def _as_sandbox(sandbox: _Sandbox) -> AsyncSandbox:
    return cast(AsyncSandbox, sandbox)


async def test_a_stop_finishes_inside_a_cancelled_request() -> None:
    # A client disconnect cancels the request's cancel scope, and anyio then cancels every later await in it.
    sandbox = _Sandbox()
    with anyio.fail_after(5):
        with anyio.CancelScope() as request_scope:
            request_scope.cancel()
            await stop_sandbox(_as_sandbox(sandbox), timeout_s=5)
            await anyio.sleep(0)
            pytest.fail("the request's cancellation did not reach the first await after the stop")

    assert sandbox.stopped
    assert request_scope.cancelled_caught


async def test_a_stop_that_runs_past_its_time_limit_raises_timeout_error() -> None:
    # Finite and short, so a stop that is not bounded finishes late and fails the test instead of hanging it.
    # No outer deadline: the shield would hold it off.
    sandbox = _Sandbox(stop_takes_s=2)
    with pytest.raises(TimeoutError):
        await stop_sandbox(_as_sandbox(sandbox), timeout_s=0.05)

    assert not sandbox.stopped


async def test_a_stop_that_fails_raises_its_error() -> None:
    sandbox = _Sandbox(error=RuntimeError("provider refused the stop"))
    with pytest.raises(RuntimeError, match="provider refused the stop"):
        await stop_sandbox(_as_sandbox(sandbox))


async def test_no_time_limit_lets_the_stop_finish() -> None:
    sandbox = _Sandbox(stop_takes_s=0.05)

    await stop_sandbox(_as_sandbox(sandbox), timeout_s=None)

    assert sandbox.stopped
