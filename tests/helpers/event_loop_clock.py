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

"""An event loop whose clock runs ahead of ``time.monotonic()``, as uvloop's does on macOS."""

import asyncio
from collections.abc import Coroutine
from typing import Any


CLOCK_LEAD_S = 1_000_000.0


class ClockAheadEventLoop(asyncio.SelectorEventLoop):
    """A selector event loop whose ``time()`` is ``CLOCK_LEAD_S`` seconds ahead of ``time.monotonic()``."""

    def time(self) -> float:
        return super().time() + CLOCK_LEAD_S


def run_on_clock_ahead_loop[T](coroutine: Coroutine[Any, Any, T]) -> T:
    """Run ``coroutine`` on a fresh ``ClockAheadEventLoop``, then clean up and close the loop as ``asyncio.run`` does."""
    with asyncio.Runner(loop_factory=ClockAheadEventLoop) as runner:
        return runner.run(coroutine)
