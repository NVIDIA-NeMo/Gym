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
"""Deadline helpers that do not depend on which clock the event loop uses."""

import asyncio
from time import monotonic


def monotonic_timeout(deadline: float | None) -> asyncio.Timeout:
    """Return an ``asyncio`` timeout that expires at a ``time.monotonic()`` deadline.

    ``asyncio.timeout_at`` compares its deadline with ``loop.time()``, which is not always ``time.monotonic()``:
    uvloop's clock also counts time the machine slept, so on macOS it can run days ahead and a monotonic deadline
    passed to ``timeout_at`` expires at once. This converts the deadline to a remaining duration instead.
    ``None`` never expires.
    """
    return asyncio.timeout(None if deadline is None else deadline - monotonic())
