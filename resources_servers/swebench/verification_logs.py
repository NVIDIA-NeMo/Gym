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
"""Verification logs for the SWE-style resources servers.

Each attempt writes its logs into a temporary folder on the node, which is removed when the attempt ends.
So a verify writes nothing into the Gym checkout or onto shared storage while it runs.
A server that keeps its logs copies the folder once into the run's results after the attempt.
"""

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TypeVar

from nemo_gym.core import async_request_temp_dir, save_dir


T = TypeVar("T")


async def run_keeping_logs(
    run: Callable[[Path], Awaitable[T]], *, keep: bool, server_name: str, instance_id: str, attempt: int
) -> tuple[T, Path | None]:
    """Run one verification attempt with a temporary log folder, and keep a copy of the folder when ``keep`` is set.

    ``run`` receives the log folder.
    The copy is made after the attempt, even when it fails.
    The second value is the kept copy, or ``None`` when logs are not kept or the copy failed.
    """
    kept = None
    async with async_request_temp_dir("verification") as logs:
        try:
            result = await run(logs)
        finally:
            if keep:
                kept = await save_dir(
                    logs, server_name=server_name, name="verification", label=f"{instance_id}-attempt{attempt}"
                )
    return result, kept
