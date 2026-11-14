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
"""``sandbox_config.files`` and ``setup_command`` for the SWE servers' agent and evaluation sandboxes.

They add toolchain setup an image lacks, such as a package mirror or a headless browser wrapper.
"""

from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path

from nemo_gym.sandbox import AsyncSandbox


# Relative sources resolve against the Gym repository root, so configs can name files shipped with Gym.
_GYM_ROOT = Path(__file__).resolve().parents[2]


def read_sandbox_files(sources: Mapping[str, str]) -> dict[str, str]:
    """Read ``sandbox_config.files`` into ``SandboxSpec.files``; raises ``FileNotFoundError`` for a missing source."""
    files: dict[str, str] = {}
    for target, source in sources.items():
        path = Path(source)
        if not path.is_absolute():
            path = _GYM_ROOT / path
        files[str(target)] = path.read_text(encoding="utf-8")
    return files


def setup_command_runner(command: str) -> Callable[[AsyncSandbox], Awaitable[None]]:
    """Return a ``start_with_setup`` hook that runs ``command`` and raises unless it exits 0."""

    async def run(sandbox: AsyncSandbox) -> None:
        result = await sandbox.exec(command)
        if result.return_code != 0 or result.error_type is not None:
            how = f"exited {result.return_code}" + (f" ({result.error_type})" if result.error_type else "")
            raise RuntimeError(f"sandbox_config.setup_command {how}: {(result.stderr or '')[-2000:]}")

    return run
