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

"""Small text read/write helpers over a nemo_gym.sandbox AsyncSandbox.

These go through exec (base64 for writes) rather than the provider upload/
download API so they need no host tempfile and work identically on every
provider, surviving arbitrary content.
"""

from __future__ import annotations

import base64
import shlex
from typing import Any, Optional


async def sandbox_write_text(sandbox: Any, path: str, content: str) -> None:
    """Write a text file into the sandbox without a host tempfile (base64 over
    exec, so it works on every provider and survives arbitrary content)."""
    b64 = base64.b64encode(content.encode()).decode()
    directory = shlex.quote(path.rsplit("/", 1)[0] or "/")
    quoted = shlex.quote(path)
    result = await sandbox.exec(
        f"mkdir -p {directory} && printf %s {shlex.quote(b64)} | base64 -d > {quoted}",
        timeout_s=30,
    )
    if result.return_code != 0:
        raise RuntimeError(f"failed to write {path} into sandbox: {(result.stderr or '')[:300]}")


async def sandbox_read_text(sandbox: Any, path: str) -> Optional[str]:
    """Read a text file from the sandbox; None if it is absent."""
    result = await sandbox.exec(f"cat {shlex.quote(path)}", timeout_s=30)
    return result.stdout if result.return_code == 0 else None
