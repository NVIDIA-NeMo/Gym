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
from contextlib import suppress

import psutil


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
