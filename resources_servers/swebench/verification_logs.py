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
"""Where the SWE-style resources servers keep verification logs.

Logs are off unless a server's ``verification_log_dir`` names a folder.
These servers used to write them next to their own source code.
A training run that mounts the Gym checkout from a shared filesystem then created a folder and a file there on every verification.
When logs are wanted, point ``verification_log_dir`` at node-local storage such as ``/tmp``.
"""

from pathlib import Path
from urllib.parse import quote
from uuid import uuid4


def verification_log_path(root: Path | None, instance_id: str) -> Path | None:
    """Return a new folder under ``root`` for one verification of ``instance_id``, or None when logs are off.

    The instance id is percent-encoded, so a slash in it cannot leave ``root``.
    The random last component keeps repeated verifications of one instance from overwriting each other.
    Nothing is created here; the verifier creates the folder only when it writes a log.
    """
    if root is None:
        return None
    return root / f"instance-{quote(instance_id, safe='')}" / uuid4().hex
