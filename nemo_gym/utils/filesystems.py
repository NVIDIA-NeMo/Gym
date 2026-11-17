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
"""Which filesystem a folder is on, for settings that belong on storage private to the host and backed by disk.

Each process checks the folders it writes into, on its own node, because the mount table differs per node.
"""

import logging
import os
import re
from pathlib import Path


logger = logging.getLogger(__name__)

# Filesystems shared by many hosts, whose metadata servers every file create and delete loads.
_SHARED_FILESYSTEM_TYPES = frozenset(
    {"lustre", "nfs", "nfs4", "gpfs", "beegfs", "ceph", "cifs", "smb3", "wekafs", "glusterfs", "panfs", "dvs"}
)
# Filesystems held in memory: what is written there counts against the job's memory, as /tmp can be in a container.
_MEMORY_FILESYSTEM_TYPES = frozenset({"tmpfs", "ramfs"})
_MOUNTINFO = Path("/proc/self/mountinfo")
_OCTAL_ESCAPE = re.compile(r"\\([0-7]{3})")


def warn_if_unsuitable(
    setting: str,
    path: str | Path | None,
    *,
    advice: str = "Point it at storage private to the host, such as the node's local disk.",
    mountinfo: Path = _MOUNTINFO,
) -> None:
    """Log a warning when ``setting`` points at a shared or memory-backed filesystem.

    The setting still applies: the warning says what holding many small files there costs.
    """
    if not path:
        return
    found = _filesystem_type(path, mountinfo=mountinfo)
    if found in _SHARED_FILESYSTEM_TYPES:
        logger.warning(
            "%s is %s, on a %s filesystem shared by many hosts. It holds many small files, and each one loads "
            "the shared metadata servers that every job on the cluster depends on. %s",
            setting,
            path,
            found,
            advice,
        )
    elif found in _MEMORY_FILESYSTEM_TYPES:
        logger.warning(
            "%s is %s, on a %s filesystem held in memory. What it holds counts against the job's memory. "
            "Point it at the node's local disk.",
            setting,
            path,
            found,
        )


def _filesystem_type(path: str | Path, *, mountinfo: Path = _MOUNTINFO) -> str | None:
    """The type of the filesystem ``path`` is on, from the mount table, or ``None`` where there is none to read.

    ``path`` need not exist yet: its location is resolved as far as it goes, and the deepest mount above it wins.
    """
    try:
        lines = mountinfo.read_text().splitlines()
    except OSError:
        return None
    target = os.path.realpath(path)
    best_mount, best_type = "", None
    for line in lines:
        fields, _, after = line.partition(" - ")
        parts = fields.split()
        if len(parts) < 5 or not after:
            continue
        # The kernel escapes spaces and other special characters in mount points as octal.
        mount = _OCTAL_ESCAPE.sub(lambda match: chr(int(match.group(1), 8)), parts[4])
        inside = target == mount or target.startswith(mount.rstrip("/") + "/")
        if inside and len(mount) >= len(best_mount):
            best_mount, best_type = mount, after.split()[0]
    return best_type
