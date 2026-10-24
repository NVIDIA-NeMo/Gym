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
"""Root-owned Linux file observer; emit events for the host collector to timestamp.

Run after profile construction and catalog staging, before prompt release.
The evaluated agent runs as a different, unprivileged user. The host captures
this process's stdout separately from all agent output.
"""

import ctypes
import json
import os
import select
import signal
import struct
import sys


ACCESS = 0x00000001
DELETE_SELF = 0x00000400
MOVE_SELF = 0x00000800
QUEUE_OVERFLOW = 0x00004000
IGNORED = 0x00008000


def observe(paths: dict[str, str]) -> None:
    """Watch exact instruction files, failing if observation becomes incomplete."""
    if sys.platform != "linux" or os.geteuid() != 0:
        raise RuntimeError("Independent observation requires root inside the Linux trial container")
    if not paths or len(set(paths.values())) != len(paths):
        raise ValueError("Expected a nonempty catalog with unique instruction paths")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.inotify_init1.argtypes = [ctypes.c_int]
    libc.inotify_init1.restype = ctypes.c_int
    libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
    libc.inotify_add_watch.restype = ctypes.c_int
    fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "Cannot start inotify")
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    previous = signal.signal(signal.SIGTERM, stop)
    watched = {}
    try:
        for skill, path in paths.items():
            if not isinstance(skill, str) or not isinstance(path, str) or not os.path.isfile(path):
                raise ValueError("Missing instruction file")
            wd = libc.inotify_add_watch(fd, path.encode(), ACCESS | DELETE_SELF | MOVE_SELF)
            if wd < 0:
                raise OSError(ctypes.get_errno(), f"Cannot watch {skill}")
            if wd in watched:
                raise ValueError("Instruction files share an inode")
            watched[wd] = skill
        print(json.dumps({"ready": True, "pid": os.getpid(), "skills": sorted(paths)}), flush=True)
        while True:
            if select.select([fd], [], [], 0 if stopping else 0.1)[0]:
                data = os.read(fd, 65536)
                offset = 0
                while offset < len(data):
                    wd, mask, _, length = struct.unpack_from("iIII", data, offset)
                    offset += 16 + length
                    if mask & (QUEUE_OVERFLOW | IGNORED | DELETE_SELF | MOVE_SELF) or wd not in watched:
                        raise RuntimeError("Instruction observation lost events or a watched file")
                    if mask & ACCESS:
                        print(json.dumps({"event": "IN_ACCESS", "skill": watched[wd]}), flush=True)
            elif stopping:
                break
        print(json.dumps({"observer_stopped": True}), flush=True)
    finally:
        signal.signal(signal.SIGTERM, previous)
        os.close(fd)


if __name__ == "__main__":
    observe(json.loads(sys.argv[1]))
