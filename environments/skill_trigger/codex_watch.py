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
"""Stream complete Codex rollout records to the host from a root-owned process.

Run before prompt release. SIGTERM performs a final drain after agent work has
stopped. Truncation, replacement, modification, symlinks, and partial final
records are errors rather than silently incomplete evidence.
"""

import json
import os
import signal
import sys
import time
from pathlib import Path


class Rollouts:
    """Track immutable prefixes while Codex appends its own records."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.seen: dict[str, tuple[int, bytes, int, int]] = {}

    def scan(self, *, final: bool = False) -> list[dict[str, object]]:
        emitted = []
        present = set()
        if self.root.is_symlink():
            raise ValueError("Codex rollout root is a symlink")
        for path in sorted(self.root.rglob("*.jsonl")):
            if path.is_symlink() or any(
                (self.root / part).is_symlink() for part in path.relative_to(self.root).parents
            ):
                raise ValueError("Codex rollout path contains a symlink")
            name = path.relative_to(self.root).as_posix()
            present.add(name)
            inode, before, offset, sequence = self.seen.get(name, (path.stat().st_ino, b"", 0, 0))
            raw = path.read_bytes()
            if path.stat().st_ino != inode or not raw.startswith(before):
                raise ValueError("Codex rollout was replaced, truncated, or modified")
            while (end := raw.find(b"\n", offset)) >= 0:
                event = json.loads(raw[offset:end])
                if not isinstance(event, dict):
                    raise ValueError("Non-object Codex rollout record")
                emitted.append({"path": name, "sequence": sequence, "event": event})
                offset = end + 1
                sequence += 1
            if final and offset != len(raw):
                raise ValueError("Codex rollout ended with an incomplete record")
            self.seen[name] = (inode, raw, offset, sequence)
        if set(self.seen) != present:
            raise ValueError("A captured Codex rollout disappeared")
        return emitted


def main(root: Path) -> None:
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    watcher = Rollouts(root)
    print(json.dumps({"ready": True, "pid": os.getpid()}), flush=True)
    while not stopped:
        for event in watcher.scan():
            print(json.dumps(event), flush=True)
        time.sleep(0.02)
    for event in watcher.scan(final=True):
        print(json.dumps(event), flush=True)
    print(json.dumps({"rollouts_stopped": True}), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
