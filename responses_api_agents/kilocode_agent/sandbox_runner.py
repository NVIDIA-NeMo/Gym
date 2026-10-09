# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone Python 3.8+ runner; Gym stays on the agent-server host."""

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Run Kilo in the task workdir while keeping all adapter files in its session directory."""
    directory = Path(sys.argv[1])
    workdir = sys.argv[2]
    payload = json.loads((directory / "input.json").read_text())
    env = dict(os.environ)
    # Do not inherit provider credentials or configuration overrides from the image.
    for key in list(env):
        if key.startswith(("KILO_", "OPENCODE_")) or key in ("OPENAI_BASE_URL", "OPENAI_API_KEY"):
            env.pop(key)
    env.update(
        HOME=str(directory / "home"),
        XDG_DATA_HOME=str(directory / "data"),
        XDG_CONFIG_HOME=str(directory / "config"),
        XDG_CACHE_HOME=str(directory / "cache"),
        KILO_NO_DAEMON="1",
        KILO_DB=str(directory / "kilo.db"),
        KILO_CONFIG=str(directory / "kilo.json"),
        KILO_DISABLE_PROJECT_CONFIG="1",
    )
    for name in ("home", "data", "config", "cache"):
        (directory / name).mkdir(exist_ok=True)
    # File-backed stdout survives supervisor termination; a pipe buffered by this runner would not.
    with (directory / "stdout.jsonl").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
        result = subprocess.run(
            payload["command"], cwd=workdir, env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr
        )
    (directory / "exit.json").write_text(json.dumps(result.returncode))
    return result.returncode


def snapshot(directory: Path) -> None:
    """Retain a consistent database, including WAL contents, after supervised cleanup."""
    with sqlite3.connect(f"file:{directory / 'kilo.db'}?mode=ro", uri=True) as source:
        with sqlite3.connect(directory / "observations.db") as destination:
            source.backup(destination)


if __name__ == "__main__":
    if sys.argv[1] == "--snapshot":
        snapshot(Path(sys.argv[2]))
    else:
        sys.exit(main())
