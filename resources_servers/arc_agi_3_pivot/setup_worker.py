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
"""What the pivot windows run on, installed on the first start under ``.runtime/`` next to this file.

- The harness, from ``harness_source`` (the config default: the Kaggle kit's harness,
  ``kaggle:eladsarafian/arc3-c237-j30vfh-20261102``, or a directory or a tarball), installed and checked file by file by
  the ``arc_agi_3`` agent's ``setup_harness.py``: the same harness that agent plays.
- The worker interpreter (``worker_python``): a uv venv with the harness's runtime (the ``arc_agi_3`` agent's
  dependencies, flask included).
- The games of the pool: each game's code from ``games.parquet``, written in the layout ``arc_agi`` loads offline
  (``<game>/<version>/<game>.py`` + ``metadata.json``).

    python resources_servers/arc_agi_3_pivot/setup_worker.py --harness-source SRC [--python 3.12]
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any

from responses_api_agents.arc_agi_3_agent.setup_harness import ensure_harness


HERE = Path(__file__).resolve().parent
DEFAULT_DEST = HERE / ".runtime"
# The harness runtime: the arc_agi_3 agent's dependencies (its pyproject.toml), without nemo-gym.
HARNESS_PROJECT = HERE.parents[1] / "responses_api_agents" / "arc_agi_3_agent" / "pyproject.toml"
# The source commit of the Kaggle kit's harness (its HARNESS-PROVENANCE.json): the harness revision the example pool was
# recorded with. Pool rows carry it as their harness revision.
HARNESS_COMMIT = "30d2cc8bb582c4126e1020d084a94468a057dea7"  # pragma: allowlist secret
DIGEST = re.compile(r"[0-9a-f]{64}")
# One component of a game's path: a game id or version (no separator, not "." or "..").
PATH_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def ensure_harness_root(dest: Path = DEFAULT_DEST, *, source: str | None = None) -> Path:
    """The harness install root (``<root>/harness/taaf/src``, ``<root>/config``): the Kaggle kit's harness from
    ``source``, installed under ``dest`` on first use."""
    if not source:
        raise FileNotFoundError(
            "no harness: set harness_source (the Kaggle kit's harness, e.g. "
            "kaggle:eladsarafian/arc3-c237-j30vfh-20261102) or harness_root (an existing install)"
        )
    return ensure_harness(source, dest / "harness")


def worker_requirements() -> list[str]:
    dependencies = tomllib.loads(HARNESS_PROJECT.read_text())["project"]["dependencies"]
    return [d for d in dependencies if not d.startswith("nemo-gym")]


def _imports_harness_runtime(python: Path) -> bool:
    probe = subprocess.run(
        [str(python), "-c", "import arc_agi, arcengine, flask, numpy, requests"], capture_output=True
    )
    return probe.returncode == 0


def ensure_worker_python(version: str, dest: Path = DEFAULT_DEST) -> Path:
    """The interpreter of a venv at ``dest/worker-py<version>`` with ``worker_requirements()``, built on first use (and
    rebuilt when the requirements change or do not import)."""
    venv = dest / f"worker-py{version}"
    python, stamp = venv / "bin" / "python", venv / ".requirements"
    requirements = worker_requirements()
    if (
        python.is_file()
        and stamp.is_file()
        and stamp.read_text().splitlines() == requirements
        and _imports_harness_runtime(python)
    ):
        return python
    uv = shutil.which("uv")
    if uv is None:
        raise FileNotFoundError(f"uv is needed to build the Python {version} worker venv at {venv}")
    dest.mkdir(parents=True, exist_ok=True)
    # --no-config: the repo's [tool.uv] settings (exclude-dependencies drops flask, which arc_agi imports) are Gym's
    # server venvs', not this one's.
    for command in (
        [uv, "--no-config", "venv", "--clear", "--python", version, str(venv)],
        [uv, "--no-config", "pip", "install", "--python", str(python), *requirements],
    ):
        result = subprocess.run(command, capture_output=True, text=True, errors="replace", cwd=dest)
        if result.returncode != 0:
            raise RuntimeError(f"{' '.join(command[:4])} failed: {result.stderr.strip()[-2000:]}")
    if not _imports_harness_runtime(python):
        raise RuntimeError(f"{python} cannot import the harness runtime after installing {requirements}")
    stamp.write_text("\n".join(requirements) + "\n")
    return python


def game_layout(entry: dict[str, Any]) -> Path:
    """The relative path ``<game>/<version>/<game>.py`` of a ``games.parquet`` entry, after checking that its
    ``source_sha256`` is a sha256 digest and its ``environment_path`` (``[environments/]<game>/<version>/<game>.py``)
    names that game and version and nothing else (no absolute path, no ``..``). Raises ValueError otherwise."""
    digest = str(entry.get("source_sha256") or "")
    if not DIGEST.fullmatch(digest):
        raise ValueError(f"games.parquet entry {entry.get('game_id')!r}: source_sha256 {digest!r} is not a sha256")
    parts = PurePosixPath(str(entry.get("environment_path") or "")).parts
    parts = parts[1:] if parts[:1] == ("environments",) else parts
    game_id = json.loads(entry["sdk_metadata_json"]).get("game_id")
    if not (
        len(parts) == 3
        and all(PATH_PART.fullmatch(p) for p in parts)
        and parts[2] == f"{parts[0]}.py"
        and game_id == entry.get("game_id") == f"{parts[0]}-{parts[1]}"
    ):
        raise ValueError(
            f"games.parquet entry {entry.get('game_id')!r}: environment_path {entry.get('environment_path')!r} is not "
            f"<game>/<version>/<game>.py of its game ({game_id!r})"
        )
    return Path(*parts)


def materialize_game(entry: dict[str, Any], cache: Path) -> Path:
    """The ``environments_dir`` holding one game of ``games.parquet`` (``environment_code``, ``environment_path``,
    ``sdk_metadata_json``), keyed by its source and code sha256. Idempotent and atomic; every path stays under
    ``cache`` (:func:`game_layout`)."""
    rel = game_layout(entry)
    code_sha = hashlib.sha256(entry["environment_code"].encode("utf-8")).hexdigest()[:16]
    root = cache / f"{entry['source_sha256']}-{code_sha}"
    if (root / ".complete").is_file():
        return root
    cache.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".game-", dir=cache))
    target = staging / rel
    if not target.resolve().is_relative_to(staging.resolve()):
        shutil.rmtree(staging, ignore_errors=True)
        raise ValueError(f"{rel} leaves the game cache")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(entry["environment_code"], encoding="utf-8")
    (target.parent / "metadata.json").write_text(json.dumps(json.loads(entry["sdk_metadata_json"])), encoding="utf-8")
    (staging / ".complete").write_text(entry["game_id"], encoding="utf-8")
    try:
        os.replace(staging, root)
    except OSError:  # another process won the race; its directory is complete by construction
        shutil.rmtree(staging, ignore_errors=True)
        if not (root / ".complete").is_file():
            raise
    return root


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument(
        "--harness-source", help="the Kaggle kit's harness: kaggle:<owner>/<prefix>, directory or tarball"
    )
    parser.add_argument("--python", help="build the worker venv for this Python version (e.g. 3.12)")
    args = parser.parse_args(argv)
    root = ensure_harness_root(args.dest, source=args.harness_source)
    print(f"harness at {root}")
    if args.python:
        print(f"worker interpreter {ensure_worker_python(args.python, args.dest)}")


if __name__ == "__main__":
    main()
