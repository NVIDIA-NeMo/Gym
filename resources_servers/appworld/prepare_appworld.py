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
"""Generate the AppWorld dataset JSONLs (one row per task id).

Rows deliberately contain **no AppWorld content** — just the task id and its
split. AppWorld's tasks, APIs and evaluation tests are its protected portion,
released under Apache 2.0 with the additional requirement that public
redistribution stay encrypted; copying instructions into a gym dataset (let
alone a dataset registry) would be exactly that. The resources server instead
reads the instruction from the locally-downloaded corpus at ``/seed_session``
time and returns it as the episode's first observation.

That also means these files are reproducible anywhere in seconds and are
gitignored rather than uploaded, except for the 5-row ``example.jsonl``.

Usage (installs/downloads AppWorld on first run):

    python resources_servers/appworld/prepare_appworld.py

    # or a single split to a chosen path
    python resources_servers/appworld/prepare_appworld.py \
        --splits train --output-dir resources_servers/appworld/data
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from resources_servers.appworld.setup_appworld import ensure_appworld, load_task_ids  # noqa: E402


# Upstream's usage policy: train is for teaching the model (demonstrations, RL),
# dev for tuning, and the two test splits are for reporting only.
ALL_SPLITS = ("train", "dev", "test_normal", "test_challenge")

# Filenames the YAML config and data/.gitignore expect.
SPLIT_FILENAMES = {
    "train": "train_appworld.jsonl",
    "dev": "dev_appworld.jsonl",
    "test_normal": "test_normal_appworld.jsonl",
    "test_challenge": "test_challenge_appworld.jsonl",
}


def make_row(task_id: str, split: str) -> dict:
    """One gym row: the task id, its split, and an empty conversation head.

    ``input`` is empty because ``/seed_session`` supplies the system prompt and
    the supervisor/instruction turn.
    """
    return {
        "task_id": task_id,
        "split": split,
        "responses_create_params": {"input": []},
    }


def write_jsonl(rows: list[dict], output_fpath: Path) -> None:
    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    with open(output_fpath, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} rows to {output_fpath}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(ALL_SPLITS),
        choices=list(ALL_SPLITS),
        help="AppWorld splits to materialize.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_HERE / "data",
        help="Directory for the generated JSONL files.",
    )
    parser.add_argument(
        "--appworld-root",
        default=os.environ.get("APPWORLD_ROOT"),
        help="AppWorld root holding data/ (default: the server's .appworld_root).",
    )
    parser.add_argument(
        "--appworld-venv",
        default=os.environ.get("APPWORLD_VENV"),
        help="Isolated venv holding the appworld package (default: the server's .appworld_venv).",
    )
    parser.add_argument(
        "--example-rows",
        type=int,
        default=5,
        help="Rows to also write to example.jsonl from the first split (0 to skip).",
    )
    args = parser.parse_args()

    install = ensure_appworld(args.appworld_root, args.appworld_venv)

    for index, split in enumerate(args.splits):
        task_ids = load_task_ids(install.root, split)
        rows = [make_row(task_id, split) for task_id in task_ids]
        write_jsonl(rows, args.output_dir / SPLIT_FILENAMES[split])
        if index == 0 and args.example_rows:
            write_jsonl(rows[: args.example_rows], args.output_dir / "example.jsonl")


if __name__ == "__main__":
    main()
