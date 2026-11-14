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

"""Rewrite train.jsonl and heldout.jsonl from generate.py's tasks.jsonl with another tool set.

    python rebuild_rows.py --geodata $GEODATA --tasks $TASKS/tasks.jsonl --tools both --out $TASKS_SQL

Ground truth is unchanged, so nothing is replayed. The SQL tables carry the Cropland Data
Layer class at each building's centroid, so raster tasks stay answerable with --tools sql.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from resources_servers.geobenchx_synth.catalog import dataset_row, offered_entries, split_states  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--geodata", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True, help="tasks.jsonl written by generate.py")
    parser.add_argument("--tools", choices=("gis", "sql", "both"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    entries = json.loads((args.geodata / "catalog.json").read_text())
    peers = split_states(entries)
    records = [json.loads(line) for line in args.tasks.open()]
    args.out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "heldout"):
        with open(args.out / f"{split}.jsonl", "w") as handle:
            for record in records:
                if record["split"] == split:
                    offered = offered_entries(entries, record["state"], peers[split])
                    handle.write(json.dumps(dataset_row(record, offered, args.tools)) + "\n")
    print(f"wrote {len(records)} rows with tools={args.tools}")


if __name__ == "__main__":
    main()
