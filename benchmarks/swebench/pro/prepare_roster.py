# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Prepare a hash-pinned roster of SWE-bench Pro instances as its own benchmark file.

A recipe names this script as its dataset's ``prepare_script`` and passes the roster through
``prepare_script_args``:

    prepare_script_args:
      roster: recipes/nooa-atom/swebench-pro-swe120.txt      # one instance id per line
      roster_sha256: <sha256 of that file>

The full benchmark is prepared first (``benchmarks/swebench/pro/prepare.py``, pinned upstream assets and image
digests), then the roster's rows are written to ``data/<roster stem>.jsonl`` (dashes become underscores), which is the path the
recipe's dataset entry must name. Every roster instance must exist in the prepared benchmark.
"""

import hashlib
import json
from pathlib import Path

from benchmarks.swebench.pro.prepare import DATA_DIR
from benchmarks.swebench.pro.prepare import prepare as prepare_full


def prepare(*, roster: str, roster_sha256: str) -> Path:
    roster_path = Path(roster)
    digest = hashlib.sha256(roster_path.read_bytes()).hexdigest()
    if digest != roster_sha256:
        raise ValueError(f"roster {roster_path} has sha256 {digest}, the recipe pins {roster_sha256}")
    wanted = {line.strip() for line in roster_path.read_text().splitlines() if line.strip()}
    full = prepare_full()
    rows = [json.loads(line) for line in full.read_text().splitlines() if line.strip()]
    selected = [row for row in rows if row.get("instance_id") in wanted]
    missing = wanted - {row.get("instance_id") for row in selected}
    if missing:
        raise ValueError(
            f"{len(missing)} roster instances are not in the prepared benchmark, e.g. {sorted(missing)[:3]}"
        )
    output = DATA_DIR / f"{roster_path.stem.replace('-', '_')}.jsonl"
    with output.open("w") as handle:
        for row in selected:
            handle.write(json.dumps(row) + "\n")
    print(f"Wrote {len(selected)} roster instances to {output}")
    return output
