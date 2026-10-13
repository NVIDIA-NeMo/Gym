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
"""Prepare the MolDeTox gym-native benchmark dataset.

The resources server ships `scripts/prepare_moldetox.py`, which fetches the
pinned dataset revision and builds gym rows: upstream's system prompt and user
question as the two input turns, plus the gold answer and scoring mode in
`verifier_metadata`. This wrapper reuses it to write one config's full test
split for the gym-native eval.

**Which config this registers, and why only one.** MolDeTox ships eight QA
configs and publishes a separate figure for each, so no single one is "the"
benchmark. This entry registers `task3_smiles_gen_single`: it is the
configuration upstream headlines -- whole-molecule detoxification written as
SMILES, single-fragment edit -- and the one whose measured accuracy agrees with
the published figure. The other seven stay reachable through the resources
server's script; see `resources_servers/moldetox/README.md`.

Usage::

    python benchmarks/moldetox/prepare.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "resources_servers" / "moldetox" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from prepare_moldetox import _load_qa_rows, _toxic_smiles_by_index, format_row  # noqa: E402


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "moldetox_benchmark.jsonl"

CONFIG = "task3_smiles_gen_single"
SPLIT = "test"


def prepare() -> Path:
    """Build the whole-split MolDeTox benchmark JSONL for the registered config."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    toxic_smiles = _toxic_smiles_by_index(SPLIT)
    rows = _load_qa_rows(CONFIG, SPLIT)

    missing = 0
    with open(OUTPUT_FPATH, "w", encoding="utf-8") as writer:
        for row in rows:
            formatted = format_row(row, CONFIG, toxic_smiles)
            # `agent_ref` in a source dataset is deprecated: routing comes from the
            # config declaration, and `gym dataset collate` strips it and warns.
            # The resources server's script still emits it, as 21 others do, so it
            # is dropped here rather than shipped and then stripped.
            formatted.pop("agent_ref", None)
            missing += formatted["verifier_metadata"]["toxic_smiles"] is None
            writer.write(json.dumps(formatted, ensure_ascii=False) + "\n")

    print(f"MolDeTox: wrote {len(rows)} rows ({CONFIG}, {SPLIT}) -> {OUTPUT_FPATH}")
    if missing:
        print(f"WARNING: {missing} rows had no toxicitycliff join for source_index", file=sys.stderr)
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
