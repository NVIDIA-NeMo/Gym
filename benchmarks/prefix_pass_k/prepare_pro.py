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

"""Prepare prefix pass@K rows for SWE-bench Pro (backticks wire).

The batch is mini-swe-agent in text mode, the wire of the Verified batch, so
the prompt, the prefix and the decisive turn come from
`prepare.decisive_turn_row`. Two things differ from Verified:
  * the task rows are Gym's SWE-bench Pro rows (`benchmarks/swebench/pro/prepare.py`),
    which pin each instance's image digest and evaluator scripts for the
    swebench_pro resources server;
  * ids keep their `instance_` prefix, which is part of a SWE-bench Pro id.

The pool is passed-only (the collecting model resolved every instance) and its
decisive turns sit deeper than Verified's: compare checkpoints' ordering on it,
not its level against another benchmark.

    python -m benchmarks.prefix_pass_k.prepare_pro
"""

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from benchmarks.prefix_pass_k.prepare import (
    DATA_DIR,
    DEFAULT_REFERENCE_ROOT,
    SkipInstance,
    decisive_turn_row,
    load_batch,
    read_pool,
    write_rows,
)
from benchmarks.swebench.pro.prepare import OUTPUT_FPATH as SWEBENCH_PRO_ROWS_FPATH
from benchmarks.swebench.pro.prepare import prepare as prepare_swebench_pro


OUTPUT_FPATH = DATA_DIR / "prefix_pass_k_pro_benchmark.jsonl"
DEFAULT_BATCH = "glm53flash-backticks-2026-09-11"


def build(
    reference_root: Path,
    batch: str,
    instances_file: Optional[Path],
    output_fpath: Path,
    swebench_pro_rows_fpath: Path = SWEBENCH_PRO_ROWS_FPATH,
) -> None:
    trace_root, bisect = load_batch(reference_root, batch)
    pool = read_pool(instances_file)

    if not swebench_pro_rows_fpath.is_file():
        swebench_pro_rows_fpath = prepare_swebench_pro()
    with swebench_pro_rows_fpath.open() as handle:
        by_instance = {row["instance_id"]: row for row in map(json.loads, handle)}

    rows: list[dict[str, Any]] = []
    skipped: Counter = Counter()
    for instance_id, entry in sorted(bisect.items()):
        if pool is not None and instance_id not in pool:
            continue
        swebench_pro_row = by_instance.get(instance_id)
        if swebench_pro_row is None:
            skipped["not_in_swebench_pro"] += 1
            continue
        try:
            messages, block = decisive_turn_row(trace_root / instance_id, entry, batch)
        except SkipInstance as reason:
            skipped[str(reason)] += 1
            continue

        # The prompt is the trace's own, not the Pro row's: the prefix was produced against it.
        rows.append(dict(swebench_pro_row) | {"responses_create_params": {"input": messages}, "prefix_pass_k": block})

    write_rows(rows, skipped, output_fpath)


def prepare() -> Path:
    """Gym entry point (`gym eval prepare`), with the environment overrides of `prepare.py`."""
    reference_root = Path(os.environ.get("PREFIX_PASS_K_REFERENCE_ROOT", DEFAULT_REFERENCE_ROOT))
    batch = os.environ.get("PREFIX_PASS_K_BATCH", DEFAULT_BATCH)
    instances = os.environ.get("PREFIX_PASS_K_INSTANCES_FILE")
    output = Path(os.environ.get("PREFIX_PASS_K_OUTPUT", OUTPUT_FPATH))
    build(reference_root, batch, Path(instances) if instances else None, output)
    return output


if __name__ == "__main__":
    print(prepare())
