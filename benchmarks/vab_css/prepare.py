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
"""Prepare the VAB-CSS test split for NeMo Gym.

Upstream does not publish the test split (THUDM/VisualAgentBench gitignores data/css_dataset), so this reads a
local copy: pass its folder as ``prepare_script_args.data_dir`` and set the resources server's ``data_dir`` to the
same folder. Rows follow upstream's filter (see resources_servers/vab_css/prepare_data.py).
"""

import json
from pathlib import Path
from typing import Optional

from resources_servers.vab_css.prepare_data import build_rows


BENCHMARK_DIR = Path(__file__).parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "vab_css_benchmark.jsonl"
TEST_SPLIT_TASKS = 165  # the paper's test split (arXiv:2408.06327)


def prepare(
    *,
    data_dir: Optional[str] = None,
    expected_tasks: Optional[int] = TEST_SPLIT_TASKS,
    min_distance: float = 0.2,
) -> Path:
    """Write one row per task folder under ``data_dir``.

    ``expected_tasks=None`` accepts any non-zero task count; ``min_distance`` is upstream's filter (0.2).
    """
    if not data_dir:
        raise ValueError(
            "vab_css needs prepare_script_args.data_dir: the folder of task folders (upstream data/css_dataset)"
        )
    rows = build_rows(Path(data_dir), min_distance=min_distance, agent_name="vab_css_benchmark_gymnasium_agent")
    if not rows:
        raise ValueError(f"no VAB-CSS task folder under {data_dir} passes the filter (min_distance={min_distance})")
    if expected_tasks is not None and len(rows) != expected_tasks:
        raise ValueError(
            f"{data_dir} has {len(rows)} VAB-CSS tasks after upstream's filter, expected {expected_tasks}; "
            "pass expected_tasks=null to prepare another task set"
        )
    OUTPUT_FPATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FPATH.open("w") as f:
        for row in rows:
            row.pop("agent_ref", None)  # routing comes from the config declaration
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} VAB-CSS tasks to {OUTPUT_FPATH}")
    return OUTPUT_FPATH
