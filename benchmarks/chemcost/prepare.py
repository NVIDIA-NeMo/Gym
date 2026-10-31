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
"""Prepare ChemCost (all 1,427 reactions of YoungerWu/ChemCost) for the ``chemcost`` resources server.

Each row carries the upstream harness's system prompt, the reaction as the agent sees it, and the two tool
definitions. The pricing database is fetched here too, so the resources server finds it without network access.

    gym eval prepare --benchmark chemcost
"""

import json
from collections import Counter
from pathlib import Path

from resources_servers.chemcost.pricing_data import RECORDS_FILE, ensure_pricing_db, ensure_release_file
from resources_servers.chemcost.prompts import build_row


OUTPUT_FPATH = Path(__file__).absolute().parent / "data" / "chemcost_benchmark.jsonl"
# The upstream README's reference results cover all records: 1,359 single-step reactions and 68 routes.
EXPECTED_TYPES = {"single_step": 1359, "multi_step": 68}


def prepare(output_path: str | Path = OUTPUT_FPATH) -> Path:
    output_path = Path(output_path)
    records_path = ensure_release_file(RECORDS_FILE)
    ensure_pricing_db()
    with records_path.open() as f:
        records = [json.loads(line) for line in f if line.strip()]
    types = Counter(r["type"] for r in records)
    if dict(types) != EXPECTED_TYPES:
        raise ValueError(f"Unexpected ChemCost release: {dict(types)}, expected {EXPECTED_TYPES}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for record in records:
            f.write(json.dumps(build_row(record)) + "\n")
    return output_path


if __name__ == "__main__":
    print(prepare())
