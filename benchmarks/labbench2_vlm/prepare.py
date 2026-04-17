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
"""Prepare LabBench2 VLM benchmark data.

Wraps the existing per-tag prepare logic in
``resources_servers/labbench2_vlm/prepare_data.py`` and writes a single
combined Gym benchmark JSONL covering all four subtasks
(figqa2-img, figqa2-pdf, tableqa2-img, tableqa2-pdf).

Media files are downloaded into ``resources_servers/labbench2_vlm/data/media/``
so they resolve against the agent's ``media_base_dir`` (inherited from
``resources_servers/labbench2_vlm/configs/labbench2_vlm.yaml``). Per-tag
metrics are emitted by the resource server's ``compute_metrics`` using the
``verifier_metadata.tag`` field.
"""

from pathlib import Path

import orjson

from resources_servers.labbench2_vlm.prepare_data import TAGS, prepare_tag


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "labbench2_vlm_benchmark.jsonl"

RS_DATA_DIR = Path(__file__).resolve().parents[2] / "resources_servers" / "labbench2_vlm" / "data"
MEDIA_DIR = RS_DATA_DIR / "media"


def prepare() -> Path:
    """Download media + questions and combine all four tags into one JSONL."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for tag in TAGS:
        safe = tag.replace("-", "_")
        per_tag_out = RS_DATA_DIR / f"{safe}_validation.jsonl"
        rows = prepare_tag(tag, per_tag_out, media_dir=MEDIA_DIR)
        all_rows.extend(rows)

    with open(OUTPUT_FPATH, "wb") as f:
        for row in all_rows:
            f.write(orjson.dumps(row) + b"\n")

    print(f"Wrote {len(all_rows)} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
