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
"""Prepare the RAGTruth gym-native benchmark dataset.

RAGTruth rows are already in Responses API shape — ``prepare_ragtruth.py`` bakes
the slice's prompt template (context + candidate response) into a single user
message — so this script only concatenates the three task slices into one
whole-dataset JSONL and retags each row with the benchmark ``agent_ref``.

Upstream reports QA / Summary / Data2txt together; the ``task_type`` field rides
on every row, so the resources server's ``compute_metrics`` still produces the
per-slice breakdown from the concatenated file.

If the split files under ``resources_servers/ragtruth/data/`` do not exist they
are built first by invoking ``prepare_ragtruth.py``, which downloads the upstream
``response.jsonl`` / ``source_info.jsonl`` into ``$XDG_CACHE_HOME/byob_ragtruth``
on first run (see that script for the offline / pre-staged options).
"""

import importlib.util
import json
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parent
GYM_ROOT = BENCHMARK_DIR.parents[1]
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "ragtruth_benchmark.jsonl"

# Whole-dataset splits built by the resources server's own prepare script.
_SERVER_DIR = GYM_ROOT / "resources_servers" / "ragtruth"
_SRC_PREPARE = _SERVER_DIR / "prepare_ragtruth.py"
_SRC_SPLITS = (
    _SERVER_DIR / "data" / "test_qa.jsonl",
    _SERVER_DIR / "data" / "test_summary.jsonl",
    _SERVER_DIR / "data" / "test_data2txt.jsonl",
)

# Agent that runs this benchmark (see config.yaml). Rows are tagged with it so
# they align with the agent selected at eval time.
_BENCHMARK_AGENT = "ragtruth_benchmark_simple_agent"


def _ensure_source() -> None:
    """Build the three test splits if any is missing."""
    if all(path.exists() for path in _SRC_SPLITS):
        return
    spec = importlib.util.spec_from_file_location("ragtruth_prepare_ragtruth", _SRC_PREPARE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


def prepare() -> Path:
    """Build the gym-native RAGTruth benchmark JSONL (all three slices, tagged)."""
    _ensure_source()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    n = 0
    with OUTPUT_FPATH.open("w", encoding="utf-8") as fout:
        for split in _SRC_SPLITS:
            with split.open(encoding="utf-8") as fin:
                for line in fin:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    row["agent_ref"] = {"type": "responses_api_agents", "name": _BENCHMARK_AGENT}
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n += 1

    print(f"RAGTruth: wrote {n} rows -> {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
