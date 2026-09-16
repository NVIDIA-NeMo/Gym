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
"""Prepare the LongMemEval gym-native benchmark dataset.

LongMemEval rows are already in Responses API shape (single user message
carrying the rendered session history + question). This script only needs to
tag each row with the benchmark ``agent_ref`` and write the benchmark JSONL.

If ``resources_servers/longmemeval/data/oracle.jsonl`` does not exist it is
built first by invoking ``prepare_longmemeval.py`` with default arguments
(``--split oracle``), which downloads the split from HuggingFace on first run.
"""

import importlib.util
import json
import sys
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parent
GYM_ROOT = BENCHMARK_DIR.parents[1]
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "longmemeval_benchmark.jsonl"

# Whole-dataset source built by the resources server's own prepare script.
_SRC_PREPARE = GYM_ROOT / "resources_servers" / "longmemeval" / "prepare_longmemeval.py"
_SRC_ORACLE = GYM_ROOT / "resources_servers" / "longmemeval" / "data" / "oracle.jsonl"

# Agent that runs this benchmark (see config.yaml). Rows are tagged with it so
# they align with the agent selected at eval time.
_BENCHMARK_AGENT = "longmemeval_benchmark_simple_agent"


def _ensure_source() -> None:
    """Build oracle.jsonl if not already present.

    ``prepare_longmemeval.main`` downloads ``xiaowu0162/longmemeval-cleaned``
    from HuggingFace and writes the full oracle split.
    """
    if _SRC_ORACLE.exists():
        return
    spec = importlib.util.spec_from_file_location("longmemeval_prepare_longmemeval", _SRC_PREPARE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old_argv = sys.argv
    sys.argv = [str(_SRC_PREPARE)]  # defaults: --split oracle, --topk-context 50
    try:
        module.main()
    finally:
        sys.argv = old_argv


def prepare() -> Path:
    """Build the gym-native LongMemEval benchmark JSONL (oracle split, tagged)."""
    _ensure_source()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    n = 0
    with _SRC_ORACLE.open(encoding="utf-8") as fin, OUTPUT_FPATH.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            row["agent_ref"] = {"type": "responses_api_agents", "name": _BENCHMARK_AGENT}
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1

    print(f"LongMemEval: wrote {n} rows -> {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
