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
"""Prepare the RoleMRC gym-native benchmark dataset (reference-metric mode).

RoleMRC rows are already in Responses API shape — ``prepare_rolemrc.py`` puts the
full multi-turn RoleMRC conversation into ``responses_create_params.input`` and
carries ``reference`` / ``task`` / ``dimension`` alongside — so this script only
retags each row with the benchmark ``agent_ref`` and writes the benchmark JSONL.

One invocation of ``prepare_rolemrc.py`` writes BOTH splits (``test.jsonl`` for
reference mode and ``test_judge.jsonl`` for judge mode), so the helpers here are
shared with ``prepare_judge.py``. The source is downloaded from
``Junrulu/RoleMRC`` on first run; set ``ROLEMRC_LOCAL_JSONL`` to convert a
pre-downloaded ``roleMRC_test.jsonl`` instead.
"""

import importlib.util
import json
import sys
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parent
GYM_ROOT = BENCHMARK_DIR.parents[1]
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "rolemrc_benchmark.jsonl"

# Whole-dataset splits built by the resources server's own prepare script.
_SERVER_DIR = GYM_ROOT / "resources_servers" / "rolemrc"
_SRC_PREPARE = _SERVER_DIR / "prepare_rolemrc.py"
_SRC_REFERENCE = _SERVER_DIR / "data" / "test.jsonl"
_SRC_JUDGE = _SERVER_DIR / "data" / "test_judge.jsonl"

# Agent that runs this benchmark (see config.yaml). Rows are tagged with it so
# they align with the agent selected at eval time.
_BENCHMARK_AGENT = "rolemrc_benchmark_simple_agent"


def ensure_source() -> None:
    """Build test.jsonl + test_judge.jsonl if either is missing.

    ``prepare_rolemrc.py`` imports the server's ``app`` module for the judge
    aspect config, so the server dir goes on ``sys.path`` for the duration.
    """
    if _SRC_REFERENCE.exists() and _SRC_JUDGE.exists():
        return
    sys.path.insert(0, str(_SERVER_DIR))
    try:
        spec = importlib.util.spec_from_file_location("rolemrc_prepare_rolemrc", _SRC_PREPARE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main()
    finally:
        sys.path.remove(str(_SERVER_DIR))


def build(source: Path, output: Path, agent_name: str, label: str) -> Path:
    """Copy ``source`` rows to ``output``, retagged with ``agent_name``."""
    ensure_source()
    output.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with source.open(encoding="utf-8") as fin, output.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            row["agent_ref"] = {"type": "responses_api_agents", "name": agent_name}
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1

    print(f"{label}: wrote {n} rows -> {output}")
    return output


def prepare() -> Path:
    """Build the gym-native RoleMRC benchmark JSONL (reference split, tagged)."""
    return build(_SRC_REFERENCE, OUTPUT_FPATH, _BENCHMARK_AGENT, "RoleMRC")


if __name__ == "__main__":
    prepare()
