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
"""Prepare the VeriSoftBench benchmark split (all 500 tasks, curated context).

Delegates to the resources server's preparation script rather than duplicating it, so
the benchmark split is produced by the same tested path as the environment split,
including the pinned checkout, the dataset digest check and the 500-row / ids 1..500
check that refuse to write a short or altered corpus.

The one difference is ``--benchmark-form``: rows carry upstream's prompts as top-level
``system_prompt``/``user_prompt`` fields with no pre-populated
``responses_create_params.input``, because this benchmark declares ``prompt.yaml`` and
Gym rejects rows that carry both. Gym renders the turns at collation time, byte-identical
to the server split's explicit input.
"""

import argparse
import importlib.util
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parents[1]
SERVER_SCRIPT = REPO_ROOT / "resources_servers" / "verisoftbench" / "scripts" / "prepare_verisoftbench.py"
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "verisoftbench_benchmark.jsonl"


def _load_server_prepare():
    spec = importlib.util.spec_from_file_location("prepare_verisoftbench", SERVER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(limit: int | None = None, upstream_dir: str | None = None) -> Path:
    OUTPUT_FPATH.parent.mkdir(parents=True, exist_ok=True)
    argv = ["--output", str(OUTPUT_FPATH), "--benchmark-form"]
    if limit is not None:
        argv += ["--limit", str(limit)]
    if upstream_dir is not None:
        argv += ["--upstream-dir", upstream_dir]
    _load_server_prepare().main(argv)
    return OUTPUT_FPATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare the VeriSoftBench benchmark split")
    parser.add_argument("--limit", type=int, default=None, help="Max tasks, for a smoke subset (positive)")
    parser.add_argument(
        "--upstream-dir", default=None, help="Pinned upstream checkout (default: the server's cache path)"
    )
    args = parser.parse_args()
    prepare(limit=args.limit, upstream_dir=args.upstream_dir)
