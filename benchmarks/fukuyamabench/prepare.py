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

"""Prepare the FukuyamaBench benchmark split (sets B and C).

Delegates to the resources server's preparation script rather than duplicating
it, so the benchmark split is produced by the same tested path as the
environment split — including the manifest check that refuses to write a short
corpus.
"""

import argparse
import importlib.util
import sys
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parents[1]
SERVER_SCRIPT = REPO_ROOT / "resources_servers" / "fukuyamabench" / "scripts" / "prepare_fukuyamabench.py"
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "fukuyamabench_benchmark.jsonl"

# Sets B and C are the tiers carrying upstream's sub-20% published figures; set
# A is a materially easier tier and is not part of this benchmark.
SETS = ("B", "C")


def _load_server_prepare():
    spec = importlib.util.spec_from_file_location("prepare_fukuyamabench", SERVER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(limit: int | None = None) -> Path:
    OUTPUT_FPATH.parent.mkdir(parents=True, exist_ok=True)
    server_prepare = _load_server_prepare()

    argv = ["prepare_fukuyamabench", "--output", str(OUTPUT_FPATH), "--sets", *SETS]
    if limit is not None:
        argv += ["--limit", str(limit)]

    original_argv = sys.argv
    try:
        sys.argv = argv
        server_prepare.main()
    finally:
        sys.argv = original_argv
    return OUTPUT_FPATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare the FukuyamaBench benchmark split")
    parser.add_argument("--limit", type=int, default=None, help="Max rows, for a smoke subset")
    args = parser.parse_args()
    prepare(limit=args.limit)
