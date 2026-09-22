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

"""Prepare the NMRArena benchmark split (105 molecules).

Delegates to the resources server's preparation script rather than duplicating
it, so the benchmark split is produced by the same tested, fail-closed path as
the environment split: pinned commit, dataset digest, row count, class structure,
unique ids and parseable gold checked before anything is written.
"""

import argparse
import importlib.util
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parents[1]
SERVER_SCRIPT = REPO_ROOT / "resources_servers" / "nmrarena" / "scripts" / "prepare_nmrarena.py"
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "nmrarena_benchmark.jsonl"


def _load_server_prepare():
    spec = importlib.util.spec_from_file_location("prepare_nmrarena", SERVER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(limit: int | None = None) -> Path:
    OUTPUT_FPATH.parent.mkdir(parents=True, exist_ok=True)
    argv = ["--output", str(OUTPUT_FPATH)]
    if limit is not None:
        argv += ["--limit", str(limit)]
    _load_server_prepare().main(argv)
    return OUTPUT_FPATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare the NMRArena benchmark split")
    parser.add_argument("--limit", type=int, default=None, help="Max rows, for a smoke subset")
    args = parser.parse_args()
    prepare(limit=args.limit)
