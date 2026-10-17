# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the ORAgentBench benchmark rows.

Delegates to the resources server's preparation script rather than duplicating it, so the
benchmark rows come from the same tested path: the pinned checkout, the per-task parser,
and the manifest check that refuses to write anything unless exactly 107 tasks
(32 easy / 41 medium / 34 hard) load. Pass ``--build-images`` to also build the base and
per-task Docker images the rows reference.
"""

import argparse
import importlib.util
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parents[1]
SERVER_SCRIPT = REPO_ROOT / "resources_servers" / "oragentbench" / "scripts" / "prepare_oragentbench.py"
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "oragentbench_benchmark.jsonl"


def _load_server_prepare():
    spec = importlib.util.spec_from_file_location("prepare_oragentbench", SERVER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(limit: int | None = None, build_images: bool = False) -> Path:
    OUTPUT_FPATH.parent.mkdir(parents=True, exist_ok=True)
    argv = ["--output", str(OUTPUT_FPATH)]
    if limit is not None:
        argv += ["--limit", str(limit)]
    if build_images:
        argv.append("--build-images")
    _load_server_prepare().main(argv)
    return OUTPUT_FPATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare the ORAgentBench benchmark rows")
    parser.add_argument("--limit", type=int, default=None, help="Max rows (sorted task order), for a smoke subset")
    parser.add_argument("--build-images", action="store_true", help="Also build the base and per-task Docker images")
    args = parser.parse_args()
    prepare(limit=args.limit, build_images=args.build_images)
