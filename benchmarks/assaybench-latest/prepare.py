# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the AssayBench LaTest cohort for the `assaybench` resources server."""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from resources_servers.assaybench.prepare import build_rows, write_jsonl  # noqa: E402


OUTPUT_FPATH = Path(__file__).parent / "data" / "assaybench_latest_benchmark.jsonl"


def prepare() -> Path:
    """Fetch the pinned dataset revision and write the 19 LaTest rows. Returns the path."""
    write_jsonl(OUTPUT_FPATH, build_rows(split="LaTest"))
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
