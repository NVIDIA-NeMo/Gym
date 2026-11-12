# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare EQ-Bench 4 from https://github.com/EQ-bench/eqbench4 at a pinned revision.

Downloads about 1.7 GB of published results once; ``source_dir`` reads a local checkout of that revision instead.
"""

from pathlib import Path

from resources_servers.eqbench4.prepare import DEFAULT_OPPONENTS_PER_SCENARIO, prepare_eqbench4


DATA_DIR = Path(__file__).parent / "data"
OUTPUT_FPATH = DATA_DIR / "eqbench4_benchmark.jsonl"
REFERENCE_COMPARISONS_FPATH = DATA_DIR / "reference_comparisons.jsonl"
CACHE_DIR = DATA_DIR / "upstream"


def prepare(*, source_dir: str | None = None, opponents_per_scenario: int = DEFAULT_OPPONENTS_PER_SCENARIO) -> Path:
    """Write the 120 scenario rows and the published comparison graph; return the rows' path."""
    return prepare_eqbench4(
        OUTPUT_FPATH,
        REFERENCE_COMPARISONS_FPATH,
        source_dir=source_dir,
        cache_dir=CACHE_DIR,
        opponents_per_scenario=opponents_per_scenario,
    )


if __name__ == "__main__":
    prepare()
