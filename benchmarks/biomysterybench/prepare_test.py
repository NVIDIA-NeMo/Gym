# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare one official BioMysteryBench task for an end-to-end test."""

from pathlib import Path

from benchmarks.biomysterybench.prepare import DATA_DIR
from benchmarks.biomysterybench.prepare import prepare as _prepare


def prepare() -> Path:
    """Prepare the fixed one-task test selection."""

    return _prepare(
        release_name="official-99",
        task_ids=["hb013"],
        output=DATA_DIR / "biomysterybench_official_test.jsonl",
    )


if __name__ == "__main__":
    prepare()
