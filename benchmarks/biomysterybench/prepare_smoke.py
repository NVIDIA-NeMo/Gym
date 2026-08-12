# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare a single official BioMysteryBench task for an end-to-end smoke run."""

from pathlib import Path

from benchmarks.biomysterybench.prepare import DATA_DIR
from benchmarks.biomysterybench.prepare import prepare as _prepare


def prepare() -> Path:
    """Gym preparation entrypoint for the small, fixed smoke selection."""

    return _prepare(
        release_name="official-99",
        task_ids=["hb013"],
        output=DATA_DIR / "biomysterybench_official_smoke.jsonl",
    )


if __name__ == "__main__":
    prepare()
