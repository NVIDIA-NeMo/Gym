# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No-op prepare for the vahid_1016 held-out test split.

Unlike benchmarks/nl2repobench/prepare.py (which re-downloads/validates the pinned upstream
NL2RepoBench source every launch), this benchmark's JSONL and task assets are already fully
materialized by sft_runs/lightning_vahid1016/testset/prepare_testset.py - a one-time derivation
from the SFT data's own held-out split, not something to regenerate from an upstream source on
every launch. This just confirms the JSONL exists.
"""

from pathlib import Path

JSONL_PATH = Path(
    "/lustre/fsw/portfolios/llmservice/projects/llmservice_nemotron_nano/users/fjia/long_code/"
    "nl2repo/sft_runs/lightning_vahid1016/testset/benchmark.jsonl"
)


def prepare() -> Path:
    if not JSONL_PATH.is_file():
        raise FileNotFoundError(
            f"{JSONL_PATH} does not exist - run "
            "sft_runs/lightning_vahid1016/testset/prepare_testset.py first."
        )
    return JSONL_PATH


if __name__ == "__main__":
    print(prepare())
