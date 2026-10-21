# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No-op prepare for the vahid_1016 verified-50 benchmark: the JSONL and task assets are checked in next to this file."""

from pathlib import Path

JSONL_PATH = Path(__file__).resolve().parent / "benchmark.jsonl"


def prepare() -> Path:
    if not JSONL_PATH.is_file():
        raise FileNotFoundError(f"{JSONL_PATH} does not exist")
    return JSONL_PATH


if __name__ == "__main__":
    print(prepare())
