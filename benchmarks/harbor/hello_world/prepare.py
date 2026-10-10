# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare rows for Harbor's hello-world task, from the Harbor dataset version pinned in config.yaml."""

from pathlib import Path

from benchmarks.harbor.prepare_utils.provisioning import prepare_rows


BENCHMARK_DIR = Path(__file__).parent


def prepare() -> Path:
    return prepare_rows(BENCHMARK_DIR / "config.yaml", BENCHMARK_DIR / "data" / "benchmark.jsonl")


if __name__ == "__main__":
    prepare()
