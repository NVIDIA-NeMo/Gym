# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the hash-pinned maintained WebVoyager population."""

from __future__ import annotations

from pathlib import Path

from benchmarks.webvoyager.prepare import BENCHMARK_DIR, prepare_native


OUTPUT = BENCHMARK_DIR / "data" / "webvoyager_native_v3.jsonl"


def prepare(source: str | Path | None = None, output: str | Path = OUTPUT) -> Path:
    """Prepare the maintained population through Gym's benchmark CLI contract."""

    return prepare_native(source, output)


if __name__ == "__main__":
    prepare()
