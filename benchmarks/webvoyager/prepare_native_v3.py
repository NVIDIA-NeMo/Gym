# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the hash-pinned maintained WebVoyager population."""

from __future__ import annotations

import os
from pathlib import Path

from benchmarks.webvoyager.prepare import BENCHMARK_DIR, prepare_native


OUTPUT = BENCHMARK_DIR / "data" / "webvoyager_native_v3.jsonl"


def prepare(source: str | Path | None = None, output: str | Path = OUTPUT) -> Path:
    """Prepare the maintained population through Gym's benchmark CLI contract."""

    source_path = str(source or os.environ.get("WEBVOYAGER_SOURCE_JSONL", "")).strip()
    if not source_path:
        raise RuntimeError(
            "WEBVOYAGER_SOURCE_JSONL must point to webvoyager.jsonl from "
            "jayl940712/webarena_benchmarks@6a2977939b157b0ab9de7799bb089c721f1ac115"
        )
    return prepare_native(Path(source_path), Path(output))


if __name__ == "__main__":
    prepare()
