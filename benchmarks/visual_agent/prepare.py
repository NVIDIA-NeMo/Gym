# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build benchmarks/visual_agent/data/benchmark.jsonl from the task specs of the visual_agent server.

The reference images are rendered ahead of time (see resources_servers/visual_agent/build_dataset.py
references); this step only assembles prompts and rows, so it needs no sandbox.
"""

from pathlib import Path


OUTPUT_FPATH = Path(__file__).parent / "data" / "benchmark.jsonl"


def prepare() -> Path:
    from resources_servers.visual_agent.build_dataset import load_specs, validate, write_rows

    specs = load_specs()
    problems = validate(specs, require_references=True)
    if problems:
        raise ValueError("visual_agent task specs are not ready:\n" + "\n".join(problems))
    write_rows(specs, OUTPUT_FPATH)
    return OUTPUT_FPATH
