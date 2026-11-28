# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare agentic CG-Bench-mini (long_acc, 3,000 questions) rows and frozen cohort."""

from pathlib import Path

from resources_servers.cgbench.prepare import CGBenchPromptStyle, prepare_cgbench_benchmark
from resources_servers.cgbench.scoring import CGBENCH_RULE_POLICY, CGBenchAnswerPolicy


BENCHMARK_DIR = Path(__file__).parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "cgbench_mini_agentic_benchmark.jsonl"
COHORT_FPATH = BENCHMARK_DIR / "data" / "cgbench_mini_agentic_cohort.json"


def prepare(
    source_dir: str | None = None,
    video_dir: str | None = None,
    annotations_path: str | None = None,
    answer_policy: CGBenchAnswerPolicy = CGBENCH_RULE_POLICY,
    prompt: CGBenchPromptStyle = "official",
    num_repeats: int = 1,
    sandbox_root: str = "/testbed",
    download: bool = False,
) -> Path:
    """Write agentic CG-Bench-mini rows to OUTPUT_FPATH and frozen cohort to COHORT_FPATH."""
    return prepare_cgbench_benchmark(
        benchmark="cgbench_mini",
        setting="agentic",
        output_fpath=OUTPUT_FPATH,
        cohort_fpath=COHORT_FPATH,
        source_dir=source_dir,
        video_dir=video_dir,
        annotations_path=annotations_path,
        answer_policy=answer_policy,
        prompt=prompt,
        num_repeats=num_repeats,
        sandbox_root=sandbox_root,
        download=download,
    )
