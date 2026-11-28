# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare agentic Video-Index v1.0 rows (840 questions) and their frozen cohort."""

from pathlib import Path

from resources_servers.videoindex.prepare import VideoIndexPromptStyle, prepare_videoindex_benchmark
from resources_servers.videoindex.scoring import VIDEOINDEX_RULE_POLICY, VideoIndexAnswerPolicy


BENCHMARK_DIR = Path(__file__).parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "videoindex_v1_0_agentic_benchmark.jsonl"
COHORT_FPATH = BENCHMARK_DIR / "data" / "videoindex_v1_0_agentic_cohort.json"


def prepare(
    source_dir: str | None = None,
    video_dir: str | None = None,
    answer_policy: VideoIndexAnswerPolicy = VIDEOINDEX_RULE_POLICY,
    prompt: VideoIndexPromptStyle = "official",
    num_repeats: int = 1,
    sandbox_root: str = "/testbed",
    download: bool = False,
) -> Path:
    """Write agentic Video-Index v1.0 rows to OUTPUT_FPATH and frozen cohort to COHORT_FPATH."""
    return prepare_videoindex_benchmark(
        benchmark="videoindex_v1_0",
        setting="agentic",
        output_fpath=OUTPUT_FPATH,
        cohort_fpath=COHORT_FPATH,
        source_dir=source_dir,
        video_dir=video_dir,
        answer_policy=answer_policy,
        prompt=prompt,
        num_repeats=num_repeats,
        sandbox_root=sandbox_root,
        download=download,
    )
