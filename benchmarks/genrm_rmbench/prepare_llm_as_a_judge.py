# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare RM-Bench with full GenRM instructions for an LLM as a judge."""

from pathlib import Path

from benchmarks.genrm_rmbench.prepare import OUTPUT as SOURCE
from benchmarks.genrm_rmbench.prepare import prepare as prepare_genrm
from resources_servers.genrm_eval.llm_as_a_judge import prepare_llm_input


OUTPUT = Path(__file__).parent / "data/rmbench_llm_as_a_judge_validation.jsonl"


def prepare(output_fpath: str | Path = OUTPUT) -> Path:
    source = SOURCE if SOURCE.is_file() else prepare_genrm()
    return prepare_llm_input(source=source, output=Path(output_fpath))
