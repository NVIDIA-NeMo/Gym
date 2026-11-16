# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare HS4 with full GenRM instructions for an LLM as a judge."""

from pathlib import Path

from benchmarks.genrm_hs4.prepare import OUTPUT as SOURCE
from benchmarks.genrm_hs4.prepare import prepare as prepare_genrm
from resources_servers.genrm_eval.llm_as_a_judge import prepare_llm_input


OUTPUT = Path(__file__).parent / "data/hs4_llm_as_a_judge_validation.jsonl"


def prepare(output_fpath: str | Path = OUTPUT) -> Path:
    source = SOURCE if SOURCE.is_file() else prepare_genrm()
    return prepare_llm_input(source=source, output=Path(output_fpath))
