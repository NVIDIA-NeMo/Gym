# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wrap saved HS3 candidates in GenRM-style instructions for an LLM as a judge."""

from pathlib import Path

from benchmarks.genrm_hs3.prepare import OUTPUT as SOURCE
from benchmarks.genrm_hs3.prepare import prepare as prepare_genrm
from resources_servers.genrm_eval.llm_as_a_judge import prepare_llm_input


OUTPUT = Path(__file__).parent / "data/hs3_llm_as_a_judge_validation.jsonl"


def prepare(output_fpath: str | Path = OUTPUT) -> Path:
    output = Path(output_fpath)
    if output.resolve() == SOURCE.resolve():
        raise ValueError("LLM preparation must not overwrite the GenRM input")
    source = SOURCE if SOURCE.is_file() else prepare_genrm()
    return prepare_llm_input(source=source, output=output)


if __name__ == "__main__":
    print(prepare())
