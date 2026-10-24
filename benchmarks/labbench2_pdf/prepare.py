# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the direct-PDF LabBench2 benchmark and its Harbor task registry."""

from __future__ import annotations

import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from environments.labbench2_pdf.data_utils import SUPPORTED_BENCHMARKS
from environments.labbench2_pdf.prepare import DEFAULT_OUTPUT_DIRS
from environments.labbench2_pdf.prepare_data import PreparationOptions, prepare_data


BENCHMARK_DIR = Path(__file__).resolve().parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "labbench2_pdf_benchmark.jsonl"
TASKS_DIR = DEFAULT_OUTPUT_DIRS["docker"]
BENCHMARK_AGENT_NAME = "labbench2_pdf_benchmark_harbor_agent"


def _optional_path(value: str | Path | None) -> Path | None:
    return Path(value) if value is not None else None


def _benchmark_tuple(value: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temp_name = tempfile.mkstemp(dir=destination.parent, prefix=f".{destination.name}.")
    temp_path = Path(temp_name)
    try:
        with source.open("rb") as input_file, os.fdopen(file_descriptor, "wb") as output_file:
            while chunk := input_file.read(1024 * 1024):
                output_file.write(chunk)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)


def prepare(
    *,
    questions_dir: str | Path | None = None,
    papers_dir: str | Path | None = None,
    benchmarks: str | list[str] | tuple[str, ...] = SUPPORTED_BENCHMARKS,
    limit_per_benchmark: int | None = None,
    build_image: bool = True,
    source_policy: str = "all",
    missing_paper: str = "skip",
    corpus_view_mode: str = "copy",
    allow_internet: bool = True,
    jobs: int = 6,
    timeout: float = 45.0,
    retries: int = 3,
    contact_email: str | None = None,
    unpaywall_email: str | None = None,
    openalex_api_key: str | None = None,
    openalex_content: bool = False,
    crossref: bool = True,
    retry_failed_papers: bool = False,
    overwrite_papers: bool = False,
    **unsupported: Any,
) -> Path:
    """Acquire inputs, build Docker tasks, and write Gym's benchmark index.

    Keyword arguments are accepted through ``+prepare_script_args.<name>=...``.
    Local question/PDF directories bypass their respective download stages.
    """
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise TypeError(f"unsupported LabBench2 PDF preparation arguments: {names}")

    defaults = PreparationOptions()
    options = replace(
        defaults,
        questions_dir=_optional_path(questions_dir),
        papers_dir=_optional_path(papers_dir),
        output_dir=TASKS_DIR,
        benchmarks=_benchmark_tuple(benchmarks),
        source_policy=source_policy,
        jobs=jobs,
        timeout=timeout,
        retries=retries,
        contact_email=contact_email,
        unpaywall_email=unpaywall_email or os.environ.get("UNPAYWALL_EMAIL"),
        openalex_api_key=openalex_api_key or os.environ.get("OPENALEX_API_KEY"),
        openalex_content=openalex_content,
        crossref=crossref,
        retry_failed_papers=retry_failed_papers,
        overwrite_papers=overwrite_papers,
        environment_type="docker",
        build_image=build_image,
        corpus_view_mode=corpus_view_mode,
        missing_paper=missing_paper,
        allow_internet=allow_internet,
        limit_per_benchmark=limit_per_benchmark,
        agent_name=BENCHMARK_AGENT_NAME,
        overwrite_tasks=True,
    )
    result = prepare_data(options)
    rollout_input = Path(result["materialization"]["output_dir"]) / "rollout_input.jsonl"
    _atomic_copy(rollout_input, OUTPUT_FPATH)
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
