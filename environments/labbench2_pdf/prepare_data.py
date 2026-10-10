#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare LabBench2 questions, papers, and filtered Harbor tasks in one command."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from environments.labbench2_pdf.data_utils import (  # noqa: E402
    DEFAULT_PAPERS_DIR,
    DEFAULT_QUESTIONS_DIR,
    SOURCE_POLICIES,
    SUPPORTED_BENCHMARKS,
    load_questions,
)
from environments.labbench2_pdf.download_papers import (  # noqa: E402
    DEFAULT_USER_AGENT,
    PaperDownloadOptions,
    download_corpus,
)
from environments.labbench2_pdf.download_questions import (  # noqa: E402
    DEFAULT_HF_REPO_ID,
    DEFAULT_HF_REVISION,
    QuestionDownloadOptions,
    download_questions,
)
from environments.labbench2_pdf.prepare import (  # noqa: E402
    DEFAULT_AGENT_NAME,
    DEFAULT_IMAGE,
    DEFAULT_OUTPUT_DIRS,
    MaterializationOptions,
    build_docker_image,
    materialize,
)


@dataclass(frozen=True, slots=True)
class PreparationOptions:
    """Inputs for the complete question/PDF/task preparation pipeline.

    A supplied ``questions_dir`` or ``papers_dir`` is used in place and is
    never populated by this command. Omitting either path enables its
    corresponding downloader and gitignored cache.
    """

    questions_dir: Path | None = None
    papers_dir: Path | None = None
    questions_cache_dir: Path = DEFAULT_QUESTIONS_DIR
    papers_cache_dir: Path = DEFAULT_PAPERS_DIR
    output_dir: Path | None = None
    benchmarks: tuple[str, ...] = SUPPORTED_BENCHMARKS
    hf_repo_id: str = DEFAULT_HF_REPO_ID
    hf_revision: str | None = DEFAULT_HF_REVISION
    refresh_questions: bool = False
    source_policy: str = "all"
    jobs: int = 6
    timeout: float = 45.0
    retries: int = 3
    max_pdf_mb: int = 100
    min_pdf_bytes: int = 1000
    user_agent: str = DEFAULT_USER_AGENT
    contact_email: str | None = None
    openalex_api_key: str | None = None
    openalex_content: bool = False
    unpaywall_email: str | None = None
    crossref: bool = True
    retry_failed_papers: bool = False
    overwrite_papers: bool = False
    environment_type: str = "docker"
    image: str = DEFAULT_IMAGE
    build_image: bool = False
    corpus_view_mode: str = "copy"
    missing_paper: str = "skip"
    allow_internet: bool = True
    limit_per_benchmark: int | None = None
    agent_name: str = DEFAULT_AGENT_NAME
    overwrite_tasks: bool = False


def _validate_options(options: PreparationOptions) -> None:
    if not options.benchmarks:
        raise ValueError("at least one benchmark is required")
    if len(set(options.benchmarks)) != len(options.benchmarks):
        raise ValueError("benchmarks must be unique")
    unsupported = sorted(set(options.benchmarks) - set(SUPPORTED_BENCHMARKS))
    if unsupported:
        raise ValueError(f"unsupported benchmarks: {unsupported}")
    if options.source_policy not in SOURCE_POLICIES:
        raise ValueError(f"unsupported source policy: {options.source_policy}")
    if options.environment_type not in DEFAULT_OUTPUT_DIRS:
        raise ValueError(f"unsupported environment type: {options.environment_type}")
    if options.corpus_view_mode not in {"hardlink", "copy"}:
        raise ValueError(f"unsupported corpus view mode: {options.corpus_view_mode}")
    if options.missing_paper not in {"skip", "error"}:
        raise ValueError(f"unsupported missing-paper policy: {options.missing_paper}")
    if options.limit_per_benchmark is not None and options.limit_per_benchmark < 1:
        raise ValueError("limit_per_benchmark must be at least 1")
    if options.build_image and options.environment_type != "docker":
        raise ValueError("build_image is only valid for the Docker environment")
    if options.papers_dir is None:
        if options.jobs < 1:
            raise ValueError("jobs must be positive")
        if options.timeout <= 0 or options.retries < 1:
            raise ValueError("timeout and retries must be positive")
        if options.max_pdf_mb < 1 or options.min_pdf_bytes < 1:
            raise ValueError("max_pdf_mb and min_pdf_bytes must be positive")
        if options.openalex_content and not options.openalex_api_key:
            raise ValueError("openalex_content requires an OpenAlex API key")


def _prepare_questions(options: PreparationOptions) -> tuple[Path, dict[str, Any]]:
    if options.questions_dir is not None:
        questions_dir = options.questions_dir.expanduser().resolve()
        if not questions_dir.is_dir():
            raise FileNotFoundError(f"questions directory not found: {questions_dir}")
        rows = load_questions(questions_dir, options.benchmarks)
        return questions_dir, {
            "mode": "provided",
            "path": str(questions_dir),
            "row_count": sum(len(values) for values in rows.values()),
        }

    questions_dir = options.questions_cache_dir.expanduser().resolve()
    manifest = download_questions(
        QuestionDownloadOptions(
            output_dir=questions_dir,
            benchmarks=options.benchmarks,
            repo_id=options.hf_repo_id,
            revision=options.hf_revision,
            overwrite=options.refresh_questions,
        )
    )
    return questions_dir, {
        "mode": "downloaded",
        "path": str(questions_dir),
        "repo_id": manifest["repo_id"],
        "revision": manifest["revision"],
        "row_count": sum(manifest["configs"][benchmark] for benchmark in options.benchmarks),
        "manifest": str(questions_dir / "source_manifest.json"),
    }


def _prepare_papers(
    options: PreparationOptions,
    questions_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    if options.papers_dir is not None:
        papers_dir = options.papers_dir.expanduser().resolve()
        if not papers_dir.is_dir():
            raise FileNotFoundError(f"papers directory not found: {papers_dir}")
        return papers_dir, {
            "mode": "provided",
            "path": str(papers_dir),
            "paper_count": len(list(papers_dir.glob("*.pdf"))),
        }

    papers_dir = options.papers_cache_dir.expanduser().resolve()
    manifest = download_corpus(
        PaperDownloadOptions(
            questions_dir=questions_dir,
            papers_dir=papers_dir,
            benchmarks=options.benchmarks,
            source_policy=options.source_policy,
            jobs=options.jobs,
            timeout=options.timeout,
            retries=options.retries,
            max_pdf_mb=options.max_pdf_mb,
            min_pdf_bytes=options.min_pdf_bytes,
            user_agent=options.user_agent,
            contact_email=options.contact_email,
            openalex_api_key=options.openalex_api_key,
            openalex_content=options.openalex_content,
            unpaywall_email=options.unpaywall_email,
            crossref=options.crossref,
            retry_failed=options.retry_failed_papers,
            overwrite=options.overwrite_papers,
        )
    )
    return papers_dir, {
        "mode": "downloaded",
        "path": str(papers_dir),
        "manifest": str(papers_dir / "download_manifest.json"),
        "doi_count": manifest["doi_count"],
        "available_count": manifest["downloaded_count"] + manifest["cached_count"],
        "failed_count": manifest["failed_count"],
    }


def prepare_data(options: PreparationOptions) -> dict[str, Any]:
    """Acquire inputs as needed, then materialize only PDF-complete tasks."""
    _validate_options(options)
    questions_dir, question_source = _prepare_questions(options)
    papers_dir, paper_source = _prepare_papers(options, questions_dir)

    if options.build_image:
        build_docker_image(options.image)

    output_dir = options.output_dir or DEFAULT_OUTPUT_DIRS[options.environment_type]
    materialization = materialize(
        MaterializationOptions(
            questions_dir=questions_dir,
            papers_dir=papers_dir,
            output_dir=output_dir,
            benchmarks=options.benchmarks,
            environment_type=options.environment_type,
            image=options.image,
            corpus_view_mode=options.corpus_view_mode,
            missing_paper=options.missing_paper,
            source_policy=options.source_policy,
            allow_internet=options.allow_internet,
            limit_per_benchmark=options.limit_per_benchmark,
            agent_name=options.agent_name,
            overwrite=options.overwrite_tasks,
        )
    )

    preparation = {
        "schema_version": "labbench2_pdf_preparation.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "generator": "environments/labbench2_pdf/prepare_data.py",
        "questions": question_source,
        "papers": paper_source,
        "source_policy": options.source_policy,
        "missing_paper_policy": options.missing_paper,
        "materialization_manifest": str(Path(materialization["output_dir"]) / "materialization_manifest.json"),
        "materialized_tasks": materialization["materialized_tasks"],
        "skipped_rows": len(materialization["skipped_rows"]),
    }
    preparation_path = Path(materialization["output_dir"]) / "preparation_manifest.json"
    preparation_path.write_text(json.dumps(preparation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"preparation": preparation, "materialization": materialization}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_argument_group("input selection")
    inputs.add_argument(
        "--questions-dir",
        type=Path,
        help="use an existing question snapshot instead of downloading one",
    )
    inputs.add_argument(
        "--papers-dir",
        type=Path,
        help="use an existing PDF folder instead of running public DOI downloaders",
    )
    inputs.add_argument("--questions-cache-dir", type=Path, default=DEFAULT_QUESTIONS_DIR)
    inputs.add_argument("--papers-cache-dir", type=Path, default=DEFAULT_PAPERS_DIR)
    inputs.add_argument(
        "--benchmarks",
        nargs="+",
        choices=SUPPORTED_BENCHMARKS,
        default=list(SUPPORTED_BENCHMARKS),
    )
    inputs.add_argument("--hf-repo-id", default=DEFAULT_HF_REPO_ID)
    inputs.add_argument("--hf-revision", default=DEFAULT_HF_REVISION)
    inputs.add_argument("--refresh-questions", action="store_true")

    downloads = parser.add_argument_group("public paper download")
    downloads.add_argument("--jobs", type=int, default=6)
    downloads.add_argument("--timeout", type=float, default=45.0)
    downloads.add_argument("--retries", type=int, default=3)
    downloads.add_argument("--max-pdf-mb", type=int, default=100)
    downloads.add_argument("--min-pdf-bytes", type=int, default=1000)
    downloads.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    downloads.add_argument("--contact-email")
    downloads.add_argument("--openalex-api-key", default=os.environ.get("OPENALEX_API_KEY"))
    downloads.add_argument("--openalex-content", action="store_true")
    downloads.add_argument("--unpaywall-email", default=os.environ.get("UNPAYWALL_EMAIL"))
    downloads.add_argument("--crossref", action=argparse.BooleanOptionalAction, default=True)
    downloads.add_argument("--retry-failed", dest="retry_failed_papers", action="store_true")
    downloads.add_argument("--overwrite-papers", action="store_true")

    tasks = parser.add_argument_group("Harbor task materialization")
    tasks.add_argument("--output-dir", type=Path)
    tasks.add_argument("--environment-type", choices=("docker", "singularity"), default="docker")
    tasks.add_argument("--image", default=DEFAULT_IMAGE)
    tasks.add_argument("--build-docker-image", action="store_true")
    tasks.add_argument("--corpus-view-mode", choices=("hardlink", "copy"), default="copy")
    tasks.add_argument("--missing-paper", choices=("skip", "error"), default="skip")
    tasks.add_argument(
        "--source-policy",
        choices=SOURCE_POLICIES,
        default="all",
        help="require every listed source PDF (default), or at least one with 'any'",
    )
    tasks.add_argument("--allow-internet", action=argparse.BooleanOptionalAction, default=True)
    tasks.add_argument("--limit-per-benchmark", type=int)
    tasks.add_argument("--agent-name", default=DEFAULT_AGENT_NAME)
    tasks.add_argument("--overwrite", dest="overwrite_tasks", action="store_true")
    return parser


def options_from_args(args: argparse.Namespace) -> PreparationOptions:
    return PreparationOptions(
        questions_dir=args.questions_dir,
        papers_dir=args.papers_dir,
        questions_cache_dir=args.questions_cache_dir,
        papers_cache_dir=args.papers_cache_dir,
        output_dir=args.output_dir,
        benchmarks=tuple(args.benchmarks),
        hf_repo_id=args.hf_repo_id,
        hf_revision=args.hf_revision,
        refresh_questions=args.refresh_questions,
        source_policy=args.source_policy,
        jobs=args.jobs,
        timeout=args.timeout,
        retries=args.retries,
        max_pdf_mb=args.max_pdf_mb,
        min_pdf_bytes=args.min_pdf_bytes,
        user_agent=args.user_agent,
        contact_email=args.contact_email,
        openalex_api_key=args.openalex_api_key,
        openalex_content=args.openalex_content,
        unpaywall_email=args.unpaywall_email,
        crossref=args.crossref,
        retry_failed_papers=args.retry_failed_papers,
        overwrite_papers=args.overwrite_papers,
        environment_type=args.environment_type,
        image=args.image,
        build_image=args.build_docker_image,
        corpus_view_mode=args.corpus_view_mode,
        missing_paper=args.missing_paper,
        allow_internet=args.allow_internet,
        limit_per_benchmark=args.limit_per_benchmark,
        agent_name=args.agent_name,
        overwrite_tasks=args.overwrite_tasks,
    )


def main() -> None:
    args = build_parser().parse_args()
    try:
        result = prepare_data(options_from_args(args))
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    preparation = result["preparation"]
    print(
        f"Prepared {preparation['materialized_tasks']} Harbor tasks "
        f"({preparation['skipped_rows']} incomplete questions skipped)."
    )
    print(f"Tasks: {result['materialization']['output_dir']}")


if __name__ == "__main__":
    main()
