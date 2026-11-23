#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize direct, relevant-paper LabBench2 questions as Harbor tasks.

The source question snapshots and PDF corpus remain external inputs. Generated
Harbor tasks contain no retrieval skill, text cache, source hint, or gold answer
outside the task-local verifier assets.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from environments.labbench2_pdf.data_utils import (
    ENV_ROOT,
    SOURCE_POLICIES,
    SUPPORTED_BENCHMARKS,
    build_coverage_audit,
    load_questions,
    question_files_digest,
)


DEFAULT_OUTPUT_DIRS = {
    "docker": ENV_ROOT / "data" / "tasks_docker",
    "singularity": ENV_ROOT / "data" / "tasks_singularity",
}
DEFAULT_DOCKERFILE = ENV_ROOT / "docker" / "labbench2-pdf-runtime.Dockerfile"
DEFAULT_IMAGE = "nemo-gym-labbench2-pdf:2.0"
DEFAULT_AGENT_NAME = "harbor_agent_general"


@dataclass(frozen=True, slots=True)
class MaterializationOptions:
    questions_dir: Path
    papers_dir: Path
    output_dir: Path
    benchmarks: tuple[str, ...] = SUPPORTED_BENCHMARKS
    environment_type: str = "docker"
    image: str = DEFAULT_IMAGE
    corpus_view_mode: str = "copy"
    missing_paper: str = "skip"
    source_policy: str = "all"
    allow_internet: bool = True
    limit_per_benchmark: int | None = None
    agent_name: str = DEFAULT_AGENT_NAME
    overwrite: bool = False


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _validate_output_path(options: MaterializationOptions) -> Path:
    output_dir = options.output_dir.expanduser().resolve()
    forbidden = {Path.cwd().resolve(), Path.cwd().resolve().parent, Path.home().resolve(), Path("/")}
    if output_dir in forbidden or len(output_dir.parts) < 4:
        raise ValueError(f"refusing unsafe output directory: {output_dir}")
    for input_dir in (options.questions_dir, options.papers_dir):
        resolved_input = input_dir.expanduser().resolve()
        if resolved_input == output_dir or resolved_input.is_relative_to(output_dir):
            raise ValueError(f"output directory must not contain input directory: {output_dir}")
    return output_dir


def _prepare_output(options: MaterializationOptions) -> Path:
    output_dir = _validate_output_path(options)
    if output_dir.exists():
        if any(output_dir.iterdir()) and not options.overwrite:
            raise FileExistsError(f"output exists and is not empty; pass --overwrite: {output_dir}")
        if options.overwrite:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _store_paper(source: Path, destination: Path, mode: str) -> None:
    if mode == "hardlink":
        try:
            os.link(source, destination)
        except OSError as exc:
            raise RuntimeError(
                "could not hard-link the generated corpus store; choose --corpus-view-mode copy "
                f"when the corpus and output are on different filesystems: {exc}"
            ) from exc
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError(f"unsupported corpus view mode: {mode}")


def _prepare_corpus_store(
    output_dir: Path,
    papers: Iterable[Path],
    mode: str,
) -> Path:
    store_dir = output_dir / "_corpus_store"
    store_dir.mkdir()
    for source in sorted(set(papers), key=lambda path: path.name):
        _store_paper(source, store_dir / source.name, mode)
    return store_dir


def _prepare_corpus_view(
    output_dir: Path,
    benchmark: str,
    papers: Iterable[Path],
    store_dir: Path,
) -> Path:
    view_dir = output_dir / "_corpus_views" / benchmark / "papers"
    view_dir.mkdir(parents=True)
    for source in sorted(set(papers), key=lambda path: path.name):
        os.link(store_dir / source.name, view_dir / source.name)
    return view_dir


def _instruction(row: dict[str, Any]) -> str:
    return f"""# Scientific literature question

Answer the question from the scientific papers mounted read-only at `/papers`.
PDF command-line tools and Python libraries are installed, including Poppler
utilities and Tesseract OCR. Inspect the PDF files directly.

Write only the final short answer to exactly `/app/answer.txt`. Use a shell
command such as `printf '%s\\n' 'your answer' > /app/answer.txt`; do not use a
patch or file-edit tool, because those tools may rewrite the absolute path as a
relative path. Verify the exact file with `cat /app/answer.txt` before finishing.

Do not include reasoning, citations, labels, or Markdown in that file.

## Question

{row["question"]}
"""


def _task_toml(
    row: dict[str, Any],
    *,
    image: str,
    allow_internet: bool,
    environment_type: str,
) -> str:
    return f"""version = "1.0"

[metadata]
benchmark = "labbench2"
tag = {_toml_string(str(row["tag"]))}
version = {_toml_string(str(row.get("version") or ""))}
item_id = {_toml_string(str(row["id"]))}
corpus_scope = "relevant"
pdf_access_mode = "direct"
container_profile = {_toml_string(environment_type)}

[agent]
timeout_sec = 1800.0

[verifier]
timeout_sec = 300.0

[verifier.env]
OPENAI_API_KEY = "${{JUDGE_API_KEY}}"
OPENAI_BASE_URL = "${{JUDGE_BASE_URL}}"
JUDGE_MODEL = "${{JUDGE_MODEL}}"
JUDGE_MAX_TOKENS = "2048"

[environment]
build_timeout_sec = 900.0
cpus = 2
memory_mb = 4096
storage_mb = 10240
gpus = 0
allow_internet = {str(allow_internet).lower()}
docker_image = {_toml_string(image)}
"""


def _docker_compose(image: str, papers_dir: Path) -> str:
    source = json.dumps(str(papers_dir.resolve()))
    return f"""services:
  main:
    image: {json.dumps(image)}
    pull_policy: never
    command: ["sh", "-c", "sleep infinity"]
    network_mode: ${{NETWORK_MODE:-bridge}}
    environment:
      - TEST_DIR=${{TEST_DIR}}
    volumes:
      - ${{HOST_VERIFIER_LOGS_PATH}}:${{ENV_VERIFIER_LOGS_PATH}}
      - ${{HOST_AGENT_LOGS_PATH}}:${{ENV_AGENT_LOGS_PATH}}
      - type: bind
        source: {source}
        target: "/papers"
        read_only: true
    deploy:
      resources:
        limits:
          cpus: ${{CPUS}}
          memory: ${{MEMORY}}
"""


def _singularity_setup() -> str:
    return """#!/bin/bash
# Expose the staged, task-local hard-link view at the same path used by Docker.
set -euo pipefail
if ! python3 -c "import fastapi, uvicorn" 2>/dev/null; then
  echo "The runtime image must include fastapi and uvicorn." >&2
  exit 1
fi
rm -rf /papers
ln -s "${HARBOR_STAGING}/papers" /papers
"""


def _test_sh() -> str:
    return """#!/bin/sh
set -eu
mkdir -p /logs/verifier
python3 /tests/verifier.py
"""


def _solution(ideal: str) -> str:
    return f"#!/bin/sh\nset -eu\nprintf '%s\\n' {shlex.quote(ideal)} > /app/answer.txt\n"


def _make_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _stage_singularity_papers(view_dir: Path, task_environment_dir: Path) -> None:
    papers_dir = task_environment_dir / "files" / "papers"
    papers_dir.mkdir(parents=True)
    for source in sorted(view_dir.glob("*.pdf")):
        os.link(source, papers_dir / source.name)
    setup_path = task_environment_dir / "files" / "setup.sh"
    setup_path.write_text(_singularity_setup(), encoding="utf-8")
    _make_executable(setup_path)


def _safe_task_name(tag: str, index: int, item_id: str) -> str:
    return f"{tag}-{index:04d}-{item_id[:8].casefold()}"


def _write_task(
    task_dir: Path,
    row: dict[str, Any],
    *,
    task_name: str,
    view_dir: Path,
    options: MaterializationOptions,
) -> None:
    environment_dir = task_dir / "environment"
    tests_dir = task_dir / "tests"
    solution_dir = task_dir / "solution"
    environment_dir.mkdir(parents=True)
    tests_dir.mkdir()
    solution_dir.mkdir()

    (task_dir / "instruction.md").write_text(_instruction(row), encoding="utf-8")
    (task_dir / "task.toml").write_text(
        _task_toml(
            row,
            image=options.image,
            allow_internet=options.allow_internet,
            environment_type=options.environment_type,
        ),
        encoding="utf-8",
    )
    if options.environment_type == "docker":
        (environment_dir / "docker-compose.yaml").write_text(
            _docker_compose(options.image, view_dir),
            encoding="utf-8",
        )
    else:
        _stage_singularity_papers(view_dir, environment_dir)

    verifier_source = (ENV_ROOT / "verifier.py").read_text(encoding="utf-8")
    (tests_dir / "verifier.py").write_text(verifier_source, encoding="utf-8")
    _write_json(
        tests_dir / "gold_metadata.json",
        {
            "benchmark": "labbench2",
            "tag": row["tag"],
            "version": row.get("version"),
            "item_id": row["id"],
            "task_name": task_name,
            "question": row["question"],
            "ideal": row["ideal"],
        },
    )
    test_path = tests_dir / "test.sh"
    test_path.write_text(_test_sh(), encoding="utf-8")
    _make_executable(test_path)
    solve_path = solution_dir / "solve.sh"
    solve_path.write_text(_solution(str(row["ideal"])), encoding="utf-8")
    _make_executable(solve_path)


def build_rollout_input(task_rows: Iterable[dict[str, Any]], agent_name: str) -> list[dict[str, Any]]:
    return [
        {
            "task_name": task["name"],
            "responses_create_params": {"input": []},
            "agent_ref": {"name": agent_name},
        }
        for task in task_rows
    ]


def build_example_input(
    task_rows: Iterable[dict[str, Any]],
    agent_name: str,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Select a deterministic, benchmark-balanced smoke sample."""
    by_benchmark: dict[str, list[dict[str, Any]]] = {}
    for task in task_rows:
        by_benchmark.setdefault(str(task["tag"]), []).append(task)

    selected: list[dict[str, Any]] = []
    offset = 0
    while len(selected) < limit:
        added = False
        for benchmark_tasks in by_benchmark.values():
            if offset < len(benchmark_tasks):
                selected.append(benchmark_tasks[offset])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        offset += 1
    return build_rollout_input(selected, agent_name)


def _select_rows(
    rows_by_benchmark: dict[str, list[dict[str, Any]]],
    coverage_by_id: dict[str, dict[str, Any]],
    *,
    missing_paper: str,
    limit_per_benchmark: int | None,
) -> dict[str, list[dict[str, Any]]]:
    if limit_per_benchmark is None:
        return rows_by_benchmark

    selected_by_benchmark: dict[str, list[dict[str, Any]]] = {}
    for benchmark, rows in rows_by_benchmark.items():
        if missing_paper == "error":
            selected_by_benchmark[benchmark] = rows[:limit_per_benchmark]
            continue

        selected: list[dict[str, Any]] = []
        answerable = 0
        for row in rows:
            selected.append(row)
            if not coverage_by_id[str(row["id"])]["missing"]:
                answerable += 1
            if answerable == limit_per_benchmark:
                break
        selected_by_benchmark[benchmark] = selected
    return selected_by_benchmark


def materialize(options: MaterializationOptions) -> dict[str, Any]:
    questions_dir = options.questions_dir.expanduser().resolve()
    papers_dir = options.papers_dir.expanduser().resolve()
    if not questions_dir.is_dir():
        raise FileNotFoundError(f"questions directory not found: {questions_dir}")
    if not papers_dir.is_dir():
        raise FileNotFoundError(f"papers directory not found: {papers_dir}")
    if options.environment_type not in {"docker", "singularity"}:
        raise ValueError(f"unsupported environment type: {options.environment_type}")
    if options.corpus_view_mode not in {"hardlink", "copy"}:
        raise ValueError(f"unsupported corpus view mode: {options.corpus_view_mode}")
    if options.missing_paper not in {"skip", "error"}:
        raise ValueError(f"unsupported missing-paper policy: {options.missing_paper}")
    if options.source_policy not in SOURCE_POLICIES:
        raise ValueError(f"unsupported source policy: {options.source_policy}")
    if options.limit_per_benchmark is not None and options.limit_per_benchmark < 1:
        raise ValueError("limit_per_benchmark must be at least 1")
    if len(set(options.benchmarks)) != len(options.benchmarks):
        raise ValueError("benchmarks must be unique")
    unsupported = sorted(set(options.benchmarks) - set(SUPPORTED_BENCHMARKS))
    if unsupported:
        raise ValueError(f"unsupported benchmarks: {unsupported}")

    all_rows_by_benchmark = load_questions(questions_dir, options.benchmarks)
    full_coverage, _ = build_coverage_audit(
        all_rows_by_benchmark,
        papers_dir,
        source_policy=options.source_policy,
    )
    rows_by_benchmark = _select_rows(
        all_rows_by_benchmark,
        {item["item_id"]: item for item in full_coverage["items"]},
        missing_paper=options.missing_paper,
        limit_per_benchmark=options.limit_per_benchmark,
    )
    coverage, resolved_by_id = build_coverage_audit(
        rows_by_benchmark,
        papers_dir,
        source_policy=options.source_policy,
    )
    if options.missing_paper == "error" and coverage["missing_count"]:
        first_missing = next(item for item in coverage["items"] if item["missing"])
        raise RuntimeError(
            f"{coverage['missing_count']} rows do not satisfy the PDF source policy; "
            f"first incomplete item: {first_missing['item_id']}"
        )

    coverage_by_id = {item["item_id"]: item for item in coverage["items"]}
    relevant_by_benchmark: dict[str, set[Path]] = {}
    for benchmark, rows in rows_by_benchmark.items():
        relevant = {
            path
            for row in rows
            if not coverage_by_id[str(row["id"])]["missing"]
            for path in resolved_by_id[str(row["id"])]
        }
        relevant_by_benchmark[benchmark] = relevant
    if not any(relevant_by_benchmark.values()):
        raise RuntimeError("no questions satisfy the PDF source policy")

    output_dir = _prepare_output(options)
    corpus_store = _prepare_corpus_store(
        output_dir,
        {path for papers in relevant_by_benchmark.values() for path in papers},
        options.corpus_view_mode,
    )
    views: dict[str, Path] = {}
    for benchmark, relevant in relevant_by_benchmark.items():
        views[benchmark] = _prepare_corpus_view(
            output_dir,
            benchmark,
            relevant,
            corpus_store,
        )

    task_rows: list[dict[str, Any]] = []
    skipped_rows = [item for item in coverage["items"] if item["missing"]]
    source_indices = {
        str(row["id"]): index for rows in all_rows_by_benchmark.values() for index, row in enumerate(rows, 1)
    }
    for benchmark, rows in rows_by_benchmark.items():
        for row in rows:
            item_id = str(row["id"])
            if coverage_by_id[item_id]["missing"]:
                continue
            task_name = _safe_task_name(benchmark, source_indices[item_id], item_id)
            _write_task(
                output_dir / task_name,
                row,
                task_name=task_name,
                view_dir=views[benchmark],
                options=options,
            )
            task_rows.append({"name": task_name, "path": task_name, "tag": benchmark, "item_id": item_id})

    registry = [
        {
            "name": "EdisonScientific/labbench2-pdf-direct-relevant",
            "description": "LitQA3, FigQA2, and TableQA2 over benchmark-relevant PDF corpora",
            "metrics": [{"type": "mean"}],
            "tasks": [{"name": task["name"], "path": task["path"]} for task in task_rows],
        }
    ]
    _write_json(output_dir / "registry.json", registry)
    rollout_rows = build_rollout_input(task_rows, options.agent_name)
    _write_jsonl(output_dir / "rollout_input.jsonl", rollout_rows)
    _write_jsonl(output_dir / "example_input.jsonl", build_example_input(task_rows, options.agent_name))
    for benchmark in options.benchmarks:
        benchmark_task_names = {task["name"] for task in task_rows if task["tag"] == benchmark}
        _write_jsonl(
            output_dir / f"{benchmark}_input.jsonl",
            [row for row in rollout_rows if row["task_name"] in benchmark_task_names],
        )
    _write_json(output_dir / "coverage_audit.json", coverage)

    manifest = {
        "schema_version": "labbench2_pdf_materialization.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "generator": "environments/labbench2_pdf/prepare.py",
        "questions_dir": str(questions_dir),
        "questions_sha256": question_files_digest(questions_dir, options.benchmarks),
        "papers_dir": str(papers_dir),
        "output_dir": str(output_dir),
        "benchmarks": list(options.benchmarks),
        "corpus_scope": "relevant",
        "pdf_access_mode": "direct",
        "environment_type": options.environment_type,
        "runtime_image": options.image,
        "corpus_view_mode": options.corpus_view_mode,
        "corpus_store": {
            "path": str(corpus_store),
            "paper_count": len(list(corpus_store.glob("*.pdf"))),
        },
        "allow_internet": options.allow_internet,
        "missing_paper_policy": options.missing_paper,
        "source_policy": options.source_policy,
        "selected_rows": sum(len(rows) for rows in rows_by_benchmark.values()),
        "materialized_tasks": len(task_rows),
        "skipped_rows": skipped_rows,
        "coverage": {key: value for key, value in coverage.items() if key != "items"},
        "corpus_views": {
            benchmark: {
                "path": str(path),
                "paper_count": len(list(path.glob("*.pdf"))),
            }
            for benchmark, path in views.items()
        },
        "tasks": task_rows,
    }
    _write_json(output_dir / "materialization_manifest.json", manifest)
    return manifest


def build_docker_image(image: str) -> None:
    subprocess.run(
        [
            "docker",
            "build",
            "--tag",
            image,
            "--file",
            str(DEFAULT_DOCKERFILE),
            str(ENV_ROOT),
        ],
        check=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions-dir", type=Path, required=True)
    parser.add_argument("--papers-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=SUPPORTED_BENCHMARKS,
        default=list(SUPPORTED_BENCHMARKS),
    )
    parser.add_argument("--environment-type", choices=("docker", "singularity"), default="docker")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--build-docker-image", action="store_true")
    parser.add_argument("--corpus-view-mode", choices=("hardlink", "copy"), default="copy")
    parser.add_argument("--missing-paper", choices=("skip", "error"), default="skip")
    parser.add_argument(
        "--source-policy",
        choices=SOURCE_POLICIES,
        default="all",
        help="require PDFs for all listed sources (default) or at least one source",
    )
    parser.add_argument("--allow-internet", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit-per-benchmark", type=int)
    parser.add_argument("--agent-name", default=DEFAULT_AGENT_NAME)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def options_from_args(args: argparse.Namespace) -> MaterializationOptions:
    output_dir = args.output_dir or DEFAULT_OUTPUT_DIRS[args.environment_type]
    return MaterializationOptions(
        questions_dir=args.questions_dir,
        papers_dir=args.papers_dir,
        output_dir=output_dir,
        benchmarks=tuple(args.benchmarks),
        environment_type=args.environment_type,
        image=args.image,
        corpus_view_mode=args.corpus_view_mode,
        missing_paper=args.missing_paper,
        source_policy=args.source_policy,
        allow_internet=args.allow_internet,
        limit_per_benchmark=args.limit_per_benchmark,
        agent_name=args.agent_name,
        overwrite=args.overwrite,
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.build_docker_image:
        if args.environment_type != "docker":
            raise SystemExit("--build-docker-image is only valid with --environment-type docker")
        build_docker_image(args.image)
    try:
        manifest = materialize(options_from_args(args))
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        f"Materialized {manifest['materialized_tasks']} tasks at {manifest['output_dir']} "
        f"({len(manifest['skipped_rows'])} missing-paper rows skipped)."
    )
    print("Condition: direct PDF access, benchmark-relevant corpus, no retrieval helper or text cache.")


if __name__ == "__main__":
    main()
