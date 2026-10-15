# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared question and paper-resolution helpers for LabBench2 PDF tasks."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


ENV_ROOT = Path(__file__).resolve().parent
DEFAULT_QUESTIONS_DIR = ENV_ROOT / "data" / "source" / "questions"
DEFAULT_PAPERS_DIR = ENV_ROOT / "data" / "source" / "papers"
SUPPORTED_BENCHMARKS = ("litqa3", "figqa2", "tableqa2")
QUESTION_FIELDS = ("id", "tag", "version", "question", "ideal", "sources")
REQUIRED_QUESTION_FIELDS = ("id", "tag", "question", "ideal", "sources")
SOURCE_POLICIES = ("all", "any")


def source_to_doi(source: str) -> str:
    """Return a DOI-like identifier from a LabBench2 source URL."""
    value = unquote(source.strip()).rstrip("/")
    parsed = urlparse(value)
    if parsed.netloc.casefold() in {"doi.org", "dx.doi.org"}:
        value = parsed.path.lstrip("/")
    if re.match(r"^10\.\d{4}-", value):
        value = value.replace("-", "/", 1)
    return value


def doi_filename_candidates(doi_or_source: str) -> tuple[str, ...]:
    doi = source_to_doi(doi_or_source)
    candidates = {
        doi.replace("/", "_") + ".pdf",
        re.sub(r"[()/]", "_", doi) + ".pdf",
    }
    return tuple(sorted(re.sub(r"_+", "_", value) for value in candidates))


def build_paper_lookup(papers_dir: Path) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    for path in sorted(papers_dir.glob("*.pdf")):
        key = path.name.casefold()
        if key in lookup:
            raise ValueError(f"case-insensitive PDF filename collision: {lookup[key]} and {path}")
        lookup[key] = path
    return lookup


def resolve_source_papers(sources: Iterable[str], lookup: dict[str, Path]) -> list[Path]:
    resolved: list[Path] = []
    for source in sources:
        for candidate in doi_filename_candidates(source):
            match = lookup.get(candidate.casefold())
            if match is not None:
                resolved.append(match)
                break
    return list(dict.fromkeys(resolved))


def load_questions(questions_dir: Path, benchmarks: Iterable[str]) -> dict[str, list[dict[str, Any]]]:
    rows_by_benchmark: dict[str, list[dict[str, Any]]] = {}
    seen_ids: set[str] = set()
    for benchmark in benchmarks:
        path = questions_dir / f"{benchmark}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"question snapshot not found: {path}")
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            missing_fields = [field for field in REQUIRED_QUESTION_FIELDS if field not in row]
            if missing_fields:
                raise ValueError(f"missing fields at {path}:{line_number}: {missing_fields}")
            if row["tag"] != benchmark:
                raise ValueError(f"tag mismatch at {path}:{line_number}: expected {benchmark!r}, got {row['tag']!r}")
            item_id = str(row["id"])
            if item_id in seen_ids:
                raise ValueError(f"duplicate question id at {path}:{line_number}: {item_id}")
            if not isinstance(row["sources"], list) or not all(isinstance(value, str) for value in row["sources"]):
                raise ValueError(f"sources must be a list of strings at {path}:{line_number}")
            if not row["sources"]:
                raise ValueError(f"sources must not be empty at {path}:{line_number}")
            seen_ids.add(item_id)
            rows.append(row)
        rows_by_benchmark[benchmark] = rows
    return rows_by_benchmark


def question_files_digest(questions_dir: Path, benchmarks: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for benchmark in benchmarks:
        path = questions_dir / f"{benchmark}.jsonl"
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_coverage_audit(
    rows_by_benchmark: dict[str, list[dict[str, Any]]],
    papers_dir: Path,
    *,
    source_policy: str = "all",
) -> tuple[dict[str, Any], dict[str, list[Path]]]:
    """Resolve papers and report which questions satisfy ``source_policy``.

    ``all`` requires a PDF for every source listed on a question. ``any`` keeps
    compatibility with the standalone materializer by accepting a question when
    at least one listed source resolves.
    """
    if source_policy not in SOURCE_POLICIES:
        raise ValueError(f"unsupported source policy: {source_policy}")

    lookup = build_paper_lookup(papers_dir)
    resolved_by_id: dict[str, list[Path]] = {}
    items: list[dict[str, Any]] = []
    by_benchmark: dict[str, dict[str, int]] = {}
    for benchmark, rows in rows_by_benchmark.items():
        counts = {"rows": 0, "answerable": 0, "missing": 0, "partial": 0, "relevant_papers": 0}
        relevant_names: set[str] = set()
        for row in rows:
            item_id = str(row["id"])
            sources = [str(value) for value in row["sources"]]
            resolved: list[Path] = []
            missing_sources: list[str] = []
            for source in sources:
                match = resolve_source_papers((source,), lookup)
                if match:
                    resolved.extend(match)
                else:
                    missing_sources.append(source)
            resolved = list(dict.fromkeys(resolved))
            resolved_by_id[item_id] = resolved
            partial = bool(resolved and missing_sources)
            missing = not resolved or (source_policy == "all" and bool(missing_sources))
            counts["rows"] += 1
            counts["missing" if missing else "answerable"] += 1
            counts["partial"] += int(partial)
            if not missing:
                relevant_names.update(path.name for path in resolved)
            items.append(
                {
                    "item_id": item_id,
                    "tag": benchmark,
                    "sources": sources,
                    "source_papers": [path.name for path in resolved],
                    "missing_sources": missing_sources,
                    "partial": partial,
                    "missing": missing,
                }
            )
        counts["relevant_papers"] = len(relevant_names)
        by_benchmark[benchmark] = counts
    audit = {
        "schema_version": "labbench2_pdf_coverage.v1",
        "papers_dir": str(papers_dir),
        "paper_count": len(lookup),
        "source_policy": source_policy,
        "row_count": len(items),
        "answerable_count": sum(not item["missing"] for item in items),
        "missing_count": sum(item["missing"] for item in items),
        "partial_count": sum(item["partial"] for item in items),
        "by_benchmark": by_benchmark,
        "items": items,
    }
    return audit, resolved_by_id
