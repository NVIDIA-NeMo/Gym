# SPDX-FileCopyrightText: Copyright (c) 2026 Harvey AI
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Rubric scoring adapted from Harvey LAB at the pinned source revision.

Upstream source:
https://github.com/harveyai/harvey-labs/blob/f46ef86e4788545622db25dcffa3aebb7a139929/evaluation/scoring.py

Intentional differences from upstream:
- uses the configured OpenAI-compatible judge instead of provider SDKs;
- creates isolated judges for parallel criteria;
- records per-criterion transcripts and judge errors;
- fails missing or unreadable deliverables without calling the judge; and
- uses deterministic filename matching without Anthropic-specific LLM fallback.
"""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable

from .judge import PROMPT_TEMPLATE, VERDICT_SCHEMA, OpenAICompatibleJudge, write_transcript_event


SKIP_DIRS = {"node_modules", ".npm", "__pycache__", ".git", "venv", ".venv"}
SKIP_EXTENSIONS = {".lock", ".map"}
SKIP_FILES = {"package-lock.json"}


class DocxTrackChanges(StrEnum):
    ACCEPT = "accept"
    ALL = "all"


@dataclass
class CriterionResult:
    id: str
    title: str
    verdict: str
    reasoning: str = ""
    judge_error: bool = False
    error_type: str | None = None


def _read_file_as_text(
    path: Path,
    *,
    track_changes: DocxTrackChanges = DocxTrackChanges.ACCEPT,
) -> str:
    suffix = path.suffix.lower()
    try:
        if suffix == ".docx":
            try:
                result = subprocess.run(
                    [
                        "pandoc",
                        str(path),
                        "-t",
                        "markdown",
                        "--wrap=none",
                        f"--track-changes={track_changes.value}",
                    ],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=30,
                )
                if result.returncode == 0:
                    return result.stdout
            except FileNotFoundError:
                pass
            return _read_docx_fallback(path)
        if suffix == ".xlsx":
            import pandas as pd

            sheets = pd.read_excel(path, sheet_name=None)
            parts = []
            for sheet_name, df in sheets.items():
                parts.append(f"=== Sheet: {sheet_name} ===")
                parts.append(df.to_string(index=False))
            return "\n".join(parts)
        if suffix == ".pptx":
            from markitdown import MarkItDown

            return MarkItDown().convert(str(path)).text_content
        if suffix == ".pdf":
            import pdfplumber

            parts = []
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        parts.append(text)
                    for table in page.extract_tables():
                        for row in table:
                            parts.append("\t".join(cell if cell else "" for cell in row))
                        parts.append("")
            return "\n".join(parts)
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"(binary file: {path.name})"
    except Exception as exc:
        return f"(error reading {path.name}: {exc})"


def _read_docx_fallback(path: Path) -> str:
    from docx import Document

    document = Document(path)
    parts = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            parts.append("\t".join(cell.text for cell in row.cells))
        parts.append("")
    return "\n".join(parts)


def _is_unreadable_output(content: str) -> bool:
    return content.startswith("(error reading ") or content.startswith("(binary file:")


def _is_thread_export(filename: str) -> bool:
    return Path(filename).stem.lower() == "output"


def _load_all_output(output_dir: Path) -> str:
    sections = []
    if output_dir.exists():
        for path in sorted(output_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(output_dir)
            if any(part in SKIP_DIRS for part in rel.parts):
                continue
            if path.suffix in SKIP_EXTENSIONS or path.name in SKIP_FILES:
                continue
            sections.append(f"## {rel}\n{_read_file_as_text(path)}")
    return "\n\n".join(sections) if sections else "(No agent output found)"


def _fuzzy_match_filename(expected: str, candidates: list[str]) -> str | None:
    expected_words = set(Path(expected).stem.lower().replace("-", " ").replace("_", " ").split())
    best_match = None
    best_score = 0
    for candidate in candidates:
        candidate_words = set(Path(candidate).stem.lower().replace("-", " ").replace("_", " ").split())
        score = len(expected_words & candidate_words)
        if score > best_score:
            best_match = candidate
            best_score = score
    return best_match


def _match_deliverables(deliverables_map: dict[str, str], actual_files: list[str]) -> dict[str, str]:
    resolved = {}
    used = set()
    for name, expected in deliverables_map.items():
        if expected in actual_files:
            resolved[name] = expected
            used.add(expected)
            continue
        expected_ext = Path(expected).suffix.lower()
        candidates = [
            filename
            for filename in actual_files
            if filename not in used
            and not _is_thread_export(filename)
            and Path(filename).suffix.lower() == expected_ext
        ]
        if len(candidates) == 1:
            resolved[name] = candidates[0]
            used.add(candidates[0])
            continue
        match = _fuzzy_match_filename(expected, candidates)
        resolved[name] = match or expected
        if match:
            used.add(match)
    return resolved


def score_rubric(
    *,
    criteria: list[dict[str, Any]],
    run_dir: Path,
    judge: OpenAICompatibleJudge,
    task_desc: str,
    judge_max_tokens: int = 1024,
    transcript_path: Path | None = None,
    judge_factory: Callable[[], OpenAICompatibleJudge] | None = None,
    parallelism: int = 1,
) -> dict[str, Any]:
    output_dir = run_dir / "output"
    filenames = {deliverable for criterion in criteria for deliverable in criterion.get("deliverables", [])}
    deliverables_map = {filename: filename for filename in filenames} if filenames else None
    if deliverables_map and output_dir.exists():
        actual_files = [path.relative_to(output_dir).as_posix() for path in output_dir.rglob("*") if path.is_file()]
        resolved_map = _match_deliverables(deliverables_map, actual_files)
    else:
        resolved_map = None

    needs_full_output = any(not (criterion.get("deliverables") and resolved_map) for criterion in criteria)
    full_output = _load_all_output(output_dir) if needs_full_output else None

    def score_one(criterion: dict[str, Any]) -> dict[str, Any]:
        criterion_deliverables = criterion.get("deliverables", [])
        trace_context = {
            "criterion_id": criterion["id"],
            "criterion_title": criterion["title"],
            "deliverables": criterion_deliverables,
        }
        write_transcript_event(transcript_path, {"type": "criterion_start", **trace_context})
        if criterion_deliverables and resolved_map:
            sections = []
            deliverable_errors = []
            for name in criterion_deliverables:
                filename = resolved_map[name]
                path = output_dir / filename
                if not path.exists():
                    deliverable_errors.append(f"missing deliverable: {filename}")
                    continue
                include_redlines = criterion.get("evaluation_options", {}).get("include_docx_redlines", False)
                track_changes = DocxTrackChanges.ALL if include_redlines else DocxTrackChanges.ACCEPT
                content = _read_file_as_text(path, track_changes=track_changes)
                if _is_unreadable_output(content):
                    deliverable_errors.append(f"invalid or unreadable deliverable {filename}: {content}")
                    continue
                sections.append(f"## Agent Output: {name}\n{content}")
            if deliverable_errors:
                criterion_result = asdict(
                    CriterionResult(
                        id=criterion["id"],
                        title=criterion["title"],
                        verdict="fail",
                        reasoning="; ".join(deliverable_errors),
                    )
                )
                write_transcript_event(
                    transcript_path,
                    {
                        "type": "criterion_complete",
                        **trace_context,
                        "resolved_deliverables": {name: resolved_map[name] for name in criterion_deliverables},
                        "skipped_judge": True,
                        "deliverable_errors": deliverable_errors,
                        "result": criterion_result,
                    },
                )
                return criterion_result
            agent_output = "\n\n".join(sections) if sections else "(No agent output found)"
        else:
            agent_output = full_output or "(No agent output found)"

        variables = {
            "task_description": task_desc,
            "agent_output": agent_output,
            "criterion_title": criterion["title"],
            "match_criteria": criterion["match_criteria"],
        }
        prompt = PROMPT_TEMPLATE.format(**variables)
        criterion_judge: OpenAICompatibleJudge | None = None
        try:
            criterion_judge = judge_factory() if judge_factory is not None else judge
            if hasattr(criterion_judge, "set_trace_context"):
                criterion_judge.set_trace_context(
                    {
                        **trace_context,
                        "resolved_deliverables": (
                            {name: resolved_map[name] for name in criterion_deliverables}
                            if criterion_deliverables and resolved_map
                            else {}
                        ),
                    }
                )
            if hasattr(criterion_judge, "evaluate_prompt"):
                result = criterion_judge.evaluate_prompt(
                    prompt,
                    VERDICT_SCHEMA,
                    max_tokens=judge_max_tokens,
                )
                raw_response = getattr(criterion_judge, "last_raw_response", None)
                structured = getattr(criterion_judge, "last_structured", None)
            else:
                result = criterion_judge.evaluate(variables)
                raw_response = None
                structured = None
        except Exception as exc:
            criterion_result = asdict(
                CriterionResult(
                    id=criterion["id"],
                    title=criterion["title"],
                    verdict="fail",
                    reasoning=f"Judge error: {type(exc).__name__}: {exc}",
                    judge_error=True,
                    error_type=type(exc).__name__,
                )
            )
            write_transcript_event(
                transcript_path,
                {
                    "type": "criterion_complete",
                    **trace_context,
                    "resolved_deliverables": (
                        {name: resolved_map[name] for name in criterion_deliverables}
                        if criterion_deliverables and resolved_map
                        else {}
                    ),
                    "prompt": prompt,
                    "raw_response": getattr(criterion_judge, "last_raw_response", None),
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "judge_error": True,
                    "result": criterion_result,
                },
            )
            return criterion_result

        criterion_result = asdict(
            CriterionResult(
                id=criterion["id"],
                title=criterion["title"],
                verdict=str(result.get("verdict", "fail")).lower(),
                reasoning=str(result.get("reasoning", "")),
            )
        )
        write_transcript_event(
            transcript_path,
            {
                "type": "criterion_complete",
                **trace_context,
                "resolved_deliverables": (
                    {name: resolved_map[name] for name in criterion_deliverables}
                    if criterion_deliverables and resolved_map
                    else {}
                ),
                "prompt": prompt,
                "raw_response": raw_response,
                "used_structured_response_format": structured,
                "parsed_response": result,
                "result": criterion_result,
            },
        )
        return criterion_result

    worker_count = max(1, min(int(parallelism), len(criteria) or 1))
    if worker_count == 1:
        criteria_results = [score_one(criterion) for criterion in criteria]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            criteria_results = list(pool.map(score_one, criteria))

    n_criteria = len(criteria_results)
    n_passed = sum(1 for item in criteria_results if item["verdict"] == "pass")
    judge_error_count = sum(1 for item in criteria_results if item.get("judge_error"))
    all_pass = n_criteria > 0 and n_passed == n_criteria
    return {
        "score": 1.0 if all_pass else 0.0,
        "max_score": 1.0,
        "summary": (
            f"{n_passed}/{n_criteria} criteria passed."
            + ("  ALL-PASS." if all_pass else f"  Missed {n_criteria - n_passed} - task FAIL.")
        ),
        "all_pass": all_pass,
        "n_criteria": n_criteria,
        "n_passed": n_passed,
        "judge_error_count": judge_error_count,
        "criteria_results": criteria_results,
    }
