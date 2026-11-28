# SPDX-FileCopyrightText: Copyright (c) 2026 Harvey AI
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# Original Harvey LAB portions remain under MIT; NVIDIA modifications follow.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Harvey LAB source (MIT): https://github.com/harveyai/harvey-labs/blob/
# f46ef86e4788545622db25dcffa3aebb7a139929/evaluation/scoring.py
"""Rubric scoring adapted from Harvey LAB at the pinned source revision.

Upstream source:
https://github.com/harveyai/harvey-labs/blob/9e9fa7fe126103de1fceb2227b7c70bf59b67f73/lab_core/evaluation/scoring.py

Intentional differences from upstream:
- uses the configured OpenAI-compatible judge instead of provider SDKs;
- creates isolated judges for parallel criteria;
- records per-criterion transcripts and judge errors;
- records missing deliverables as failures and extraction errors as ungraded criteria;
- resolves verified task document IDs to filenames in judge prompts; and
- uses deterministic filename matching without Anthropic-specific LLM fallback.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable
from xml.etree import ElementTree

from .judge import PROMPT_TEMPLATE, VERDICT_SCHEMA, OpenAICompatibleJudge, write_transcript_event


SKIP_DIRS = {"node_modules", ".npm", "__pycache__", ".git", "venv", ".venv"}
SKIP_EXTENSIONS = {".lock", ".map"}
SKIP_FILES = {"package-lock.json"}
# Keep coverage ranges such as DOC_001 through DOC_025 intact: they describe
# the whole production, rather than a single document's identity.
_DOCUMENT_ID_REFERENCE = re.compile(r"\bDOC_[0-9]{3}(?:\s+through\s+DOC_[0-9]{3})?\b")


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


_WORDML_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_COMMENT_PASSAGE_MAX_CHARS = 200
# pandoc's markdown writer prints a Word comment as a span opening with `{.comment-start id="<w:id>"`.
_PANDOC_COMMENT_START_RE = re.compile(r'\{\.comment-start id="([^"]*)"')
# Claude model that matches deliverable names to output files when name matching fails.


@dataclass(frozen=True)
class _DocxComment:
    comment_id: str
    author: str
    text: str
    # Accepted text inside the comment's range; None when the comment has no range.
    passage: str | None


def _docx_comment_passages(document_root: ElementTree.Element) -> dict[str, str]:
    """Map each comment range id in a .docx body (`word/document.xml`) to the text inside the range.

    The text leaves out tracked deletions and moved-from runs and is cut to
    `_COMMENT_PASSAGE_MAX_CHARS` characters.
    """
    removed_text = {
        run_text
        for tag in ("del", "moveFrom")
        for container in document_root.iter(f"{_WORDML_NS}{tag}")
        for run_text in container.iter(f"{_WORDML_NS}t")
    }
    open_ids: set[str] = set()
    chunks: dict[str, list[str]] = {}
    for element in document_root.iter():
        comment_id = element.get(f"{_WORDML_NS}id")
        if element.tag == f"{_WORDML_NS}commentRangeStart" and comment_id is not None:
            open_ids.add(comment_id)
            chunks.setdefault(comment_id, [])
        elif element.tag == f"{_WORDML_NS}commentRangeEnd":
            open_ids.discard(comment_id or "")
        elif element.tag == f"{_WORDML_NS}t" and element not in removed_text:
            for open_id in open_ids:
                chunks[open_id].append(element.text or "")
        elif element.tag in (f"{_WORDML_NS}p", f"{_WORDML_NS}tab", f"{_WORDML_NS}br", f"{_WORDML_NS}cr"):
            for open_id in open_ids:
                chunks[open_id].append(" ")
    passages: dict[str, str] = {}
    for comment_id, parts in chunks.items():
        passage = " ".join("".join(parts).split())
        if len(passage) > _COMMENT_PASSAGE_MAX_CHARS:
            passage = passage[:_COMMENT_PASSAGE_MAX_CHARS].rstrip() + "..."
        passages[comment_id] = passage
    return passages


def _read_docx_comments(path: Path) -> list[_DocxComment]:
    """Read the Word margin comments that a .docx file's body references with `w:commentReference`.

    Returns an empty list when the file is not a readable .docx package.
    """
    try:
        with zipfile.ZipFile(path) as package:
            names = package.namelist()
            if "word/comments.xml" not in names or "word/document.xml" not in names:
                return []
            comments_root = ElementTree.fromstring(package.read("word/comments.xml"))
            document_root = ElementTree.fromstring(package.read("word/document.xml"))
    except Exception:
        return []
    passages = _docx_comment_passages(document_root)
    # The reference is a comment's position in the document. The file format lets readers
    # ignore a comment without a reference, along with any comment range it has.
    referenced_ids = {
        reference.get(f"{_WORDML_NS}id") for reference in document_root.iter(f"{_WORDML_NS}commentReference")
    }
    comments: list[_DocxComment] = []
    for comment in comments_root.iter(f"{_WORDML_NS}comment"):
        comment_id = comment.get(f"{_WORDML_NS}id")
        if comment_id is None or comment_id not in referenced_ids:
            continue
        paragraphs = (
            "".join(run_text.text or "" for run_text in paragraph.iter(f"{_WORDML_NS}t"))
            for paragraph in comment.iter(f"{_WORDML_NS}p")
        )
        text = " ".join(" ".join(paragraphs).split())
        if not text:
            continue
        author = (comment.get(f"{_WORDML_NS}author") or "").strip()
        comments.append(
            _DocxComment(comment_id=comment_id, author=author, text=text, passage=passages.get(comment_id))
        )
    return comments


def _format_docx_comments(comments: list[_DocxComment]) -> str:
    """Render comments as a "Margin comments" section to append to a converted .docx body, or "" for none."""
    if not comments:
        return ""
    lines: list[str] = []
    for comment in comments:
        author = f"[{comment.author}] " if comment.author else ""
        passage = f'on "{comment.passage}": ' if comment.passage else ""
        lines.append(f"- {author}{passage}{comment.text}")
    return "\n\n## Margin comments\n\n" + "\n".join(lines)


def pandoc_version() -> str | None:
    """Return the version that `pandoc --version` reports, such as "3.11", or None when pandoc is not on PATH."""
    if shutil.which("pandoc") is None:
        return None
    result = subprocess.run(["pandoc", "--version"], capture_output=True, text=True, timeout=30)
    first_line_words = result.stdout.partition("\n")[0].split()
    return first_line_words[-1] if first_line_words else "unknown"


class DeliverableReadError(RuntimeError):
    """A deliverable could not be converted to text, so it cannot be graded."""


def _extract_text(path: Path, *, track_changes: DocxTrackChanges) -> str:
    """Convert a file to plain text, letting conversion failures raise.

    Uses the same extraction methods as the agent harness (harness/tools.py):
    pandoc for .docx, pandas for .xlsx, markitdown for .pptx, pdfplumber for .pdf. The
    text of a .docx ends with the Word margin comments that pandoc's output leaves out.
    """
    suffix = path.suffix.lower()
    if suffix == ".docx":
        result = subprocess.run(
            ["pandoc", str(path), "-t", "markdown", "--wrap=none", f"--track-changes={track_changes.value}"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"pandoc failed: {result.stderr}")
        # pandoc prints comments only in `--track-changes=all` mode, and skips some of them
        # there, such as a comment whose range starts inside a tracked insertion.
        printed_ids = set(_PANDOC_COMMENT_START_RE.findall(result.stdout))
        comments = [comment for comment in _read_docx_comments(path) if comment.comment_id not in printed_ids]
        return result.stdout + _format_docx_comments(comments)
    if suffix == ".xlsx":
        import pandas as pd

        sheets = pd.read_excel(path, sheet_name=None)
        parts = []
        for sheet_name, df in sheets.items():
            parts.append(f"=== Sheet: {sheet_name} ===")
            parts.append(df.to_string(index=False))
        return "\n".join(parts)
    if suffix == ".pptx":
        # A .pptx is a zip package. markitdown converts any other content as plain
        # text, which would grade a corrupt deck as its raw bytes.
        if not zipfile.is_zipfile(path):
            raise ValueError("not a .pptx package (not a zip archive)")
        from markitdown import MarkItDown

        md = MarkItDown()
        result = md.convert(str(path))
        return result.text_content
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


def read_file_as_text(path: Path, *, track_changes: DocxTrackChanges = DocxTrackChanges.ACCEPT) -> str:
    """Read a file as plain text, reporting an unreadable file as a text line.

    An unreadable file returns an "(error reading <name>: ...)" line. Grading must not
    use this, because that line would be graded as the agent's work; it is for places
    that only describe a file, such as matching deliverable names. Grading reads
    deliverables with `read_deliverable_as_text`.
    """
    try:
        return _extract_text(path, track_changes=track_changes)
    except UnicodeDecodeError:
        return f"(binary file: {path.name})"
    except Exception as e:
        return f"(error reading {path.name}: {e})"


def read_deliverable_as_text(path: Path, *, track_changes: DocxTrackChanges = DocxTrackChanges.ACCEPT) -> str:
    """Read a deliverable for grading, raising `DeliverableReadError` if conversion fails.

    A file whose bytes are not UTF-8 text still returns "(binary file: <name>)". That is
    a fact about the agent's output rather than a converter failure, so the judge grades
    it as it always has.
    """
    try:
        return _extract_text(path, track_changes=track_changes)
    except UnicodeDecodeError:
        return f"(binary file: {path.name})"
    except Exception as e:
        raise DeliverableReadError(f"could not read {path.name}: {e}") from e


def _read_file_as_text(path: Path, *, track_changes: DocxTrackChanges = DocxTrackChanges.ACCEPT) -> str:
    """Read gradeable text; extraction failure is never the agent's answer."""
    return read_deliverable_as_text(path, track_changes=track_changes)


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
    document_id_aliases: dict[str, str] | None = None,
) -> dict[str, Any]:
    output_dir = run_dir / "output"
    pandoc = pandoc_version()
    parsed_pandoc = re.match(r"(\d+)\.(\d+)", pandoc or "")
    if any(path.suffix.lower() == ".docx" for path in output_dir.rglob("*")) and (
        parsed_pandoc is None or tuple(map(int, parsed_pandoc.groups())) < (3, 5)
    ):
        raise RuntimeError("LAB grading requires pandoc 3.5 or later in the verifier image")
    filenames = {deliverable for criterion in criteria for deliverable in criterion.get("deliverables", [])}
    deliverables_map = {filename: filename for filename in filenames} if filenames else None
    if deliverables_map and output_dir.exists():
        actual_files = [path.relative_to(output_dir).as_posix() for path in output_dir.rglob("*") if path.is_file()]
        resolved_map = _match_deliverables(deliverables_map, actual_files)
    else:
        resolved_map = None

    needs_full_output = any(not (criterion.get("deliverables") and resolved_map) for criterion in criteria)
    full_output = None
    full_output_error = None
    if needs_full_output:
        try:
            full_output = _load_all_output(output_dir)
        except DeliverableReadError as exc:
            full_output_error = str(exc)

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
            extraction_errors = []
            for name in criterion_deliverables:
                filename = resolved_map[name]
                path = output_dir / filename
                if not path.exists():
                    deliverable_errors.append(f"missing deliverable: {filename}")
                    continue
                include_redlines = criterion.get("evaluation_options", {}).get("include_docx_redlines", False)
                track_changes = DocxTrackChanges.ALL if include_redlines else DocxTrackChanges.ACCEPT
                try:
                    content = _read_file_as_text(path, track_changes=track_changes)
                except DeliverableReadError as exc:
                    extraction_errors.append(str(exc))
                    continue
                sections.append(f"## Agent Output: {name}\n{content}")
            if deliverable_errors or extraction_errors:
                criterion_result = asdict(
                    CriterionResult(
                        id=criterion["id"],
                        title=criterion["title"],
                        verdict="error" if extraction_errors else "fail",
                        reasoning="; ".join(deliverable_errors + extraction_errors),
                        judge_error=bool(extraction_errors),
                        error_type="DeliverableReadError" if extraction_errors else None,
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
        elif full_output_error is not None:
            return asdict(
                CriterionResult(
                    id=criterion["id"],
                    title=criterion["title"],
                    verdict="error",
                    reasoning=full_output_error,
                    judge_error=True,
                    error_type="DeliverableReadError",
                )
            )
        else:
            agent_output = full_output or "(No agent output found)"

        def resolve_identity(text: str) -> str:
            return _DOCUMENT_ID_REFERENCE.sub(
                lambda match: (document_id_aliases or {}).get(match.group(), match.group()), text
            )

        variables = {
            "task_description": task_desc,
            "agent_output": agent_output,
            "criterion_title": resolve_identity(criterion["title"]),
            "match_criteria": resolve_identity(criterion["match_criteria"]),
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
                    verdict="error",
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
        "pandoc_version": pandoc,
    }
