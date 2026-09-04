# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deliverable extraction and rubric scoring for JobBench.

A port of upstream ``eval/judge.py`` (Job-Bench/job-bench-eval). The extraction
rules, judge prompt, JSON-recovery ladder and weighted scorecard are kept
byte-compatible with upstream so Gym scores stay comparable to the leaderboard;
the transport is Gym's model server instead of a direct OpenAI client.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile


MAX_CHARS_PER_FILE = 200_000
SQLITE_EXTS = {"db", "sqlite", "sqlite3"}
SQLITE_ROWS_PER_TABLE = 500

VISION_IMAGE_EXTS = {"png", "jpg", "jpeg", "gif", "webp"}
MAX_VISION_IMAGES = 8
VISUAL_RUBRIC_PATTERN = re.compile(
    r"\b(plot|figure|visualization|visualisation|visualize|visualise|"
    r"heatmap|histogram|scatter ?plot|biplot|diagram|q[- ]?q)\b",
    re.IGNORECASE,
)
VISION_MIME = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
}

PLAINTEXT_EXTS = {
    "txt",
    "md",
    "csv",
    "py",
    "json",
    "sh",
    "log",
    "xml",
    "html",
    "css",
    "js",
    "ts",
    "yaml",
    "yml",
    "ini",
    "cfg",
    "conf",
    "sql",
    "rules",
    "geojson",
}

JUDGE_SYSTEM_PROMPT = (
    "You are an evaluation judge. You must return valid JSON only, with no markdown formatting or extra text."
)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"[ERROR: Failed to read text file: {path.name}: {exc}]"


def _excel_to_text(path: Path) -> str:
    try:
        import pandas as pd

        excel = pd.ExcelFile(str(path))
        parts = []
        for sheet in excel.sheet_names:
            frame = pd.read_excel(excel, sheet_name=sheet)
            parts.append(f"=== Sheet: {sheet} ===\n{frame.to_csv(index=False)}")
        return "\n".join(parts)
    except ImportError:
        return f"[ERROR: pandas/openpyxl not available for {path.name}]"
    except Exception as exc:
        return f"[ERROR: Failed to read Excel {path.name}: {exc}]"


def _docx_to_text(path: Path) -> str:
    try:
        import mammoth

        def embedded_image_placeholder(image):
            return {"src": "embedded-image", "alt": f"Embedded image: {image.content_type}"}

        with open(str(path), "rb") as handle:
            result = mammoth.convert_to_markdown(
                handle,
                convert_image=mammoth.images.img_element(embedded_image_placeholder),
            )
        return result.value
    except ImportError:
        return f"[ERROR: mammoth not available for {path.name}]"
    except Exception as exc:
        return f"[ERROR: Failed to read DOCX {path.name}: {exc}]"


def _pdf_to_text(path: Path) -> str:
    try:
        import pdfplumber

        with pdfplumber.open(str(path)) as pdf:
            parts = []
            for index, page in enumerate(pdf.pages):
                parts.append(f"=== Page {index + 1} ===\n{page.extract_text(layout=True) or ''}")
        return "\n".join(parts)
    except ImportError:
        return f"[ERROR: pdfplumber not available for {path.name}]"
    except Exception as exc:
        return f"[ERROR: Failed to read PDF {path.name}: {exc}]"


def _sqlite_to_text(path: Path) -> str:
    try:
        connection = sqlite3.connect(str(path))
    except Exception as exc:
        return f"[ERROR: Failed to read SQLite {path.name}: {exc}]"
    try:
        cursor = connection.cursor()
        schema = connection.execute("SELECT sql FROM sqlite_master WHERE sql IS NOT NULL").fetchall()
        parts = ["=== Schema ==="]
        parts.extend(row[0] for row in schema if row[0])
        tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        for table in tables:
            parts.append(f"\n=== Table: {table} ===")
            try:
                total_rows = connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
                rows = cursor.execute(f'SELECT * FROM "{table}" LIMIT {SQLITE_ROWS_PER_TABLE}').fetchall()
                columns = [description[0] for description in cursor.description]
                parts.append(f"-- total_rows: {total_rows}; shown: {len(rows)} (LIMIT {SQLITE_ROWS_PER_TABLE})")
                parts.append(",".join(columns))
                for row in rows:
                    parts.append(",".join("" if value is None else str(value) for value in row))
            except Exception as exc:
                parts.append(f"[ERROR reading table {table}: {exc}]")
        return "\n".join(parts)
    except Exception as exc:
        return f"[ERROR: Failed to read SQLite {path.name}: {exc}]"
    finally:
        connection.close()


def _pptx_to_text(path: Path) -> str:
    try:
        from pptx import Presentation

        presentation = Presentation(str(path))
        parts = []
        for index, slide in enumerate(presentation.slides):
            parts.append(f"=== Slide {index + 1} ===")
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    parts.append(shape.text)
        return "\n".join(parts)
    except ImportError:
        return f"[ERROR: python-pptx not available for {path.name}]"
    except Exception as exc:
        return f"[ERROR: Failed to read PowerPoint {path.name}: {exc}]"


def _notebook_to_text(path: Path) -> str:
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        parts = []
        for cell in notebook.get("cells", []):
            parts.append(f"=== {cell['cell_type']} ===")
            parts.append("".join(cell.get("source", [])))
            for output in cell.get("outputs", []):
                if "text" in output:
                    parts.append("".join(output["text"]))
        return "\n".join(parts)
    except Exception as exc:
        return f"[ERROR: Failed to read notebook {path.name}: {exc}]"


def convert_file_to_text(path: Path) -> str:
    """Render one deliverable as judge-readable text."""
    ext = path.suffix.lower().lstrip(".")

    if ext in PLAINTEXT_EXTS:
        return _read_text(path)
    if ext in ("xlsx", "xls"):
        return _excel_to_text(path)
    if ext == "docx":
        return _docx_to_text(path)
    if ext == "pdf":
        return _pdf_to_text(path)
    if ext in SQLITE_EXTS:
        return _sqlite_to_text(path)
    if ext == "pptx":
        return _pptx_to_text(path)
    if ext == "ipynb":
        return _notebook_to_text(path)
    if ext in ("png", "jpg", "jpeg", "gif", "svg", "bmp"):
        return f"[Image file: {path.name} — cannot extract text content]"
    return f"[Binary or unsupported file type: {ext} — {path.name}]"


def extract_all_file_contents(output_dir: Path, *, max_chars_per_file: int = MAX_CHARS_PER_FILE) -> str:
    """Concatenate every deliverable under ``output_dir`` as text."""
    if not output_dir.is_dir():
        return ""
    parts = []
    for file_path in sorted(output_dir.rglob("*")):
        if not file_path.is_file():
            continue
        content = convert_file_to_text(file_path)
        ext = file_path.suffix.lower().lstrip(".")
        # The cap guards against a model copying a multi-MB input into its output;
        # SQLite renderings are already row-limited, so they are exempt upstream.
        if ext not in SQLITE_EXTS and len(content) > max_chars_per_file:
            content = content[:max_chars_per_file] + f"\n... [Content truncated at {max_chars_per_file} characters]"
        parts.append(f"=== FILE: {file_path.name} ===\n{content}\n")
    return "\n".join(parts)


def rubric_needs_vision(rubric: dict) -> bool:
    """True when a rubric mentions a visual artifact and so warrants image attachments."""
    text = rubric.get("rubric", "") or ""
    criterion = rubric.get("criterion", [])
    if isinstance(criterion, list):
        text = text + " " + " ".join(str(item) for item in criterion)
    elif isinstance(criterion, str):
        text = text + " " + criterion
    return bool(VISUAL_RUBRIC_PATTERN.search(text))


def collect_image_paths(output_dir: Path, cap: int | None = MAX_VISION_IMAGES) -> list[Path]:
    if not output_dir.exists():
        return []
    images = [
        path
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.suffix.lower().lstrip(".") in VISION_IMAGE_EXTS
    ]
    return images if cap is None else images[:cap]


def collect_image_attachments(output_dir: Path, cap: int = MAX_VISION_IMAGES) -> list[tuple[str, str]]:
    """Collect standalone, docx-embedded and notebook-embedded images as deduplicated data URLs."""
    if not output_dir.exists() or cap <= 0:
        return []

    attachments: list[tuple[str, str]] = []
    seen_hashes: set[str] = set()

    def add_attachment(name: str, mime: str, image_bytes: bytes) -> bool:
        """Append unless a duplicate; return True once the cap is reached."""
        digest = hashlib.sha256(image_bytes).hexdigest()
        if digest in seen_hashes:
            return False
        seen_hashes.add(digest)
        encoded = base64.b64encode(image_bytes).decode("ascii")
        attachments.append((name, f"data:{mime};base64,{encoded}"))
        return len(attachments) >= cap

    for path in collect_image_paths(output_dir, cap=None):
        mime = VISION_MIME.get(path.suffix.lower().lstrip("."))
        if mime is None:
            continue
        try:
            image_bytes = path.read_bytes()
        except OSError:
            continue
        if add_attachment(path.relative_to(output_dir).as_posix(), mime, image_bytes):
            return attachments

    for docx_path in sorted(output_dir.rglob("*.docx")):
        try:
            with ZipFile(docx_path) as archive:
                media_names = [
                    name
                    for name in sorted(archive.namelist())
                    if name.startswith("word/media/") and Path(name).suffix.lower().lstrip(".") in VISION_IMAGE_EXTS
                ]
                for media_name in media_names:
                    mime = VISION_MIME.get(Path(media_name).suffix.lower().lstrip("."))
                    if mime is None:
                        continue
                    try:
                        image_bytes = archive.read(media_name)
                    except (KeyError, OSError):
                        continue
                    display_name = f"{docx_path.relative_to(output_dir).as_posix()}:{media_name}"
                    if add_attachment(display_name, mime, image_bytes):
                        return attachments
        except (BadZipFile, OSError):
            continue

    for notebook_path in sorted(output_dir.rglob("*.ipynb")):
        try:
            notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for cell_index, cell in enumerate(notebook.get("cells", [])):
            for output_index, output in enumerate(cell.get("outputs", [])):
                data = output.get("data", {})
                if not isinstance(data, dict):
                    continue
                for mime in ("image/png", "image/jpeg", "image/gif", "image/webp"):
                    encoded = data.get(mime)
                    if isinstance(encoded, list):
                        encoded = "".join(str(part) for part in encoded)
                    if not isinstance(encoded, str):
                        continue
                    try:
                        image_bytes = base64.b64decode("".join(encoded.split()), validate=True)
                    except (ValueError, binascii.Error):
                        continue
                    display_name = (
                        f"{notebook_path.relative_to(output_dir).as_posix()}:"
                        f"cell-{cell_index}-output-{output_index}:{mime}"
                    )
                    if add_attachment(display_name, mime, image_bytes):
                        return attachments

    return attachments


def normalize_criteria(rubric: dict) -> list[str]:
    criterion_raw = rubric.get("criterion", [])
    if isinstance(criterion_raw, str):
        return [criterion_raw]
    return list(criterion_raw)


def build_rubric_prompt(rubric: dict, file_contents: str, *, vision_used: bool) -> str:
    """Render upstream's rubric-level judge prompt verbatim."""
    rubric_text = rubric.get("rubric", "")
    criteria = normalize_criteria(rubric)
    criterion_count = len(criteria)
    criteria_list_text = "\n".join(f"Criterion {index}: {criterion}" for index, criterion in enumerate(criteria))

    return f"""You are an evaluation judge. Your task is to evaluate ALL criteria for a single rubric.

## Rubric Description
{rubric_text}

## Criteria to Evaluate (Judge ALL of them)
{criteria_list_text}

## Output Files Content
The following are the contents of all output files to evaluate:

{file_contents}

## Evaluation Rules
- Evaluate EACH criterion listed above independently
- For each criterion: determine if it PASSES or FAILS
- Semantic matching is acceptable (you don't need exact wording match)
- Binary judgment for each criterion: PASS or FAIL only
- The rubric passes ONLY if ALL criteria pass

## Output Format
Return your judgment as a JSON object with EXACTLY this structure (no markdown, no extra text):
{{
  "criteria_results": [
    {{"index": 0, "passed": true/false, "reasoning": "...", "evidence": "..."}},
    {{"index": 1, "passed": true/false, "reasoning": "...", "evidence": "..."}}
  ],
  "rubric_passed": true/false,
  "overall_reasoning": "Summary of why the rubric passed or failed"
}}

IMPORTANT:
- criteria_results array must have exactly {criterion_count} items (one for each criterion)
- rubric_passed should be true ONLY if ALL criteria passed
- Include specific evidence from the output files{" and the attached images" if vision_used else ""}
"""


def build_user_content(prompt: str, attachments: list[tuple[str, str]]) -> str | list[dict[str, Any]]:
    """Return a plain string, or multimodal content parts when images are attached."""
    if not attachments:
        return prompt
    plural = "s" if len(attachments) != 1 else ""
    content: list[dict[str, Any]] = [
        {"type": "text", "text": prompt},
        {"type": "text", "text": f"\n## Attached Images ({len(attachments)} file{plural})"},
    ]
    for index, (filename, url) in enumerate(attachments, start=1):
        content.append({"type": "text", "text": f"Image {index}: {filename}"})
        content.append({"type": "image_url", "image_url": {"url": url}})
    return content


def parse_judge_json(content: str) -> tuple[dict, str]:
    """Recover the judge's JSON object, widening the search on each failure."""
    try:
        return json.loads(content), "direct_json"
    except json.JSONDecodeError:
        pass

    fence = re.search(r"```(?:json)?\s*\n(.*?)\n\s*```", content, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1).strip()), "markdown_fence"
        except json.JSONDecodeError:
            pass

    first = content.find("{")
    last = content.rfind("}")
    if first != -1 and last > first:
        try:
            return json.loads(content[first : last + 1]), "first_last_brace"
        except json.JSONDecodeError:
            pass

    for candidate in reversed(re.findall(r"\{.*?\"criteria_results\"\s*:\s*\[.*?\].*?\}", content, re.DOTALL)):
        try:
            return json.loads(candidate), "regex_extract"
        except json.JSONDecodeError:
            continue

    raise ValueError(f"Could not extract JSON from response: {content[:500]}")


def build_rubric_result(rubric_index: int, rubric: dict, parsed: dict) -> dict:
    """Score one rubric from the judge's parsed verdict; all criteria must pass."""
    criteria = normalize_criteria(rubric)
    weight = rubric.get("weight", 0)
    model_criteria = parsed.get("criteria_results", [])
    rubric_passed = bool(parsed.get("rubric_passed", False))

    enriched = []
    for index, criterion in enumerate(criteria):
        item = model_criteria[index] if index < len(model_criteria) else {}
        if not isinstance(item, dict):
            item = {}
        enriched.append(
            {
                "index": index,
                "criterion": criterion,
                "passed": bool(item.get("passed", False)),
                "reasoning": item.get("reasoning", ""),
                "evidence": item.get("evidence", ""),
            }
        )

    return {
        "index": rubric_index,
        "rubric": rubric.get("rubric", ""),
        "weight": weight,
        "result": {
            "passed": rubric_passed,
            "score": weight if rubric_passed else 0,
            "criteria_count": len(criteria),
            "criteria_passed": sum(1 for item in enriched if item["passed"]),
            "criteria_results": enriched,
            "overall_reasoning": parsed.get("overall_reasoning", ""),
        },
    }


def build_failed_rubric_result(rubric_index: int, rubric: dict, overall_reasoning: str) -> dict:
    """A zero-score result for a rubric the judge could not decide."""
    criteria = normalize_criteria(rubric)
    return {
        "index": rubric_index,
        "rubric": rubric.get("rubric", ""),
        "weight": rubric.get("weight", 0),
        "result": {
            "passed": False,
            "score": 0,
            "criteria_count": len(criteria),
            "criteria_passed": 0,
            "criteria_results": [
                {
                    "index": index,
                    "criterion": criterion,
                    "passed": False,
                    "reasoning": overall_reasoning,
                    "evidence": "",
                }
                for index, criterion in enumerate(criteria)
            ],
            "overall_reasoning": overall_reasoning,
        },
    }


def build_scorecard(results: list[dict]) -> dict[str, float | int]:
    """Aggregate rubric results into JobBench's weighted scorecard.

    ``normalized_score`` (weighted score / max weight) is the JobBench headline
    metric and becomes the Gym reward; ``pass_rate`` is the fraction of rubrics
    that passed every criterion.
    """
    total_score = sum(result["result"]["score"] for result in results)
    max_score = sum(result["weight"] for result in results)
    passed_count = sum(1 for result in results if result["result"]["passed"])
    total_count = len(results)

    return {
        "total_score": total_score,
        "max_score": max_score,
        "normalized_score": round(total_score / max_score, 4) if max_score > 0 else 0.0,
        "pass_rate": round(passed_count / total_count, 4) if total_count > 0 else 0.0,
        "passed_count": passed_count,
        "total_count": total_count,
    }
