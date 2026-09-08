# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the public GDP.pdf test set according to AA methodology v4.3."""

from __future__ import annotations

import hashlib
import json
import re
from importlib.metadata import version
from pathlib import Path
from typing import Any, Optional


DATASET_ID = "surgeai/GDP.pdf"
DATASET_REVISION = "73e94c87235e0477f8a65996086acd3f47c98d2e"  # pragma: allowlist secret
LITEPARSE_VERSION = "2.14.4"
SOURCE_DPI = 150
BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
SOURCE_DIR = DATA_DIR / "source"
DOCUMENTS_DIR = DATA_DIR / "documents"
OUTPUT_FPATH = DATA_DIR / "gdp_pdf_benchmark.jsonl"

_RUBRIC_ATTRIBUTES = (
    "criterion",
    "criterion_type",
    "criterion_severity",
    "criterion_implicitness",
    "criterion_subjectiveness",
    "criterion_failure_mode",
)


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "as_py"):
        return value.as_py()
    return str(value)


def extract_rubric_criteria(row: dict[str, Any]) -> list[dict[str, Any]]:
    criteria: list[dict[str, Any]] = []
    for index in range(1, 31):
        criterion = _json_value(row.get(f"rubric - {index}. criterion"))
        if not isinstance(criterion, str) or not criterion.strip():
            continue
        item: dict[str, Any] = {"id": f"rubric-{index}", "criterion": criterion.strip()}
        for attribute in _RUBRIC_ATTRIBUTES[1:]:
            value = _json_value(row.get(f"rubric - {index}. {attribute}"))
            if value is not None:
                item[attribute] = value
        criteria.append(item)
    if not criteria:
        raise ValueError(f"GDP.pdf task {row.get('task_id')!r} has no rubric criteria")
    return criteria


def _safe_task_dir(task_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", task_id).strip("._")
    if not normalized:
        normalized = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:16]
    return normalized


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cached_document_is_valid(manifest_path: Path, *, source_sha256: str, parser_version: str) -> bool:
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pages = manifest["pages"]
        return (
            manifest["source_sha256"] == source_sha256
            and manifest["parser"]["name"] == "liteparse"
            and manifest["parser"]["version"] == parser_version
            and manifest["source_dpi"] == SOURCE_DPI
            and len(pages) == manifest["page_count"]
            and all((manifest_path.parent / page["image"]).is_file() for page in pages)
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def prepare_document(pdf_path: Path, document_dir: Path, *, force: bool = False, num_workers: int = 4) -> Path:
    """Parse all pages with OCR enabled and persist 150-DPI page screenshots."""
    try:
        from liteparse import LiteParse
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"GDP.pdf preparation requires liteparse=={LITEPARSE_VERSION}; follow the command in "
            "benchmarks/gdp_pdf/README.md."
        ) from exc

    parser_version = version("liteparse")
    if parser_version != LITEPARSE_VERSION:
        raise RuntimeError(f"GDP.pdf preparation requires liteparse=={LITEPARSE_VERSION}, found {parser_version}")
    source_sha256 = _sha256(pdf_path)
    manifest_path = document_dir / "manifest.json"
    if not force and _cached_document_is_valid(
        manifest_path,
        source_sha256=source_sha256,
        parser_version=parser_version,
    ):
        return manifest_path

    parser = LiteParse(
        ocr_enabled=True,
        dpi=SOURCE_DPI,
        output_format="text",
        extract_screenshots=True,
        continue_on_page_error=False,
        max_pages=10_000,
        num_workers=num_workers,
        quiet=True,
    )
    result = parser.parse(pdf_path)
    if result.page_errors:
        raise RuntimeError(f"LiteParse reported page errors for {pdf_path}: {result.page_errors}")
    if len(result.pages) != result.total_pages:
        raise RuntimeError(f"LiteParse returned {len(result.pages)} of {result.total_pages} pages for {pdf_path}")
    ordered_pages = sorted(result.pages, key=lambda page: page.page_num)
    page_numbers = [page.page_num for page in ordered_pages]
    if page_numbers != list(range(1, result.total_pages + 1)):
        raise RuntimeError(f"LiteParse returned invalid page numbering for {pdf_path}: {page_numbers}")

    screenshots = {screenshot.page_num: screenshot for screenshot in result.screenshots}
    if len(screenshots) != result.total_pages:
        screenshots = {screenshot.page_num: screenshot for screenshot in parser.screenshot(pdf_path)}
    if len(screenshots) != result.total_pages:
        raise RuntimeError(f"LiteParse rendered {len(screenshots)} of {result.total_pages} pages for {pdf_path}")

    pages_dir = document_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    pages = []
    for page in ordered_pages:
        screenshot = screenshots.get(page.page_num)
        if screenshot is None:
            raise RuntimeError(f"missing screenshot for page {page.page_num} of {pdf_path}")
        image_name = f"page_{page.page_num:04d}.png"
        (pages_dir / image_name).write_bytes(screenshot.image_bytes)
        pages.append(
            {
                "page_number": page.page_num,
                "text": page.text,
                "image": f"pages/{image_name}",
                "width": screenshot.width,
                "height": screenshot.height,
            }
        )

    expected_images = {Path(page["image"]).name for page in pages}
    for stale_image in pages_dir.glob("page_*.png"):
        if stale_image.name not in expected_images:
            stale_image.unlink()

    manifest = {
        "source_pdf": pdf_path.name,
        "source_sha256": source_sha256,
        "source_dpi": SOURCE_DPI,
        "page_count": result.total_pages,
        "parser": {"name": "liteparse", "version": parser_version, "ocr_enabled": True},
        "pages": pages,
    }
    document_dir.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest_path


def _download_source(source_dir: Path, revision: str, allow_patterns: list[str]) -> Path:
    from huggingface_hub import snapshot_download

    snapshot_path = snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        revision=revision,
        allow_patterns=allow_patterns,
        local_dir=source_dir,
    )
    return Path(snapshot_path)


def prepare(
    output_path: Path = OUTPUT_FPATH,
    source_dir: Path = SOURCE_DIR,
    documents_dir: Path = DOCUMENTS_DIR,
    revision: str = DATASET_REVISION,
    limit: Optional[int] = None,
    force: bool = False,
    num_workers: int = 4,
) -> Path:
    """Download, parse, render, and convert GDP.pdf to Gym JSONL."""
    import pyarrow.parquet as parquet

    snapshot = _download_source(Path(source_dir), revision, ["data.parquet"])
    source_rows = parquet.read_table(snapshot / "data.parquet").to_pylist()
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        source_rows = source_rows[:limit]

    pdf_paths = sorted({str(row["pdf_path"]) for row in source_rows})
    snapshot = _download_source(Path(source_dir), revision, pdf_paths)

    output_path = Path(output_path)
    documents_dir = Path(documents_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared_rows: list[dict[str, Any]] = []

    for index, row in enumerate(source_rows, start=1):
        task_id = str(row.get("task_id") or index)
        task_prompt = str(row["prompt"]).strip()
        domain = str(row["domain"]).strip()
        relative_pdf = Path(str(row["pdf_path"]))
        pdf_path = (snapshot / relative_pdf).resolve()
        if not pdf_path.is_relative_to(snapshot.resolve()):
            raise ValueError(f"GDP.pdf source path escapes the pinned snapshot: {relative_pdf}")
        if not pdf_path.is_file():
            raise FileNotFoundError(f"missing GDP.pdf source document: {pdf_path}")

        document_dir = documents_dir / _safe_task_dir(task_id)
        manifest_path = prepare_document(pdf_path, document_dir, force=force, num_workers=num_workers)
        manifest_relative = manifest_path.relative_to(output_path.parent).as_posix()
        prepared_rows.append(
            {
                "question": task_prompt,
                "responses_create_params": {"tools": [], "parallel_tool_calls": False},
                "verifier_metadata": {
                    "task_id": task_id,
                    "task_prompt": task_prompt,
                    "domain": domain,
                    "rubric_criteria": extract_rubric_criteria(row),
                    "document_manifest": manifest_relative,
                    "source_pdf": relative_pdf.as_posix(),
                    "source_revision": revision,
                    "worker_id": _json_value(row.get("worker_id")),
                    "task_response_id": _json_value(row.get("task_response_id")),
                },
            }
        )

    temporary = output_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in prepared_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(output_path)

    criteria_count = sum(len(row["verifier_metadata"]["rubric_criteria"]) for row in prepared_rows)
    print(f"Prepared {len(prepared_rows)} GDP.pdf tasks with {criteria_count} criteria at {output_path}")
    return output_path


if __name__ == "__main__":
    prepare()
