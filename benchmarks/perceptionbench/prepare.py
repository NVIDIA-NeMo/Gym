# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Prompt construction adapted from:
#   Repository: https://github.com/MoonshotAI/PerceptionBench
#   Commit: ba032c06e9b6ee3679171ff6ba643b7a0cfebe2e
#   File: eval/eval.py (build_messages)
#   License: Apache-2.0
"""Download and prepare PerceptionBench (``moonshotai/PerceptionBench``) into NeMo Gym JSONL."""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import re
from pathlib import Path
from typing import Any

import orjson


BENCHMARK_DIR = Path(__file__).resolve().parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "perceptionbench_benchmark.jsonl"

SOURCE_ENV = "PERCEPTIONBENCH_DATA_ROOT"
HF_REPO_ID = "moonshotai/PerceptionBench"
HF_REVISION = "6ba8c3135c7675ad6a5c141536a86b9460c70960"  # pragma: allowlist secret
HF_FILENAME = "PerceptionBench.jsonl"
EXPECTED_SHA256 = "f84afb46c3c150b572481ca34b351d6afb1c31b98fef0c0c46d7659333573c47"  # pragma: allowlist secret
EXPECTED_TASK_COUNT = 3000

_IMAGE_PLACEHOLDER_RE = re.compile(r"<\|image_(\d+)\|>")
_DATA_URL_RE = re.compile(r"^data:([^;,]+);base64,(.+)$", re.DOTALL)
_IMAGE_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _detect_mime_type(image_bytes: bytes, fallback: str = "image/jpeg") -> str:
    for signature, mime in _IMAGE_SIGNATURES:
        if image_bytes.startswith(signature):
            return mime
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return fallback


def normalize_image_data_url(raw_image: str) -> str:
    """Ensure an image data URL carries the MIME type matching its decoded magic bytes.

    Some upstream PerceptionBench records label PNG bytes as ``data:image/jpeg;base64,...``;
    preserving the original bytes under their true MIME type avoids strict provider rejections.
    """
    stripped = raw_image.strip()
    match = _DATA_URL_RE.match(stripped)
    if match:
        declared_mime, encoded = match.group(1), match.group(2)
    else:
        declared_mime, encoded = "image/jpeg", stripped
    compact = "".join(encoded.split())
    # Decode only the leading 24 bytes (32 base64 chars) needed for magic-byte inspection.
    header_b64 = compact[:32]
    header_b64 += "=" * ((4 - len(header_b64) % 4) % 4)
    mime = _detect_mime_type(base64.b64decode(header_b64), fallback=declared_mime)
    return f"data:{mime};base64,{compact}"


def build_content_items(problem: str, images: list[str]) -> list[dict[str, str]]:
    """Interleave text segments and ``input_image`` blocks at ``<|image_N|>`` placeholders.

    Matches ``build_messages`` in ``MoonshotAI/PerceptionBench/eval/eval.py``: non-empty text
    segments are emitted unchanged, out-of-range placeholders remain in text, and unreferenced
    images are appended in list order at the end.
    """
    image_items = [
        {"type": "input_image", "image_url": normalize_image_data_url(img), "detail": "high"} for img in images
    ]
    items: list[dict[str, str]] = []
    last = 0
    used: set[int] = set()
    for match in _IMAGE_PLACEHOLDER_RE.finditer(problem):
        idx = int(match.group(1))
        if 1 <= idx <= len(image_items):
            used.add(idx - 1)
            if problem[last : match.start()]:
                items.append({"type": "input_text", "text": problem[last : match.start()]})
            items.append(image_items[idx - 1])
            last = match.end()
    if problem[last:]:
        items.append({"type": "input_text", "text": problem[last:]})
    for k, image_item in enumerate(image_items):
        if k not in used:
            items.append(image_item)
    return items


def convert_record(record: dict[str, Any]) -> dict[str, Any]:
    """Convert one upstream PerceptionBench JSONL record into a NeMo Gym row."""
    if "index" not in record or record["index"] is None:
        raise ValueError("PerceptionBench record missing 'index'")
    index = int(record["index"])
    problem = str(record.get("problem") or "")
    if not problem.strip():
        raise ValueError(f"PerceptionBench record {index} has an empty problem")
    raw_images = record.get("image") or []
    if not isinstance(raw_images, list):
        raw_images = []

    raw_answer = record.get("answer")
    if isinstance(raw_answer, dict):
        raw_answer = raw_answer.get("answer")
    answer = "" if raw_answer is None else str(raw_answer)
    if not answer.strip():
        raise ValueError(f"PerceptionBench record {index} has an empty answer")

    category = str(record.get("error_category") or "").strip()
    if not category:
        raise ValueError(f"PerceptionBench record {index} has no error_category")

    source_bmk_raw = record.get("source_bmk")
    source_bmk = str(source_bmk_raw) if source_bmk_raw is not None else "NA"

    content = build_content_items(problem, [str(img) for img in raw_images])
    messages: list[dict[str, Any]] = []
    system = record.get("system")
    if isinstance(system, str) and system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content if content else problem})

    return {
        "task_id": f"perceptionbench_{index}",
        "responses_create_params": {"input": messages},
        "verifier_metadata": {
            "index": index,
            "question": problem,
            "answer": answer,
            "category": category,
            "source_bmk": source_bmk,
            "source_idx": record.get("source_idx"),
        },
    }


def _resolve_source_jsonl(source_path: str | Path | None) -> Path:
    configured = source_path or os.environ.get(SOURCE_ENV)
    if configured is not None:
        candidate = Path(configured).expanduser()
        if candidate.is_dir():
            candidate = candidate / HF_FILENAME
        if not candidate.is_file():
            raise FileNotFoundError(f"PerceptionBench source file not found: {candidate}")
        return candidate

    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        HF_REPO_ID,
        HF_FILENAME,
        repo_type="dataset",
        revision=HF_REVISION,
    )
    return Path(downloaded)


def prepare(
    source_path: str | Path | None = None,
    output_path: str | Path = OUTPUT_FPATH,
    limit: int | None = None,
) -> Path:
    """Prepare ``perceptionbench_benchmark.jsonl`` from the pinned release and return its path."""
    if limit is not None and limit <= 0:
        raise ValueError(f"limit must be positive when provided, got {limit}")

    source_jsonl = _resolve_source_jsonl(source_path)
    if limit is None:
        actual_sha256 = _sha256_file(source_jsonl)
        if actual_sha256 != EXPECTED_SHA256:
            raise ValueError(
                f"{source_jsonl} has sha256 {actual_sha256}, expected pinned {EXPECTED_SHA256} "
                f"({HF_REPO_ID}@{HF_REVISION})"
            )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = destination.with_name(destination.name + ".tmp")

    seen_ids: set[str] = set()
    written = 0
    with source_jsonl.open("rb") as src, staged.open("wb") as dst:
        for raw_line in src:
            if not raw_line.strip():
                continue
            if limit is not None and written >= limit:
                break
            row = convert_record(orjson.loads(raw_line))
            task_id = row["task_id"]
            if task_id in seen_ids:
                raise ValueError(f"Duplicate task_id {task_id} in {source_jsonl}")
            seen_ids.add(task_id)
            dst.write(orjson.dumps(row) + b"\n")
            written += 1

    if limit is None and written != EXPECTED_TASK_COUNT:
        staged.unlink(missing_ok=True)
        raise ValueError(f"Prepared {written} PerceptionBench items, expected {EXPECTED_TASK_COUNT}")

    staged.replace(destination)
    print(f"Wrote {written} PerceptionBench tasks to {destination}")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=None, help=f"Local JSONL or directory (default: HF or ${SOURCE_ENV})")
    parser.add_argument("--output", default=str(OUTPUT_FPATH), help=f"Output JSONL (default: {OUTPUT_FPATH})")
    parser.add_argument("--limit", type=int, default=None, help="Prepare only the first N items (smoke testing)")
    args = parser.parse_args()
    prepare(source_path=args.source, output_path=args.output, limit=args.limit)
