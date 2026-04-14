# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Prepare labbench2 VLM benchmark data for NeMo Gym.

Downloads questions from HuggingFace (EdisonScientific/labbench2) and media
files from GCS (gs://labbench2-data-public) into data/media/. Writes lightweight
JSONL files with references to the local media directories — no base64 embedding.

Images and PDFs are embedded at rollout time by the labbench2_vlm_agent via
``embed_media_into_row``.

Produces one JSONL per benchmark tag:
  data/figqa2_img_validation.jsonl
  data/figqa2_pdf_validation.jsonl
  data/tableqa2_img_validation.jsonl
  data/tableqa2_pdf_validation.jsonl

Run with --example to also write data/example.jsonl and populate data/test_media/
with a small set of media files (committed to git for smoke tests).

Usage:
    python prepare_data.py
    python prepare_data.py --tags figqa2-img figqa2-pdf
    python prepare_data.py --limit 10 --example   # quick smoke-test
"""

import argparse
import base64
import shutil
from copy import deepcopy
from pathlib import Path

import httpx
import orjson
from datasets import load_dataset

LABBENCH2_HF_DATASET = "EdisonScientific/labbench2"
GCS_BUCKET = "labbench2-data-public"
GCS_API_URL = "https://storage.googleapis.com/storage/v1/b/{bucket}/o"
GCS_DOWNLOAD_URL = "https://storage.googleapis.com/{bucket}/{path}"

TAGS = ["figqa2-img", "figqa2-pdf", "tableqa2-img", "tableqa2-pdf"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
PDF_EXTENSION = ".pdf"
MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

MEDIA_DIR_NAME = "media"
TEST_MEDIA_DIR_NAME = "test_media"


# ---------------------------------------------------------------------------
# GCS download helpers (public bucket, no auth required)
# ---------------------------------------------------------------------------


def _list_gcs_objects(prefix: str) -> list[str]:
    objects = []
    page_token = None
    prefix_with_slash = prefix.strip("/") + "/"
    while True:
        params: dict = {"prefix": prefix_with_slash}
        if page_token:
            params["pageToken"] = page_token
        resp = httpx.get(GCS_API_URL.format(bucket=GCS_BUCKET), params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        for item in data.get("items", []):
            objects.append(item["name"])
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return objects


def _download_question_files(gcs_prefix: str, media_dir: Path) -> Path:
    """Download all files for a question from GCS into the local media directory."""
    dest_dir = media_dir / gcs_prefix.strip("/")
    dest_dir.mkdir(parents=True, exist_ok=True)

    prefix_with_slash = gcs_prefix.strip("/") + "/"
    for blob_name in _list_gcs_objects(gcs_prefix):
        if blob_name.endswith("/"):
            continue
        relative = blob_name[len(prefix_with_slash):]
        if not relative:
            continue
        dest = dest_dir / relative
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = GCS_DOWNLOAD_URL.format(bucket=GCS_BUCKET, path=blob_name)
        with httpx.stream("GET", url, timeout=120) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_bytes():
                    f.write(chunk)

    return dest_dir


# ---------------------------------------------------------------------------
# Image / PDF → input_image content blocks
#
# These helpers are imported by the labbench2_vlm_agent at rollout time.
# ---------------------------------------------------------------------------


def _image_block(path: Path) -> dict:
    mime = MIME_TYPES.get(path.suffix.lower(), "image/png")
    b64 = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return {"type": "input_image", "image_url": f"data:{mime};base64,{b64}", "detail": "high"}


def _pdf_to_image_blocks(pdf_path: Path, dpi: int) -> list[dict]:
    """Render every page of a PDF to PNG at ``dpi`` and return input_image blocks."""
    import fitz  # pymupdf

    doc = fitz.open(str(pdf_path))
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    blocks = []
    for page in doc:
        png_bytes = page.get_pixmap(matrix=matrix).tobytes("png")
        b64 = base64.standard_b64encode(png_bytes).decode("utf-8")
        blocks.append({"type": "input_image", "image_url": f"data:image/png;base64,{b64}", "detail": "high"})
    doc.close()
    return blocks


def _files_to_image_blocks(files_dir: Path, dpi: int) -> list[dict]:
    """Convert all image/PDF files in a directory to input_image content blocks."""
    blocks = []
    for f in sorted(files_dir.iterdir()):
        if not f.is_file():
            continue
        ext = f.suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            blocks.append(_image_block(f))
        elif ext == PDF_EXTENSION:
            blocks.extend(_pdf_to_image_blocks(f, dpi=dpi))
    return blocks


def embed_media_into_row(row: dict, media_base_dir: Path, dpi: int = 170) -> dict:
    """Resolve ``verifier_metadata.media_dir`` and inject base64 input_image blocks.

    Used by ``labbench2_vlm_agent`` at rollout time to enrich lightweight JSONL
    rows with the actual image/PDF content before sending to the model.

    Returns a deep copy with ``input_image`` blocks inserted before any
    ``input_text`` blocks in ``responses_create_params.input[0].content``.
    Rows without ``media_dir`` are returned unchanged.
    """
    meta = row.get("verifier_metadata") or {}
    media_dir = meta.get("media_dir")
    if not media_dir:
        return row

    files_dir = media_base_dir / media_dir
    image_blocks = _files_to_image_blocks(files_dir, dpi=dpi)
    if not image_blocks:
        return row

    row = deepcopy(row)
    content = row["responses_create_params"]["input"][0]["content"]
    non_image_blocks = [b for b in content if b.get("type") != "input_image"]
    row["responses_create_params"]["input"][0]["content"] = image_blocks + non_image_blocks
    return row


# ---------------------------------------------------------------------------
# Core prepare logic
# ---------------------------------------------------------------------------


def prepare_tag(tag: str, output_path: Path, media_dir: Path, limit: int | None = None) -> list[dict]:
    """
    Download questions + media for ``tag`` and write lightweight Gym JSONL.

    Returns the list of written rows (useful for building example.jsonl).
    """
    print(f"[{tag}] Loading from HuggingFace...")
    hf_dataset = load_dataset(LABBENCH2_HF_DATASET, tag, split="train")
    questions = [row for row in hf_dataset if row["tag"] == tag]

    if limit is not None:
        questions = questions[:limit]

    print(f"[{tag}] {len(questions)} questions. Downloading media and building JSONL...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = 0

    with open(output_path, "wb") as f:
        for i, row in enumerate(questions):
            if (i + 1) % 100 == 0:
                print(f"[{tag}]   {i + 1}/{len(questions)}...")

            gcs_prefix = (row.get("files") or "").strip()
            if not gcs_prefix:
                skipped += 1
                continue

            files_dir = _download_question_files(gcs_prefix, media_dir)
            has_media = any(
                f.is_file() and (f.suffix.lower() in IMAGE_EXTENSIONS or f.suffix.lower() == PDF_EXTENSION)
                for f in files_dir.iterdir()
            )
            if not has_media:
                print(f"[{tag}]   WARNING: no images for {row['id']} ({gcs_prefix}), skipping")
                skipped += 1
                continue

            question_text = row["question"]
            if row.get("prompt_suffix"):
                question_text += "\n\n" + row["prompt_suffix"]

            gym_row = {
                "responses_create_params": {
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": question_text},
                            ],
                        }
                    ]
                },
                "verifier_metadata": {
                    "ideal": row["ideal"],
                    "tag": row["tag"],
                    "id": row["id"],
                    "media_dir": f"{MEDIA_DIR_NAME}/{gcs_prefix.strip('/')}",
                },
            }

            line = orjson.dumps(gym_row) + b"\n"
            f.write(line)
            rows.append(gym_row)

    print(f"[{tag}] Wrote {len(rows)} rows to {output_path} ({skipped} skipped)")
    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare labbench2 VLM data for NeMo Gym")
    parser.add_argument(
        "--tags", nargs="+", default=TAGS, choices=TAGS, metavar="TAG",
        help="Which benchmark tags to prepare (default: all four)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of questions per tag (useful for quick smoke-tests)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent / "data",
        help="Output directory for JSONL files (default: ./data)",
    )
    parser.add_argument(
        "--example", action="store_true",
        help="Also write data/example.jsonl and populate data/test_media/ (committed to git)",
    )
    args = parser.parse_args()

    media_dir = args.output_dir / MEDIA_DIR_NAME
    example_rows: list[dict] = []

    for tag in args.tags:
        safe = tag.replace("-", "_")
        out = args.output_dir / f"{safe}_validation.jsonl"
        rows = prepare_tag(tag, out, media_dir=media_dir, limit=args.limit)
        if args.example and rows:
            n = 2 if not example_rows else 1
            example_rows.extend(rows[:n])

    if args.example:
        example_rows = example_rows[:5]
        test_media_dir = args.output_dir / TEST_MEDIA_DIR_NAME

        test_rows = []
        for row in example_rows:
            row = deepcopy(row)
            media_path = row["verifier_metadata"]["media_dir"]
            rel = media_path.removeprefix(MEDIA_DIR_NAME + "/")
            src = args.output_dir / media_path
            dst = test_media_dir / rel
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src, dst, dirs_exist_ok=True)
            row["verifier_metadata"]["media_dir"] = f"{TEST_MEDIA_DIR_NAME}/{rel}"
            test_rows.append(row)

        example_path = args.output_dir / "example.jsonl"
        with open(example_path, "wb") as f:
            for row in test_rows:
                f.write(orjson.dumps(row) + b"\n")
        print(f"Wrote {len(test_rows)} rows to {example_path}")
