# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert the public FACTS Multimodal CSV release to NeMo Gym JSONL."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import mimetypes
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.error import URLError
from urllib.request import Request, urlopen

from google.protobuf import text_format
from google.protobuf.wrappers_pb2 import StringValue
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


BENCHMARK_DIR = Path(__file__).parent
SOURCE_FPATH = BENCHMARK_DIR / "facts_multimodal_public.csv"
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "facts_multimodal_benchmark.jsonl"
EXAMPLE_OUTPUT_FPATH = Path(__file__).parents[2] / "resources_servers" / "facts_multimodal" / "data" / "example.jsonl"
IMAGE_DIR = BENCHMARK_DIR / "data" / "images"
DOWNLOAD_REPORT_FPATH = BENCHMARK_DIR / "data" / "image_download_report.csv"
SOURCE_URL = "https://www.kaggle.com/api/v1/datasets/download/deepmind/facts-multimodal-v2-public-data"
SOURCE_SHA256 = "140b09d46cf8907703b4c96833ea8b966db59e065c19ca18666bfb6025974e17"  # pragma: allowlist secret
SOURCE_ARCHIVE_MEMBER = "facts_multimodal_public.csv"
EXAMPLE_ITEM_IDS = (
    "3025679385787018338",
    "17735436854864542395",
    "15871630253925770903",
    "16312682473540558304",
    "9127976910108645475",
)
_ITEM_RE = re.compile(r"rubric_items\s*\{(.*?)\s*\}", re.DOTALL)
_FACT_RE = re.compile(r'fact_text:\s*"((?:\\.|[^"\\])*)"', re.DOTALL)
_TAG_RE = re.compile(r'tags:\s*"([^"]*)"')
_MAX_IMAGE_BYTES = 25 * 1024 * 1024
_MIN_IMAGE_EDGE = 16
_MIME_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
}


@dataclass(frozen=True)
class ImageDownloadResult:
    """Outcome of materializing one source image."""

    image_path: str = ""
    mime_type: str = ""
    error: str = ""

    @property
    def succeeded(self) -> bool:
        return bool(self.image_path and self.mime_type and not self.error)


def ensure_source_csv(source_fpath: Path = SOURCE_FPATH) -> Path:
    """Download the pinned public Kaggle release when it is not already present."""
    if source_fpath.exists():
        digest = hashlib.sha256(source_fpath.read_bytes()).hexdigest()
        if digest != SOURCE_SHA256:
            raise ValueError(f"FACTS Multimodal source SHA-256 mismatch: expected {SOURCE_SHA256}, got {digest}")
        return source_fpath

    request = Request(SOURCE_URL, headers={"User-Agent": "NeMo-Gym-FACTS-Multimodal/1.0"})
    with urlopen(request, timeout=60) as response:  # noqa: S310 - pinned public benchmark archive
        archive = response.read()
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        source_bytes = bundle.read(SOURCE_ARCHIVE_MEMBER)
    digest = hashlib.sha256(source_bytes).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f"FACTS Multimodal source SHA-256 mismatch: expected {SOURCE_SHA256}, got {digest}")
    source_fpath.parent.mkdir(parents=True, exist_ok=True)
    source_fpath.write_bytes(source_bytes)
    return source_fpath


def parse_rubric_items(serialized_rubric: str) -> list[dict[str, object]]:
    """Parse the public release's protobuf-text rubric field."""
    items: list[dict[str, object]] = []
    for block in _ITEM_RE.findall(serialized_rubric):
        fact_match = _FACT_RE.search(block)
        if fact_match is None:
            continue
        # The release uses protobuf, rather than JSON, string escaping. For
        # example, protobuf accepts \\' for an apostrophe, which JSON rejects.
        # Delegate escape handling to protobuf so every source literal follows
        # the format it was authored in.
        parsed_fact = StringValue()
        text_format.Parse(f'value: "{fact_match.group(1)}"', parsed_fact)
        fact = parsed_fact.value
        items.append({"fact": fact, "tags": _TAG_RE.findall(block)})
    if not items:
        raise ValueError("Rubric has no parseable rubric_items")
    return items


def _mime_type(content_type: str, image_url: str, data: bytes) -> str:
    """Return a supported image MIME type, or an empty string for invalid media."""
    supplied = content_type.split(";", 1)[0].strip().lower()
    if supplied in _MIME_EXTENSIONS:
        return supplied
    guessed, _ = mimetypes.guess_type(image_url)
    if guessed in _MIME_EXTENSIONS:
        return guessed
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return ""


def _image_dimensions_error(data: bytes) -> str:
    """Return an error for undecodable or implausibly small image bytes."""
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
    except (OSError, UnidentifiedImageError) as error:
        return f"invalid image: {error}"
    if width < _MIN_IMAGE_EDGE or height < _MIN_IMAGE_EDGE:
        return f"image dimensions {width}x{height} are below {_MIN_IMAGE_EDGE}x{_MIN_IMAGE_EDGE}"
    return ""


def download_image(image_url: str, item_id: str, image_dir: Path) -> ImageDownloadResult:
    """Download one image once and return a repository-relative local reference.

    The request is deliberately performed during preparation, not rollout. This
    makes provider-side URL fetching irrelevant to evaluation and records all
    unavailable or non-image responses in the preparation report.
    """
    existing = sorted(image_dir.glob(f"{item_id}.*"))
    if len(existing) == 1 and existing[0].stat().st_size > 0:
        mime_type, _ = mimetypes.guess_type(existing[0].name)
        if mime_type in _MIME_EXTENSIONS and not _image_dimensions_error(existing[0].read_bytes()):
            return ImageDownloadResult(image_path=existing[0].name, mime_type=mime_type)

    request = Request(image_url, headers={"User-Agent": "NeMo-Gym-FACTS-Multimodal/1.0"})
    try:
        with urlopen(request, timeout=30) as response:  # noqa: S310 - public benchmark URLs
            data = response.read(_MAX_IMAGE_BYTES + 1)
            mime_type = _mime_type(response.headers.get_content_type(), image_url, data)
    except (OSError, URLError) as error:
        return ImageDownloadResult(error=f"{type(error).__name__}: {error}")

    if len(data) > _MAX_IMAGE_BYTES:
        return ImageDownloadResult(error=f"image exceeds {_MAX_IMAGE_BYTES} byte limit")
    if not data:
        return ImageDownloadResult(error="empty response body")
    if not mime_type:
        return ImageDownloadResult(error="response is not a supported JPEG, PNG, GIF, or WebP image")
    if dimensions_error := _image_dimensions_error(data):
        return ImageDownloadResult(error=dimensions_error)

    image_dir.mkdir(parents=True, exist_ok=True)
    destination = image_dir / f"{item_id}{_MIME_EXTENSIONS[mime_type]}"
    with tempfile.NamedTemporaryFile(dir=image_dir, delete=False) as temporary:
        temporary.write(data)
        temporary_path = Path(temporary.name)
    temporary_path.replace(destination)
    return ImageDownloadResult(image_path=destination.name, mime_type=mime_type)


def clear_aggregate_metrics(output_fpath: Path) -> None:
    """Remove derived metrics that no longer describe a rewritten JSONL file."""
    for suffix in ("_metrics.json", "_metrics_conflict.json"):
        output_fpath.with_name(f"{output_fpath.stem}{suffix}").unlink(missing_ok=True)


def prepare(
    source_fpath: Path = SOURCE_FPATH,
    output_fpath: Path = OUTPUT_FPATH,
    example_output_fpath: Path = EXAMPLE_OUTPUT_FPATH,
    image_dir: Path = IMAGE_DIR,
    download_report_fpath: Path = DOWNLOAD_REPORT_FPATH,
    use_base64_images: bool = True,
    download_image_fn: Callable[[str, str, Path], ImageDownloadResult] = download_image,
) -> Path:
    """Materialize images and write runnable tasks only for successful downloads.

    ``use_base64_images`` selects whether generated model requests contain an
    inline data URL or the original remote image URL. Images are always
    downloaded and checked during preparation so unavailable sources are
    excluded in either mode.
    """
    if source_fpath == SOURCE_FPATH:
        ensure_source_csv(source_fpath)
    with source_fpath.open(encoding="utf-8", newline="") as source:
        # The protobuf rubrics contain quoted newlines, so physical file lines
        # vastly overcount tasks. Count parsed CSV records for the progress bar.
        total_rows = sum(1 for _ in csv.DictReader(source))
        source.seek(0)
        reader = csv.DictReader(source)
        required_columns = {
            "prompt",
            "item_id",
            "rubrics",
            "img_url",
            "prompt_category",
            "image_category",
            "user_intent_majority",
            "external_information_majority",
            "reasoning_requirement_majority",
        }
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"Expected CSV columns {sorted(required_columns)}, got {reader.fieldnames}")

        rows = []
        report_rows = []
        with tqdm(total=total_rows, desc="Materializing FACTS images", unit="image") as progress:
            for index, source_row in enumerate(reader):
                prompt = source_row["prompt"].strip()
                image_url = source_row["img_url"].strip()
                item_id = source_row["item_id"].strip()
                if not prompt or not image_url or not item_id:
                    raise ValueError(f"Row {index} is missing prompt, image URL, or item id")
                download = download_image_fn(image_url, item_id, image_dir)
                report_rows.append(
                    {
                        "item_id": item_id,
                        "image_url": image_url,
                        "status": "ok" if download.succeeded else "failed",
                        "image_path": download.image_path,
                        "mime_type": download.mime_type,
                        "error": download.error,
                    }
                )
                if download.succeeded:
                    image_data_url = ""
                    policy_image_url = image_url
                    if use_base64_images:
                        image_data_url = (
                            f"data:{download.mime_type};base64,"
                            f"{base64.standard_b64encode((image_dir / download.image_path).read_bytes()).decode('ascii')}"
                        )
                        policy_image_url = image_data_url
                    rows.append(
                        {
                            "id": f"facts_multimodal_{item_id}",
                            "prompt": prompt,
                            "image_url": image_url,
                            "image_path": download.image_path,
                            "image_mime_type": download.mime_type,
                            "image_data_url": image_data_url,
                            "rubric_items": parse_rubric_items(source_row["rubrics"]),
                            "prompt_category": source_row["prompt_category"].strip(),
                            "image_category": source_row["image_category"].strip(),
                            "user_intent_majority": source_row["user_intent_majority"].strip(),
                            "external_information_majority": source_row["external_information_majority"].strip(),
                            "reasoning_requirement_majority": source_row["reasoning_requirement_majority"].strip(),
                            "responses_create_params": {
                                "input": [
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "input_image",
                                                "image_url": policy_image_url,
                                                "detail": "auto",
                                            },
                                            {"type": "input_text", "text": prompt},
                                        ],
                                    }
                                ]
                            },
                        }
                    )
                progress.set_postfix(kept=len(rows), failed=len(report_rows) - len(rows), refresh=False)
                progress.update()

    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    with output_fpath.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    example_output_fpath.parent.mkdir(parents=True, exist_ok=True)
    example_rows = rows[:5]
    if source_fpath == SOURCE_FPATH:
        rows_by_item_id = {row["id"].removeprefix("facts_multimodal_"): row for row in rows}
        missing_examples = set(EXAMPLE_ITEM_IDS) - rows_by_item_id.keys()
        if missing_examples:
            raise ValueError(f"Prepared dataset is missing example items: {sorted(missing_examples)}")
        example_rows = [rows_by_item_id[item_id] for item_id in EXAMPLE_ITEM_IDS]
    with example_output_fpath.open("w", encoding="utf-8") as example_output:
        for row in example_rows:
            example_output.write(json.dumps(row, ensure_ascii=False) + "\n")
    download_report_fpath.parent.mkdir(parents=True, exist_ok=True)
    with download_report_fpath.open("w", encoding="utf-8", newline="") as report:
        writer = csv.DictWriter(
            report,
            fieldnames=["item_id", "image_url", "status", "image_path", "mime_type", "error"],
        )
        writer.writeheader()
        writer.writerows(report_rows)
    clear_aggregate_metrics(output_fpath)
    return output_fpath


if __name__ == "__main__":
    print(prepare())
