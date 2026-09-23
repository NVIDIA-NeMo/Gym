# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the public FACTS Grounding set scored under the v2 protocol (FACTS Benchmark Suite, arXiv:2512.10791).

FACTS Grounding v2 keeps the v1 prompts and changes only the judges (section 6). The source of record for the
public prompts is the Kaggle dataset ``deepmind/FACTS-grounding-examples`` (Apache 2.0 / CC-BY 4.0, see its
``LICENSE.txt``), version 17 (2026-01-07), ``examples.csv`` with 856 rows and the columns ``system_instruction``,
``user_request``, ``context_document``, ``full_prompt``, ``domain``, ``type``, ``high_level_type``. The official
starter notebook (``prathameshbang/facts-grounding-v2-benchmark-starter``) reads exactly this file and prompts
the evaluated model with ``full_prompt`` as a single user message, which is what the emitted rows carry.

The Hugging Face mirror ``google/FACTS-grounding-public`` (revision ``11b6961``) is the December 2024 v1 release
with 860 rows; 856 of them are content-identical to the Kaggle file and 4 are absent from the current official
release. This adapter follows the Kaggle release because the leaderboard pipeline does, and records the delta
in ``METRICS.md``. The 859-row private half is held by Kaggle and is not available.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional


BENCHMARK_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "facts_grounding_v2_public.jsonl"

KAGGLE_DATASET = "deepmind/FACTS-grounding-examples"
KAGGLE_DATASET_VERSION = 17
KAGGLE_DOWNLOAD_URL = (
    f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_DATASET}?datasetVersionNumber={KAGGLE_DATASET_VERSION}"
)
KAGGLE_DATASET_URL = f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}"
CSV_MEMBER = "examples.csv"
CSV_SHA256 = "66990aea86a91d40edcda6ebcd0d686674246dc7c07338ed8c17e1c275b1e096"  # pragma: allowlist secret
PROMPTS_MEMBER = "evaluation_prompts.csv"
PROMPTS_SHA256 = "43a4bc5083275ca1a3d95eed8b26c0760a50dacac5ee19856af887b36d120069"  # pragma: allowlist secret
EXPECTED_ROWS = 856
LICENSE = "Apache 2.0"
COLUMNS = (
    "system_instruction",
    "user_request",
    "context_document",
    "full_prompt",
    "domain",
    "type",
    "high_level_type",
)
HF_V1_PUBLIC_ROWS = 860
HF_V1_REVISION = "11b6961370aa0ac73c91d58e35317e724b2ac765"

csv.field_size_limit(sys.maxsize)


def _download_members(timeout: float = 300.0) -> tuple[bytes, bytes]:
    request = urllib.request.Request(
        KAGGLE_DOWNLOAD_URL, headers={"User-Agent": "nemo-gym-facts-grounding-prepare/1.0"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        archive = response.read()
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        names = zf.namelist()
        for member in (CSV_MEMBER, PROMPTS_MEMBER):
            if member not in names:
                raise ValueError(f"{KAGGLE_DATASET} v{KAGGLE_DATASET_VERSION} archive does not contain {member}")
        return zf.read(CSV_MEMBER), zf.read(PROMPTS_MEMBER)


def _verify_sha256(content: bytes, expected: str, label: str) -> str:
    digest = hashlib.sha256(content).hexdigest()
    if digest != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, got {digest}; refusing to prepare an unpinned file"
        )
    return digest


def _render(content: bytes) -> tuple[str, int]:
    rows = list(csv.DictReader(io.StringIO(content.decode("utf-8"))))
    if not rows or tuple(rows[0].keys()) != COLUMNS:
        raise ValueError(f"unexpected FACTS Grounding columns: {list(rows[0].keys()) if rows else 'no rows'}")
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} FACTS Grounding rows, found {len(rows)}")
    seen: set[str] = set()
    output: list[str] = []
    for index, row in enumerate(rows, start=1):
        full_prompt = row["full_prompt"]
        if not full_prompt.strip() or not row["user_request"].strip() or not row["context_document"].strip():
            raise ValueError(f"row {index} has an empty prompt field")
        row_sha256 = hashlib.sha256("\t".join(row[column] for column in COLUMNS).encode("utf-8")).hexdigest()
        if row_sha256 in seen:
            raise ValueError(f"duplicate FACTS Grounding row at CSV index {index}")
        seen.add(row_sha256)
        output.append(
            json.dumps(
                {
                    "id": f"facts_grounding_v2_public_{index:04d}",
                    "system_instruction": row["system_instruction"],
                    "user_request": row["user_request"],
                    "context_document": row["context_document"],
                    "full_prompt": full_prompt,
                    "domain": row["domain"],
                    "type": row["type"],
                    "high_level_type": row["high_level_type"],
                    "context_document_chars": len(row["context_document"]),
                    "full_prompt_chars": len(full_prompt),
                    "row_sha256": row_sha256,
                    # The official starter sends `full_prompt` as the single user message (no system message).
                    "responses_create_params": {"input": [{"role": "user", "content": full_prompt}]},
                    "upstream": {
                        "dataset": KAGGLE_DATASET,
                        "version": KAGGLE_DATASET_VERSION,
                        "file": CSV_MEMBER,
                        "csv_sha256": CSV_SHA256,
                        "license": LICENSE,
                    },
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
    return "".join(output), len(output)


def _atomic_write(content: str, output_fpath: Path) -> None:
    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(dir=output_fpath.parent, prefix=f".{output_fpath.name}.")
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, output_fpath)
    finally:
        temp_path.unlink(missing_ok=True)


def prepare(source_csv: Optional[str] = None, output_fpath: Optional[str] = None) -> Path:
    """Download (or read ``source_csv``), verify the pinned SHA-256 values, and write the Gym JSONL."""
    if source_csv:
        content = Path(source_csv).read_bytes()
    else:
        content, prompts = _download_members()
        _verify_sha256(prompts, PROMPTS_SHA256, PROMPTS_MEMBER)
    _verify_sha256(content, CSV_SHA256, CSV_MEMBER)
    rendered, count = _render(content)
    target = Path(output_fpath) if output_fpath else OUTPUT_FPATH
    _atomic_write(rendered, target)
    print(f"Wrote {count} FACTS Grounding public rows to {target}")
    return target


if __name__ == "__main__":
    prepare()
