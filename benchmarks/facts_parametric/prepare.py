# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the public FACTS Parametric set (FACTS Benchmark Suite, arXiv:2512.10791).

Source of record: the Kaggle dataset ``kaggle/facts-parametric-public-examples`` (Apache 2.0),
version 2 (2025-12-08), file ``FACTS-Parametric-public.csv`` with 1,052 rows and the columns
``url``, ``query``, ``answer``, ``topic``. The private 1,052-item half is held by Kaggle and is not
available; this adapter scores the public half only.

The CSV is downloaded through Kaggle's public dataset download endpoint (no credentials are needed for
public datasets) and verified against a pinned SHA-256 before any row is emitted. Row ids are the
CSV order (``facts_parametric_public_0001`` ...), which is stable because the file content is pinned.

Rows are emitted raw (no ``responses_create_params.input``); the benchmark config applies
``benchmarks/prompts/generic/default.yaml``, whose ``user: "{question}"`` template is byte-identical to
the official starter notebook's ``QUERY_TEMPLATE = "{question}"``.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional


BENCHMARK_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "facts_parametric_public.jsonl"

KAGGLE_DATASET = "kaggle/facts-parametric-public-examples"
KAGGLE_DATASET_VERSION = 2
KAGGLE_DOWNLOAD_URL = (
    f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_DATASET}?datasetVersionNumber={KAGGLE_DATASET_VERSION}"
)
KAGGLE_DATASET_URL = f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}"
CSV_MEMBER = "FACTS-Parametric-public.csv"
CSV_SHA256 = "23fdc39d681656c87c6790b158f3d3a54f335903691a3285599c67a1578439f8"  # pragma: allowlist secret
EXPECTED_ROWS = 1052
LICENSE = "Apache 2.0"
COLUMNS = ("url", "query", "answer", "topic")


def _download_csv_bytes(timeout: float = 120.0) -> bytes:
    request = urllib.request.Request(
        KAGGLE_DOWNLOAD_URL, headers={"User-Agent": "nemo-gym-facts-parametric-prepare/1.0"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        archive = response.read()
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        if CSV_MEMBER not in zf.namelist():
            raise ValueError(f"{KAGGLE_DATASET} v{KAGGLE_DATASET_VERSION} archive does not contain {CSV_MEMBER}")
        return zf.read(CSV_MEMBER)


def _verify_sha256(content: bytes) -> str:
    digest = hashlib.sha256(content).hexdigest()
    if digest != CSV_SHA256:
        raise ValueError(
            f"{CSV_MEMBER} SHA-256 mismatch: expected {CSV_SHA256}, got {digest}; refusing to prepare an unpinned file"
        )
    return digest


def _render(content: bytes) -> tuple[str, int]:
    rows = list(csv.DictReader(io.StringIO(content.decode("utf-8"))))
    if not rows or tuple(rows[0].keys()) != COLUMNS:
        raise ValueError(f"unexpected FACTS Parametric columns: {list(rows[0].keys()) if rows else 'no rows'}")
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} FACTS Parametric rows, found {len(rows)}")
    seen: set[tuple[str, str]] = set()
    output: list[str] = []
    for index, row in enumerate(rows, start=1):
        question, answer = row["query"], row["answer"]
        if not question.strip() or not answer.strip():
            raise ValueError(f"row {index} has an empty query or answer")
        key = (question, answer)
        if key in seen:
            raise ValueError(f"duplicate FACTS Parametric row at CSV index {index}: {question!r}")
        seen.add(key)
        row_sha256 = hashlib.sha256("\t".join(row[column] for column in COLUMNS).encode("utf-8")).hexdigest()
        output.append(
            json.dumps(
                {
                    "id": f"facts_parametric_public_{index:04d}",
                    "question": question,
                    "expected_answer": answer,
                    "source_url": row["url"],
                    "topic": row["topic"],
                    "row_sha256": row_sha256,
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
    """Download (or read ``source_csv``), verify the pinned SHA-256, and write the Gym JSONL."""
    content = Path(source_csv).read_bytes() if source_csv else _download_csv_bytes()
    _verify_sha256(content)
    rendered, count = _render(content)
    target = Path(output_fpath) if output_fpath else OUTPUT_FPATH
    _atomic_write(rendered, target)
    print(f"Wrote {count} FACTS Parametric public rows to {target}")
    return target


if __name__ == "__main__":
    prepare()
