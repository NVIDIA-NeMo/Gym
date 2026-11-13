# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the authorized ZeroBench main split without exposing references to the agent."""

import hashlib
import io
import json
from pathlib import Path

from PIL import Image

from benchmarks.opencode_multimodal_benchmarks.data import sha256_file
from benchmarks.opencode_multimodal_benchmarks.prepare import task_row
from resources_servers.opencode_multimodal_benchmarks.zerobench import ANSWER_INSTRUCTION


DATASET_REVISION = "8fcdfd9138c802cac52b774b36946e70f738f500"
SOURCE_SHA256 = "d6e775b73e93e09eb9ef11f99c0f1e4083fcbf15c41886ae3f3e231c7d39e80a"


def prepare(parquet: str, prepared_dir: str | None = None, limit: int | str | None = None) -> Path:
    """Validate the pinned 100-question split and materialize its original image bytes."""
    import pyarrow.parquet as pq

    source = Path(parquet)
    if sha256_file(source) != SOURCE_SHA256:
        raise ValueError("ZeroBench source checksum mismatch; use the pinned official main split")
    rows = pq.read_table(source).to_pylist()
    identifiers = [str(row["question_id"]) for row in rows]
    if len(rows) != 100 or set(identifiers) != {str(i) for i in range(1, 101)}:
        raise ValueError("ZeroBench main split must contain each question ID 1..100 exactly once")
    if isinstance(limit, str):
        limit = None if limit.lower() in ("none", "null") else int(limit)
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100):
        raise ValueError("limit must be an integer in 1..100 or null")
    output = Path("benchmarks/opencode_multimodal_benchmarks/data/zerobench.jsonl")
    media_root = (Path(prepared_dir) if prepared_dir else output.with_suffix("")).resolve()
    media_root.mkdir(parents=True, exist_ok=True)
    prepared = []
    selected = rows[:limit]
    for row in selected:
        if not row["question_text"] or not row["question_answer"] or not row["question_images_decoded"]:
            raise ValueError(f"Incomplete ZeroBench question {row['question_id']}")
        images = []
        for image in row["question_images_decoded"]:
            raw = image["bytes"]
            with Image.open(io.BytesIO(raw)) as decoded:
                extension = (decoded.format or "").lower()
                decoded.verify()
            if extension not in ("png", "jpeg", "webp", "gif"):
                raise ValueError(f"Unsupported ZeroBench image format: {extension}")
            target = media_root / f"{hashlib.sha256(raw).hexdigest()}.{extension}"
            if not target.exists():
                target.write_bytes(raw)
            elif target.read_bytes() != raw:
                raise ValueError(f"Materialized ZeroBench image checksum mismatch: {target.name}")
            images.append(str(target))
        prepared.append(
            task_row(
                {
                    "index": str(row["question_id"]),
                    "benchmark": "ZeroBench",
                    "question": row["question_text"],
                    "prompt": row["question_text"] + ANSWER_INSTRUCTION,
                    "answer": row["question_answer"],
                    "images": images,
                    "split": "zerobench",
                    "expected_samples": len(selected),
                    "dataset_revision": DATASET_REVISION,
                }
            )
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prepared))
    temporary.replace(output)
    return output
