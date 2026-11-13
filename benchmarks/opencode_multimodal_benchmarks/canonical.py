# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate released task identities independently of local paths and TSV encoding."""

import hashlib
import json
from pathlib import Path

from benchmarks.opencode_multimodal_benchmarks.data import sha256_file


def fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def manifest(benchmark: str) -> dict:
    path = Path(__file__).with_name("manifests") / f"{benchmark}.json"
    return json.loads(path.read_text())


def mcq_identity(task: dict) -> dict:
    """Include scoring fields and media bytes; exclude deployment paths and harness prompts."""
    return {
        key: task[key]
        for key in ("index", "question", "answer", "options", "category", "l2_category", "cycle_category", "split")
    } | {"images": [Path(path).stem for path in task["images"]]}


def validate_mcq_tasks(tasks: list[dict], benchmark: str) -> None:
    reference = manifest(benchmark)
    ids = [task["index"] for task in tasks]
    if len(ids) != len(set(ids)) or set(ids) != set(reference["tasks"]):
        raise ValueError(f"{benchmark} IDs differ from the complete pinned public split")
    for image in {path for task in tasks for path in task["images"]}:
        path = Path(image)
        if sha256_file(path) != path.stem:
            raise ValueError(f"{benchmark} prepared image checksum mismatch: {path}")
    for task in tasks:
        if fingerprint(mcq_identity(task)) != reference["tasks"][task["index"]]:
            raise ValueError(f"{benchmark} public task/media mismatch at ID {task['index']}")
