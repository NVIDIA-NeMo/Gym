# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download the ether0 benchmark from its pinned revision and write it as Gym rows."""

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from environments.ether0.prepare import format_row


OUTPUT_FPATH = Path(__file__).resolve().parent / "data" / "test.jsonl"

DATA_REVISION = "c7d5e59960087f360bc32a5006bb994324b38c35"  # pragma: allowlist secret
NUM_TASKS = 325


def prepare() -> Path:
    from datasets import load_dataset

    source = load_dataset("futurehouse/ether0-benchmark", split="test", revision=DATA_REVISION)
    # format_row is the environment's own formatter: it holds the system prompt and the
    # verifier_metadata fields the resources server reads back.
    rows = [format_row(row) for row in source]
    identities = {row["verifier_metadata"]["id"] for row in rows}
    if len(rows) != NUM_TASKS or len(identities) != NUM_TASKS:
        raise ValueError(f"Expected {NUM_TASKS} distinct ether0 tasks, got {len(rows)} rows and {len(identities)} ids")

    OUTPUT_FPATH.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=OUTPUT_FPATH.parent, delete=False) as stream:
        stream.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    os.replace(stream.name, OUTPUT_FPATH)
    return OUTPUT_FPATH


if __name__ == "__main__":
    print(prepare())
