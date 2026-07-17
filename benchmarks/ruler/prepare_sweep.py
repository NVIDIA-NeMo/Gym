# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare and combine RULER data for multiple sequence lengths."""

import json
import os
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from benchmarks.ruler.prepare_utils import DATA_DIR, prepare_helper


RULER_TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
)


def _normalize_lengths(lengths: Iterable[int]) -> list[int]:
    normalized = [int(length) for length in lengths]
    if not normalized:
        raise ValueError("At least one RULER sequence length is required")
    if any(length <= 0 for length in normalized):
        raise ValueError("RULER sequence lengths must be positive")
    if len(set(normalized)) != len(normalized):
        raise ValueError("RULER sequence lengths must be unique")
    return normalized


def _length_label(length: int) -> str:
    return f"{length // 1024}k" if length % 1024 == 0 else str(length)


def prepare(model: str, lengths: Iterable[int], **prepare_kwargs) -> Path:
    """Prepare each length with the single-length adapter and combine the rows."""
    lengths = _normalize_lengths(lengths)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_fpath = DATA_DIR / "ruler_sweep.jsonl"
    temporary = output_fpath.with_name(f".{output_fpath.name}.{os.getpid()}.tmp")
    part_fpaths = []
    task_order = {task: index for index, task in enumerate(RULER_TASKS)}

    try:
        with temporary.open("w", encoding="utf-8") as output_file:
            for length in lengths:
                part_fpath = prepare_helper(
                    output_name=f".ruler_sweep_{length}_{os.getpid()}.jsonl",
                    model=model,
                    length=length,
                    **prepare_kwargs,
                )
                part_fpaths.append(part_fpath)
                with part_fpath.open(encoding="utf-8") as part_file:
                    rows = [json.loads(line) for line in part_file]
                rows.sort(key=lambda row: task_order[row["subset"]])

                source_indices = defaultdict(int)
                for row in rows:
                    subset = row["subset"]
                    row["sequence_length"] = _length_label(length)
                    row["sequence_length_tokens"] = length
                    row["source_index"] = source_indices[subset]
                    source_indices[subset] += 1
                    output_file.write(json.dumps(row) + "\n")

        temporary.replace(output_fpath)
    finally:
        temporary.unlink(missing_ok=True)
        for part_fpath in part_fpaths:
            part_fpath.unlink(missing_ok=True)

    return output_fpath


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit("Use `gym eval prepare --benchmark ruler/config_sweep` to prepare this benchmark")
