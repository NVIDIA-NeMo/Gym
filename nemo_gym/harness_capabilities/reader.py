# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict JSONL reading without merging or comparing duplicate evidence."""

import copy
import hashlib
import json
from pathlib import Path
from typing import Iterator


PAYLOADS = ("request", "response", "request_raw", "response_raw")


def _reject_constant(value: str) -> None:
    raise ValueError("nonfinite JSON number")


def json_rows(path: Path) -> Iterator[tuple[int, dict]]:
    """Read JSONL strictly; malformed/truncated rows are checker errors."""
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    value = json.loads(line, parse_constant=_reject_constant)
                except (ValueError, RecursionError) as exc:
                    raise ValueError(f"{path.name}:{number}: invalid JSON") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"{path.name}:{number}: expected an object")
                yield number, value


def digest_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def hydrate_record(record: dict, *, capture_dir: Path | None = None) -> dict:
    """Copy a record without synthesizing evidence at designated paths.

    capture_dir is retained for CLI compatibility and input provenance only.
    Capture sidecars neither fill missing canonical fields nor invalidate them.
    """
    return copy.deepcopy(record)
