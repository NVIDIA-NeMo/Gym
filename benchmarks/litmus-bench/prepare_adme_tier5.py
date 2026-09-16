# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the ADME Tier-5 sub10 splits for evaluation with the litmus_agent verifier.

Companion to ``prepare.py``. The committed 10-question examples support
out-of-the-box smoke runs. Larger splits are local exports (not pinned HF
revisions), so their rows are validated and copied rather than downloaded.

Three subsets, each its own benchmark dataset so their metrics stay separate:

* ``direct``     -- predict a property for one molecule (float regression)
* ``analogue``   -- predict a property for an analogue, given a reference (float)
* ``comparison`` -- do two molecules differ by less than a threshold (bool)

Rows are dropped when a ``float`` row carries no ``match`` rule. Without one the
verifier falls back to ``isclose(rel_tol=1e-6)``, which no regression prediction
can satisfy, so such rows would score a guaranteed 0.0 and depress every
aggregate they appear in. Affected properties are Log MPPB / MBPB / MGMB.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


BENCHMARK_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCHMARK_DIR / "data"
BUNDLED_EXAMPLE_DIR = DATA_DIR / "examples"
SOURCE_DIR_ENV = "ADME_TIER5_SOURCE_DIR"
SOURCE_DIR = Path(os.environ.get(SOURCE_DIR_ENV, Path.home() / "chemLLM_setup/adme_tier5/data"))

# Which source split to materialize. `example` (10 rows) is handy for smoke runs.
SPLIT = os.environ.get("ADME_TIER5_SPLIT", "validation")

SUBSETS = {
    "direct": "adme_tier5_direct_sub10",
    "analogue": "adme_tier5_analogue_sub10",
    "comparison": "adme_tier5_comparison_sub10",
}

SUPPORTED_ANSWER_TYPES = frozenset({"float", "bool"})
REQUIRED_FIELDS = frozenset(
    {
        "responses_create_params",
        "expected_answer",
        "answer_type",
        "property",
        "method",
        "output_regex",
        "uuid",
    }
)


def _row_error(subset: str, row_number: int, message: str) -> ValueError:
    return ValueError(f"ADME Tier-5 {subset} row {row_number}: {message}")


def _validate_row(subset: str, row: Mapping[str, Any], row_number: int) -> None:
    missing = sorted(REQUIRED_FIELDS - row.keys())
    if missing:
        raise _row_error(subset, row_number, f"missing required fields: {missing}")

    answer_type = row["answer_type"]
    if answer_type not in SUPPORTED_ANSWER_TYPES:
        raise _row_error(
            subset,
            row_number,
            f"unsupported answer_type {answer_type!r}; expected one of {sorted(SUPPORTED_ANSWER_TYPES)}",
        )

    expected_answer = row["expected_answer"]
    if isinstance(expected_answer, bool) or not isinstance(expected_answer, (int, float)):
        raise _row_error(subset, row_number, f"expected_answer must be numeric, got {expected_answer!r}")
    if answer_type == "bool" and expected_answer not in {0, 1}:
        raise _row_error(subset, row_number, f"bool expected_answer must be 0 or 1, got {expected_answer}")

    # The verifier requires exactly one capture group; fail here rather than at
    # scoring time, where it would surface as a 500 mid-run.
    try:
        pattern = re.compile(row["output_regex"])
    except re.error as exc:
        raise _row_error(subset, row_number, f"invalid output_regex: {exc}") from exc
    if pattern.groups != 1:
        raise _row_error(subset, row_number, f"output_regex must have exactly one capture group, got {pattern.groups}")

    responses_create_params = row["responses_create_params"]
    if not isinstance(responses_create_params, Mapping):
        raise _row_error(subset, row_number, "responses_create_params must be an object")
    input_messages = responses_create_params.get("input")
    if not isinstance(input_messages, list) or not input_messages:
        raise _row_error(subset, row_number, "responses_create_params.input must be a non-empty list")
    for message_number, message in enumerate(input_messages, start=1):
        if not isinstance(message, Mapping):
            raise _row_error(subset, row_number, f"input message {message_number} must be an object")
        if not isinstance(message.get("role"), str) or not isinstance(message.get("content"), str):
            raise _row_error(subset, row_number, f"input message {message_number} must have string role and content")
    if responses_create_params.get("tools"):
        raise _row_error(subset, row_number, "rows must not expose tools")


def _is_unscorable(row: Mapping[str, Any]) -> bool:
    """A float row with no reward rule can only ever score 0.0."""
    return row["answer_type"] == "float" and not row.get("match")


def _source_fpath(subset: str, source_dir: Path) -> Path:
    if SPLIT == "example" and SOURCE_DIR_ENV not in os.environ:
        return BUNDLED_EXAMPLE_DIR / f"adme-tier5-{subset}_example.jsonl"
    return source_dir / SUBSETS[subset] / "nemo_gym_data" / f"{SPLIT}.jsonl"


def _render_subset(subset: str, source_dir: Path) -> tuple[str, int, int, dict[str, int]]:
    source_fpath = _source_fpath(subset, source_dir)
    if not source_fpath.is_file():
        raise FileNotFoundError(f"ADME Tier-5 {subset}: no source file at {source_fpath}")

    rendered: list[str] = []
    seen_uuids: set[str] = set()
    dropped: dict[str, int] = {}

    with source_fpath.open(encoding="utf-8") as source:
        for row_number, line in enumerate(source, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise _row_error(subset, row_number, "row must be an object")
            _validate_row(subset, row, row_number)

            uuid = row["uuid"]
            if uuid in seen_uuids:
                raise _row_error(subset, row_number, f"duplicate uuid {uuid!r}")
            seen_uuids.add(uuid)

            if _is_unscorable(row):
                prop = row["property"]
                dropped[prop] = dropped.get(prop, 0) + 1
                continue

            row = dict(row)
            # Gym's benchmark collation assigns the agent; a stale reference in
            # the prepared artifact would override it.
            row.pop("agent_ref", None)
            rendered.append(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")

    if not rendered:
        raise ValueError(f"ADME Tier-5 {subset}: no rows survived preparation")

    return "".join(rendered), len(rendered), len(seen_uuids), dropped


def _atomic_write(output_path: Path, content: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temp_name = tempfile.mkstemp(dir=output_path.parent, prefix=f".{output_path.name}.")
    temp_path = Path(temp_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp_path, output_path)
    finally:
        temp_path.unlink(missing_ok=True)


def prepare(subset: str) -> Path:
    """Validate and materialize one ADME subset for ``gym eval prepare``."""
    if subset not in SUBSETS:
        raise ValueError(f"Unknown ADME Tier-5 subset {subset!r}; expected one of {sorted(SUBSETS)}")

    content, kept, total, dropped = _render_subset(subset, SOURCE_DIR)
    output_fpath = DATA_DIR / f"adme-tier5-{subset}_benchmark.jsonl"
    _atomic_write(output_fpath, content)

    print(f"Wrote {kept}/{total} ADME Tier-5 {subset} ({SPLIT}) rows to {output_fpath}")
    for prop, count in sorted(dropped.items(), key=lambda item: -item[1]):
        print(f"  dropped {count} unscorable rows (no match rule): {prop}")
    return output_fpath


def prepare_all() -> list[Path]:
    """Convenience entry point for materializing all three local subsets."""
    return [prepare(subset) for subset in SUBSETS]


if __name__ == "__main__":
    requested_subset = os.environ.get("ADME_TIER5_SUBSET")
    if requested_subset:
        prepare(requested_subset)
    else:
        prepare_all()
