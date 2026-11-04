# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the content-pinned Litmus-Bench held-out data for NeMo Gym."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download


BENCHMARK_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCHMARK_DIR / "data"
HF_DATASET = "nvidia/Nemotron-RL-litmus-bench-v0.2"

# Pin the public source content and select subsets by the published question families.
RELEASE_FILE = (
    "validation.jsonl",
    2056,
    "5a46a7412e0ca6ac92091fde426f639bb87e8de1312ed7e9829635d44f9dbb96",  # pragma: allowlist secret
)
SUBSET_FAMILIES = {
    "tier12": {"tier1_rdkit": 419, "tier2_rdkit": 581},
    "direct": {"tier5_admet_direct": 598},
    "analogue": {"tier5_admet_analog": 358},
    "comparison": {"tier5_admet_comparison": 100},
}
REQUIRED_FIELDS = frozenset(
    {
        "dataset",
        "question_license",
        "responses_create_params",
        "expected_answer",
        "answer_type",
        "output_regex",
        "property",
        "method",
        "metadata",
        "question_family",
    }
)


def _row_error(row_number: int, message: str) -> ValueError:
    return ValueError(f"Litmus-Bench held-out row {row_number}: {message}")


def _validate_row(row: Mapping[str, Any], row_number: int) -> None:
    missing = sorted(REQUIRED_FIELDS - row.keys())
    if missing:
        raise _row_error(row_number, f"missing required fields: {missing}")
    if row["method"] != "direct":
        raise _row_error(row_number, f"method must be 'direct', got {row['method']!r}")
    if row["answer_type"] not in {"float", "bool", "string"}:
        raise _row_error(row_number, f"unsupported answer_type {row['answer_type']!r}")
    expected_answer = row["expected_answer"]
    if row["answer_type"] == "string":
        if not isinstance(expected_answer, str):
            raise _row_error(row_number, f"expected_answer must be a string, got {expected_answer!r}")
    elif (
        isinstance(expected_answer, bool)
        or not isinstance(expected_answer, (int, float))
        or not math.isfinite(expected_answer)
    ):
        raise _row_error(row_number, f"expected_answer must be finite and numeric, got {expected_answer!r}")
    if row["answer_type"] == "bool" and expected_answer not in {0, 1}:
        raise _row_error(row_number, "bool expected_answer must be 0 or 1")

    for field in ("dataset", "question_license", "property", "output_regex", "question_family"):
        if not isinstance(row[field], str) or not row[field]:
            raise _row_error(row_number, f"{field} must be a non-empty string")
    try:
        pattern = re.compile(row["output_regex"])
    except re.error as exc:
        raise _row_error(row_number, f"invalid output_regex: {exc}") from exc
    if pattern.groups != 1:
        raise _row_error(row_number, "output_regex must have exactly one capture group")

    params = row["responses_create_params"]
    if not isinstance(params, Mapping):
        raise _row_error(row_number, "responses_create_params must be an object")
    messages = params.get("input")
    if not isinstance(messages, list) or not messages:
        raise _row_error(row_number, "responses_create_params.input must be a non-empty list")
    for message in messages:
        if not isinstance(message, Mapping):
            raise _row_error(row_number, "input messages must be objects")
        if not isinstance(message.get("role"), str) or not isinstance(message.get("content"), str):
            raise _row_error(row_number, "input messages must have string role and content")
    if "tools" in params and (not isinstance(params["tools"], list) or params["tools"]):
        raise _row_error(row_number, "direct-answer rows must not expose tools")

    metadata = row["metadata"]
    if not isinstance(metadata, Mapping):
        raise _row_error(row_number, "metadata must be an object")
    if not isinstance(metadata.get("uuid"), str) or not metadata["uuid"]:
        raise _row_error(row_number, "metadata.uuid must be a non-empty string")
    if not isinstance(metadata.get("sources"), list) or not metadata["sources"]:
        raise _row_error(row_number, "metadata.sources must be a non-empty list")


def _render_rows(dataset: Iterable[Mapping[str, Any]], *, expected_rows: int) -> str:
    rendered: list[str] = []
    seen_uuids: set[str] = set()
    for row_number, source_row in enumerate(dataset, start=1):
        if not isinstance(source_row, Mapping):
            raise _row_error(row_number, "row must be an object")
        row = dict(source_row)
        _validate_row(row, row_number)
        uuid = row["metadata"]["uuid"]
        if uuid in seen_uuids:
            raise _row_error(row_number, f"duplicate uuid {uuid!r}")
        seen_uuids.add(uuid)
        # Collation assigns the configured agent. Preserve prompts, numeric
        # precision, extraction regexes, match rules, licenses, and provenance.
        row.pop("agent_ref", None)
        rendered.append(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    if len(rendered) != expected_rows:
        raise ValueError(f"Litmus-Bench held-out data has {len(rendered)} rows; expected {expected_rows}")
    return "".join(rendered)


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


def _selected_families(subset: str | None) -> dict[str, int]:
    if subset is None:
        return {family: count for families in SUBSET_FAMILIES.values() for family, count in families.items()}
    if subset not in SUBSET_FAMILIES:
        raise ValueError(f"Unknown Litmus-Bench subset {subset!r}; expected one of {sorted(SUBSET_FAMILIES)}")
    return SUBSET_FAMILIES[subset]


def _source_fingerprint(subset: str | None) -> str:
    source = [HF_DATASET, RELEASE_FILE, _selected_families(subset)]
    return hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()


def _output_filename(subset: str | None) -> str:
    if subset is None:
        return "litmus-bench_benchmark.jsonl"
    if subset == "tier12":
        return "litmus-tier12_benchmark.jsonl"
    return f"adme-tier5-{subset}_benchmark.jsonl"


def is_prepared_data_current(output_path: str | Path) -> bool:
    """Reject cached artifacts from a different release or modified prepared data."""
    path = Path(output_path)
    subset = next((key for key in SUBSET_FAMILIES if path.name == _output_filename(key)), None)
    try:
        provenance = json.loads(path.with_suffix(".source.json").read_text(encoding="utf-8"))
        return provenance == {
            "source": _source_fingerprint(subset),
            "prepared_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    except (OSError, ValueError):
        return False


def prepare(*, subset: str | None = None, source_dir: str | None = None) -> Path:
    """Materialize all held-out subsets, or one subset, from the Hub or an exact local export."""
    selected = _selected_families(subset)
    filename, count, sha256 = RELEASE_FILE
    source = (
        Path(source_dir) / filename
        if source_dir is not None
        else Path(hf_hub_download(repo_id=HF_DATASET, filename=filename, repo_type="dataset"))
    )
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != sha256:
        raise ValueError(f"Litmus-Bench source checksum mismatch: {filename}; review the release before updating pins")
    source_rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    # Validate the full release before filtering so unknown or missing families
    # cannot silently disappear from an individual subset evaluation.
    _render_rows(source_rows, expected_rows=count)
    family_counts = Counter(row["question_family"] for row in source_rows)
    expected_counts = _selected_families(None)
    if family_counts != expected_counts:
        raise ValueError(f"Litmus-Bench question family counts are {dict(family_counts)}; expected {expected_counts}")
    rows = [row for row in source_rows if row["question_family"] in selected]
    content = _render_rows(rows, expected_rows=sum(selected.values()))
    output_path = DATA_DIR / _output_filename(subset)
    _atomic_write(output_path, content)
    _atomic_write(
        output_path.with_suffix(".source.json"),
        json.dumps(
            {"source": _source_fingerprint(subset), "prepared_sha256": hashlib.sha256(content.encode()).hexdigest()},
            sort_keys=True,
        )
        + "\n",
    )
    print(f"Wrote {len(rows)} Litmus-Bench rows to {output_path}")
    return output_path


if __name__ == "__main__":
    prepare()
