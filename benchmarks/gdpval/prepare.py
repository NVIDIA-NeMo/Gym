# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Prepare the GDPVal benchmark JSONL.

By default downloads the ``openai/gdpval`` HuggingFace dataset; alternatively
converts a local GDPVal CSV export. Either way each row has
``responses_create_params`` (an empty input — the Stirrup agent builds the
actual prompt from the top-level ``prompt`` / ``sector`` / ``occupation``
fields) plus task metadata at the top level so the GDPVal resources server can
pick them up via /verify.

The CSV path is *policy-safe*: the ``deliverable_*`` answer files are
validated but never emitted into the policy-visible row (a deliverable would
leak the answer to the policy model).
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "gdpval_benchmark.jsonl"

HF_DATASET = "openai/gdpval"
HF_SPLIT = "train"

# Columns a GDPVal CSV export must carry. ``deliverable_*`` are
# required (validated) but deliberately NOT emitted into the gym row.
CSV_REQUIRED_COLUMNS = (
    "task_id",
    "sector",
    "occupation",
    "prompt",
    "reference_files",
    "reference_file_urls",
    "deliverable_files",
    "deliverable_file_urls",
    "rubric_pretty",
    "rubric_json",
)

# CSV cells holding JSON lists.
_JSON_FIELDS = (
    "reference_files",
    "reference_file_urls",
    "deliverable_files",
    "deliverable_file_urls",
    "rubric_json",
)


def _parse_json_list(value: str | None, *, record_num: int, field: str) -> list[Any]:
    """Decode a required JSON-list cell and raise a record-located error."""
    if value is None:
        raise ValueError(f"record {record_num}: missing JSON field {field}")
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"record {record_num}: invalid JSON in {field}: {exc}") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"record {record_num}: {field} must decode to a list")
    return parsed


def _require_string_list(value: list[Any], *, record_num: int, field: str) -> list[str]:
    if not all(isinstance(item, str) for item in value):
        raise ValueError(f"record {record_num}: {field} must contain only strings")
    return value


def prepare_gdpval_csv(source_csv: str | Path, output_fpath: str | Path) -> Path:
    """Convert a GDPVal CSV export into policy-safe NeMo-Gym JSONL.

    One output row per task carrying only the metadata the Stirrup agent + GDPVal
    resources server need. ``deliverable_files`` / ``deliverable_file_urls`` (the
    answer) are validated for consistency but never written into the row.
    """
    source_csv = Path(source_csv)
    output_fpath = Path(output_fpath)
    output_fpath.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    seen_task_ids: set[str] = set()
    with source_csv.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, strict=True)
        fieldnames = reader.fieldnames or []
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError("CSV has duplicate column names")
        columns = set(fieldnames)
        missing = [column for column in CSV_REQUIRED_COLUMNS if column not in columns]
        if missing:
            raise ValueError(f"missing required columns: {', '.join(missing)}")

        for record_num, row in enumerate(reader, start=1):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"record {record_num}: wrong number of CSV columns")
            parsed = {
                field: _parse_json_list(row.get(field), record_num=record_num, field=field) for field in _JSON_FIELDS
            }
            for field in ("reference_files", "reference_file_urls", "deliverable_files", "deliverable_file_urls"):
                parsed[field] = _require_string_list(parsed[field], record_num=record_num, field=field)
            if not all(isinstance(item, dict) for item in parsed["rubric_json"]):
                raise ValueError(f"record {record_num}: rubric_json must contain only objects")

            n_ref, n_ref_url = len(parsed["reference_files"]), len(parsed["reference_file_urls"])
            if n_ref != n_ref_url:
                raise ValueError(f"record {record_num}: {n_ref} reference files but {n_ref_url} reference URLs")
            n_del, n_del_url = len(parsed["deliverable_files"]), len(parsed["deliverable_file_urls"])
            if n_del != n_del_url:
                raise ValueError(f"record {record_num}: {n_del} deliverable files but {n_del_url} deliverable URLs")

            task_id = row["task_id"]
            prompt = row["prompt"]
            if not task_id.strip() or not prompt.strip():
                raise ValueError(f"record {record_num}: task_id and prompt must be non-empty")
            if task_id in seen_task_ids:
                raise ValueError(f"record {record_num}: duplicate task_id '{task_id}'")
            seen_task_ids.add(task_id)

            records.append(
                {
                    # Empty input: the Stirrup agent constructs the user prompt
                    # from the top-level ``prompt`` field at runtime.
                    "responses_create_params": {"input": []},
                    "task_id": task_id,
                    "sector": row.get("sector", ""),
                    "occupation": row.get("occupation", ""),
                    "prompt": prompt,
                    "reference_files": parsed["reference_files"],
                    "reference_file_urls": parsed["reference_file_urls"],
                    "rubric_json": parsed["rubric_json"],
                    "rubric_pretty": row.get("rubric_pretty", ""),
                }
            )

    if not records:
        raise ValueError(f"CSV contains no data records: {source_csv}")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=output_fpath.parent,
            prefix=f".{output_fpath.name}.",
            suffix=".tmp",
            delete=False,
        ) as out:
            temporary_path = Path(out.name)
            for record in records:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary_path.chmod(0o600)
        temporary_path.replace(output_fpath)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return output_fpath


def _prepare_from_hf(output_fpath: Path) -> Path:
    from datasets import load_dataset

    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    # Pass HF_TOKEN explicitly — ``load_dataset`` doesn't always pick it up from
    # the env, and GDPVal's bucket aggressively rate-limits anonymous IPs.
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    ds = load_dataset(HF_DATASET, split=HF_SPLIT, token=hf_token)

    with output_fpath.open("w") as f:
        for row in ds:
            record = {
                "responses_create_params": {"input": []},
                "task_id": row["task_id"],
                "sector": row.get("sector", ""),
                "occupation": row.get("occupation", ""),
                "prompt": row["prompt"],
                "reference_files": row.get("reference_files", []),
                "reference_file_urls": row.get("reference_file_urls", []),
                "rubric_json": row.get("rubric_json", {}),
                "rubric_pretty": row.get("rubric_pretty", ""),
            }
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {len(ds)} tasks to {output_fpath}")
    return output_fpath


def prepare(source_csv: str | Path | None = None, output_fpath: str | Path | None = None) -> Path:
    """Prepare the GDPVal benchmark JSONL.

    With ``source_csv`` set, convert a local CSV export to
    ``output_fpath`` (default ``OUTPUT_FPATH``) and restrict it to owner-only
    (0600) since it may carry private reference data. Otherwise download the
    ``openai/gdpval`` HF dataset into ``output_fpath``.
    """
    output_fpath = Path(output_fpath) if output_fpath is not None else OUTPUT_FPATH
    if source_csv is not None:
        result = prepare_gdpval_csv(source_csv, output_fpath)
        os.chmod(result, 0o600)
        return result
    return _prepare_from_hf(output_fpath)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare the GDPVal benchmark JSONL.")
    parser.add_argument(
        "--source-csv",
        default=None,
        help="GDPVal CSV export to convert (default: download from HuggingFace).",
    )
    parser.add_argument(
        "--output",
        "--output-fpath",
        "--output_fpath",  # the spelling sibling benchmarks' prepare.py accept
        dest="output",
        default=None,
        help="Output JSONL path (default: data/gdpval_benchmark.jsonl).",
    )
    args = parser.parse_args()
    prepare(source_csv=args.source_csv, output_fpath=args.output)
