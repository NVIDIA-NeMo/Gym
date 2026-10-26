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
"""Prepare the FrontierFinance benchmark data.

The ``gym eval prepare`` entry point, which also converts an explicit file via
``--input/--output``.

Each query becomes one sample that runs on the Finance Agent v2 tools and prompts,
with the prompt's "current date" set to the query's ``query_date``. Rubrics are
copied through verbatim: they are the scoring input.

The dataset is gated on Hugging Face and its terms forbid resharing, so the source
file is downloaded at prepare time with ``HF_TOKEN`` and never committed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

from benchmarks.finance_agent_v2.prepare import QUESTION_PROMPT, SYSTEM_PROMPT, build_tools


ENV_DIR = Path(__file__).parent
DATA_DIR = ENV_DIR / "data"

# Must match the benchmark config's `jsonl_fpath`. Anchored to the CWD because the
# CLI compares the returned path against that config value.
_BENCHMARK_REL_FPATH = "benchmarks/frontier_finance/data/frontier_finance_220q.jsonl"

HF_REPO_ID = "samaya-ai/FrontierFinance"
HF_REVISION = "36afd6dc1cbb2c4512536daf917dc28b1225a720"  # pragma: allowlist secret
SOURCE_FNAME = "frontier_finance_public.jsonl"
SOURCE_URL = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/{HF_REVISION}/{SOURCE_FNAME}"

DATE_SENTENCE = "You should answer all questions as if the current date is March 1, 2026."

RUBRIC_FIELDS = (
    "rubric_id",
    "rubric_text",
    "must_have",
    "rubric_type",
    "rubric_subtype",
    "data_source_type",
    "data_source_subtype",
)


def format_as_of_date(iso_date: str) -> str:
    """``2025-06-23`` -> ``June 23, 2025``. Raises ValueError on a malformed date."""
    d = date.fromisoformat(iso_date)
    return f"{d:%B} {d.day}, {d.year}"


def build_system_prompt(as_of_date: str) -> str:
    if DATE_SENTENCE not in SYSTEM_PROMPT:
        raise ValueError(
            f"The Finance Agent v2 system prompt no longer contains {DATE_SENTENCE!r}, so the per-query "
            "date cannot be set. Update DATE_SENTENCE to match benchmarks/finance_agent_v2/upstream_spec.json."
        )
    dated = f"You should answer all questions as if the current date is {format_as_of_date(as_of_date)}."
    return SYSTEM_PROMPT.replace(DATE_SENTENCE, dated)


def _convert_rubric(query_id: str, rubric: dict) -> dict:
    if not isinstance(rubric, dict) or not str(rubric.get("rubric_text") or "").strip():
        raise ValueError(f"query {query_id}: rubric without rubric_text: {rubric!r}")
    if rubric.get("rubric_id") is None:
        raise ValueError(f"query {query_id}: rubric without rubric_id: {rubric!r}")
    converted = {field: rubric.get(field) for field in RUBRIC_FIELDS}
    converted["must_have"] = bool(converted["must_have"])
    converted["data_source_detailed"] = list(rubric.get("data_source_detailed") or [])
    return converted


def convert_row(row: dict, tools: list[dict]) -> dict:
    query_id = str(row.get("query_id") or "").strip()
    query = str(row.get("query") or "").strip()
    as_of_date = str(row.get("query_date") or "").strip()
    if not query_id or not query or not as_of_date:
        raise ValueError(f"row is missing query_id, query or query_date: {str(row)[:200]}")
    rubrics = row.get("rubrics") or []
    if not rubrics:
        raise ValueError(f"query {query_id}: no rubrics, so it cannot be scored")

    return {
        "query_id": query_id,
        "question": query,
        "as_of_date": as_of_date,
        "use_cases": list(row.get("use_cases") or []),
        "capabilities": list(row.get("capabilities") or []),
        "rubrics": [_convert_rubric(query_id, r) for r in rubrics],
        "responses_create_params": {
            "input": [
                {"role": "system", "content": build_system_prompt(as_of_date), "type": "message"},
                {"role": "user", "content": QUESTION_PROMPT.format(question=query), "type": "message"},
            ],
            "tools": tools,
        },
    }


def convert_file(input_file: Path, output_file: Path) -> tuple[int, int]:
    """Convert the FrontierFinance JSONL to benchmark JSONL. Returns (queries, rubrics)."""
    tools = build_tools()
    samples: list[dict] = []
    seen: set[str] = set()
    for line_no, line in enumerate(input_file.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            sample = convert_row(json.loads(line), tools)
        except ValueError as e:
            raise ValueError(f"{input_file}:{line_no}: {e}") from e
        if sample["query_id"] in seen:
            raise ValueError(f"{input_file}:{line_no}: duplicate query_id {sample['query_id']}")
        seen.add(sample["query_id"])
        samples.append(sample)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    return len(samples), sum(len(s["rubrics"]) for s in samples)


def download_source(dest: Path) -> Path:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            f"{HF_REPO_ID} is gated. Accept its terms at https://huggingface.co/datasets/{HF_REPO_ID}, "
            f"then export HF_TOKEN, or place {SOURCE_FNAME} at {dest}."
        )
    request = urllib.request.Request(SOURCE_URL, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise RuntimeError(
                f"Hugging Face refused the download (HTTP {e.code}). Make sure the HF_TOKEN account has "
                f"accepted the terms at https://huggingface.co/datasets/{HF_REPO_ID}."
            ) from e
        raise
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(payload)
    return dest


def _default_source() -> Path:
    source = DATA_DIR / SOURCE_FNAME
    if source.exists():
        print(f"Using dataset: {source}")
        return source
    print(f"Downloading {SOURCE_URL} -> {source}")
    return download_source(source)


def prepare() -> Path:
    """``gym eval prepare`` entry point: convert the default source, return the output path."""
    out = Path(_BENCHMARK_REL_FPATH)
    queries, rubrics = convert_file(_default_source(), out)
    print(f"Wrote {queries} queries ({rubrics} rubrics) to {out}")
    return out.absolute()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare the FrontierFinance benchmark data.")
    parser.add_argument("--input", "-i", default=None, help="FrontierFinance JSONL. Default: download it.")
    parser.add_argument("--output", "-o", default=None, help="Output JSONL (only with --input).")
    args = parser.parse_args(argv)

    if args.input is None:
        prepare()
        return 0

    input_file = Path(args.input)
    if not input_file.exists():
        parser.error(f"Input file not found: {input_file}")
    output_file = Path(args.output) if args.output else input_file.with_name(input_file.stem + "_prepared.jsonl")
    queries, rubrics = convert_file(input_file, output_file)
    print(f"Converted {input_file} -> {output_file}: {queries} queries, {rubrics} rubrics")
    return 0


if __name__ == "__main__":
    sys.exit(main())
