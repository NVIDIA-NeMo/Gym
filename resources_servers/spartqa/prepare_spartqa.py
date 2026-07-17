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
"""Build the SpartQA Gym dataset from the public ``mteb/SpartQA`` HF dataset.

Ports ``benchmarks/spartqa/byob_spartqa.py::build_records``: joins the MTEB
``queries`` / ``corpus`` / ``qrels`` splits into one row per query whose
``target`` is the accepted answer phrase (all accepted phrases in
``all_targets``), then renders the shared ``PROMPT`` and writes a Gym task per
query. ``target`` / ``all_targets`` ride along as extra fields consumed by
``app.py``'s verify().

Requires the ``datasets`` package (HF) at prep time only.

Usage::

    python resources_servers/spartqa/prepare_spartqa.py --split test \\
        --output data/spartqa_test.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, List

from app import PROMPT


_HF_DATASET = "mteb/SpartQA"
_DEFAULT_SPLIT = "test"
_AGENT = {"type": "responses_api_agents", "name": "spartqa_simple_agent"}


def _load_split(config: str, split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "The 'datasets' package is required to build mteb/SpartQA. "
            "Install it (pip install datasets) or stage a prepared JSONL."
        ) from exc
    return load_dataset(_HF_DATASET, config, split=split)


def _unique_preserve_order(values: List[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for value in values:
        key = value.strip().casefold()
        if key and key not in seen:
            seen.add(key)
            result.append(value.strip())
    return result


def build_records(split: str = _DEFAULT_SPLIT) -> List[dict[str, Any]]:
    """Join the MTEB ``queries`` / ``corpus`` / ``qrels`` splits into rows."""
    corpus = {
        row["_id"]: str(row["text"]).strip()
        for row in _load_split("corpus", split)
        if str(row.get("text", "")).strip()
    }
    queries = {
        row["_id"]: str(row["text"]).strip()
        for row in _load_split("queries", split)
        if str(row.get("text", "")).strip()
    }

    qrels_by_query: dict[str, List[str]] = {}
    for row in _load_split("qrels", split):
        try:
            score = int(row.get("score", 1))
        except (TypeError, ValueError):
            score = 1
        if score <= 0:
            continue
        qrels_by_query.setdefault(row["query-id"], []).append(row["corpus-id"])

    records: List[dict[str, Any]] = []
    for query_id in sorted(queries):
        answer_ids = [
            doc_id for doc_id in qrels_by_query.get(query_id, []) if doc_id in corpus
        ]
        answers = _unique_preserve_order([corpus[doc_id] for doc_id in answer_ids])
        if not answers:
            continue
        records.append(
            {
                "question": queries[query_id],
                "target": answers[0],
                "all_targets": answers,
            }
        )
    return records


def _to_task(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "responses_create_params": {
            "input": [{"role": "user", "content": PROMPT.format(question=record["question"])}]
        },
        "target": record["target"],
        "all_targets": record["all_targets"],
        # ``all_targets`` is a list and is dropped by the nemo-evaluator
        # ``gym://...protocol=native`` driver (it forwards only top-level scalar
        # fields to /verify). Mirror the full accepted set into verifier_metadata,
        # which the driver forwards intact, so verify() sees every phrase.
        "verifier_metadata": {
            "target": record["target"],
            "all_targets": record["all_targets"],
        },
        "agent_ref": _AGENT,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the SpartQA Gym dataset.")
    parser.add_argument("--split", default=_DEFAULT_SPLIT, help="MTEB split (default: test)")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "data" / "spartqa_test.jsonl"),
        help="Output JSONL path",
    )
    args = parser.parse_args()

    records = build_records(args.split)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(_to_task(record), ensure_ascii=False) + "\n")
    print(f"SpartQA: wrote {len(records)} rows -> {out_path}")


if __name__ == "__main__":
    main()
