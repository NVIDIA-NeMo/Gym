# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate-only validation of the frozen final comparison; no network calls."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import jsonschema

from .core import ROOT, Dataset, bounded_schema, digest, judge, read, summarize
from .transport import delivered, replay_sse


def replay(
    dataset: Dataset,
    evaluation: Path,
    *,
    expected_counts: dict[str, tuple[int, int]],
    old_root: str | None = None,
    new_root: Path | None = None,
) -> dict:
    """Remap receipt paths without rewriting bytes or original payload manifest bindings."""
    pin = read(ROOT / "source_manifest.json")
    index = evaluation / "provenance.private.jsonl"
    if digest(index) != pin["source_receipt_index_sha256"]:
        raise ValueError("Historical receipt index hash mismatch")

    def resolve(value: str) -> Path:
        p = Path(value)
        if old_root is not None:
            if new_root is None:
                raise ValueError("Path remapping requires both roots")
            p = new_root / p.relative_to(old_root)
        return p

    records = [__import__("json").loads(line) for line in index.read_text().splitlines()]
    results, seen, failures = {}, set(), Counter()
    bounded_count = 0
    for record in records:
        try:
            key = (record["variant"], record["item_id"])
            if key in seen:
                raise ValueError("duplicate_cell")
            seen.add(key)
            files = {}
            for kind in ["grade", "payload", "inference", "raw_response"]:
                p = resolve(record[kind])
                if digest(p) != record[kind + "_sha256"]:
                    raise ValueError("receipt_hash_mismatch")
                files[kind] = p
            grade, payload, inference = (read(files[k]) for k in ["grade", "payload", "inference"])
            identifier = record["item_id"]
            if payload["input_request"] != dataset.request(identifier):
                raise ValueError("input_mismatch")
            if payload["constraints"] != dataset.criteria[identifier]["constraints"]:
                raise ValueError("criteria_mismatch")
            if payload["benchmark_manifest_sha256"] != record["source_manifest_sha256"]:
                raise ValueError("original_manifest_mismatch")
            if grade["status"] != "graded" or grade["inference_sha256"] != record["inference_sha256"]:
                raise ValueError("grade_binding_mismatch")
            if inference["status"] != "generated" or inference.get("delivery_error"):
                raise ValueError("incomplete_historical_delivery")
            if digest(resolve(inference["raw_response_path"])) != inference["raw_response_sha256"]:
                raise ValueError("raw_binding_mismatch")
            response = inference["response"]
            body = inference["request_body"]
            if {k: body[k] for k in ["messages", "tools"] if k in body} != dataset.request(identifier):
                raise ValueError("actual_request_mismatch")
            if (
                body["model"] != record["model"]
                or body["temperature"] != 0.7
                or body.get("seed") != record["seed"]
                or body["max_tokens"] != record["max_tokens"]
            ):
                raise ValueError("generation_controls_mismatch")
            if body.get("stream") and replay_sse(files["raw_response"]) != response:
                raise ValueError("SSE_replay_mismatch")
            if payload["response"] != delivered(response):
                raise ValueError("response_binding_mismatch")
            review = grade["review_receipt"]
            if review.get("raw_path") and digest(resolve(review["raw_path"])) != review["raw_sha256"]:
                raise ValueError("judge_raw_hash_mismatch")
            if review.get("raw_review_is_mechanical_projection"):
                raise ValueError("unexpected_mechanical_projection")
            if record.get("judge_delivery_schema"):
                bounded_count += 1
                schema_path = resolve(record["judge_delivery_schema"])
                freeze_path = resolve(record["judge_delivery_schema_freeze"])
                freeze = read(freeze_path)
                if (
                    digest(schema_path) != record["judge_delivery_schema_sha256"]
                    or digest(freeze_path) != record["judge_delivery_schema_freeze_sha256"]
                    or not freeze["frozen_before_call"]
                    or digest(schema_path) != freeze["schema_sha256"]
                    or digest(files["payload"]) != freeze["source_payload_sha256"]
                ):
                    raise ValueError("transport_schema_binding_mismatch")
                # Locate the canonical source by its recorded hash, not a guessed payload sibling.
                matches = [
                    p
                    for p in schema_path.parents
                    if (p / "judge.schema.json").exists()
                    and digest(p / "judge.schema.json") == freeze["source_judge_schema_sha256"]
                ]
                if not matches or read(schema_path) != bounded_schema(read(matches[0] / "judge.schema.json"), payload):
                    raise ValueError("transport_schema_mismatch")
                jsonschema.validate(review["raw_review"], read(schema_path))
            checked = judge.validate_review(review["raw_review"], payload)
            if any(grade.get(k) != v for k, v in checked.items()):
                raise ValueError("recorded_grade_mismatch")
            results.setdefault(record["variant"], []).append(checked)
        except Exception as error:  # noqa: BLE001 - deliberate per-cell/transport accounting boundary
            # Do not expose transcript/rationale or jsonschema instance text.
            reason = str(error) if type(error) is ValueError and str(error).isidentifier() else type(error).__name__
            failures[reason] += 1
    expected = expected_counts
    summaries = {name: summarize(rows, dataset) for name, rows in results.items()}
    passed = (
        len(records) == len(expected) * len(dataset.inputs)
        and not failures
        and set(summaries) == set(expected)
        and all(
            (summaries[n]["pass"], summaries[n]["fail"]) == pair and summaries[n]["not_scored"] == 0
            for n, pair in expected.items()
        )
        and bounded_count == 4
    )
    return {
        "passed": passed,
        "records": len(records),
        "models": summaries,
        "mismatches": dict(failures),
        "bound_transport_schemas": bounded_count,
        "new_inference_or_judge_calls": 0,
        "receipt_index_sha256": digest(index),
    }
