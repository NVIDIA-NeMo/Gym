# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from collections import Counter
from pathlib import Path

import pytest

from resources_servers.synthetic_tool_use_simulation.scripts.sample_materialized_dataset import (
    SampleVariant,
    extract_sampling_fields,
    sample_materialized_datasets,
)


def make_row(source_index: int, row_index: int, *, gt_transfer: bool) -> dict:
    original_dataset_name = "full_materialized"
    return {
        "id": f"{original_dataset_name}_{source_index}_{row_index:06d}",
        "domain_name": f"domain-{source_index}-{row_index % 4}",
        "policy": "Use the appropriate tool.",
        "tools": [],
        "customer_scenario": {
            "reason_for_contact": f"request-{row_index}",
            "outside_policy_scope": gt_transfer,
        },
        "metadata": {
            "dataset_name": original_dataset_name,
            "source_name": f"source-{source_index}",
            "source_index": source_index,
            "outside_policy_scope": gt_transfer,
        },
        "responses_create_params": {"input": []},
        "agent_ref": {"name": "test-agent"},
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row) + "\n")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_extract_sampling_fields_supports_compact_json() -> None:
    raw_line = b'{"metadata":{"source_index":7,"outside_policy_scope":false}}\n'
    assert extract_sampling_fields(raw_line, line_number=1) == (7, False)


def test_extract_sampling_fields_rejects_conflicting_copies() -> None:
    raw_line = (
        b'{"customer_scenario":{"outside_policy_scope":true},'
        b'"metadata":{"source_index":7,"outside_policy_scope":false}}\n'
    )
    with pytest.raises(ValueError, match="conflicting outside_policy_scope"):
        extract_sampling_fields(raw_line, line_number=1)


def test_sample_materialized_datasets_balances_sources_and_removes_gt_transfer(tmp_path: Path) -> None:
    input_path = tmp_path / "full.jsonl"
    rows = [
        make_row(source_index, row_index, gt_transfer=row_index % 4 == 0)
        for source_index in (0, 1)
        for row_index in range(20)
    ]
    write_rows(input_path, rows)
    all_output = tmp_path / "sample.jsonl"
    no_transfer_output = tmp_path / "sample_no_transfer.jsonl"
    variants = [
        SampleVariant(all_output, "sample", seed=123),
        SampleVariant(no_transfer_output, "sample_no_transfer", seed=124, exclude_gt_transfer=True),
    ]

    reports = sample_materialized_datasets(
        input_path,
        variants,
        rows_per_source=10,
        bucket_count=4,
        temp_dir=tmp_path,
        progress_every=0,
    )

    all_rows = read_rows(all_output)
    no_transfer_rows = read_rows(no_transfer_output)
    assert Counter(row["metadata"]["source_index"] for row in all_rows) == {0: 10, 1: 10}
    assert Counter(row["metadata"]["source_index"] for row in no_transfer_rows) == {0: 10, 1: 10}
    assert all(not row["customer_scenario"]["outside_policy_scope"] for row in no_transfer_rows)
    assert reports[1]["gt_transfer_rows_written"] == 0
    assert reports[0]["selected"]["domain_count"] > 2

    for row in all_rows:
        assert row["id"].startswith("sample_")
        assert row["metadata"]["dataset_name"] == "sample"
        assert row["metadata"]["sampled_from_dataset_name"] == "full_materialized"
        assert row["metadata"]["sampled_from_row_id"].startswith("full_materialized_")
        assert row["metadata"]["sampling_seed"] == 123


def test_sample_materialized_datasets_is_deterministic(tmp_path: Path) -> None:
    input_path = tmp_path / "full.jsonl"
    write_rows(
        input_path,
        [make_row(source_index, row_index, gt_transfer=False) for source_index in (0, 1) for row_index in range(12)],
    )

    first_variants = [
        SampleVariant(tmp_path / "first.jsonl", "sample", seed=9),
        SampleVariant(tmp_path / "first_no_transfer.jsonl", "sample_no_transfer", seed=10, exclude_gt_transfer=True),
    ]
    second_variants = [
        SampleVariant(tmp_path / "second.jsonl", "sample", seed=9),
        SampleVariant(tmp_path / "second_no_transfer.jsonl", "sample_no_transfer", seed=10, exclude_gt_transfer=True),
    ]
    for variants in (first_variants, second_variants):
        sample_materialized_datasets(
            input_path,
            variants,
            rows_per_source=6,
            bucket_count=3,
            temp_dir=tmp_path,
            progress_every=0,
        )

    assert (tmp_path / "first.jsonl").read_bytes() == (tmp_path / "second.jsonl").read_bytes()
    assert (tmp_path / "first_no_transfer.jsonl").read_bytes() == (tmp_path / "second_no_transfer.jsonl").read_bytes()


def test_sample_materialized_datasets_rejects_insufficient_no_transfer_rows(tmp_path: Path) -> None:
    input_path = tmp_path / "full.jsonl"
    write_rows(
        input_path,
        [
            make_row(source_index, row_index, gt_transfer=row_index >= 2)
            for source_index in (0, 1)
            for row_index in range(5)
        ],
    )
    variants = [
        SampleVariant(tmp_path / "sample.jsonl", "sample", seed=1),
        SampleVariant(
            tmp_path / "sample_no_transfer.jsonl",
            "sample_no_transfer",
            seed=2,
            exclude_gt_transfer=True,
        ),
    ]

    with pytest.raises(ValueError, match="eligible non-transfer rows"):
        sample_materialized_datasets(
            input_path,
            variants,
            rows_per_source=3,
            temp_dir=tmp_path,
            progress_every=0,
        )
