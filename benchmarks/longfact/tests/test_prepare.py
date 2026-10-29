# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from benchmarks.longfact.prepare import (
    BENCHMARK_VARIANT,
    TOPICS,
    _parse_rows,
    _source_path,
    build_rows,
    select_rows,
)


def test_source_path_uses_official_layout():
    assert _source_path("objects", "longfact-objects_gpt4_01-12-2024_noduplicates", "physics") == (
        "longfact/longfact-objects_gpt4_01-12-2024_noduplicates/longfact-objects_physics.jsonl"
    )


def test_topic_order_matches_case_sensitive_source_file_sort():
    assert TOPICS[:3] == ("20th-century-events", "US-foreign-policy", "accounting")


def test_parse_rows_requires_thirty_valid_prompts():
    payload = b"".join(json.dumps({"prompt": f"prompt {index}"}).encode() + b"\n" for index in range(30))
    assert len(_parse_rows(payload, "fixture.jsonl")) == 30
    with pytest.raises(ValueError, match="Expected 30"):
        _parse_rows(
            b"".join(json.dumps({"prompt": f"prompt {index}"}).encode() + b"\n" for index in range(29)),
            "fixture.jsonl",
        )


def test_build_rows_preserves_subset_topic_and_row_order(tmp_path, monkeypatch):
    import benchmarks.longfact.prepare as module

    monkeypatch.setattr(module, "SUBSETS", (("concepts", "concepts"),))
    monkeypatch.setattr(module, "TOPICS", ("alpha", "beta"))
    for topic in module.TOPICS:
        path = tmp_path / "longfact" / "concepts"
        path.mkdir(parents=True, exist_ok=True)
        (path / f"longfact-concepts_{topic}.jsonl").write_text(
            "".join(json.dumps({"prompt": f"{topic} {index}"}) + "\n" for index in range(30)),
            encoding="utf-8",
        )

    rows = build_rows(tmp_path)

    assert len(rows) == 60
    assert [row["id"] for row in rows[:2]] == ["longfact-concepts-alpha-00", "longfact-concepts-alpha-01"]
    assert rows[30]["id"] == "longfact-concepts-beta-00"
    assert all(row["benchmark_variant"] == BENCHMARK_VARIANT for row in rows)


def test_select_rows_is_reproducible_and_retains_original_rows():
    rows = [{"id": str(index)} for index in range(100)]

    selected = select_rows(rows, sample_size=100, sample_seed=17)

    assert selected == select_rows(rows, sample_size=100, sample_seed=17)
    assert len(selected) == 100
    assert len({row["id"] for row in selected}) == 100
    assert selected == rows
