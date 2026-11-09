# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.longfact.prepare import (
    BENCHMARK_VARIANT,
    TOPICS,
    _parse_rows,
    _source_path,
    build_rows,
    select_rows,
)
from nemo_gym.data.task_materialization import materialize_task


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
    assert all(row["task_id"] == row["id"] for row in rows)
    original = [materialize_task(row, taskset="longfact") for row in rows]
    reordered = [materialize_task(row, taskset="longfact") for row in reversed(rows)]
    assert {task["task_id"]["task_id"] for task in original} == {row["id"] for row in rows}
    assert original == list(reversed(reordered))


def test_select_rows_is_reproducible_and_retains_original_rows():
    rows = [{"id": str(index)} for index in range(100)]

    selected = select_rows(rows, sample_size=17, sample_seed=17)

    assert selected == select_rows(rows, sample_size=17, sample_seed=17)
    assert selected != select_rows(rows, sample_size=17, sample_seed=104729)
    assert selected != rows[:17]
    assert len(selected) == 17
    assert len({row["id"] for row in selected}) == 17
    assert all(row in rows for row in selected)


@pytest.mark.parametrize("custom_output", [False, True])
def test_preparation_worker_writes_under_the_callers_directory(tmp_path, monkeypatch, custom_output):
    from benchmarks.longfact import prepare as source
    from nemo_gym.cli._prepare_worker import load_preparer
    from nemo_gym.environment.benchmark_preparation import _multiprocess_benchmark_prepare_fn

    installation = tmp_path / "installation"
    installation.mkdir()
    script = installation / "prepare.py"
    script.write_bytes(Path(source.__file__).read_bytes())
    module = load_preparer(str(script))
    monkeypatch.setattr(module, "SUBSETS", (("concepts", "concepts"),))
    monkeypatch.setattr(module, "TOPICS", ("alpha", "beta"))
    source_dir = tmp_path / "source"
    for topic in module.TOPICS:
        directory = source_dir / "longfact" / "concepts"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"longfact-concepts_{topic}.jsonl").write_text(
            "".join(json.dumps({"prompt": f"{topic} {index}"}) + "\n" for index in range(30)),
            encoding="utf-8",
        )
    working = tmp_path / "working"
    working.mkdir()
    monkeypatch.chdir(working)
    output = Path("custom/output.jsonl" if custom_output else "benchmarks/longfact/data/longfact_benchmark.jsonl")
    args = {"source_dir": source_dir, "sample_size": 0}
    if custom_output:
        args["output_fpath"] = output
    benchmark = SimpleNamespace(name="longfact", dataset=SimpleNamespace(jsonl_fpath=output))

    try:
        _multiprocess_benchmark_prepare_fn((benchmark, module.__name__, args, None))
    finally:
        sys.modules.pop(module.__name__, None)

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 60
    assert [row["id"] for row in rows[:2]] == ["longfact-concepts-alpha-00", "longfact-concepts-alpha-01"]
    assert rows[30]["id"] == "longfact-concepts-beta-00"
    assert all(row["task_id"] == row["id"] for row in rows)
    assert [row["prompt"] for row in rows] == [
        f"{topic} {index}" for topic in ("alpha", "beta") for index in range(30)
    ]
    assert not (installation / "data").exists()
