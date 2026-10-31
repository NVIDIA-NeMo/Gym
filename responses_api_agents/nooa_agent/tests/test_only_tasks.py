# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[3] / "recipes" / "nooa-atom" / "only_tasks.py"
spec = importlib.util.spec_from_file_location("only_tasks", SCRIPT)
only_tasks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(only_tasks)


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_filter_datasets_keeps_matching_rows_and_points_the_run_at_them(tmp_path: Path) -> None:
    tb = tmp_path / "tb" / "benchmark.jsonl"
    _write_rows(
        tb,
        [
            {"task_id": "code-from-image", "task_name": "terminal-bench/code-from-image"},
            {"task_id": "regex-log", "task_name": "terminal-bench/regex-log"},
            {"task_id": "fp8-gemm", "task_name": "terminal-bench/fp8-gemm"},
        ],
    )
    swe = tmp_path / "swe" / "swe120.jsonl"
    _write_rows(swe, [{"instance_id": "instance_navidrome__navidrome-7073"}, {"instance_id": "instance_x__y-1"}])
    config = {
        "benchmark": {"resources_servers": {"tb": {"datasets": [{"type": "benchmark", "jsonl_fpath": str(tb)}]}}},
        "swe_server": {"resources_servers": {"swe": {"datasets": [{"type": "benchmark", "jsonl_fpath": str(swe)}]}}},
        "policy_model": {"responses_api_models": {}},
    }
    out = tmp_path / "only"
    overlay = only_tasks.filter_datasets(config, out, ["code-from-image", "fp8-*", "*navidrome*"])

    assert overlay == {
        "benchmark": {
            "resources_servers": {
                "tb": {"datasets": [{"type": "benchmark", "jsonl_fpath": str(out / "benchmark.jsonl")}]}
            }
        },
        "swe_server": {
            "resources_servers": {
                "swe": {"datasets": [{"type": "benchmark", "jsonl_fpath": str(out / "swe120.jsonl")}]}
            }
        },
    }
    # The source config is not mutated: a later `--config` carries the filtered list.
    assert config["benchmark"]["resources_servers"]["tb"]["datasets"][0]["jsonl_fpath"] == str(tb)
    kept = [json.loads(line)["task_id"] for line in (out / "benchmark.jsonl").read_text().splitlines()]
    assert kept == ["code-from-image", "fp8-gemm"]
    assert [json.loads(line)["instance_id"] for line in (out / "swe120.jsonl").read_text().splitlines()] == [
        "instance_navidrome__navidrome-7073"
    ]


def test_filter_datasets_matches_nested_task_ids_and_rejects_empty_selections(tmp_path: Path) -> None:
    src = tmp_path / "rows.jsonl"
    _write_rows(src, [{"task_id": {"taskset": "tb", "task_id": "sparql-university"}}])
    config = {"s": {"resources_servers": {"r": {"datasets": [{"type": "benchmark", "jsonl_fpath": str(src)}]}}}}
    assert only_tasks.filter_datasets(config, tmp_path / "o", ["sparql-*"])
    with pytest.raises(SystemExit, match="no task"):
        only_tasks.filter_datasets(config, tmp_path / "o2", ["nothing-here"])
