# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks/nemotron_3.5_super/analyze_prefill_timing.py"
SPEC = importlib.util.spec_from_file_location("analyze_prefill_timing", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
timing = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = timing
SPEC.loader.exec_module(timing)


def record(*, timestamp: float = 10, count: int = 1, total: float = 100, **overrides: object) -> dict[str, object]:
    value = {
        "component": "worker_v2",
        "scope": "prefill_engine",
        "host": "host-a",
        "pid": 10,
        "dp_rank": 0,
        "rank": 0,
        "timestamp": timestamp,
        "window_seconds": 10,
        "sample_every": 10,
        "cpu_ms": {
            "forward": {
                "count": count,
                "total": total,
                "max": total / count,
                "p50": total / count,
                "p90": total / count,
                "p99": total / count,
                "percentile_samples": count,
            }
        },
        "cuda_elapsed_ms": {},
        "batch": {},
        "counts": {"real_steps": count * 10, "sampled_steps": count},
        "pending_cuda_ranges": 0,
    }
    value.update(overrides)
    return value


def line(value: dict[str, object]) -> str:
    return "[logger] VLLM_PREFILL_TIMING " + json.dumps(value) + "\n"


def test_weighted_means_do_not_scale_sampled_totals_or_merge_quantiles(tmp_path: Path) -> None:
    path = tmp_path / "run.log"
    path.write_text(line(record()) + line(record(timestamp=30, count=9, total=90)))
    report, windows = timing.analyze([path])
    group = next(iter(report["groups"].values()))
    assert group["cpu_ms"]["forward"] == {"count": 10, "total": 190, "mean": 19, "max": 100}
    assert group["counts"] == {"real_steps": 100, "sampled_steps": 10}
    # Original p90s remain attached to their windows; no invented global p90.
    csv_path = tmp_path / "windows.csv"
    timing.write_csv(csv_path, windows)
    with csv_path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert [float(r["p90"]) for r in rows] == [100, 10]
    assert [int(r["count"]) for r in rows] == [1, 9]


def test_recovery_after_broken_fragment_and_duplicate_inputs(tmp_path: Path) -> None:
    path = tmp_path / "partial.log"
    path.write_text(
        'VLLM_PREFILL_TIMING {"cpu_ms": BROKEN '
        + line(record())
        + line(record(timestamp=30))
        + 'VLLM_PREFILL_TIMING {"component":'
    )
    report, _ = timing.analyze([path, path])
    assert report["parsing"]["valid_records"] == 2
    assert report["parsing"]["malformed_fragments"] == 4
    assert report["parsing"]["duplicates"] == 2


def test_process_identity_pending_gauges_and_sampling_groups(tmp_path: Path) -> None:
    path = tmp_path / "ranks.log"
    path.write_text(
        line(record(pending_cuda_ranges=7))
        + line(record(timestamp=30, pending_cuda_ranges=2))
        + line(record(host="host-b", pending_cuda_ranges=3))
        + line(record(host="host-c", sample_every=1))
    )
    report, _ = timing.analyze([path])
    assert len(report["processes"]) == 3  # Same PID/ranks on different hosts.
    assert len(report["groups"]) == 2  # Sampling policies stay separate.
    group = report["groups"]["worker_v2 / prefill_engine / sample_every=10"]
    assert group["processes"] == 2
    assert group["max_pending_in_one_record"] == 7
    assert sorted(group["pending_at_last_record_per_process"].values()) == [2, 3]


def test_time_filter_uses_window_start_and_handles_out_of_order_lines(tmp_path: Path) -> None:
    path = tmp_path / "run.log"
    path.write_text(line(record(timestamp=30)) + line(record(timestamp=10)) + line(record(timestamp=20)))
    report, windows = timing.analyze([path], skip_first_seconds=15)
    # [10,20] crosses the cutoff; including it would retain excluded samples.
    assert [w.timestamp for w in windows] == [30]
    assert report["excluded_records"] == 2
    report, windows = timing.analyze([path], skip_first_seconds=100)
    assert not windows and report["groups"] == {}


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_filter_is_rejected(tmp_path: Path, value: float) -> None:
    with pytest.raises(ValueError):
        timing.analyze([tmp_path / "unused.log"], skip_first_seconds=value)


def test_malformed_schema_is_counted_without_poisoning_summary(tmp_path: Path) -> None:
    path = tmp_path / "run.log"
    path.write_text(line(record(timestamp=float("nan"))) + line(record(cpu_ms=[])) + line(record()))
    report, _ = timing.analyze([path])
    assert report["parsing"]["malformed_fragments"] == 2
    assert report["included_records"] == 1
