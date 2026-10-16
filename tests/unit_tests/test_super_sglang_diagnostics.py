# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise diagnostic control requests, accounting, and recipe preservation without GPUs."""

import argparse
import gzip
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "benchmarks/nemotron_3.5_super/diagnostics"


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, DIRECTORY / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collect = load("collect")
analyze = load("analyze")
prepare = load("prepare_recipe")


def prom(path: Path, content: str) -> dict:
    with gzip.open(path, "wt") as output:
        output.write(content)
    return analyze.parse_prom(path)


def test_endpoint_sources_and_ambiguity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "inference-metrics.yaml"
    path.write_text(
        json.dumps(
            {"inference_metrics": {"endpoints": {"prefill0": "http://p:1/metrics", "decode0": "http://d:2/metrics"}}}
        )
    )
    args = argparse.Namespace(endpoints_json=str(path), engine=[])
    assert collect.engines(args) == {"prefill0": "http://p:1", "decode0": "http://d:2"}
    args.endpoints_json = None
    monkeypatch.setenv("SRT_PREFILL_ENDPOINTS", "p:1,p:2")
    monkeypatch.setenv("SRT_DECODE_ENDPOINTS", "d:1,d:2")
    assert len(collect.engines(args)) == 4
    for entries in (
        ["router=http://router:1"],
        ["prefill0=p:1", "decode0=p:1"],
        ["prefill0=p:1", "prefill0=p:2"],
        ["prefill0=http://user:token@p:1"],
    ):
        args.engine = entries
        with pytest.raises(ValueError):
            collect.engines(args)


def schema(fields: set[str]) -> str:
    return json.dumps(
        {
            "paths": {
                "/start_profile": {
                    "post": {
                        "requestBody": {
                            "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Profile"}}}
                        }
                    }
                }
            },
            "components": {"schemas": {"Profile": {"properties": {field: {} for field in fields}}}},
        }
    )


@pytest.mark.parametrize("failure", [None, "schema", "timeout"])
def test_profile_preflights_all_engines_and_never_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    endpoints = {name: f"http://{name}:1" for name in ("prefill0", "prefill1", "decode0", "decode1")}
    calls = []
    fields = {"output_dir", "profile_id", "num_steps", "activities", "profile_by_stage", "with_stack", "record_shapes"}

    def http(url: str, payload: dict | None = None) -> tuple[str, int]:
        calls.append((url, payload))
        if url.endswith("/server_info"):
            return json.dumps({"version": "test", "server_args": {"tp_size": 4}, "api_key": "secret"}), 200
        if url.endswith("/openapi.json"):
            return schema(fields - {"profile_by_stage"} if failure == "schema" and "decode1" in url else fields), 200
        assert sum(url.endswith("/openapi.json") for url, _ in calls) == 4
        if failure == "timeout" and "decode0" in url:
            raise TimeoutError("start may already be active")
        return "OK", 200

    monkeypatch.setattr(collect, "http", http)
    args = argparse.Namespace(server_profile_dir="/logs/profiles", steps=20)
    if failure:
        with pytest.raises(ValueError if failure == "schema" else RuntimeError):
            collect.collect_profile(args, endpoints, tmp_path)
    else:
        collect.collect_profile(args, endpoints, tmp_path)
    starts = [(url, body) for url, body in calls if body is not None]
    assert len(starts) == (0 if failure == "schema" else 4)
    assert len({url for url, _ in starts}) == len(starts)
    for url, body in starts:
        assert body["num_steps"] == 20
        assert body["profile_by_stage"] is True
        assert body["activities"] == ["CPU", "GPU"]
        assert body["with_stack"] is False and body["record_shapes"] is False
        assert "start_step" not in body
    assert "secret" not in (tmp_path / "prefill0-server-info.json").read_text()
    if failure == "timeout":
        responses = json.loads((tmp_path / "profile-responses.json").read_text())
        assert next(row for row in responses if row["engine"] == "decode0")["outcome"].startswith("unknown")


def test_metrics_preserves_scrape_errors_labels_and_liveness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    tick = [0.0]
    count = {"prefill0": 0, "decode0": 0}
    monkeypatch.setattr(collect.time, "monotonic", lambda: tick[0])
    monkeypatch.setattr(collect.time, "sleep", lambda seconds: tick.__setitem__(0, tick[0] + seconds))

    def http(url: str, payload: dict | None = None) -> tuple[str, int]:
        assert payload is None
        if url.endswith("/server_info"):
            return '{"api_key":"private","tp_size":4}', 200
        assert url.endswith("/metrics")
        name = url.split("/")[2]
        count[name] += 1
        if name == "prefill0" and count[name] == 2:
            raise OSError("temporary scrape failure")
        value = count[name] if name == "prefill0" else 1
        return f'sglang:work_total{{tp_rank="0",pool="kv"}} {value}\n', 200

    monkeypatch.setattr(collect, "http", http)
    collect.collect_metrics(
        argparse.Namespace(seconds=20, interval=5), {name: f"http://{name}" for name in count}, tmp_path
    )
    records = [json.loads(line) for line in (tmp_path / "scrapes.jsonl").read_text().splitlines()]
    assert len(records) == 10
    assert [row["error"] for row in records if "error" in row] == ["temporary scrape failure"]
    assert not (tmp_path / "prefill0-0001.prom.gz").exists()
    with gzip.open(tmp_path / "prefill0-0004.prom.gz", "rt") as stream:
        assert 'tp_rank="0",pool="kv"' in stream.read()
    warning = capsys.readouterr().err
    assert "WARNING decode0" in warning and "WARNING prefill0" not in warning
    assert json.loads((tmp_path / "liveness.json").read_text())["prefill0"]["ok_scrapes"] == 4


def test_counters_detect_midwindow_reset_and_missing_labels(tmp_path: Path) -> None:
    snaps = [
        prom(tmp_path / f"{index}.gz", f'sglang:work_total{{tp_rank="0"}} {value}\nsglang:queue 0\n')
        for index, value in enumerate((100, 2, 150))
    ]
    first, last, invalid = analyze.counter_window(snaps)
    assert invalid == ["sglang:work_total"]
    assert analyze.delta(first, last, "sglang:work_total") is None
    assert analyze.gauge_range(snaps, "sglang:queue")["max"] == 0
    assert analyze.gauge_range(snaps, "sglang:missing") is None
    a = prom(tmp_path / "a.gz", 'sglang:work_total{pool="kv"} 1\n')
    b = prom(tmp_path / "b.gz", 'sglang:work_total{pool="mamba"} 100\n')
    assert analyze.delta(a, b, "sglang:work_total") is None


def test_hicache_requires_verified_tp_semantics_and_complete_shares(tmp_path: Path) -> None:
    template = (
        'sglang:prefill_effective_tokens_total{{mode="input",tp_rank="0"}} {input}\n'
        'sglang:hicache_backup_tokens_total{{pool="kv"}} {tokens}\n'
        "sglang:hicache_backup_bytes_total {nbytes}\n"
        "sglang:hicache_backup_duration_seconds_sum {duration}\n"
        "sglang:hicache_backup_duration_seconds_count {count}\n"
    )
    first = prom(tmp_path / "first.gz", template.format(input=10, tokens=0, nbytes=0, duration=0, count=0))
    last = prom(tmp_path / "last.gz", template.format(input=110, tokens=400, nbytes=4096, duration=0.04, count=2))
    unknown = analyze.hicache(first, last, [first, last], 10, 4)
    assert "shares" not in unknown
    assert unknown["computed_input_tok_s"] == 10
    assert unknown["backup"]["logical_tokens_by_pool"] == {"kv": None}
    assert unknown["backup"]["raw_tokens_by_pool"] == {"kv": 400}
    confirmed = analyze.hicache(first, last, [first, last], 10, 4, tp_summed=True)
    assert confirmed["backup"]["logical_tokens_by_pool"] == {"kv": 100}
    assert confirmed["backup"]["bytes"] == 4096
    assert confirmed["backup"]["mean_ms"] == 20
    assert confirmed["load_back"]["operations"] is None


def test_startup_retains_full_logs_and_missing_memory_is_unknown(tmp_path: Path) -> None:
    source = tmp_path / "prefill.log"
    source.write_text(
        "unmatched evidence\n[2026-10-08 TP0 EP0] Mamba Cache is allocated. max_mamba_cache_size: 320\n"
        "[2026-10-08 TP0 EP0] max_total_num_tokens=1000, max_running_requests=256, available_gpu_mem=1.5\n"
    )
    output = tmp_path / "startup"
    output.mkdir()
    collect.collect_startup(argparse.Namespace(log=[str(source)]), output)
    assert (output / "worker-logs/0-prefill.log").read_bytes() == source.read_bytes()
    report = analyze.analyze_startup(output)["prefill.log [TP0 EP0]"]
    assert report["mamba_slots"] == 320
    assert report["max_running_requests"] == 256
    assert report["mamba_state_gb"] is None
    assert report["graphs_gb"] is None
    assert report["accounted_gb"] is None


def test_trace_union_does_not_sum_overlapping_streams(tmp_path: Path) -> None:
    path = tmp_path / "busy-prefill0-TP-0-EXTEND.trace.json"
    events = [
        {"cat": "kernel", "name": "bmm_Bfloat16", "ts": 0, "dur": 2000},
        {"cat": "kernel", "name": "bmm_chunk_scan", "ts": 1000, "dur": 2000},
        {"cat": "gpu_memcpy", "name": "Memcpy HtoD", "ts": 4000, "dur": 1000},
    ]
    path.write_text(json.dumps({"traceEvents": [{"ph": "X", "pid": 0, **event} for event in events]}))
    report = analyze.analyze_trace(path)
    assert report["window_ms"] == 5
    assert report["gpu_busy_ms"] == 4
    assert report["gpu_idle_pct"] == 20
    assert report["kernel_ms_by_category"] == {"gemm": 2, "mamba/ssm": 2}
    assert report["nonzero_kernel_events"] == 2
    assert report["exposed_copy_or_collective_wait_ms"] is None


@pytest.mark.parametrize("recipe_name", ["2P2D.yaml", "2P2D_mooncake.yaml", "2P2D_ddes256.yaml"])
def test_recipe_wrapper_preserves_workload_and_serving(recipe_name: str) -> None:
    recipe = yaml.safe_load((DIRECTORY.parent / "sglang_configs" / recipe_name).read_text())
    result = prepare.instrument(recipe, profile=True, delay=300, seconds=300, steps=20)
    for field in ("roles", "frontend", "model", "setup_script", "resources"):
        assert result[field] == recipe[field]
    assert result.get("services", [])[:-1] == recipe.get("services", [])
    assert (
        result["benchmark"]["command"].split("gym eval run", 1)[1]
        == recipe["benchmark"]["command"].split("gym eval run", 1)[1]
    )
    assert result["benchmark"]["env"].items() >= recipe["benchmark"]["env"].items()
    assert result["services"][-1]["placement"] == {"node": "workers"}
    assert result["services"][-1]["critical"] is False
    assert " --profile -- gym eval run" in result["benchmark"]["command"]
    syntax = subprocess.run(["bash", "-n"], input=result["benchmark"]["command"], text=True, capture_output=True)
    assert syntax.returncode == 0, syntax.stderr


def test_runner_preserves_workload_exit_before_collection(tmp_path: Path) -> None:
    root = tmp_path / "diagnostics"
    result = subprocess.run(
        [
            sys.executable,
            str(DIRECTORY / "run.py"),
            "benchmark",
            "--root",
            str(root),
            "--endpoints",
            str(tmp_path / "unused.json"),
            "--delay",
            "30",
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(7)",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 7, result.stderr
    assert not (root / "metrics-window.json").exists()
    assert json.loads((root / "done.json").read_text())["metric_window_complete"] is False
    assert (root / "analysis/SUMMARY.md").exists()


def test_existing_capture_is_not_overwritten(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(DIRECTORY / "collect.py"), "gpu", "--out", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert not (tmp_path / "gpu-query.json").exists()


@pytest.mark.parametrize("scenario", ["complete", "collector-failed", "workload-finished"])
def test_runner_orders_capture_and_preserves_workload_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scenario: str
) -> None:
    monkeypatch.setitem(sys.modules, "collect", collect)
    runner = load("run")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "collect.py").write_text(
        "import os, pathlib, sys, time\n"
        "mode = sys.argv[1]\n"
        "events = pathlib.Path(os.environ['DIAG_TEST_EVENTS'])\n"
        "with events.open('a') as f: f.write(mode + '-start\\n')\n"
        "if mode == 'metrics':\n"
        "    if os.environ['DIAG_TEST_SCENARIO'] == 'collector-failed': raise SystemExit(9)\n"
        "    time.sleep(2 if os.environ['DIAG_TEST_SCENARIO'] == 'workload-finished' else 0.05)\n"
        "    with events.open('a') as f: f.write('metrics-end\\n')\n"
        "else:\n"
        "    assert 'metrics-end' in events.read_text()\n"
    )
    events = tmp_path / "events"
    monkeypatch.setenv("DIAG_TEST_EVENTS", str(events))
    monkeypatch.setenv("DIAG_TEST_SCENARIO", scenario)
    monkeypatch.setattr(runner, "HERE", scripts)
    monkeypatch.setattr(runner, "finalize", lambda root, logs_root: None)
    args = argparse.Namespace(
        root=tmp_path / "diag",
        endpoints=tmp_path / "endpoints.json",
        delay=0,
        seconds=0.05,
        profile=True,
        steps=20,
        command=[sys.executable, "-c", "import time; time.sleep(0.6); raise SystemExit(7)"],
    )
    assert runner.benchmark(args) == 7
    rows = events.read_text().splitlines()
    if scenario == "complete":
        assert rows == ["metrics-start", "metrics-end", "profile-start"]
    else:
        assert rows == ["metrics-start"]
    if scenario == "collector-failed":
        assert "9" in (args.root / "controller/ERROR.json").read_text()
    assert json.loads((args.root / "done.json").read_text())["metric_window_complete"] == (scenario == "complete")


def test_trace_coverage_reports_missing_tp_ranks(tmp_path: Path) -> None:
    control = tmp_path / "profile-control"
    control.mkdir()
    collect.save(control / "profile-requests.json", [{"engine": "prefill0"}])
    collect.save(control / "prefill0-server-info.json", {"server_args": {"tp_size": 4}})
    report = analyze.profile_coverage(tmp_path, {"prefill0 EXTEND TP0": {"nonzero_kernel_events": 10}})
    row = report["profile-control/prefill0"]
    assert row["ranks"]["0"] == "GPU kernels present"
    assert all(row["ranks"][str(rank)] == "missing or no GPU kernels" for rank in (1, 2, 3))
    assert row["complete_warmed_steps_verified"] is False
