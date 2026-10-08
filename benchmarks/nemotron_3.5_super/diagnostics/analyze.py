#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Turn collect.py output into the Task 2/3/4 tables: SUMMARY.md and summary.json.

Python standard library only. Usage:
    python3 analyze.py /logs/nemotron-diag [--traces DIR ...] [--out DIR]
Reads every collect.py capture below the root (metrics, gpu, startup) and every *.trace.json.gz
below the root and any --traces directory (the --server-profile-dir of the profile capture).
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+(\S+)(?:\s+\d+)?\s*$")
LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')
TRACE_NAME = re.compile(r"-(prefill\d+|decode\d+)-TP-(\d+)(?:-EP-\d+)?-(\w+)\.trace\.json(?:\.gz)?$")
KERNEL_CATEGORIES = [  # Heuristic, by kernel name; first match wins.
    (
        "communication",
        r"nccl|all_?reduce|all_?gather|reduce_?scatter|cross_device|custom_?ar|oneshot|twoshot|lamport|mnnvl|a2a|deepep",
    ),
    ("moe", r"moe|expert|topk_?gating|routing|permute|grouped_gemm"),
    ("attention", r"attention|attn|fmha|flash|xqa|\bmha|mla|paged|qkv"),
    (
        "mamba/ssm",
        r"mamba|ssm|selective|causal_conv|conv1d|chunk_scan|chunk_state|state_passing|chunk_cumsum|bmm_chunk|ssd",
    ),
    ("gemm", r"gemm|gemv|cutlass|cublas|nvjet|matmul|sm90_|sm100_|xmma|tensorop|^bmm_"),
    ("sampling/spec", r"sampl|verify|tree|argmax|top_?k|top_?p|softmax|eagle|draft"),
    (
        "norm/elementwise",
        r"norm|rms|silu|gelu|act_and_mul|act_kernel|activation|elementwise|vectorized|reduce_kernel|copy|cat|index|fill|scatter|gather|rotary|rope|quant",
    ),
]


def r2(value: float | None, digits: int = 2) -> float | None:
    return None if value is None else round(value, digits)


# ---------- metrics ----------


def parse_prom(path: Path) -> dict:
    series = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            match = LINE.match(line) if not line.startswith("#") else None
            if match:
                try:
                    value = float(match.group(3))
                    if math.isfinite(value):
                        series[(match.group(1), tuple(sorted(LABEL.findall(match.group(2) or ""))))] = value
                except ValueError:
                    pass
    return series


def select(series: dict, name: str, **where: str) -> list[tuple[dict[str, str], float]]:
    """Rows of one family; per-rank families are restricted to tp_rank 0 (logical counts are reported there)."""
    rows = [(dict(labels), value) for (n, labels), value in series.items() if n == name]
    rows = [(labels, v) for labels, v in rows if all(labels.get(k) == w for k, w in where.items())]
    if any("tp_rank" in labels for labels, _ in rows):
        rows = [(labels, v) for labels, v in rows if labels.get("tp_rank") == "0"]
    return rows


def total(series: dict, name: str, **where: str) -> float | None:
    rows = select(series, name, **where)
    return sum(v for _, v in rows) if rows else None


def delta(first: dict, last: dict, name: str, **where: str) -> float | None:
    a_rows, b_rows = select(first, name, **where), select(last, name, **where)
    if {tuple(sorted(labels.items())) for labels, _ in a_rows} != {
        tuple(sorted(labels.items())) for labels, _ in b_rows
    }:
        return None
    a, b = total(first, name, **where), total(last, name, **where)
    if a is None or b is None:
        return None
    return b - a if b >= a else None  # None on a counter reset


def counter_window(snaps: list[dict]) -> tuple[dict, dict, list[str]]:
    """Exclude counter families with resets or coverage changes anywhere in the window."""
    invalid = set()
    keys = set().union(*(snap.keys() for snap in snaps))
    for key in keys:
        if not key[0].endswith(("_total", "_count", "_sum", "_bucket")):
            continue
        values = [snap.get(key) for snap in snaps]
        if any(value is None for value in values) or any(b < a for a, b in zip(values, values[1:])):
            invalid.add(re.sub(r"_(?:bucket|sum|count)$", "", key[0]))

    def valid(snap: dict) -> dict:
        return {
            key: value for key, value in snap.items() if re.sub(r"_(?:bucket|sum|count)$", "", key[0]) not in invalid
        }

    return valid(snaps[0]), valid(snaps[-1]), sorted(invalid)


def histogram(first: dict, last: dict, name: str, group: str | None = None) -> dict[str, dict]:
    """Window histogram per group label: count, mean, p50, p90, p99 from bucket deltas."""
    out = {}
    buckets = defaultdict(lambda: defaultdict(float))
    for sign, snap in ((-1, first), (1, last)):
        for labels, value in select(snap, name + "_bucket"):
            key = labels.get(group, "all") if group else "all"
            le = float("inf") if labels["le"] == "+Inf" else float(labels["le"])
            buckets[key][le] += sign * value
    for key, cum in buckets.items():
        n = cum.get(float("inf"), 0.0)
        s = delta(first, last, name + "_sum", **({group: key} if group else {}))
        if n <= 0:
            continue
        stats = {"count": int(n), "mean": r2(s / n, 4) if s is not None else None}
        for q in (0.5, 0.9, 0.99):
            target, prev_le, prev_c, value = q * n, 0.0, 0.0, None
            for le in sorted(cum):
                if cum[le] >= target:
                    if prev_c == 0 and le != float("inf"):
                        value = f"<={le:g}"  # inside the lowest non-empty bucket: only the bound is known
                    else:
                        value = (
                            prev_le
                            if le == float("inf")
                            else prev_le + (le - prev_le) * (target - prev_c) / max(cum[le] - prev_c, 1e-12)
                        )
                    break
                prev_le, prev_c = le, cum[le]
            stats[f"p{int(q * 100)}"] = value if isinstance(value, str) else r2(value, 4)
        out[key] = stats
    return out


def gauge_range(snaps: list[dict], name: str, **where: str) -> dict[str, float] | None:
    values = [total(s, name, **where) for s in snaps]
    values = [v for v in values if v is not None]
    return (
        {"min": r2(min(values), 4), "max": r2(max(values), 4), "mean": r2(statistics.mean(values), 4)}
        if values
        else None
    )


def analyze_metrics(capture: Path, *, tp_summed_hicache_tokens: bool = False) -> dict[str, dict]:
    records = [json.loads(labels) for labels in (capture / "scrapes.jsonl").read_text().splitlines()]
    liveness = json.loads((capture / "liveness.json").read_text()) if (capture / "liveness.json").exists() else {}
    report = {}
    for engine in sorted({r["engine"] for r in records}):
        ok = [r for r in records if r["engine"] == engine and "file" in r]
        if len(ok) < 2:
            report[engine] = {"error": "fewer than two successful scrapes"}
            continue
        snaps = [parse_prom(capture / r["file"]) for r in ok]
        first, last, invalid = counter_window(snaps)
        seconds = _utc(ok[-1]["start_utc"]) - _utc(ok[0]["start_utc"])
        info_path = capture / f"{engine}-server-info.json"
        info = json.loads(info_path.read_text()) if info_path.exists() else {}
        tp = info.get("tp_size") or (info.get("server_args") or {}).get("tp_size")
        tp = int(tp) if tp is not None and int(tp) > 0 else None
        e = {
            "window_seconds": r2(seconds, 1),
            "start_utc": ok[0]["start_utc"],
            "end_utc": ok[-1]["start_utc"],
            "tp_size": tp,
            "liveness": liveness.get(engine),
            "excluded_counter_families": invalid,
            "failed_scrapes": [r for r in records if r["engine"] == engine and "error" in r],
        }
        rate = (
            lambda name, **w: r2(delta(first, last, name, **w) / seconds, 1)
            if delta(first, last, name, **w) is not None and seconds > 0
            else None
        )
        e["tokens_per_second"] = {
            m: rate("sglang:realtime_tokens_total", mode=m) for m in ("prefill_compute", "prefill_cache", "decode")
        }
        e["latency_seconds"] = {
            n: histogram(first, last, "sglang:" + n)
            for n in (
                "time_to_first_token_seconds",
                "inter_token_latency_seconds",
                "e2e_request_latency_seconds",
                "queue_time_seconds",
            )
        }
        if engine.startswith("prefill"):
            for name in ("time_to_first_token_seconds", "inter_token_latency_seconds"):
                e["latency_seconds"].pop(name)
        e["retractions"] = {
            name: delta(first, last, "sglang:" + name)
            for name in (
                "num_retracted_requests_total",
                "num_retracted_input_tokens_total",
                "num_retracted_output_tokens_total",
            )
        }
        e["per_stage_request_latency_seconds"] = histogram(
            first, last, "sglang:per_stage_req_latency_seconds", group="stage"
        )
        e["kv_transfer"] = {
            n: histogram(first, last, "sglang:" + n)
            for n in (
                "kv_transfer_latency_ms",
                "kv_transfer_bootstrap_ms",
                "kv_transfer_alloc_ms",
                "kv_transfer_speed_gb_s",
                "kv_transfer_total_mb",
            )
        }
        e["gauges"] = {
            n: gauge_range(snaps, "sglang:" + n)
            for n in (
                "num_running_reqs",
                "num_queue_reqs",
                "num_prefill_bootstrap_queue_reqs",
                "num_prefill_inflight_queue_reqs",
                "num_decode_prealloc_queue_reqs",
                "num_decode_transfer_queue_reqs",
                "full_token_usage",
                "kv_available_tokens",
                "kv_evictable_tokens",
                "mamba_usage",
                "mamba_available_tokens",
                "mamba_evictable_tokens",
                "num_retracted_reqs",
            )
        }
        if engine.startswith("decode"):
            e["gauges"]["spec_accept_length"] = gauge_range(snaps, "sglang:spec_accept_length")
            e["spec_verify_calls"] = delta(first, last, "sglang:spec_verify_calls_total")
        else:
            e["hicache"] = hicache(first, last, snaps, seconds, tp, tp_summed=tp_summed_hicache_tokens)
        report[engine] = {k: v for k, v in e.items() if v not in (None, {}, [])}
    return report


def hicache(
    first: dict, last: dict, snaps: list[dict], seconds: float, tp: int | None, *, tp_summed: bool = False
) -> dict[str, object]:
    reuse = {
        m: delta(first, last, "sglang:prefill_effective_tokens_total", mode=m)
        for m in ("input", "device_hit", "host_hit", "storage_hit")
    }
    known = {m: v for m, v in reuse.items() if v is not None}
    out = {"effective_tokens": reuse}
    if len(known) == len(reuse) and sum(known.values()) > 0:
        out["shares"] = {m: r2(v / sum(known.values()), 4) for m, v in known.items()}
    out["computed_input_tok_s"] = r2(known["input"] / seconds, 1) if seconds > 0 and "input" in known else None
    out["completed_work"] = {
        "prompt_tokens": delta(first, last, "sglang:prompt_tokens_histogram_sum"),
        "requests": delta(first, last, "sglang:prompt_tokens_histogram_count"),
        "cached_tokens_by_source": {
            source: delta(first, last, "sglang:cached_tokens_total", cache_source=source)
            for source in ("device", "host", "storage")
        },
    }
    out["token_normalization"] = (
        "TP-summed semantics explicitly confirmed" if tp_summed and tp else "unknown; raw movement tokens retained"
    )
    if (
        total(last, "sglang:hicache_backup_bytes_total") is None
        and total(last, "sglang:load_back_bytes_total") is None
    ):
        out["note"] = "no HiCache movement metrics exported (HiCache off or not instrumented)"
    for kind, prefix in (("backup", "sglang:hicache_backup"), ("load_back", "sglang:load_back")):
        dsum, dcount = (
            delta(first, last, prefix + "_duration_seconds_sum"),
            delta(first, last, prefix + "_duration_seconds_count"),
        )
        tokens, raw_tokens = {}, {}
        for labels, _ in select(last, prefix + "_tokens_total"):
            d = delta(first, last, prefix + "_tokens_total", pool=labels.get("pool"))
            raw_tokens[labels.get("pool", "all")] = d
            tokens[labels.get("pool", "all")] = (
                r2(d / tp, 0) if d is not None and tp and tp_summed and "tp_rank" not in labels else None
            )
        nbytes = delta(first, last, prefix + "_bytes_total")
        out[kind] = {
            "logical_tokens_by_pool": tokens,
            "raw_tokens_by_pool": raw_tokens,
            "bytes": nbytes,
            "gib": r2(nbytes / 2**30) if nbytes is not None else None,
            "duration_seconds_sum": dsum,
            "operations": dcount,
            "mean_ms": r2(1000 * dsum / dcount, 3) if dsum is not None and dcount else None,
        }
    out["dropped_tokens"] = {
        f"{labels.get('pool', '?')}/{labels.get('reason', '?')}": delta(
            first, last, "sglang:hicache_dropped_tokens_total", **labels
        )
        for labels, _ in select(last, "sglang:hicache_dropped_tokens_total")
    }
    used, cap = gauge_range(snaps, "sglang:hicache_host_used_tokens"), total(last, "sglang:hicache_host_total_tokens")
    out["host_occupancy_peak"] = r2(used["max"] / cap, 4) if used and cap else None
    out["storage_prefetch_hit_tokens"] = delta(first, last, "sglang:storage_prefetch_hit_tokens_total")
    out["storage"] = {
        name: delta(first, last, "sglang:" + name) for name in ("prefetched_tokens_total", "backuped_tokens_total")
    }
    out["storage_unfulfilled_by_reason"] = {
        labels.get("reason", "?"): delta(first, last, "sglang:storage_prefetch_unfulfilled_tokens_total", **labels)
        for labels, _ in select(last, "sglang:storage_prefetch_unfulfilled_tokens_total")
    }
    out["evicted_tokens_unlabelled"] = delta(first, last, "sglang:evicted_tokens_total")
    return out


def _utc(text: str) -> float:
    from datetime import datetime

    return datetime.fromisoformat(text).timestamp()


# ---------- traces ----------


def category(name: str) -> str:
    lower = name.lower()
    for label, pattern in KERNEL_CATEGORIES:
        if re.search(pattern, lower):
            return label
    return "other"


def pct(values: list[float], q: float) -> float | None:
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))] if values else None


def analyze_trace(path: Path) -> dict[str, object]:
    with gzip.open(path, "rt") if path.name.endswith(".gz") else open(path) as f:
        doc = json.load(f)
    events = doc.get("traceEvents", []) if isinstance(doc, dict) else doc
    busy, kernels, memcpy, runtime = (
        [],
        defaultdict(float),
        defaultdict(lambda: [0, 0.0, 0]),
        defaultdict(lambda: [0, 0.0]),
    )
    gpu_ranges, cpu_ranges, batch_sizes, step_starts = defaultdict(list), defaultdict(list), [], []
    step_tokens = defaultdict(list)
    kernel_events = 0
    devices = set()
    for e in events:
        if not isinstance(e, dict) or e.get("ph") != "X" or "dur" not in e:
            continue
        cat, name, ts, dur = e.get("cat"), e.get("name", ""), float(e.get("ts", 0)), float(e["dur"])
        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            devices.add((e.get("args") or {}).get("device", e.get("pid")))
            busy.append((ts, ts + dur))
            if cat == "kernel":
                kernel_events += int(dur > 0)
                kernels[category(name)] += dur / 1000
            elif cat == "gpu_memcpy":
                kind = re.search(r"Memcpy (\w+)", name)
                row = memcpy[kind.group(1) if kind else name]
                row[0] += 1
                row[1] += dur / 1000
                row[2] += int((e.get("args") or {}).get("bytes", 0))
        elif cat in ("gpu_user_annotation", "user_annotation"):
            bs, toks = re.search(r"bs=(\d+)", name), re.search(r"step\[(\w+)[^\]]*?toks=(\d+)", name)
            if bs and cat == "user_annotation":
                batch_sizes.append(int(bs.group(1)))
            if toks:
                step_tokens[(cat, toks.group(1))].append(int(toks.group(2)))
            key = re.sub(r"\s*(bs|toks)=\d+", "", name)
            (gpu_ranges if cat == "gpu_user_annotation" else cpu_ranges)[key].append(dur / 1000)
            if cat == "user_annotation" and key == "scheduler.run_batch":
                step_starts.append(ts)
        elif cat in ("cuda_runtime", "cuda_driver"):
            runtime[name][0] += 1
            runtime[name][1] += dur / 1000
    if not busy:
        return {"error": "no GPU activity events"}
    if len(devices) > 1:
        return {"error": "multiple GPUs in one trace; separate devices before computing busy/idle time"}
    busy.sort()
    merged = [list(busy[0])]
    for start, end in busy[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    window = (merged[-1][1] - merged[0][0]) / 1000
    union = sum(b - a for a, b in merged) / 1000
    gaps = sorted(((merged[i + 1][0] - merged[i][1]) / 1000 for i in range(len(merged) - 1)), reverse=True)
    kernel_total = sum(kernels.values())
    phase = lambda ranges: {
        k: {
            "count": len(v),
            "total_ms": r2(sum(v)),
            "median_ms": r2(statistics.median(v), 3),
            "p90_ms": r2(pct(v, 0.9), 3),
        }
        for k, v in sorted(ranges.items(), key=lambda kv: -sum(kv[1]))
    }
    step_starts.sort()
    intervals = [(b - a) / 1000 for a, b in zip(step_starts, step_starts[1:])]
    sync = {k: v for k, v in runtime.items() if "Synchronize" in k}
    launch = {k: v for k, v in runtime.items() if "Launch" in k}
    return {
        "nonzero_kernel_events": kernel_events,
        "window_definition": "first GPU activity start to last GPU activity end; excludes leading/trailing idle",
        "context_lengths": None,
        "accepted_tokens_per_iteration": None,
        "exposed_copy_or_collective_wait_ms": None,
        "complete_warmed_steps_verified": False,
        "window_ms": r2(window),
        "gpu_busy_ms": r2(union),
        "gpu_idle_pct": r2(100 * (1 - union / window), 1) if window else None,
        "largest_idle_gaps_ms": [r2(g, 3) for g in gaps[:3]],
        "kernel_ms_by_category": {k: r2(v) for k, v in sorted(kernels.items(), key=lambda kv: -kv[1])},
        "kernel_share_by_category": {k: r2(v / kernel_total, 3) for k, v in kernels.items()} if kernel_total else {},
        "memcpy": {k: {"count": v[0], "ms": r2(v[1]), "mib": r2(v[2] / 2**20)} for k, v in memcpy.items()},
        "gpu_annotation_ranges": phase(gpu_ranges),
        "cpu_annotation_ranges": phase(cpu_ranges),
        "scheduler_steps": len(step_starts),
        "step_interval_ms": {"median": r2(statistics.median(intervals), 3), "p90": r2(pct(intervals, 0.9), 3)}
        if intervals
        else None,
        "batch_size": {"median": statistics.median(batch_sizes), "max": max(batch_sizes)} if batch_sizes else None,
        "tokens_per_step": {
            k[1]: {"median": statistics.median(v), "max": max(v), "steps": len(v)}  # CPU ranges if present, else GPU
            for k, v in step_tokens.items()
            if k[0] == "user_annotation" or ("user_annotation", k[1]) not in step_tokens
        },
        "cpu_launch_calls": {
            "count": sum(v[0] for v in launch.values()),
            "ms": r2(sum(v[1] for v in launch.values())),
        },
        "cpu_blocked_in_sync_ms": r2(sum(v[1] for v in sync.values())),
    }


# ---------- gpu / startup ----------


def analyze_gpu(capture: Path) -> dict[str, dict]:
    per_gpu = defaultdict(lambda: {"max_used_mib": 0, "min_free_mib": None, "util": []})
    host = json.loads((capture / "gpu-query.json").read_text()).get("host", capture.name)
    for line in (capture / "gpu-samples.jsonl").read_text().splitlines():
        for row in json.loads(line).get("csv", "").splitlines():
            cols = [c.strip() for c in row.split(",")]
            if len(cols) < 9:
                continue
            g = per_gpu[cols[1]]
            g["uuid"], g["name"], g["total_mib"] = cols[2], cols[3], int(cols[4])
            g["max_used_mib"] = max(g["max_used_mib"], int(cols[5]))
            g["min_free_mib"] = int(cols[6]) if g["min_free_mib"] is None else min(g["min_free_mib"], int(cols[6]))
            g["util"].append(float(cols[7]))
    return {
        host: {
            i: {
                "uuid": g["uuid"],
                "name": g["name"],
                "total_mib": g["total_mib"],
                "max_used_mib": g["max_used_mib"],
                "min_free_mib": g["min_free_mib"],
                "mean_util_pct": r2(statistics.mean(g["util"]), 1) if g["util"] else None,
            }
            for i, g in sorted(per_gpu.items())
        }
    }


def analyze_startup(capture: Path) -> dict[str, dict]:
    report = {}
    for log, ranks in json.loads((capture / "startup-memory.json").read_text()).items():
        for rank, r in ranks.items():
            parts = (
                "target_weights_gb",
                "draft_weights_gb",
                "mamba_state_gb",
                "spec_scratch_gb",
                "kv_target_gb",
                "kv_draft_gb",
            )
            graph_values = list((r.get("graphs_gb") or {}).values())
            graphs = sum(graph_values) if graph_values and all(value is not None for value in graph_values) else None
            row = {k: r.get(k) for k in parts}
            known = [r[k] for k in parts if r.get(k) is not None] + ([graphs] if graphs is not None else [])
            row.update(
                graphs_gb=r2(graphs),
                accounted_gb=r2(sum(known)) if len(known) == len(parts) + 1 else None,
                known_allocations_gb=r2(sum(known)) if known else None,
                available_after_startup_gb=r.get("available_after_startup_gb"),
                kv_tokens=r.get("kv_target_tokens"),
                kv_draft_tokens=r.get("kv_draft_tokens"),
                mamba_slots_raised_to=r.get("mamba_slots_raised_to"),
                mamba_slots=r.get("mamba_slots"),
                max_running_requests=r.get("max_running_requests"),
                running_capped_to=r.get("running_capped_to"),
            )
            report[f"{Path(log).name} [{rank}]"] = row
    return report


def profile_coverage(root: Path, trace_results: dict[str, dict]) -> dict[str, dict]:
    """Check requested engines and TP ranks; a successful HTTP response is not trace validation."""
    coverage = {}
    for requests in root.rglob("profile-requests.json"):
        for request in json.loads(requests.read_text()):
            engine = request["engine"]
            info_path = requests.parent / f"{engine}-server-info.json"
            info = json.loads(info_path.read_text()) if info_path.exists() else {}
            tp = info.get("tp_size") or (info.get("server_args") or {}).get("tp_size")
            stage = "EXTEND" if engine.startswith("prefill") else "DECODE"
            ranks = {}
            for rank in range(int(tp)) if tp else []:
                key = f"{engine} {stage} TP{rank}"
                traces = [value for name, value in trace_results.items() if name == key or name.startswith(key + " [")]
                ranks[str(rank)] = (
                    "GPU kernels present"
                    if any(value.get("nonzero_kernel_events", 0) for value in traces)
                    else "missing or no GPU kernels"
                )
            coverage[f"{requests.parent.relative_to(root)}/{engine}"] = {
                "tp_size": tp,
                "stage": stage,
                "ranks": ranks,
                "complete_warmed_steps_verified": False,
                "note": "Review step boundaries in the original traces; TP size unknown means rank coverage cannot be checked.",
            }
    return coverage


# ---------- report ----------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("--traces", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--tp-summed-hicache-tokens",
        action="store_true",
        help="Confirm the installed build sums unlabelled HiCache movement tokens across TP ranks",
    )
    args = parser.parse_args()
    out = args.out or args.root / "analysis"
    summary = {"metrics": {}, "gpu": {}, "startup": {}, "traces": {}}
    for cap in sorted(args.root.rglob("capture.json")):
        mode = json.loads(cap.read_text()).get("mode")
        where = cap.parent
        try:
            if mode == "metrics" and (where / "scrapes.jsonl").exists():
                summary["metrics"][str(where.relative_to(args.root))] = analyze_metrics(
                    where, tp_summed_hicache_tokens=args.tp_summed_hicache_tokens
                )
            elif mode == "gpu" and (where / "gpu-samples.jsonl").exists():
                summary["gpu"].update(analyze_gpu(where))
            elif mode == "startup" and (where / "startup-memory.json").exists():
                summary["startup"].update(analyze_startup(where))
        except Exception as exc:  # Keep going; report what could not be read.
            summary.setdefault("errors", {})[str(where)] = str(exc)
    for base in [args.root] + args.traces:
        for trace in sorted(base.rglob("*.trace.json*")):
            match = TRACE_NAME.search(trace.name)
            key = f"{match.group(1)} {match.group(3)} TP{match.group(2)}" if match else trace.name
            if key in summary["traces"]:
                key += f" [{trace.parent.relative_to(base)}]"
            try:
                summary["traces"][key] = analyze_trace(trace)
            except Exception as exc:
                summary.setdefault("errors", {})[str(trace)] = str(exc)
    summary["profile_coverage"] = profile_coverage(args.root, summary["traces"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "SUMMARY.md").write_text(render(summary))
    print(out / "SUMMARY.md")


def fmt(value: object) -> str:
    return "–" if value is None else (f"{value:g}" if isinstance(value, (int, float)) else str(value))


def render(s: dict[str, dict]) -> str:
    lines = [
        "# Collection summary",
        "",
        "Generated by analyze.py from raw collect.py output. Times in the source files are UTC.",
        "Percentiles are interpolated inside histogram buckets; `<=x` means only the lowest bucket bound is known. "
        "Means are exact. Trace GPU ranges nest (e.g. `scheduler.run_batch` contains the phases) and communication "
        "kernels include time spent waiting for the slowest TP rank.",
        "",
    ]
    lines += [
        "Unknown values are shown as –. GPU idle spans only the first-to-last GPU activity in each trace. "
        "Copy/kernel duration sums are not exposed critical-path waits. Context lengths, accepted tokens per iteration, "
        "and complete warmed-step coverage require inspection of the raw traces and matching workload timestamps.",
        "",
    ]
    for cap, engines in s["metrics"].items():
        lines += [f"## Metrics window `{cap}`", ""]
        for name, e in engines.items():
            if "error" in e:
                lines += [f"### {name}", "", e["error"], ""]
                continue
            live = e.get("liveness") or {}
            lines += [
                f"### {name} — {fmt(e.get('window_seconds'))} s, TP{fmt(e.get('tp_size'))}, "
                f"last scheduler progress {fmt(live.get('last_scheduler_progress_utc'))}",
                "",
            ]
            tps = e.get("tokens_per_second", {})
            if live.get("warnings"):
                lines += [
                    "WARNING: "
                    + "; ".join(live["warnings"])
                    + ". Check worker logs before using this engine's metrics.",
                    "",
                ]
            lines += [f"UTC window: {e.get('start_utc')} to {e.get('end_utc')}.", ""]
            if e.get("excluded_counter_families"):
                lines += [
                    "Counters excluded for resets or incomplete coverage: "
                    + ", ".join(e["excluded_counter_families"]),
                    "",
                ]
            lines.append("Tokens/s: " + ", ".join(f"{k} {fmt(v)}" for k, v in tps.items() if v is not None))
            lat = {k: v.get("all") for k, v in e.get("latency_seconds", {}).items() if v.get("all")}
            if lat:
                lines += [
                    "",
                    "| Request latency (s) | count | mean | p50 | p90 | p99 |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
                lines += [
                    f"| {k} | {fmt(v['count'])} | {fmt(v['mean'])} | {fmt(v['p50'])} | {fmt(v['p90'])} | {fmt(v['p99'])} |"
                    for k, v in lat.items()
                ]
            stages = e.get("per_stage_request_latency_seconds", {})
            if stages:
                lines += ["", "| Stage (s) | count | mean | p50 | p90 |", "|---|---:|---:|---:|---:|"]
                lines += [
                    f"| {k} | {fmt(v['count'])} | {fmt(v['mean'])} | {fmt(v['p50'])} | {fmt(v['p90'])} |"
                    for k, v in stages.items()
                ]
            kv = {k: v.get("all") for k, v in e.get("kv_transfer", {}).items() if v.get("all")}
            if kv:
                lines += ["", "| KV transfer | count | mean | p50 | p90 |", "|---|---:|---:|---:|---:|"]
                lines += [
                    f"| {k} | {fmt(v['count'])} | {fmt(v['mean'])} | {fmt(v['p50'])} | {fmt(v['p90'])} |"
                    for k, v in kv.items()
                ]
            g = {k: v for k, v in e.get("gauges", {}).items() if v}
            if g:
                lines += ["", "| Gauge | min | mean | max |", "|---|---:|---:|---:|"]
                lines += [f"| {k} | {fmt(v['min'])} | {fmt(v['mean'])} | {fmt(v['max'])} |" for k, v in g.items()]
            h = e.get("hicache")
            if h:
                lines += [
                    "",
                    "HiCache (prefill): " + (h.get("note") or ""),
                    "",
                    "| Effective tokens | Delta | Share |",
                    "|---|---:|---:|",
                ]
                lines += [
                    f"| {mode} | {fmt(value)} | {fmt(h.get('shares', {}).get(mode))} |"
                    for mode, value in h["effective_tokens"].items()
                ]
                lines += [
                    "",
                    h["token_normalization"],
                    "",
                    f"Completed work (separate accounting): {h['completed_work']}",
                    "",
                ]
                for kind in ("backup", "load_back"):
                    if kind in h:
                        k = h[kind]
                        lines.append(
                            f"- {kind}: {fmt(k['gib'])} GiB, {fmt(k['operations'])} ops, mean {fmt(k['mean_ms'])} ms, "
                            f"duration sum {fmt(k['duration_seconds_sum'])} s, logical tokens {k['logical_tokens_by_pool']}, "
                            f"raw tokens {k['raw_tokens_by_pool']}"
                        )
                if "dropped_tokens" in h:
                    lines.append(
                        f"- dropped tokens: {h['dropped_tokens']}; peak host occupancy {fmt(h.get('host_occupancy_peak'))}; "
                        f"storage prefetch hits {fmt(h.get('storage_prefetch_hit_tokens'))}"
                    )
            lines.append("")
    if s["traces"]:
        lines += [
            "## Profiler traces (per engine, stage and TP rank)",
            "",
            "| Trace | window ms | GPU idle % | top idle gap ms | steps | median step ms | batch | launch calls | CPU sync ms |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for key, t in s["traces"].items():
            if "error" in t:
                lines.append(f"| {key} | {t['error']} |||||||")
                continue
            step = t.get("step_interval_ms") or {}
            lines.append(
                f"| {key} | {fmt(t['window_ms'])} | {fmt(t['gpu_idle_pct'])} | {fmt((t['largest_idle_gaps_ms'] or [None])[0])} | "
                f"{t['scheduler_steps']} | {fmt(step.get('median'))} | {fmt((t.get('batch_size') or {}).get('median'))} | "
                f"{t['cpu_launch_calls']['count']} | {fmt(t['cpu_blocked_in_sync_ms'])} |"
            )
        for key, t in s["traces"].items():
            if "error" in t:
                continue
            lines += [
                "",
                f"### {key}",
                "",
                "| GPU range (nested) | count | median ms | p90 ms | total ms |",
                "|---|---:|---:|---:|---:|",
            ]
            lines += [
                f"| {k} | {v['count']} | {fmt(v['median_ms'])} | {fmt(v['p90_ms'])} | {fmt(v['total_ms'])} |"
                for k, v in list(t["gpu_annotation_ranges"].items())[:12]
            ]
            lines += [
                "",
                "Kernel time by category (heuristic): "
                + ", ".join(
                    f"{k} {fmt(v)} ms ({100 * t['kernel_share_by_category'].get(k, 0):.0f}%)"
                    for k, v in t["kernel_ms_by_category"].items()
                ),
            ]
            if t.get("tokens_per_step"):
                lines.append(
                    "Tokens per step: "
                    + ", ".join(
                        f"{k} median {fmt(v['median'])} (max {v['max']}, {v['steps']} steps)"
                        for k, v in t["tokens_per_step"].items()
                    )
                )
            lines += [
                "Memcpy: "
                + ", ".join(f"{k} {v['count']}× {fmt(v['ms'])} ms {fmt(v['mib'])} MiB" for k, v in t["memcpy"].items())
            ]
    if s["startup"]:
        cols = (
            "target_weights_gb",
            "draft_weights_gb",
            "mamba_state_gb",
            "spec_scratch_gb",
            "kv_target_gb",
            "kv_draft_gb",
            "graphs_gb",
            "accounted_gb",
            "available_after_startup_gb",
            "kv_tokens",
            "kv_draft_tokens",
            "mamba_slots",
            "max_running_requests",
        )
        lines += [
            "",
            "## Startup allocation per GPU (GB as logged)",
            "",
            "| Log [rank] | " + " | ".join(cols) + " |",
            "|---|" + "---:|" * len(cols),
        ]
        lines += [f"| {k} | " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for k, r in s["startup"].items()]
    if s.get("profile_coverage"):
        lines += [
            "",
            "## Profile coverage",
            "",
            "| Capture / engine | TP | Stage | Rank coverage |",
            "|---|---:|---|---|",
        ]
        lines += [
            f"| {name} | {fmt(row['tp_size'])} | {row['stage']} | {row['ranks']} |"
            for name, row in s["profile_coverage"].items()
        ]
        lines += [
            "",
            "GPU events do not prove that all captured steps were complete and warmed; inspect the original traces.",
        ]
    if s["gpu"]:
        lines += [
            "",
            "## GPU memory during the window (MiB)",
            "",
            "| Host | GPU | max used | min free | mean util % |",
            "|---|---:|---:|---:|---:|",
        ]
        lines += [
            f"| {h} | {i} | {g['max_used_mib']} | {g['min_free_mib']} | {fmt(g['mean_util_pct'])} |"
            for h, gpus in s["gpu"].items()
            for i, g in gpus.items()
        ]
    if s.get("errors"):
        lines += ["", "## Could not read", ""] + [f"- {k}: {v}" for k, v in s["errors"].items()]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
