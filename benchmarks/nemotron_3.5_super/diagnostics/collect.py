#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect existing SGLang evidence; profiling requires the explicit profile mode.

Python standard library only. No inference requests are generated, no cache is flushed,
and no server configuration is changed by metrics/gpu modes.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def sanitize(value: object) -> object:
    """Redact credential fields recursively before persisting server metadata."""
    if isinstance(value, dict):
        return {
            k: (
                "<redacted>"
                if any(
                    secret in k.lower()
                    for secret in (
                        "api_key",
                        "password",
                        "secret",
                        "access_token",
                        "auth_token",
                        "credential",
                        "launch_command",
                    )
                )
                else sanitize(v)
            )
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    return value


def http(url: str, payload: dict[str, object] | None = None) -> tuple[str, int]:
    headers = {}
    token = os.environ.get("SGLANG_DIAG_AUTH_TOKEN")
    if token:
        headers["Authorization"] = "Bearer " + token
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode(), response.status


def engines(args: argparse.Namespace) -> dict[str, str]:
    """Resolve named, direct engine endpoints without accepting credentials in URLs."""
    result = {}
    if args.endpoints_json:
        doc = json.loads(Path(args.endpoints_json).read_text())
        result.update(doc.get("inference_metrics", doc).get("endpoints", doc))
    for item in args.engine:
        name, url = item.split("=", 1)
        if name in result:
            raise ValueError(f"Duplicate engine name: {name}")
        result[name] = url
    if not result:
        for role in ("prefill", "decode"):
            addresses = os.environ.get(f"SRT_{role.upper()}_ENDPOINTS", "").split(",")
            for index, url in enumerate(filter(None, (a.strip() for a in addresses))):
                result[f"{role}{index}"] = url
    if not result:
        raise ValueError(
            "Pass --engine NAME=URL, --endpoints-json, or the SRT_PREFILL_ENDPOINTS/SRT_DECODE_ENDPOINTS environment variables."
        )
    for name, url in list(result.items()):
        if not re.fullmatch(r"(?:prefill|decode)\d+", name):
            raise ValueError(f"Expected a direct engine name such as prefill0, not {name!r}")
        url = url.strip().rstrip("/")
        # SRT exports bare host:port addresses; the Gym recipe adds http:// itself.
        if "://" not in url:
            url = "http://" + url
        if url.endswith("/metrics"):
            url = url[: -len("/metrics")]
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Engine URL must include http:// or https://: {name}")
        parts = urlsplit(url)
        if not parts.hostname or parts.username or parts.password or parts.query or parts.fragment or parts.path:
            raise ValueError(f"Expected a direct engine origin without credentials, query or path: {name}")
        result[name] = url
    if len(set(result.values())) != len(result):
        raise ValueError("Each engine must have a distinct endpoint")
    return result


def read_info(name: str, url: str, out: Path) -> None:
    errors = []
    for route in ("/server_info", "/get_server_info"):
        try:
            body, _ = http(url + route)
            save(out / f"{name}-server-info.json", sanitize(json.loads(body)))
            return
        except Exception as exc:
            errors.append({"route": route, "error": str(exc)})
    save(out / f"{name}-server-info-errors.json", errors)


def scheduler_counters(body: str) -> dict[str, float]:
    # Scheduler-emitted series carry tp_rank; tokenizer/process counters keep moving after a scheduler crash.
    values = {}
    for line in body.splitlines():
        if line.startswith("sglang:") and 'tp_rank="' in line:
            key, value, *_ = line.split()
            if key.split("{", 1)[0].endswith(("_total", "_count")):
                try:
                    number = float(value)
                except ValueError:
                    continue
                if math.isfinite(number):
                    values[key] = number
    return values


def report_liveness(out: Path, progress: dict[str, dict], end: float, stall_seconds: float) -> None:
    # A crashed scheduler can leave /metrics answering 200 with frozen counters,
    # so judge liveness by when scheduler counters last moved, not by HTTP status.
    report = {}
    for name, p in progress.items():
        quiet = None if p["last_progress"] is None else round(end - p["last_progress"], 1)
        entry = {
            "server_info_ok": (out / f"{name}-server-info.json").exists(),
            "ok_scrapes": p["ok_scrapes"],
            "last_scheduler_progress_utc": p["last_progress_utc"],
            "seconds_without_progress_at_end": quiet,
        }
        report[name] = entry
        reasons = [] if entry["server_info_ok"] else ["/server_info failed"]
        if quiet is None:
            reasons.append("no scheduler counter advanced")
        elif quiet > stall_seconds:
            reasons.append(f"no scheduler progress in the last {quiet:.0f} s")
        entry["warnings"] = reasons
        if reasons:
            print(
                f"WARNING {name}: {'; '.join(reasons)}. The engine is idle or its scheduler is not running; "
                "check the worker log before using these metrics.",
                file=sys.stderr,
            )
    save(out / "liveness.json", report)


def collect_metrics(args: argparse.Namespace, endpoints: dict[str, str], out: Path) -> None:
    inventory = {name: set() for name in endpoints}
    progress = {
        name: {"ok_scrapes": 0, "prev": None, "last_progress": None, "last_progress_utc": None} for name in endpoints
    }
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
        list(pool.map(lambda item: read_info(*item, out), endpoints.items()))

        def scrape(item, sample):
            name, url = item
            record = {"engine": name, "sample": sample, "start_utc": utc()}
            begin = time.monotonic()
            try:
                body, status = http(url + "/metrics")
                filename = f"{name}-{sample:04d}.prom.gz"
                with gzip.open(out / filename, "wt") as f:
                    f.write(body)
                names = {
                    line.split("{", 1)[0].split()[0] for line in body.splitlines() if line and not line.startswith("#")
                }
                record.update(
                    status=status,
                    file=filename,
                    metric_names=sorted(names),
                    counters=scheduler_counters(body),
                    done=time.monotonic(),
                )
            except Exception as exc:
                record["error"] = str(exc)
            record.update(end_utc=utc(), elapsed_seconds=time.monotonic() - begin)
            return record

        start = time.monotonic()
        if getattr(args, "window_marker", None):
            marker = args.window_marker
            pending = marker.with_suffix(".pending")
            save(pending, {"start_utc": utc(), "start_epoch": time.time(), "end_epoch": time.time() + args.seconds})
            pending.replace(marker)
        sample = 0
        with (out / "scrapes.jsonl").open("w") as log:
            while True:
                records = list(pool.map(lambda item: scrape(item, sample), endpoints.items()))
                for record in records:
                    inventory[record["engine"]].update(record.pop("metric_names", []))
                    counters, done = record.pop("counters", None), record.pop("done", None)
                    if counters is not None:
                        p = progress[record["engine"]]
                        p["ok_scrapes"] += 1
                        if p["prev"] is not None and any(
                            p["prev"].get(k) not in (None, v) for k, v in counters.items()
                        ):
                            p["last_progress"], p["last_progress_utc"] = done, record["start_utc"]
                        p["prev"] = counters
                    log.write(json.dumps(record) + "\n")
                log.flush()
                elapsed = time.monotonic() - start
                if elapsed >= args.seconds:
                    break
                sample += 1
                time.sleep(min(args.interval, args.seconds - elapsed))
    save(out / "metric-inventory.json", {name: sorted(values) for name, values in inventory.items()})
    report_liveness(out, progress, time.monotonic(), max(15.0, 3 * args.interval))


def schema_properties(schema: dict, api: dict, seen: set[str] | None = None) -> set[str]:
    seen = set() if seen is None else seen
    if "$ref" in schema:
        ref = schema["$ref"]
        if ref in seen or not ref.startswith("#/"):
            return set()
        seen.add(ref)
        node = api
        for part in ref[2:].split("/"):
            node = node[part.replace("~1", "/").replace("~0", "~")]
        return schema_properties(node, api, seen)
    fields = set(schema.get("properties", {}))
    for key in ("anyOf", "oneOf", "allOf"):
        for branch in schema.get(key, []):
            fields |= schema_properties(branch, api, seen.copy())
    return fields


def collect_profile(args: argparse.Namespace, endpoints: dict[str, str], out: Path) -> None:
    """Preflight every engine, then issue one bounded request per engine without retries."""
    if not args.server_profile_dir or not args.server_profile_dir.startswith("/"):
        raise ValueError("profile mode needs an absolute --server-profile-dir visible in the serving containers")
    planned = []
    # Preflight every endpoint before starting any profiler.
    for name, url in endpoints.items():
        read_info(name, url, out)
        body, _ = http(url + "/openapi.json")
        api = json.loads(body)
        save(out / f"{name}-openapi.json", sanitize(api))
        operation = api["paths"]["/start_profile"]["post"]
        schema = operation["requestBody"]["content"]["application/json"]["schema"]
        payload = {
            "output_dir": f"{args.server_profile_dir.rstrip('/')}/{name}",
            "profile_id": f"busy-{name}",
            "num_steps": args.steps,
            "activities": ["CPU", "GPU"],
            "profile_by_stage": True,
            "with_stack": False,
            "record_shapes": False,
        }
        missing = set(payload) - schema_properties(schema, api)
        if missing:
            raise ValueError(
                f"{name}: installed /start_profile schema does not advertise {sorted(missing)}; no profiling was started"
            )
        planned.append((name, url, payload))
    save(out / "profile-requests.json", [{"engine": n, "url": u, "payload": p} for n, u, p in planned])

    def start(item):
        name, url, payload = item
        record = {"engine": name, "start_utc": utc()}
        try:
            body, status = http(url + "/start_profile", payload)
            record.update(status=status, response=body)
        except Exception as exc:
            record.update(error=str(exc), outcome="unknown; inspect worker logs before attempting another start")
        record["end_utc"] = utc()
        return record

    with ThreadPoolExecutor(max_workers=len(planned)) as pool:
        results = list(pool.map(start, planned))
    save(out / "profile-responses.json", results)
    if any("error" in result for result in results):
        raise RuntimeError("A profile request failed; inspect profile-responses.json and worker logs before retrying")
    print(
        "Control responses saved. Confirm actual trace files and GPU events on every TP rank; HTTP 200 is not capture validation."
    )


def collect_gpu(args: argparse.Namespace, out: Path) -> None:
    fields = "timestamp,index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory"
    command = ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"]
    save(
        out / "gpu-query.json",
        {"host": socket.gethostname(), "fields": fields, "memory_unit": "MiB", "utilization_unit": "percent"},
    )
    for source in [
        "/proc/meminfo",
        "/sys/fs/cgroup/memory.current",
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
    ]:
        p = Path(source)
        if p.is_file():
            (out / (source.strip("/").replace("/", "_") + ".txt")).write_text(p.read_text())
    start = time.monotonic()
    with (out / "gpu-samples.jsonl").open("w") as log:
        while True:
            record = {"utc": utc()}
            proc = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=15)
            record.update(returncode=proc.returncode, csv=proc.stdout, stderr=proc.stderr)
            log.write(json.dumps(record) + "\n")
            log.flush()
            if proc.returncode:
                raise RuntimeError("nvidia-smi failed; inspect gpu-samples.jsonl")
            elapsed = time.monotonic() - start
            if elapsed >= args.seconds:
                break
            time.sleep(min(args.interval, args.seconds - elapsed))


STARTUP_MARKERS = (
    "Load weight begin",
    "Load weight end",
    "Mamba Cache is allocated",
    "raising effective_mamba_size",
    "max_running_requests is capped",
    "KV Cache is allocated",
    "Memory pool end",
    "CUDA graph end",
    "max_total_num_tokens=",
    "Tree cache initialized",
    "transfer_layer_num",
)
RANK_TAG = re.compile(r"^\[[^\]]*?((?:DP\d+ )?(?:PP\d+ )?TP\d+(?: EP\d+)?)\]")


def number(pattern: str, line: str) -> float | None:
    match = re.search(pattern, line)
    return float(match.group(1)) if match else None


def parse_startup_line(rank: dict[str, object], line: str) -> None:
    if "Load weight end" in line:
        kind = re.search(r"type=([^,]+)", line)
        kind = kind.group(1) if kind else "?"
        key = "draft_weights_gb" if re.search(r"MTP|Draft|Eagle|NextN", kind) else "target_weights_gb"
        rank[key] = number(r"mem usage=([\d.]+)", line)
    elif "Mamba Cache is allocated" in line:
        rank["mamba_slots"] = number(r"max_mamba_cache_size: (\d+)", line)
        for key, patterns in (
            ("mamba_state_gb", (r" conv_state size: ([\d.]+)", r" ssm_state size: ([\d.]+)")),
            (
                "spec_scratch_gb",
                (r"intermediate_ssm_state_cache size: ([\d.]+)", r"intermediate_conv_window_cache size: ([\d.]+)"),
            ),
        ):
            values = [number(pattern, line) for pattern in patterns]
            rank[key] = round(sum(values), 2) if all(value is not None for value in values) else None
    elif "raising effective_mamba_size" in line:
        rank["mamba_slots_raised_to"] = number(r"raising effective_mamba_size to (\d+)", line)
    elif "max_running_requests is capped" in line:
        rank["running_capped_to"] = number(r"capped to (\d+)", line)
    elif "KV Cache is allocated" in line:
        # The target pool is allocated first, the draft pool second.
        key = "kv_draft" if "kv_target_gb" in rank else "kv_target"
        values = [number(pattern, line) for pattern in (r"K size: ([\d.]+)", r"V size: ([\d.]+)")]
        rank[key + "_gb"] = round(sum(values), 2) if all(value is not None for value in values) else None
        rank[key + "_tokens"] = number(r"#tokens: (\d+)", line)
    elif "CUDA graph end" in line:
        name = re.search(r"Capture (.*?) [Cc][Uu][Dd][Aa] graph end", line)
        rank.setdefault("graphs_gb", {})[name.group(1) if name else "?"] = number(r"mem usage=([\d.]+)", line)
    elif "max_total_num_tokens=" in line:
        for key in (
            "max_total_num_tokens",
            "max_running_requests",
            "context_len",
            "chunked_prefill_size",
            "max_prefill_tokens",
        ):
            rank[key] = number(key + r"=(\d+)", line)
        rank["available_after_startup_gb"] = number(r"available_gpu_mem=([\d.]+)", line)
    elif "Tree cache initialized" in line:
        rank["tree_cache"] = line.split("Tree cache initialized:", 1)[1].strip()
    elif "transfer_layer_num" in line:
        rank["hicache_transfer_layer_num"] = number(r"transfer_layer_num=(\d+)", line)


def collect_startup(args: argparse.Namespace, out: Path) -> None:
    # Reads worker logs only; run it wherever the P/D worker logs are visible.
    report = {}
    logs = out / "worker-logs"
    logs.mkdir()
    sources = {}
    with (out / "startup-lines.txt").open("w") as kept:
        for index, path in enumerate(args.log):
            saved = logs / f"{index}-{Path(path).name}"
            shutil.copyfile(path, saved)
            sources[str(path)] = str(saved.relative_to(out))
            ranks = {}
            with saved.open(errors="replace") as log:
                for line in log:
                    if not any(marker in line for marker in STARTUP_MARKERS):
                        continue
                    tag = RANK_TAG.match(line)
                    kept.write(f"{path}: {line}")
                    if tag:
                        parse_startup_line(ranks.setdefault(tag.group(1), {}), line)
            report[str(path)] = ranks
    save(out / "startup-memory.json", report)
    save(out / "worker-log-sources.json", sources)
    if not any(report.values()):
        print("WARNING: no allocation lines found; pass the full worker startup logs.", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["metrics", "gpu", "profile", "startup"])
    parser.add_argument(
        "--out", type=Path, required=True, help="New output directory; never overwrites a previous capture"
    )
    parser.add_argument("--engine", action="append", default=[], help="Direct engine NAME=URL; repeat for P0/P1/D0/D1")
    parser.add_argument(
        "--endpoints-json", help="JSON endpoint map, including SRT's JSON-formatted inference-metrics.yaml"
    )
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--server-profile-dir")
    parser.add_argument(
        "--window-marker", type=Path, help="metrics mode: publish the sampling window for GPU sidecars"
    )
    parser.add_argument(
        "--log", action="append", default=[], help="startup mode: a full P/D worker log; repeat per worker"
    )
    args = parser.parse_args()
    if (
        not math.isfinite(args.seconds)
        or not math.isfinite(args.interval)
        or args.seconds <= 0
        or args.interval <= 0
        or args.steps <= 0
    ):
        parser.error("seconds, interval and steps must be positive")
    if args.mode == "startup" and not args.log:
        parser.error("startup mode needs at least one --log")
    endpoints = {} if args.mode in ("gpu", "startup") else engines(args)
    args.out.mkdir(parents=True, exist_ok=False)
    save(
        args.out / "capture.json",
        {
            "mode": args.mode,
            "start_utc": utc(),
            "endpoints": endpoints,
            "seconds": args.seconds,
            "interval": args.interval,
        },
    )
    try:
        if args.mode == "metrics":
            collect_metrics(args, endpoints, args.out)
        elif args.mode == "gpu":
            collect_gpu(args, args.out)
        elif args.mode == "startup":
            collect_startup(args, args.out)
        else:
            collect_profile(args, endpoints, args.out)
    except Exception as exc:
        save(args.out / "ERROR.json", {"utc": utc(), "error": str(exc)})
        raise
    save(args.out / "finished.json", {"end_utc": utc()})
    print(args.out)


if __name__ == "__main__":
    main()
