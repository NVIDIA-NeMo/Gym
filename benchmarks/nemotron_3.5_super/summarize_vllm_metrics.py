# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize decode-tier vLLM metrics from a combined Slurm log."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


API_SERVER_RE = re.compile(r"\(APIServer pid=(?P<pid>\d+)\).*?non-default args: (?P<args>.*)")
PORT_RE = re.compile(r"'port': (?P<port>\d+)")
DP_SIZE_RE = re.compile(r"'data_parallel_size': (?P<size>\d+)")
KV_ROLE_RE = re.compile(r"kv_role='(?P<role>[^']+)'")
METRIC_RE = re.compile(
    r"\(APIServer pid=(?P<pid>\d+)\).*?INFO "
    r"(?P<date>\d{2}-\d{2}) (?P<time>\d{2}:\d{2}:\d{2}).*?"
    r"Engine (?P<engine>\d+): "
    r"Avg prompt throughput: (?P<prompt>[\d.]+) tokens/s, "
    r"Avg generation throughput: (?P<generation>[\d.]+) tokens/s, "
    r"Running: (?P<running>\d+) reqs, "
    r"Waiting: (?P<waiting>\d+) reqs, "
    r"GPU KV cache usage: (?P<kv>[\d.]+)%"
)


@dataclass(frozen=True)
class Server:
    port: int
    dp_size: int
    kv_role: str | None


@dataclass(frozen=True)
class EngineMetric:
    prompt_tps: float
    generation_tps: float
    running: int
    waiting: int
    kv_percent: float


@dataclass(frozen=True)
class Sample:
    timestamp: str
    engines: int
    prompt_tps: float
    generation_tps: float
    running: int
    waiting: int
    kv_average: float
    kv_minimum: float
    kv_maximum: float

    @property
    def generation_tps_per_request(self) -> float | None:
        if self.running == 0:
            return None
        return self.generation_tps / self.running


def parse_log(path: Path, decode_port: int) -> tuple[list[Sample], int, set[int]]:
    servers: dict[int, Server] = {}
    raw_samples: dict[tuple[int, str], dict[int, EngineMetric]] = defaultdict(dict)

    with path.open(errors="replace") as log:
        for line in log:
            if server_match := API_SERVER_RE.search(line):
                args = server_match.group("args")
                port_match = PORT_RE.search(args)
                if port_match:
                    pid = int(server_match.group("pid"))
                    dp_size_match = DP_SIZE_RE.search(args)
                    role_match = KV_ROLE_RE.search(args)
                    servers[pid] = Server(
                        port=int(port_match.group("port")),
                        dp_size=int(dp_size_match.group("size")) if dp_size_match else 1,
                        kv_role=role_match.group("role") if role_match else None,
                    )
                continue

            metric_match = METRIC_RE.search(line)
            if not metric_match:
                continue
            pid = int(metric_match.group("pid"))
            server = servers.get(pid)
            if server is None or server.port != decode_port:
                continue

            timestamp = f"{metric_match.group('date')} {metric_match.group('time')}"
            engine = int(metric_match.group("engine"))
            raw_samples[(pid, timestamp)][engine] = EngineMetric(
                prompt_tps=float(metric_match.group("prompt")),
                generation_tps=float(metric_match.group("generation")),
                running=int(metric_match.group("running")),
                waiting=int(metric_match.group("waiting")),
                kv_percent=float(metric_match.group("kv")),
            )

    decode_pids = {pid for pid, server in servers.items() if server.port == decode_port}
    if not decode_pids:
        raise ValueError(f"No decode APIServer using port {decode_port} was found in {path}")

    expected_engines = sum(servers[pid].dp_size for pid in decode_pids)
    snapshots = sorted(
        (
            datetime.strptime(timestamp, "%m-%d %H:%M:%S"),
            timestamp,
            pid,
            by_engine,
        )
        for (pid, timestamp), by_engine in raw_samples.items()
    )

    # A coupled DP server reports all engines under one PID. Independent replicas
    # report one engine under each PID, and their ten-second logger ticks can differ
    # slightly. Emit a sample after every decode server has supplied one new snapshot.
    latest_by_pid: dict[int, tuple[datetime, dict[int, EngineMetric]]] = {}
    last_emitted_by_pid: dict[int, datetime] = {}
    samples = []
    for timestamp_value, timestamp, pid, by_engine in snapshots:
        latest_by_pid[pid] = (timestamp_value, by_engine)
        if latest_by_pid.keys() != decode_pids:
            continue
        if any(
            latest_by_pid[decode_pid][0] <= last_emitted_by_pid.get(decode_pid, datetime.min)
            for decode_pid in decode_pids
        ):
            continue

        metrics = [metric for decode_pid in decode_pids for metric in latest_by_pid[decode_pid][1].values()]
        kv_values = [metric.kv_percent for metric in metrics]
        samples.append(
            Sample(
                timestamp=timestamp,
                engines=len(metrics),
                prompt_tps=sum(metric.prompt_tps for metric in metrics),
                generation_tps=sum(metric.generation_tps for metric in metrics),
                running=sum(metric.running for metric in metrics),
                waiting=sum(metric.waiting for metric in metrics),
                kv_average=sum(kv_values) / len(kv_values),
                kv_minimum=min(kv_values),
                kv_maximum=max(kv_values),
            )
        )
        last_emitted_by_pid = {decode_pid: latest_by_pid[decode_pid][0] for decode_pid in decode_pids}

    return samples, expected_engines, decode_pids


def print_samples(samples: list[Sample], expected_engines: int) -> None:
    print("timestamp       engines  output TPS  running  TPS/request  waiting  KV avg       KV range")
    for sample in samples:
        tps_per_request = sample.generation_tps_per_request
        tps_text = f"{tps_per_request:11.1f}" if tps_per_request is not None else "          —"
        engine_text = f"{sample.engines}/{expected_engines}"
        print(
            f"{sample.timestamp:14} {engine_text:>7} "
            f"{sample.generation_tps:11.1f} {sample.running:8d} {tps_text} "
            f"{sample.waiting:8d} {sample.kv_average:7.1f}% "
            f"{sample.kv_minimum:5.1f}–{sample.kv_maximum:5.1f}%"
        )


def print_summary(samples: list[Sample], expected_engines: int, decode_pids: set[int]) -> None:
    if not samples:
        print("No matching active decode samples found.")
        return

    total_generation_tps = sum(sample.generation_tps for sample in samples)
    total_running = sum(sample.running for sample in samples)
    weighted_tps_per_request = total_generation_tps / total_running if total_running else 0.0
    kv_weight = sum(sample.engines for sample in samples)
    kv_average = sum(sample.kv_average * sample.engines for sample in samples) / kv_weight

    print()
    print(f"Decode API PID(s): {', '.join(str(pid) for pid in sorted(decode_pids))}")
    print(f"Active samples: {len(samples)} ({sum(s.engines == expected_engines for s in samples)} complete)")
    print(f"Average aggregate output TPS: {total_generation_tps / len(samples):.1f}")
    print(f"Weighted output TPS/request: {weighted_tps_per_request:.1f}")
    print(
        "Waiting requests: "
        f"average {sum(sample.waiting for sample in samples) / len(samples):.2f}, "
        f"maximum {max(sample.waiting for sample in samples)}"
    )
    print(
        f"GPU KV-cache use: average {kv_average:.1f}%, "
        f"range {min(sample.kv_minimum for sample in samples):.1f}–"
        f"{max(sample.kv_maximum for sample in samples):.1f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="Combined Slurm log produced by sbatch_external_vllm.sh")
    parser.add_argument("--decode-port", type=int, default=8002, help="Decode API port (default: 8002)")
    parser.add_argument("--last", type=int, default=20, help="Number of recent active samples to show (default: 20)")
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Include timestamps that do not contain every expected decode engine",
    )
    args = parser.parse_args()
    if args.last < 1:
        parser.error("--last must be positive")

    try:
        samples, expected_engines, decode_pids = parse_log(args.log, args.decode_port)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    selected = [sample for sample in samples if sample.running > 0]
    if not args.include_incomplete:
        selected = [sample for sample in selected if sample.engines == expected_engines]
    selected = selected[-args.last :]

    print_samples(selected, expected_engines)
    print_summary(selected, expected_engines, decode_pids)


if __name__ == "__main__":
    main()
