# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Diagnose vLLM prefill/decode worker imbalance from a Slurm job log.

`sbatch_external_vllm.sh` funnels every vLLM node's stdout into one Slurm log, so a
single file already contains the whole cluster's engine stats. This reads them back
and answers whether the router spread decode work evenly, and what it cost.

Three views, each taking `LABEL=LOG` pairs (the label is cosmetic; a bare path works):

    workers      per-worker load summary and a time-bucketed decode split, one run
    compare      decode max/min ratio across runs, gated on actual saturation
    throughput   rollout completion rate, by wall clock and by matched completion

Read `compare` before `workers`: imbalance only develops while the decode nodes are
saturated, so a run whose queue drains early looks perfectly balanced no matter what
the router did. `--min-load` is what keeps that from reading as a clean result.

Examples:
    decode_balance.py workers slurm-logs/<jobid>-gym-<exp>/<bench>/<model>.log
    decode_balance.py compare baseline=old.log patched=new.log
    decode_balance.py throughput baseline=old.log patched=new.log --bands
"""

import argparse
import collections
import datetime
import re
import statistics
import sys


# vLLM's per-engine stats line, one per API server, all interleaved in the Slurm log.
ENGINE_STAT = re.compile(
    r"\(APIServer pid=(\d+)\) INFO (\d\d-\d\d \d\d:\d\d:\d\d) \[loggers\.py:\d+\] "
    r"Engine \d+: Avg prompt throughput: ([\d.]+) tokens/s, "
    r"Avg generation throughput: ([\d.]+) tokens/s, "
    r"Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%"
)
# vLLM echoes its resolved args at startup; that is where a pid gets a hostname.
HOST = re.compile(r"\(APIServer pid=(\d+)\).*'host': '([^']+)'")
# Gym's rollout progress bar. It carries its own elapsed clock, which is what lets
# runs that started at different times be compared without aligning wall clocks.
ROLLOUT_BAR = re.compile(r"Collecting rollouts:\s+\d+%\|[^|]*\|\s*(\d+)/(\d+) \[(\d+:\d+(?::\d+)?)<")

Sample = collections.namedtuple("Sample", "time prompt gen running waiting kv")


def _elapsed_seconds(hms):
    parts = [int(p) for p in hms.split(":")]
    return parts[0] * 60 + parts[1] if len(parts) == 2 else parts[0] * 3600 + parts[1] * 60 + parts[2]


def parse_engine_stats(path):
    """Return (hosts_by_pid, samples_by_pid, decode_pids).

    Roles are inferred rather than configured: a decode worker spends its time
    generating, a prefill worker consuming prompt tokens. Averaging over the whole
    run avoids misclassifying either during startup, when neither is doing much.
    """
    hosts, rows = {}, collections.defaultdict(list)
    with open(path, errors="replace") as handle:
        for line in handle:
            host_match = HOST.search(line)
            if host_match:
                hosts.setdefault(host_match.group(1), host_match.group(2))
            stat_match = ENGINE_STAT.search(line)
            if stat_match:
                pid, timestamp, prompt, gen, running, waiting, kv = stat_match.groups()
                # The log timestamps carry no year; only differences are ever used.
                when = datetime.datetime.strptime(f"2000-{timestamp}", "%Y-%m-%d %H:%M:%S")
                rows[pid].append(Sample(when, float(prompt), float(gen), int(running), int(waiting), float(kv)))
    decode_pids = [
        pid
        for pid, samples in rows.items()
        if sum(s.gen for s in samples) / len(samples) > sum(s.prompt for s in samples) / len(samples)
    ]
    return hosts, rows, decode_pids


def decode_grid(rows, decode_pids, bucket_seconds=60):
    """Align decode workers onto a shared time grid; their loggers do not tick together.

    Buckets missing any worker are dropped, so a ratio is never computed from a
    partially observed instant.
    """
    if not decode_pids:
        return []
    start = min(s.time for pid in decode_pids for s in rows[pid])
    grid = collections.defaultdict(dict)
    for pid in decode_pids:
        per_bucket = collections.defaultdict(list)
        for sample in rows[pid]:
            per_bucket[int((sample.time - start).total_seconds() // bucket_seconds)].append(sample)
        for bucket, samples in per_bucket.items():
            grid[bucket][pid] = (
                sum(s.running for s in samples) / len(samples),
                max(s.waiting for s in samples),
                max(s.kv for s in samples),
            )
    return [(b, grid[b]) for b in sorted(grid) if len(grid[b]) == len(decode_pids)]


def parse_rollouts(path):
    """Return (sorted [(elapsed_seconds, completed)], total) from the progress bar."""
    earliest = {}
    total = 0
    with open(path, errors="replace") as handle:
        for line in handle:
            for done, run_total, elapsed in ROLLOUT_BAR.findall(line):
                done, elapsed, total = int(done), _elapsed_seconds(elapsed), int(run_total)
                if done not in earliest or elapsed < earliest[done]:
                    earliest[done] = elapsed
    return sorted((elapsed, done) for done, elapsed in earliest.items()), total


def split_label(spec):
    label, _, path = spec.partition("=")
    return (label, path) if path else (spec, spec)


def cmd_workers(args):
    hosts, rows, decode_pids = parse_engine_stats(args.log)
    if not rows:
        sys.exit(f"no vLLM engine stat lines in {args.log}")
    roles = {pid: ("decode" if pid in decode_pids else "prefill") for pid in rows}
    print(f"{'worker':<28} {'role':<8} {'n':>5} {'avgRun':>7} {'maxRun':>7} {'maxWait':>8} {'maxKV%':>7}")
    for pid, samples in sorted(rows.items(), key=lambda kv: roles[kv[0]]):
        running = [s.running for s in samples]
        print(
            f"{f'{hosts.get(pid, chr(63))} (pid {pid})':<28} {roles[pid]:<8} {len(samples):>5} "
            f"{sum(running) / len(running):>7.1f} {max(running):>7} "
            f"{max(s.waiting for s in samples):>8} {max(s.kv for s in samples):>7.1f}"
        )
    if len(decode_pids) < 2:
        return
    print(f"\ndecode running-request split, {args.bucket}-minute buckets")
    labels = [hosts.get(pid, pid) for pid in decode_pids]
    print("  min  " + "  ".join(f"{name:>16}" for name in labels) + "   max/min")
    grid = decode_grid(rows, decode_pids, bucket_seconds=args.bucket * 60)
    for bucket, per_pid in grid:
        loads = [per_pid[pid] for pid in decode_pids]
        low = min(load[0] for load in loads)
        ratio = max(load[0] for load in loads) / low if low > 0.5 else float("nan")
        cells = "  ".join(f"{load[0]:>9.1f} kv{load[2]:>4.0f}%" for load in loads)
        print(f"{bucket * args.bucket:>5}  {cells}   {ratio:>6.2f}")


def cmd_compare(args):
    print(
        "decode max/min running-request ratio, sampled per minute, "
        f"only while combined decode load >= {args.min_load}\n"
    )
    print(
        f"{'run':<22} {'loaded':>6}  {'median':>7} {'max':>7} {'early':>7} {'late':>7} "
        f"{'maxKV%':>7} {'maxWait':>7}  decode nodes"
    )
    for spec in args.logs:
        label, path = split_label(spec)
        hosts, rows, decode_pids = parse_engine_stats(path)
        if len(decode_pids) < 2:
            print(f"{label:<22} no decode pair found yet in {path}")
            continue
        grid = decode_grid(rows, decode_pids)
        loaded = []
        for _, per_pid in grid:
            loads = [per_pid[pid][0] for pid in decode_pids]
            if sum(loads) >= args.min_load and min(loads) > 0:
                loaded.append(
                    (
                        max(loads) / min(loads),
                        max(per_pid[pid][2] for pid in decode_pids),
                        max(per_pid[pid][1] for pid in decode_pids),
                    )
                )
        if not loaded:
            peak = max((sum(p[pid][0] for pid in decode_pids) for _, p in grid), default=0)
            print(f"{label:<22} never reached combined load {args.min_load} (peak {peak:.0f}) - inconclusive")
            continue
        ratios = [entry[0] for entry in loaded]
        third = max(1, len(loaded) // 3)
        nodes = "/".join(hosts.get(pid, pid).split("-")[-1] for pid in decode_pids)
        print(
            f"{label:<22} {len(loaded):>5}m  {statistics.median(ratios):>7.2f} {max(ratios):>7.2f} "
            f"{statistics.median(r for r, _, _ in loaded[:third]):>7.2f} "
            f"{statistics.median(r for r, _, _ in loaded[-third:]):>7.2f} "
            f"{max(e[1] for e in loaded):>7.1f} {max(e[2] for e in loaded):>7.0f}  {nodes}"
        )
    print(
        "\nearly/late are the median ratio over the first and last third of the saturated\n"
        "window. A rising early->late is the vllm-project/router#197 runaway."
    )


def cmd_throughput(args):
    runs = []
    for spec in args.logs:
        label, path = split_label(spec)
        samples, total = parse_rollouts(path)
        if samples:
            runs.append((label, samples, total))
        else:
            print(f"{label}: no rollout progress bar found yet")
    if not runs:
        return

    print(f"rollout completion rate, per {args.window}-minute window of eval wall clock\n")
    header = f"{'window (min)':>13}" + "".join(f" | {label:^22}" for label, _, _ in runs)
    print(header)
    print(f"{'':>13}" + "".join(f" | {'done':>7} {'rate/min':>12}" for _ in runs))
    print("-" * len(header))
    tables = []
    for _, samples, _ in runs:
        table, previous, edge = {}, 0, args.window * 60
        for elapsed, done in samples:
            while elapsed > edge:
                table[edge // 60] = (previous, done - previous)
                previous, edge = done, edge + args.window * 60
        tables.append(table)
    for edge in sorted({e for table in tables for e in table}):
        row = f"{edge:>13}"
        for table in tables:
            if edge in table:
                done, delta = table[edge]
                row += f" | {done:>7} {delta / args.window:>12.2f}"
            else:
                row += f" | {'-':>7} {'-':>12}"
        print(row)

    if args.bands:
        # Rate against wall clock is confounded by run phase: every run slows near the
        # end as only stragglers remain. Time-per-band compares at equal progress.
        print(f"\nminutes to cross each {args.band_size}-rollout band (matched completion)\n")
        header = f"{'band':>13}" + "".join(f"{label[:12]:>14}" for label, _, _ in runs)
        print(header)
        print("-" * len(header))

        def reached(samples, count):
            return next((elapsed for elapsed, done in samples if done >= count), None)

        total = max(t for _, _, t in runs)
        for low in range(0, total, args.band_size):
            high = low + args.band_size
            row = f"{f'{low}-{high}':>13}"
            for _, samples, _ in runs:
                start, end = reached(samples, low or 1), reached(samples, high)
                row += f"{(end - start) / 60:>14.1f}" if start is not None and end is not None else f"{'-':>14}"
            print(row)

    print()
    for label, samples, total in runs:
        elapsed, done = samples[-1]
        print(f"{label:<22} reached {done}/{total} at {elapsed // 60}m")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    workers = sub.add_parser("workers", help="per-worker summary and decode split for one run")
    workers.add_argument("log")
    workers.add_argument("--bucket", type=int, default=5, help="bucket size in minutes (default 5)")
    workers.set_defaults(func=cmd_workers)

    compare = sub.add_parser("compare", help="decode balance across runs, gated on saturation")
    compare.add_argument("logs", nargs="+", metavar="LABEL=LOG")
    compare.add_argument(
        "--min-load",
        type=int,
        default=200,
        help="only count minutes where combined decode running requests reach this (default 200)",
    )
    compare.set_defaults(func=cmd_compare)

    throughput = sub.add_parser("throughput", help="rollout completion rate across runs")
    throughput.add_argument("logs", nargs="+", metavar="LABEL=LOG")
    throughput.add_argument("--window", type=int, default=15, help="window size in minutes (default 15)")
    throughput.add_argument("--bands", action="store_true", help="also compare at matched completion")
    throughput.add_argument("--band-size", type=int, default=100, help="rollouts per band (default 100)")
    throughput.set_defaults(func=cmd_throughput)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
