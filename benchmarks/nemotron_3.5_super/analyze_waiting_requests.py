# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


parser = ArgumentParser()
parser.add_argument("--log-fpath", type=str, required=True)
parser.add_argument("--output-fpath", type=str, help="Defaults to <log-fpath>.waiting.png")
args = parser.parse_args()

pattern = re.compile(
    r"Avg prompt throughput: (\d+)(?:\.\d+)? tokens/s, "
    r"Avg generation throughput: (\d+)(?:\.\d+)? tokens/s, "
    r"Running: \d+ reqs, Waiting: (\d+) reqs"
)
timestamp_pattern = re.compile(r"\b(?:(\d{4})-)?(\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2}(?:\.\d+)?)")
waiting_by_engine = defaultdict(list)
with open(args.log_fpath, errors="replace") as f:
    for line in f:
        match = pattern.search(line)
        if match is None:
            continue

        timestamp_match = timestamp_pattern.search(line)
        if timestamp_match is None:
            continue

        year, date, time = timestamp_match.groups()
        # vLLM normally logs MM-DD without a year; use a leap year to allow February 29.
        timestamp = datetime.fromisoformat(f"{year or '2000'}-{date}T{time}")
        prompt, generation, waiting = map(int, match.groups())
        engine_type = "prefill" if prompt > generation else "decode"
        waiting_by_engine[engine_type].append((timestamp, waiting))

if not waiting_by_engine:
    parser.error("No waiting-request metrics with timestamps found in the log")

fig, ax = plt.subplots(figsize=(12, 5))
for engine_type, samples in sorted(waiting_by_engine.items()):
    timestamps, waiting = zip(*sorted(samples, key=lambda sample: sample[0]))
    (line,) = ax.plot(timestamps, waiting, label=engine_type)
    average_waiting = sum(waiting) / len(waiting)
    ax.axhline(
        average_waiting,
        color=line.get_color(),
        linestyle="--",
        label=f"{engine_type} average: {average_waiting:.2f}",
    )

locator = mdates.AutoDateLocator()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.set(xlabel="Time", ylabel="Waiting requests", title="Waiting requests by engine", ylim=(0, None))
ax.legend(title="Engine")
ax.grid(True, alpha=0.3)
fig.tight_layout()

output_fpath = Path(args.output_fpath or f"{args.log_fpath}.waiting.png")
fig.savefig(output_fpath, dpi=150)
plt.close(fig)
print(f"Saved plot to {output_fpath}")
