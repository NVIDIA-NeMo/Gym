# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Plot the structured outputs v4 train-data distribution."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_INPUT = REPO_ROOT / "resources_servers/structured_outputs/data/structured_outputs_v4_tool_call.jsonl"
DEFAULT_OUTPUT = SCRIPT_DIR / "structured_outputs_v4_train_distribution.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", type=Path, default=DEFAULT_INPUT, help="Input v4 train JSONL")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT, help="Output PNG path")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_value(value: object) -> str:
    if value is None:
        return "None"
    return str(value)


def count_key(rows: Iterable[dict], key: str) -> Counter:
    return Counter(normalize_value(row.get(key)) for row in rows)


def annotate_barh(ax: plt.Axes, values: list[int], total: int) -> None:
    max_value = max(values) if values else 0
    for i, value in enumerate(values):
        pct = 100.0 * value / total if total else 0.0
        ax.text(max_value * 0.01 + value, i, f"{value} ({pct:.1f}%)", va="center", fontsize=8)


def plot_barh(ax: plt.Axes, counter: Counter, title: str, color: str, total: int) -> None:
    items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    labels = [label for label, _ in items]
    values = [value for _, value in items]
    ax.barh(labels, values, color=color)
    ax.invert_yaxis()
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("rows")
    annotate_barh(ax, values, total)
    if values:
        ax.set_xlim(0, max(values) * 1.22)
    ax.grid(axis="x", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)


def plot_num_distractors(ax: plt.Axes, counter: Counter, total: int) -> None:
    numeric_items = sorted((int(label), value) for label, value in counter.items())
    labels = [str(label) for label, _ in numeric_items]
    values = [value for _, value in numeric_items]
    ax.bar(labels, values, color="#E45756")
    ax.set_title("Number of distractors", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("num_distractors")
    ax.set_ylabel("rows")
    max_value = max(values) if values else 0
    for i, value in enumerate(values):
        pct = 100.0 * value / total if total else 0.0
        label = f"{value}" if pct < 5 else f"{value}\n({pct:.1f}%)"
        ax.text(i, value + max_value * 0.015, label, ha="center", va="bottom", fontsize=7, rotation=90)
    if values:
        ax.set_ylim(0, max(values) * 1.18)
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    total = len(rows)

    fig, axes = plt.subplots(5, 2, figsize=(18, 22), dpi=150)
    fig.suptitle(f"Structured Outputs v4 Train Distribution, {total:,} rows", fontsize=16, fontweight="bold")

    plot_barh(axes[0, 0], count_key(rows, "tool_schema_mode"), "Tool schema mode", "#54A24B", total)
    plot_barh(axes[0, 1], count_key(rows, "distractor_style"), "Distractor style", "#F58518", total)
    plot_barh(axes[1, 0], count_key(rows, "tool_union_mode"), "Tool union mode", "#B279A2", total)
    plot_barh(axes[1, 1], count_key(rows, "tool_name_style"), "Tool name style", "#4C78A8", total)
    plot_barh(axes[2, 0], count_key(rows, "num_tools"), "Number of tools", "#E45756", total)
    plot_barh(axes[2, 1], count_key(rows, "parallel_tool_calls"), "Parallel tool calls", "#72B7B2", total)
    plot_barh(axes[3, 0], count_key(rows, "instruction_layout"), "Instruction layout", "#72B7B2", total)
    plot_barh(axes[3, 1], count_key(rows, "instruction_detail_level"), "Instruction detail level", "#9D755D", total)
    plot_barh(axes[4, 0], count_key(rows, "system_instruction_style"), "System instruction style", "#BAB0AC", total)
    plot_num_distractors(axes[4, 1], count_key(rows, "num_distractors"), total)

    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.0, w_pad=2.8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(f"Wrote {args.output} from {total} rows")


if __name__ == "__main__":
    main()
