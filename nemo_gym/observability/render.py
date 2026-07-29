# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Static HTML and PNG timeline reports for NeMo Gym observability."""

from __future__ import annotations

import html
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from nemo_gym.observability.summary import load_jsonl


_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")

_CATEGORY_COLORS = {
    "foreground_compute": "#c8102e",
    "read_write_edit": "#8ee68e",
    "build": "#f29e2e",
    "sleep_poll": "#c7c7c7",
    "other_bash": "#ffcc00",
    "llm": "#77b7e5",
    "background": "#7f0000",
    "setup": "#9a6fd0",
    "cleanup": "#6a8fba",
}
_LEGEND_LABELS = {
    "background": "Background tool process running",
    "llm": "LLM inference (waiting on inference endpoint)",
    "foreground_compute": "Foreground compute (python/binary)",
    "build": "Build (gcc/make/cmake)",
    "read_write_edit": "Read / Write / Edit",
    "sleep_poll": "Foreground sleep/poll",
    "other_bash": "Other bash",
}
_SEMANTIC_TOOL_NAMES = {"trajectory.tool", "agent.tool"}


def safe_report_name(value: str) -> str:
    """Return a filesystem-safe report stem."""
    cleaned = _SAFE_FILENAME_RE.sub("_", value)[:80].strip("._-")
    return f"trajectory-{cleaned or 'unknown'}"


def _event_trajectory_id(event: dict[str, Any]) -> str | None:
    attrs = event.get("attributes") or {}
    value = attrs.get("trajectory_id") or attrs.get("trial_name") or attrs.get("environment_name")
    return str(value) if value else None


def _span_bars(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bars = []
    has_semantic_tools = any(
        event.get("event_type") == "span_end" and event.get("name") in _SEMANTIC_TOOL_NAMES for event in events
    )
    for event in events:
        if event.get("event_type") != "span_end":
            continue
        attrs = event.get("attributes") or {}
        duration_s = attrs.get("duration_s")
        timestamp = event.get("timestamp_unix_s")
        if not isinstance(duration_s, (int, float)) or not isinstance(timestamp, (int, float)):
            continue
        name = str(event.get("name") or "unknown")
        if (
            has_semantic_tools
            and name not in _SEMANTIC_TOOL_NAMES
            and name not in {"llm.request", "trajectory.background_process"}
        ):
            continue
        command_class = attrs.get("command_class")
        if name == "llm.request":
            lane = "LLM inference"
            category = "llm"
        elif name == "sandbox.resource_sample":
            lane = "Background tool process"
            category = "background"
        elif name in _SEMANTIC_TOOL_NAMES:
            lane = "Foreground tool / agent"
            category = str(command_class or "other_bash")
        else:
            lane = "Foreground tool / agent"
            category = str(command_class or attrs.get("phase") or "other_bash")
        bars.append(
            {
                "start": float(timestamp) - float(duration_s),
                "duration": float(duration_s),
                "lane": lane,
                "category": category,
                "name": name,
                "status": attrs.get("status") or "ok",
            }
        )
    return bars


def _render_timeline_png(events: list[dict[str, Any]], output_path: Path, title: str) -> bool:
    bars = _span_bars(events)
    if not bars:
        return False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    lanes = ["Foreground tool / agent", "LLM inference", "Background tool process"]
    y_by_lane = {lane: idx for idx, lane in enumerate(reversed(lanes))}
    t0 = min(bar["start"] for bar in bars)
    max_end = max(bar["start"] + bar["duration"] for bar in bars)
    fig, ax = plt.subplots(figsize=(14, 2.8))
    for bar in bars:
        y = y_by_lane[bar["lane"]]
        color = _CATEGORY_COLORS.get(bar["category"], _CATEGORY_COLORS["other_bash"])
        ax.broken_barh(
            [(bar["start"] - t0, max(bar["duration"], 0.01))],
            (y - 0.18, 0.36),
            facecolors=color,
            edgecolors=color,
            linewidth=0.4,
            alpha=0.95 if bar["status"] == "ok" else 0.55,
        )
    ax.set_yticks([y_by_lane[lane] for lane in lanes], labels=lanes)
    ax.set_xlim(0, max(max_end - t0, 1.0))
    ax.set_xlabel("Wall-clock time since trajectory start (s)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)
    legend_keys = [
        "background",
        "llm",
        "foreground_compute",
        "build",
        "read_write_edit",
        "sleep_poll",
        "other_bash",
    ]
    ax.legend(
        handles=[Patch(color=_CATEGORY_COLORS[key], label=_LEGEND_LABELS[key]) for key in legend_keys],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.32),
        ncols=4,
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    return True


def _render_aggregate_png(
    events: list[dict[str, Any]],
    resources: list[dict[str, Any]],
    output_path: Path,
) -> bool:
    span_durations = [
        float((event.get("attributes") or {}).get("duration_s"))
        for event in events
        if event.get("event_type") == "span_end"
        and isinstance((event.get("attributes") or {}).get("duration_s"), (int, float))
    ]
    memory_values = [
        int(sample["memory_usage_bytes"]) for sample in resources if isinstance(sample.get("memory_usage_bytes"), int)
    ]
    if not span_durations and not memory_values:
        return False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if span_durations:
        axes[0].hist(span_durations, bins=min(30, max(5, len(span_durations) // 2)), color="#77b7e5")
        axes[0].set_title("Operation Durations")
        axes[0].set_xlabel("seconds")
        axes[0].set_ylabel("count")
    else:
        axes[0].axis("off")
    if memory_values:
        axes[1].plot([value / (1024 * 1024) for value in memory_values], color="#c8102e")
        axes[1].set_title("Sandbox Memory Samples")
        axes[1].set_xlabel("sample")
        axes[1].set_ylabel("MiB")
    else:
        axes[1].axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    return True


def _write_html(path: Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html>",
                "<head>",
                '<meta charset="utf-8">',
                f"<title>{html.escape(title)}</title>",
                "<style>body{font-family:Arial,sans-serif;margin:24px;}"
                "table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:6px;}"
                "img{max-width:100%;height:auto}code{background:#f4f4f4;padding:2px 4px}</style>",
                "</head>",
                "<body>",
                body,
                "</body>",
                "</html>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def render_reports(
    output_dir: Path,
    *,
    max_rendered_trajectories: int,
    render_html: bool,
    render_png: bool,
) -> None:
    """Render static observability reports under ``reports/``."""
    reports_dir = output_dir / "reports"
    events = load_jsonl(output_dir / "events.jsonl")
    resources = load_jsonl(output_dir / "resource_samples.jsonl")
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    aggregate_png = reports_dir / "aggregate.png"
    aggregate_png_written = render_png and _render_aggregate_png(events, resources, aggregate_png)

    events_by_trajectory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        trajectory_id = _event_trajectory_id(event)
        if trajectory_id:
            events_by_trajectory[trajectory_id].append(event)

    trajectory_links = []
    for trajectory_id, trajectory_events in list(events_by_trajectory.items())[:max_rendered_trajectories]:
        stem = safe_report_name(trajectory_id)
        png_path = reports_dir / f"{stem}.png"
        png_written = render_png and _render_timeline_png(
            trajectory_events,
            png_path,
            title=trajectory_id,
        )
        html_path = reports_dir / f"{stem}.html"
        if render_html:
            image_html = f'<p><img src="{html.escape(png_path.name)}"></p>' if png_written else ""
            _write_html(
                html_path,
                trajectory_id,
                f"<h1>{html.escape(trajectory_id)}</h1>{image_html}<p>Events: {len(trajectory_events)}</p>",
            )
            trajectory_links.append(html_path.name)

    if render_html:
        aggregate_image = '<p><img src="aggregate.png"></p>' if aggregate_png_written else ""
        rows = "\n".join(
            f'<li><a href="{html.escape(link)}">{html.escape(link.removesuffix(".html"))}</a></li>'
            for link in trajectory_links
        )
        _write_html(
            reports_dir / "index.html",
            "NeMo Gym Observability",
            "<h1>NeMo Gym Observability</h1>"
            f"{aggregate_image}"
            "<h2>Summary</h2>"
            f"<pre>{html.escape(json.dumps(summary, indent=2, sort_keys=True))}</pre>"
            "<h2>Trajectories</h2>"
            f"<ul>{rows}</ul>",
        )
        _write_html(
            reports_dir / "aggregate.html",
            "NeMo Gym Aggregate",
            "<h1>NeMo Gym Aggregate</h1>"
            f"{aggregate_image}"
            f"<pre>{html.escape(json.dumps(summary, indent=2, sort_keys=True))}</pre>",
        )
