# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-contained HTML for one benchmark run, rendered from ``benchmark_summary.json`` only.

Inline CSS, inline SVG, one inline script for table sorting; no third-party code and no external URL, so the
page opens from a Lustre path in a browser.
"""

from __future__ import annotations

import html
import json
import math
from typing import Any, Optional


SECTIONS = (
    "Limitations",
    "Run",
    "Outcome at a glance",
    "Gym's native metrics",
    "Run timeline",
    "Where the time goes",
    "Distributions",
    "Repeats and budgets",
    "Tokens",
    "Tail",
    "Per-task table",
    "Data quality",
)
BUDGET_COLORS = {"passes lost": "#c0392b", "saved": "#4c78a8"}
STATUS_COLORS = {"ok": "#2a9d3f", "error": "#c0392b", "timeout": "#e67e22", "unknown": "#999999"}
BAND_COLORS = {"in_model": "#4c78a8", "in_tool": "#f58518", "elsewhere": "#bab0ac"}
SPLIT_COLORS = {"model": "#4c78a8", "tool": "#f58518", "other": "#bab0ac", "unexplained": "#e45756"}
ENGINE_COLORS = {"queue": "#e45756", "prefill": "#72b7b2", "decode": "#4c78a8", "gym_hop": "#bab0ac"}
STAT_HEADERS = ("metric", "n", "mean", "median", "p90", "p99", "max")
STYLE = (
    "body{font:13px/1.4 system-ui,sans-serif;margin:16px;color:#222;max-width:1200px}"
    "h2{margin-top:28px;border-bottom:1px solid #ddd}h3{margin:14px 0 6px}"
    ".cards{display:flex;flex-wrap:wrap;gap:8px}.card{border:1px solid #ddd;padding:8px 12px;min-width:120px}"
    ".card b{display:block;font-size:18px}table{border-collapse:collapse;margin:4px 0}"
    "th,td{padding:3px 8px;border-bottom:1px solid #eee;text-align:left;vertical-align:top}"
    "th{cursor:pointer;background:#f5f5f5}.trust{padding:8px;border-radius:4px}.trust.good{background:#e8f5e9}"
    ".trust.bad{background:#fdecea;color:#a00;font-weight:600}.note{color:#555}svg text{font-size:11px}"
    "details{margin-top:20px}summary{cursor:pointer;list-style:none;border-bottom:1px solid #ddd;padding:4px 0}"
    "summary::-webkit-details-marker{display:none}summary::before{content:'\\25B8';color:#888;margin-right:6px}"
    "details[open]>summary::before{content:'\\25BE'}summary h2{display:inline;margin:0;border:0}"
    "#glance td:first-child{font-weight:600;white-space:nowrap}"
    "#glance td:nth-child(2){text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}"
    "#glance td:nth-child(3){color:#555}"
    ".tip{position:absolute;display:none;background:#222;color:#fff;padding:4px 8px;border-radius:3px;font-size:12px;"
    "pointer-events:none;z-index:9;white-space:nowrap}.grid{display:flex;flex-wrap:wrap;gap:12px 24px}"
    "figure{margin:0}figcaption{font-weight:600;margin-bottom:2px}h4{margin:10px 0 4px}"
    ".scope{color:#555;font-size:12px;margin:2px 0 8px}.scope b{color:#333}svg text.pct{fill:#fff;font-size:10px}"
)
SCOPES = {
    "recorded": ("All recorded rollouts", "every rollout the collector wrote, timeouts and errors included"),
    "agent": (
        "Rollouts with agent-side data",
        "recorded rollouts whose agent events were captured; a timeout counts with the time it was active, an error "
        "with what it did before failing; rollouts without agent-side spans are left out",
    ),
    "completed": ("Completed rollouts", "rollouts whose agent finished on its own: no timeout, no error"),
    "outcome": (
        "Resolved vs failed",
        "rollouts with agent-side data split by reward; a rollout without a reward is in neither group",
    ),
    "calls": ("Per call", "every model call or tool call of the rollouts with agent-side data, pooled"),
    "solved": (
        "Always-solved tasks",
        "tasks with at least two repeats, every one resolved; repeats are compared within the task, so the "
        "variation is the policy's, not the task's",
    ),
    "steps": ("Per step", "every main-agent step of the rollouts with agent-side data"),
}
BAND_LABELS = {
    "in_model": "in a model call",
    "in_tool": "in a tool call",
    "elsewhere": "elsewhere (harness work, waiting)",
}
PURPOSE_PALETTE = ("#4c78a8", "#f58518", "#e45756", "#72b7b2", "#54a24b", "#b279a2")
TIP_SCRIPT = (
    "(function(){var g=document.getElementById('gantt'),tip=document.getElementById('tip');if(!g||!tip)return;"
    "var rows=JSON.parse(g.dataset.rows),rh=parseFloat(g.dataset.rowH),top=parseFloat(g.dataset.top);"
    "g.addEventListener('mousemove',function(e){var b=g.getBoundingClientRect(),i=Math.floor((e.clientY-b.top-top)/rh);"
    "if(i<0||i>=rows.length){tip.style.display='none';return;}var d=rows[i];"
    "tip.textContent=d[0]+'/'+d[1]+'  '+d[4]+'  '+Math.round(d[2])+'\u2013'+Math.round(d[3])+' s';"
    "tip.style.display='block';tip.style.left=(e.pageX+12)+'px';tip.style.top=(e.pageY+12)+'px';});"
    "g.addEventListener('mouseleave',function(){tip.style.display='none';});})();"
)
SORT_SCRIPT = (
    "function sortBy(id,i){const t=document.getElementById(id);const r=Array.from(t.tBodies[0].rows);"
    "const asc=t.dataset.col!==String(i)||t.dataset.dir!=='asc';t.dataset.col=i;t.dataset.dir=asc?'asc':'desc';"
    "r.sort(function(a,b){const x=a.cells[i].textContent,y=b.cells[i].textContent;"
    "const nx=parseFloat(x.replace(/,/g,'')),ny=parseFloat(y.replace(/,/g,''));"
    "const c=(!isNaN(nx)&&!isNaN(ny))?nx-ny:(isNaN(nx)&&isNaN(ny))?x.localeCompare(y):(isNaN(nx)?-1:1);"
    "return asc?c:-c;});"
    "r.forEach(function(row){t.tBodies[0].appendChild(row);});}"
)


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def num(value: Any, digits: int = 1) -> str:
    if value is None or isinstance(value, bool):
        return "–"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    return esc(value)


def pct(value: Any) -> str:
    return "–" if not isinstance(value, (int, float)) or isinstance(value, bool) else f"{100 * value:.1f} %"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _section(title: str, body: str) -> str:
    """One foldable section, closed by default: the title is the fold handle and keeps its anchor."""
    anchor = title.lower().replace("'", "").replace(" ", "-")
    return f'<details><summary><h2 id="{anchor}">{esc(title)}</h2></summary>\n{body}\n</details>\n'


def _scope_counts(summary: dict[str, Any]) -> dict[str, str]:
    outcomes, where = summary["outcomes"], summary["where_time_goes"]
    calls = _entry(where.get("model_call_time"), "pooled", "count")
    tools = _entry(where.get("tool_execution_time"), "pooled", "count")
    return {
        "recorded": num(outcomes["rollouts"]),
        "agent": num(outcomes.get("with_agent_data")),
        "completed": num(outcomes.get("successful")),
        "outcome": f"{num(outcomes['success'])} / {num(outcomes['failure'])}",
        "calls": f"{num(calls)} model calls, {num(tools)} tool calls",
        "solved": num(((summary.get("repeats") or {}).get("repeatability") or {}).get("n_tasks", 0)),
        "steps": num((where.get("reasoning_clusters") or {}).get("steps", 0)),
    }


def _scope(summary: dict[str, Any], key: str) -> str:
    """The one-line data label at the top of a metric group: which rollouts its numbers cover."""
    label, description = SCOPES[key]
    return f'<p class="scope">Data: <b>{esc(label)} ({esc(_scope_counts(summary)[key])})</b>: {esc(description)}.</p>'


def _scope_legend(summary: dict[str, Any]) -> str:
    counts = _scope_counts(summary)
    rows = [
        [f"<b>{esc(label)}</b>", esc(counts[key]), esc(description)] for key, (label, description) in SCOPES.items()
    ]
    return "<h3>Data labels used on this page</h3>" + table(
        ("label", "count", "meaning"), rows, table_id="scopes", sortable=False
    )


def table(headers: tuple[str, ...], rows: list[list[str]], *, table_id: str, sortable: bool = True) -> str:
    """Cells are already-rendered HTML strings (escaped text or a link)."""
    head = "".join(
        f"<th onclick=\"sortBy('{table_id}',{i})\">{esc(h)}</th>" if sortable else f"<th>{esc(h)}</th>"
        for i, h in enumerate(headers)
    )
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f'<table id="{esc(table_id)}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def stat_row(name: str, st: Optional[dict[str, Any]], *, title: str = "") -> list[str]:
    st = st or {}
    return [
        f'<span title="{esc(title)}">{esc(name)}</span>',
        num(st.get("count")),
        num(st.get("mean"), 2),
        num(st.get("median"), 2),
        num(st.get("p90"), 2),
        num(st.get("p99"), 2),
        num(st.get("max"), 2),
    ]


def bar_chart(
    items: list[tuple[str, Any, str]],
    *,
    width: int = 640,
    bar_h: int = 16,
    label_w: int = 220,
    marker: Optional[float] = None,
    suffix: str = "",
) -> str:
    """Horizontal bars: ``(label, value, colour)`` per row; ``marker`` draws a dashed reference line at that value."""
    drawn = [(label, float(value), color) for label, value, color in items if _is_number(value)]
    if not drawn:
        return '<p class="note">no data</p>'
    top = max([v for _, v, _ in drawn] + ([marker] if _is_number(marker) else [])) or 1.0
    height = len(drawn) * (bar_h + 4)
    parts = [f'<svg width="{width}" height="{height}">']
    for i, (label, value, color) in enumerate(drawn):
        y = i * (bar_h + 4)
        w = max(1.0, (width - label_w - 80) * value / top)
        parts.append(
            f'<text x="0" y="{y + bar_h - 4}">{esc(label)}</text>'
            f'<rect x="{label_w}" y="{y}" width="{w:.1f}" height="{bar_h}" fill="{color}">'
            f"<title>{esc(label)}: {num(value, 2)}{esc(suffix)}</title></rect>"
            f'<text x="{label_w + w + 4:.1f}" y="{y + bar_h - 4}">{num(value, 1)}{esc(suffix)}</text>'
        )
    if _is_number(marker):
        x = label_w + (width - label_w - 80) * marker / top
        parts.append(
            f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{height}" stroke="#222" stroke-dasharray="3,3">'
            f"<title>baseline {num(marker, 1)}{esc(suffix)}</title></line>"
        )
    parts.append("</svg>")
    return "".join(parts)


def _short(seconds: Any) -> str:
    if not _is_number(seconds):
        return "–"
    if seconds < 60:
        return f"{seconds:.0f} s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def dot_strip(tasks: list[dict[str, Any]], *, svg_id: str, width: int = 1180) -> str:
    """One row per task (in the given order), a dot per repeat on a log time axis and a tick at the task median;
    hover a dot for the task, the repeat's time, the median and the spread."""
    times = [v for t in tasks for v in t.get("agent_times") or [] if _is_number(v) and v > 0]
    if not times:
        return f'<svg id="{esc(svg_id)}" width="{width}" height="40"><text x="2" y="18">no data</text></svg>'
    lo, hi = math.log10(min(times)), math.log10(max(times))
    span = (hi - lo) or 1.0
    left, right, top_pad, row_h = 10, 20, 18, max(3, min(8, 900 // max(1, len(tasks))))
    height = top_pad + len(tasks) * row_h + 24

    def px(value: float) -> float:
        return left + (math.log10(value) - lo) / span * (width - left - right)

    out = [f'<svg id="{esc(svg_id)}" width="{width}" height="{height}">']
    for tick in (10, 30, 60, 180, 600, 1800, 3600, 7200):
        if min(times) <= tick <= max(times):
            out.append(
                f'<line x1="{px(tick):.1f}" y1="{top_pad}" x2="{px(tick):.1f}" y2="{height - 24}" stroke="#eee"/>'
                f'<text x="{px(tick) - 10:.1f}" y="{height - 8}">{esc(_short(tick))}</text>'
            )
    for i, task in enumerate(tasks):
        y = top_pad + i * row_h + row_h / 2
        median, spread = task.get("median"), task.get("spread")
        if _is_number(median) and median > 0:
            out.append(
                f'<line x1="{px(median):.1f}" y1="{y - row_h / 2:.1f}" x2="{px(median):.1f}" y2="{y + row_h / 2:.1f}" stroke="#222"/>'
            )
        for value in task.get("agent_times") or []:
            if _is_number(value) and value > 0:
                title = (
                    f"{task.get('task_id')}: {_short(value)} (median {_short(median)}, "
                    f"spread ×{num(spread, 1) if _is_number(spread) else '–'})"
                )
                out.append(
                    f'<circle cx="{px(value):.1f}" cy="{y:.1f}" r="2.2" fill="#4c78a8" fill-opacity="0.7">'
                    f"<title>{esc(title)}</title></circle>"
                )
    out.append(f'<text x="2" y="12">{len(tasks)} tasks, sorted by median agent time (log axis)</text></svg>')
    return "".join(out)


def stacked_bar(
    rows: list[tuple[str, dict[str, Any]]],
    colors: dict[str, str],
    *,
    width: int = 640,
    bar_h: int = 18,
    label_w: int = 120,
) -> str:
    """One bar per row, segments proportional to their share of the row's sum; hover shows value and share."""
    drawn = [(label, {k: float(v) for k, v in parts.items() if _is_number(v) and v > 0}) for label, parts in rows]
    drawn = [(label, parts) for label, parts in drawn if parts]
    if not drawn:
        return '<p class="note">no data</p>'
    out = [f'<svg width="{width}" height="{len(drawn) * (bar_h + 6) + 18}">']
    for i, (label, parts) in enumerate(drawn):
        total = sum(parts.values())
        x = float(label_w)
        y = i * (bar_h + 6)
        out.append(f'<text x="0" y="{y + bar_h - 4}">{esc(label)}</text>')
        for key, value in parts.items():
            w = (width - label_w - 10) * value / total
            color = colors.get(key, "#999999")
            out.append(
                f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{bar_h}" fill="{color}">'
                f"<title>{esc(key)}: {num(value, 2)} ({pct(value / total)})</title></rect>"
            )
            if w >= 44:  # room for "12.3 %"; narrower segments keep their share in the hover
                out.append(
                    f'<text class="pct" x="{x + w / 2:.1f}" y="{y + bar_h - 5}" text-anchor="middle" fill="#fff">'
                    f"{pct(value / total)}</text>"
                )
            x += w
    legend_y = len(drawn) * (bar_h + 6) + 12
    x = float(label_w)
    for key, color in colors.items():
        out.append(f'<rect x="{x:.0f}" y="{legend_y - 9}" width="10" height="10" fill="{color}"/>')
        out.append(f'<text x="{x + 14:.0f}" y="{legend_y}">{esc(key)}</text>')
        x += 90
    out.append("</svg>")
    return "".join(out)


def histogram(values: list[Any], marks: dict[str, Any], *, bins: int = 24, width: int = 640, height: int = 140) -> str:
    drawn = [float(v) for v in values if _is_number(v)]
    if not drawn:
        return '<p class="note">no data</p>'
    lo, hi = min(drawn), max(drawn)
    span = (hi - lo) or 1.0
    counts = [0] * bins
    for v in drawn:
        counts[min(bins - 1, int((v - lo) / span * bins))] += 1
    top = max(counts)
    bw = (width - 40) / bins
    base = height - 20
    out = [f'<svg width="{width}" height="{height}">']
    for i, c in enumerate(counts):
        h = (base - 10) * c / top if top else 0.0
        out.append(
            f'<rect x="{20 + i * bw:.1f}" y="{base - h:.1f}" width="{bw - 1:.1f}" height="{h:.1f}" fill="#4c78a8">'
            f"<title>{num(lo + i * span / bins, 1)}–{num(lo + (i + 1) * span / bins, 1)}: {c}</title></rect>"
        )
    for name, value in marks.items():
        if _is_number(value):
            x = 20 + (float(value) - lo) / span * (width - 40)
            out.append(
                f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{base}" stroke="#e45756" stroke-dasharray="3,2"/>'
                f'<text x="{x + 2:.1f}" y="10">{esc(name)}</text>'
            )
    out.append(
        f'<text x="20" y="{height - 4}">{num(lo, 1)}</text>'
        f'<text x="{width - 60}" y="{height - 4}">{num(hi, 1)}</text></svg>'
    )
    return "".join(out)


def _tick_step(horizon_s: float) -> float:
    for step in (60.0, 300.0, 600.0, 1800.0, 3600.0, 7200.0, 14400.0):
        if horizon_s / step <= 12:
            return step
    return 28800.0


def gantt(
    rollouts: list[list[Any]],
    links: dict[str, str],
    *,
    width: int = 960,
    row_h: Optional[int] = None,
    breaks: list[dict[str, Any]] | tuple = (),
    svg_id: str = "gantt",
) -> str:
    """One thin bar per rollout on the run's active-time axis; ``links`` maps ``"<replica>/<rollout_id>"`` to a
    page; ``breaks`` are the collapsed gaps between collector sessions, drawn as dashed lines. ``row_h`` adapts to
    the rollout count (2–6 px) when not given. The svg carries its rows as data so TIP_SCRIPT can show the rollout
    under the pointer anywhere on its row (the bars themselves are too thin to hover reliably)."""
    if not rollouts:
        return '<p class="note">no data</p>'
    if row_h is None:
        row_h = max(2, min(6, 1200 // len(rollouts)))
    horizon = max(float(r[3]) for r in rollouts) or 1.0
    scale = (width - 20) / horizon
    height = len(rollouts) * row_h + 24
    data = json.dumps(
        [[str(r), str(i), round(float(a), 1), round(float(b), 1), str(st)] for r, i, a, b, st in rollouts]
    )
    out = [
        f'<svg id="{esc(svg_id)}" width="{width}" height="{height}" data-top="20" data-row-h="{row_h}" '
        f'data-rows="{esc(data)}">'
    ]
    for i, (replica, rid, start, end, status) in enumerate(rollouts):
        x = 10 + float(start) * scale
        w = max(1.0, (float(end) - float(start)) * scale)
        y = 20 + i * row_h
        color = STATUS_COLORS.get(str(status), STATUS_COLORS["unknown"])
        rect = (
            f'<rect class="ro" x="{x:.1f}" y="{y}" width="{w:.1f}" height="{max(1, row_h - 1)}" fill="{color}">'
            f"<title>{esc(replica)}/{esc(rid)} {esc(status)} {num(float(start), 0)}–{num(float(end), 0)} s</title>"
            "</rect>"
        )
        page = links.get(f"{replica}/{rid}")
        out.append(f'<a href="{esc(page)}">{rect}</a>' if page else rect)
    step = _tick_step(horizon)
    t = 0.0
    while t <= horizon:
        x = 10 + t * scale
        out.append(
            f'<line x1="{x:.1f}" y1="12" x2="{x:.1f}" y2="{height}" stroke="#eeeeee"/>'
            f'<text x="{x + 2:.1f}" y="10">{int(t // 60)} min</text>'
        )
        t += step
    out.append(_break_lines(list(breaks), scale, height))
    out.append("</svg>")
    return "".join(out)


def stacked_area(
    bins: list[dict[str, Any]],
    bin_s: float,
    *,
    width: int = 960,
    height: int = 160,
    breaks: list[dict[str, Any]] | tuple = (),
) -> str:
    """Rollouts active over time, stacked by what they were doing (``in_model``, ``in_tool``, ``elsewhere``)."""
    if not bins:
        return '<p class="note">no data</p>'
    top = max(b["active"] for b in bins) or 1
    # Bins carry their own t (collapsed seconds); after a session break they are not i * bin_s, so the axis
    # is laid out from t, and every bin runs to the next bin's t (the last one to t + bin_s).
    starts = [float(b["t"]) for b in bins]
    horizon = starts[-1] + bin_s
    ends = starts[1:] + [horizon]
    scale_x = (width - 20) / horizon
    base = height - 16

    def y(v: float) -> float:
        return base - (base - 10) * v / top

    def x(t: float) -> float:
        return 10 + t * scale_x

    out = [f'<svg width="{width}" height="{height}">']
    lower = [0.0] * len(bins)
    for key in ("in_model", "in_tool", "elsewhere"):
        upper = [lower[i] + b[key] for i, b in enumerate(bins)]
        forward = " ".join(
            f"{x(starts[i]):.1f},{y(upper[i]):.1f} {x(ends[i]):.1f},{y(upper[i]):.1f}" for i in range(len(bins))
        )
        back = " ".join(
            f"{x(ends[i]):.1f},{y(lower[i]):.1f} {x(starts[i]):.1f},{y(lower[i]):.1f}"
            for i in reversed(range(len(bins)))
        )
        out.append(f'<polygon points="{forward} {back}" fill="{BAND_COLORS[key]}"><title>{esc(key)}</title></polygon>')
        lower = upper
    out.append(_break_lines(list(breaks), scale_x, height))
    out.append(f'<text x="10" y="10">{top} rollouts active at peak</text>')
    lx = width - 470
    for key, label in BAND_LABELS.items():
        out.append(
            f'<rect x="{lx}" y="2" width="10" height="10" fill="{BAND_COLORS[key]}"/>'
            f'<text x="{lx + 14}" y="10">{esc(label)}</text>'
        )
        lx += 120 if key == "in_tool" else 150
    out.append("</svg>")
    return "".join(out)


OUTCOME_COLORS = {"success": "#2a9d3f", "failure": "#c0392b"}


def line_chart(
    xs: list[Any],
    series: dict[str, list[Any]],
    colors: dict[str, str],
    *,
    svg_id: str,
    width: int = 640,
    height: int = 170,
    rollouts: Optional[dict[str, list[Any]]] = None,
) -> str:
    """One polyline per series over shared x positions; None values break the line. Hover a point for its value
    and, when ``rollouts`` gives per-x counts, how many rollouts stand behind it. Tick labels are thinned so they
    stay at least 48 px apart."""
    points = [(k, [(x, v) for x, v in zip(xs, vals) if _is_number(v)]) for k, vals in series.items()]
    points = [(k, pts) for k, pts in points if pts]
    if not points or len(xs) == 0:
        return f'<svg id="{esc(svg_id)}" width="{width}" height="40"><text x="2" y="18">no data</text></svg>'
    behind = {k: dict(zip(xs, counts or [])) for k, counts in (rollouts or {}).items()}
    last_x = max(x for _, pts in points for x, _ in pts)
    xs = [x for x in xs if x <= last_x]  # the axis ends at the last drawn point, not at the last hidden bucket
    top = max(v for _, pts in points for _, v in pts) or 1.0
    span_x = (max(xs) - min(xs)) or 1.0
    left, base = 50, height - 24

    def px(x: Any) -> float:
        return left + (float(x) - min(xs)) / span_x * (width - left - 10)

    def py(v: float) -> float:
        return base - (base - 14) * v / top

    out = [f'<svg id="{esc(svg_id)}" width="{width}" height="{height}">']
    out.append(f'<line x1="{left}" y1="{base}" x2="{width - 10}" y2="{base}" stroke="#ccc"/>')
    out.append(f'<text x="2" y="18">{num(top, 0)}</text><text x="2" y="{base}">0</text>')
    for key, pts in points:
        color = colors.get(key, "#999999")
        out.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2" points="'
            + " ".join(f"{px(x):.1f},{py(v):.1f}" for x, v in pts)
            + '"/>'
        )
        for x, v in pts:
            n = behind.get(key, {}).get(x)
            tail = f" ({num(n)} rollouts)" if _is_number(n) else ""
            out.append(
                f'<circle cx="{px(x):.1f}" cy="{py(v):.1f}" r="3" fill="{color}">'
                f"<title>{esc(key)} at {esc(x)}: {num(v, 1)}{tail}</title></circle>"
            )
    per_x = (width - left - 10) / max(1, len(xs) - 1)
    every = max(1, math.ceil(48 / per_x))
    for i, x in enumerate(xs):
        if i % every == 0:
            out.append(f'<text x="{px(x) - 6:.1f}" y="{height - 8}">{esc(x)}</text>')
    lx = left
    for key, _ in points:
        out.append(
            f'<rect x="{lx}" y="2" width="10" height="10" fill="{colors.get(key, "#999999")}"/><text x="{lx + 14}" y="11">{esc(key)}</text>'
        )
        lx += 90
    out.append("</svg>")
    return "".join(out)


def _cards(items: list[tuple[str, str]]) -> str:
    cards = "".join(f'<div class="card"><b>{value}</b>{esc(label)}</div>' for label, value in items)
    return f'<div class="cards">{cards}</div>'


def _duration_text(seconds: Any) -> str:
    if not _is_number(seconds):
        return "–"
    if seconds >= 3600:
        return f"{seconds / 3600:.1f} h"
    if seconds >= 60:
        return f"{seconds / 60:.0f} min"
    return f"{seconds:.0f} s"


def _break_lines(breaks: list[dict[str, Any]], scale: float, height: int, *, x0: float = 10.0) -> str:
    out = []
    for b in breaks:
        x = x0 + float(b["at_s"]) * scale
        label = f"resumed after {_duration_text(b.get('downtime_s'))}, job {b.get('resumed_by') or 'unknown'}"
        out.append(
            f'<line class="break" x1="{x:.1f}" y1="12" x2="{x:.1f}" y2="{height}" stroke="#444444" stroke-dasharray="4 3"/>'
            f'<text x="{x + 3:.1f}" y="{height - 4}" fill="#444444">{esc(label)}</text>'
        )
    return "".join(out)


def _trust_line(summary: dict[str, Any]) -> str:
    validation = summary["validation"]
    replicas = validation["replicas"]
    passed = sum(1 for r in replicas if r["check_passed"] is True)
    coverages = [r["server_coverage"] for r in replicas if _is_number(r["server_coverage"])]
    excluded = validation["excluded_trials"]
    without_agent = excluded.get("no_timeline", 0) + excluded.get("agent_artifact_unavailable", 0)
    text = (
        f"checks passed on {passed} of {len(replicas)} replicas; model calls with a server record "
        f"{pct(min(coverages)) if coverages else '–'}; {without_agent} rollout{'' if without_agent == 1 else 's'} "
        "without agent-side spans excluded from agent metrics"
    )
    if validation["trusted"]:
        return f'<p class="trust good">{esc(text + ".")}</p>'

    def failed(r: dict[str, Any]) -> str:
        details = r.get("failed_details") or {}
        return ", ".join(
            f"{name} ({'; '.join(details[name])})" if details.get(name) else name for name in r["failed_checks"]
        )

    reasons = [f"{r['replica']}: failed {failed(r)}" for r in replicas if r["failed_checks"]]
    reasons += [f"{r['replica']}: no latency data" for r in replicas if not r["latency_data"]]
    reasons += [f"{r['replica']}: no check file" for r in replicas if r["latency_data"] and r["check_passed"] is None]
    notes = summary["data_quality"]["notes"]
    reasons += [n for n in notes if n.startswith("old layout") or n.startswith("no run_info")]
    listed = ("; ".join(reasons) + ".") if reasons else ""
    return f'<p class="trust bad">{esc("DO NOT QUOTE: " + text + ". " + listed)}</p>'


def _run_section(summary: dict[str, Any]) -> str:
    run = summary["run"]
    keys = (
        "benchmark",
        "instance",
        "agent",
        "model",
        "model_name",
        "vllm_config",
        "container",
        "concurrency",
        "num_replicas",
        "sandbox_timeout_s",
        "walltime",
        "max_num_seqs",
        "experiment_name",
        "slurm_job_id",
        "code_commit",
        "layout",
    )
    rows = [[esc(k), esc(run.get(k))] for k in keys]
    rows += [
        [
            esc(f"replica {r['name']}"),
            esc(f"node {r['node']}, started {r['started_at']}, shard {r.get('shard_config', 'unknown')}"),
        ]
        for r in run["replicas"]
    ]
    return (
        _trust_line(summary) + _scope_legend(summary) + table(("fact", "value"), rows, table_id="run", sortable=False)
    )


def _seconds(st: dict[str, Any], key: str) -> str:
    value = st.get(key)
    return f"{num(value, 1)} s" if _is_number(value) else "–"


def _glance_rows(summary: dict[str, Any]) -> list[tuple[str, str, str]]:
    """The outcome table as (metric, rendered value, note); the note says what the number is and is not."""
    outcomes, timing, tokens, run = summary["outcomes"], summary["time"], summary["tokens"], summary["run"]
    agent = (timing.get("e2e_agent_time") or {}).get("all") or {}
    completed_only = timing.get("e2e_agent_time_completed") or {}
    agent_def = (summary.get("definitions") or {}).get("e2e_agent_time") or "agent span duration"
    planned, recorded, completed = outcomes.get("planned"), outcomes["rollouts"], outcomes.get("successful")

    def with_share(count: Any) -> str:
        share = f" ({pct(count / planned)})" if _is_number(planned) and planned and _is_number(count) else ""
        return num(count) + share

    run_time = timing.get("benchmark_run_time_s")
    run_time_note = (
        "first rollout start to last rollout end across replicas, on the collectors' clocks; model-server startup "
        "before the first rollout and the post-processing after the last are not included"
    )
    timeouts = outcomes["agent_status"].get("timeout", 0)
    limit = run.get("sandbox_timeout_s")
    limit_text = f"{num(limit)} s" if _is_number(limit) else "SANDBOX_TIMEOUT"
    totals = tokens["totals"]
    completed_note = (
        "over completed rollouts only: agents that finished on their own, no timeout and no error "
        f"({num(completed)} of {num(recorded)})"
    )
    return [
        (
            "Total rollouts",
            num(planned),
            "rollouts the run was asked to produce: every task of each replica's shard dataset times its repeats "
            "(rows of rollouts_materialized_inputs.jsonl)",
        ),
        (
            "Dispatched rollouts",
            num(outcomes.get("dispatched")),
            "rollouts the collector sent to the agent: recorded rows plus rows of the failures sidecar",
        ),
        (
            "Recorded rollouts",
            with_share(recorded),
            "rollouts the collector recorded, with the share of the total in parentheses; rollouts in flight when "
            "a collector stopped are never recorded",
        ),
        (
            "Scoring denominator",
            num(outcomes.get("scoring_denominator")),
            "recorded rollouts that carry a reward: Mean reward and every aggregate on this page are over these, "
            "never over dispatched rollouts",
        ),
        (
            "Successful rollouts",
            with_share(completed),
            "rollouts whose agent finished on its own (agent status ok): no timeout, no error, with the share of "
            "the total in parentheses; says nothing about the reward, which Mean reward covers",
        ),
        (
            "Evaluated rollouts",
            num(outcomes.get("evaluated")),
            "rollouts for which evaluation completed successfully (evaluation_completed in the rollout row)",
        ),
        (
            "Mean reward",
            num(outcomes["reward_mean"], 3),
            "mean verifier reward over recorded rollouts that have one; SWE-bench and Terminal-Bench rewards are "
            "0 or 1, so this is also the success rate",
        ),
        (
            "Benchmark run time",
            f"{_duration_text(run_time)} ({num(run_time, 1)} s)" if _is_number(run_time) else "–",
            run_time_note,
        ),
        ("Median agent time", _seconds(agent, "median"), f"median over recorded rollouts of the {agent_def}"),
        ("Agent time p90", _seconds(agent, "p90"), f"90th percentile over recorded rollouts of the {agent_def}"),
        ("Median agent time (completed only)", _seconds(completed_only, "median"), completed_note),
        ("Agent time p90 (completed only)", _seconds(completed_only, "p90"), completed_note),
        (
            "Total tokens",
            num(tokens["total"]),
            "input + output + reasoning over every captured model call, including subagent and compaction calls; "
            "cached tokens are a subset of input and are not added",
        ),
        (
            "Input tokens",
            num(totals.get("input")),
            "prompt tokens as vLLM saw them on every call, cache reads included",
        ),
        (
            "Output tokens",
            num(totals.get("output")),
            f"visible output tokens; reasoning tokens ({num(totals.get('reasoning'))}) are counted separately in "
            "the total",
        ),
        (
            "Cached tokens",
            num(totals.get("cached")),
            "prompt tokens served from vLLM's prefix cache; a subset of input tokens",
        ),
        (
            "Cache hit share",
            pct(totals["cached"] / totals["input"]) if totals.get("input") else "–",
            "cached ÷ input tokens over the whole run; the Tokens section shows the per-rollout mean instead",
        ),
        ("Timeouts", num(timeouts), f"rollouts whose agent was stopped at the sandbox time limit ({limit_text})"),
        (
            "Errors",
            num(outcomes["agent_status"].get("error", 0)),
            "rollouts whose agent run ended in an error instead of finishing; the breakdown by type is charted "
            "below when there are any",
        ),
        (
            "Collector failures",
            num(outcomes["collector_failures"])
            + (
                " ("
                + ", ".join(f"{k}: {num(v)}" for k, v in (outcomes.get("collector_failures_by_class") or {}).items())
                + ")"
                if outcomes.get("collector_failures_by_class")
                else ""
            ),
            "rollouts whose /run call failed at the collector (by failure class) and were written to the failures "
            "sidecar instead of stopping the run; they have no trajectory and no timing"
            + (f"; sidecar: {outcomes['failures_path']}" if outcomes.get("failures_path") else "; no sidecar file"),
        ),
    ]


def _outcome_section(summary: dict[str, Any]) -> str:
    outcomes = summary["outcomes"]
    rows = [[esc(metric), value, esc(note)] for metric, value, note in _glance_rows(summary)]
    body = _scope(summary, "recorded") + table(("metric", "value", "notes"), rows, table_id="glance", sortable=False)
    body += "<h3>Agent status</h3>" + bar_chart(
        [(s, c, STATUS_COLORS.get(s, STATUS_COLORS["unknown"])) for s, c in outcomes["agent_status"].items()]
    )
    timeouts = outcomes.get("timeouts") or []
    if timeouts:
        silent = sum(1 for t in timeouts if t.get("hang"))
        capped = sum(1 for t in timeouts if t.get("verdict") == "model call ran to the length cap")
        limit = summary["run"].get("sandbox_timeout_s")
        limit_text = f"{num(limit)} s" if _is_number(limit) else "the sandbox time limit"

        def last_call(t: dict[str, Any]) -> str:
            tokens, seconds = t.get("last_call_tokens"), t.get("last_call_s")
            if not (_is_number(tokens) or _is_number(seconds)):
                return "–"
            return f"{num(tokens)} tok, {num(seconds, 0)} s"

        rows = [
            [
                esc(t["replica"]),
                esc(t["rollout_id"]),
                esc(t["task_id"]),
                num(t["agent_time"], 0),
                num(t["stall_s"], 0),
                num(t["model_calls"]),
                esc(t["last_action"]),
                last_call(t),
                esc(t.get("verdict") or ("hang" if t.get("hang") else "ran to the cap")),
            ]
            for t in timeouts
        ]
        body += "<h3>Timeouts</h3>" + (
            f'<p class="note">{silent} of {len(timeouts)} timeouts were silent for at least five minutes before the '
            f"kill; {capped} of them were waiting on a model call that generated up to its length cap. A timed-out "
            f"agent is killed by the sandbox: the agent runs as one command under the time limit ({limit_text}), "
            "the sandbox ends the command when it reaches the limit, and the harness records the timeout, exports "
            "what the agent persisted and evaluates it anyway. Silent means the agent persisted no model call, tool "
            "call or other event in that time. <b>model call ran to the length cap</b>: the last request generated "
            "until its token cap (its tokens and vLLM seconds are in the last call column); the harness never "
            "recorded a response, did not retry, and idled until the kill. <b>waiting on a model call</b>: a request "
            "was in flight that never completed. <b>waiting on a tool</b>: a tool call never returned. <b>ran to the "
            "cap</b>: the agent was still working when the limit hit. <b>silent, cause unknown</b>: no open call "
            "explains the silence.</p>"
        )
        body += table(
            ("replica", "rollout", "task", "agent s", "silent s", "calls", "last action", "last call", "verdict"),
            rows,
            table_id="timeouts",
        )
    if outcomes["error_types"]:
        errors = [(e, c, STATUS_COLORS["error"]) for e, c in outcomes["error_types"].items()]
        body += "<h3>Errors by type</h3>" + bar_chart(errors)
        body += (
            '<p class="note"><b>failed</b>: the agent process exited abnormally before the time limit; the process\'s '
            "last recorded words follow when the harness kept them (<i>signal: killed</i> is the process being "
            "killed inside a sandbox that stayed alive, typically its memory limit). <b>sandbox ended</b>: the "
            "sandbox itself died under the agent, with the reason the sandbox service gave. Other labels are the "
            "exception the harness raised.</p>"
        )
    if outcomes["by_group"]:
        rows = [[esc(g), num(e["count"]), num(e["reward_mean"], 3)] for g, e in outcomes["by_group"].items()]
        body += '<h3 id="groups">Reward by task group</h3>'
        body += table(("group", "rollouts", "mean reward"), rows, table_id="groups-table")
    return body


def _value_text(value: Any) -> str:
    return esc(value if not isinstance(value, (dict, list)) else json.dumps(value, sort_keys=True))


def _aggregate_tables(agg: Any) -> list[tuple[str, list[list[str]]]]:
    """Gym's ``rollouts_aggregate_metrics.json`` is a list with one entry per agent (nested dicts such as
    ``agent_metrics`` and ``perf_summary``); older files are a flat dict. Every value is shown, nested dicts one
    level down as ``section/key`` rows and deeper values as JSON."""
    if isinstance(agg, dict):
        return [("", [[esc(k), _value_text(v)] for k, v in sorted(agg.items())])]
    tables: list[tuple[str, list[list[str]]]] = []
    for index, entry in enumerate(agg if isinstance(agg, list) else [agg]):
        if not isinstance(entry, dict):
            tables.append((f"entry {index}", [[esc(index), _value_text(entry)]]))
            continue
        ref = entry.get("agent_ref")
        title = (ref.get("name") if isinstance(ref, dict) else None) or f"entry {index}"
        rows: list[list[str]] = []
        for key, value in sorted(entry.items()):
            if key == "agent_ref":
                continue
            if isinstance(value, dict):
                rows.extend([esc(f"{key}/{sub}"), _value_text(sub_value)] for sub, sub_value in sorted(value.items()))
            else:
                rows.append([esc(key), _value_text(value)])
        tables.append((str(title), rows))
    return tables


NATIVE_INTRO = (
    "These are Gym's own evaluation metrics, as Gym reports them. When a collection finishes, Gym sends every "
    "recorded rollout, minus the trajectories, to the agent's aggregate endpoint, which computes the mean, max, min, "
    "median and standard deviation of each metric and confidence intervals across repeats, and writes them to "
    "rollouts_aggregate_metrics.json next to rollouts.jsonl. A collection that stopped early has no file, and the "
    "numbers cover only the rollouts it recorded. The latency pipeline neither changes nor recomputes them."
)


def _native_section(summary: dict[str, Any]) -> str:
    parts = [_scope(summary, "recorded"), f'<p class="note source">{esc(NATIVE_INTRO)}</p>']
    for rep, agg in summary["native_aggregate"].items():
        if not agg:
            parts.append(f'<p class="note">{esc(rep)}: no rollouts_aggregate_metrics.json</p>')
            continue
        for index, (title, rows) in enumerate(_aggregate_tables(agg)):
            heading = f"{rep}: {title}" if title else rep
            parts.append(
                f"<h3>{esc(heading)}</h3>" + table(("metric", "value"), rows, table_id=f"native-{rep}-{index}")
            )
    return "".join(parts)


def _timeline_section(summary: dict[str, Any]) -> str:
    occ = summary.get("occupancy")
    if not occ:
        return '<p class="note">no occupancy data (the report ran without occupancy, or no replica has a timeline)</p>'
    links = {f"{r['replica']}/{r['rollout_id']}": r["page"] for r in summary["per_task"] if r.get("page")}
    legend = " ".join(f'<span style="color:{c}">■ {esc(s)}</span>' for s, c in STATUS_COLORS.items())
    return (
        _scope(summary, "recorded")
        + f'<p class="note">One bar per rollout, sorted by start; colour is the agent status: {legend}. '
        "A bar spans the whole rollout: sandbox start, the agent, then the evaluation after the agent ended or was "
        "killed, so two agents killed at the same instant can end at different points. "
        "Hover a bar for its id, status and times. Bars link to the rollout's timeline on the replica's "
        "timelines.html page when the analyze step has written it.</p>"
        + gantt(occ["rollouts"], links, width=1180, breaks=[])
        + f'<p class="note">Rollouts active over time, split by what they were doing (bins of {num(occ["bin_s"], 0)} s).</p>'
        + stacked_area(occ["bins"], occ["bin_s"], width=1180, height=260, breaks=[])
        + _state_bands(summary)
    )


def _state_bands(summary: dict[str, Any]) -> str:
    bands = summary.get("state_bands")
    if not bands:
        return ""
    xs = [b["progress"] for b in bands["bins"]]
    series = {k: [b[k] for b in bands["bins"]] for k in ("model", "tool", "other")}
    colors = {"model": BAND_COLORS["in_model"], "tool": BAND_COLORS["in_tool"], "other": BAND_COLORS["elsewhere"]}
    return (
        f'<h3>What a rollout is doing as it progresses</h3><p class="note">Over {num(bands["rollouts"])} rollouts, '
        "each divided into equal parts from its start (0) to its end (1): the share of rollouts in a model call, "
        "in a tool, or elsewhere at each part. Every rollout weighs the same, so this is the shape of a typical "
        "rollout, not of the run's wall clock above.</p>"
        + line_chart(xs, series, colors, svg_id="state-bands", width=1180, height=200)
    )


def _entry(metric: Optional[dict[str, Any]], *keys: str) -> Any:
    value: Any = metric or {}
    for key in keys:
        value = (value or {}).get(key) if isinstance(value, dict) else None
    return value


def _total(metric: Optional[dict[str, Any]]) -> Any:
    """Sum over the run of a scalar metric (``all.sum``) or of a per-trial distribution (``per_trial_sum.sum``)."""
    metric = metric or {}
    if "per_trial_sum" in metric:
        return _entry(metric, "per_trial_sum", "sum")
    if "all" in metric:
        return _entry(metric, "all", "sum")
    return metric.get("total")


def _share_text(value: Any, total: Any) -> str:
    """Share of ``total`` as a rendered table cell; a share above zero that would print as 0.0 % reads "<0.1 %"
    (escaped, since cells are HTML)."""
    if not (_is_number(value) and _is_number(total) and total):
        return "–"
    fraction = value / total
    return "&lt;0.1 %" if 0 < fraction < 0.001 else pct(fraction)


def _where_section(summary: dict[str, Any]) -> str:
    where = summary["where_time_goes"]
    definitions = summary.get("definitions") or {}

    def critical_path_rows(stat: str) -> list[tuple[str, dict[str, Any]]]:
        rows = []
        for group in ("all", "success", "failure"):
            other = _entry(where.get("cp_other_time"), group, stat)
            unexplained = _entry(where.get("unexplained_no_model_tool_time"), group, stat)
            other_only = max(0.0, other - unexplained) if _is_number(other) and _is_number(unexplained) else other
            rows.append(
                (
                    group,
                    {
                        "model": _entry(where.get("cp_model_time"), group, stat),
                        "tool": _entry(where.get("cp_tool_time"), group, stat),
                        "other": other_only,
                        "unexplained": unexplained,
                    },
                )
            )
        return rows

    body = _scope(summary, "agent")
    body += "<h3>Critical path, mean seconds per rollout</h3>" + stacked_bar(critical_path_rows("mean"), SPLIT_COLORS)
    body += '<h3>Critical path, median seconds per rollout</h3><div id="cp-median">'
    body += stacked_bar(critical_path_rows("median"), SPLIT_COLORS) + "</div>"
    body += (
        '<p class="note">Each segment is the median of that part over the rollouts of the row; the medians of the '
        "parts do not sum to the median agent time. The percentage printed on a segment is its share of the bar.</p>"
    )
    body += (
        '<p class="note">The critical path is the longest chain of dependent work from the agent\'s first event to '
        "its last, waits included: a tool call cannot start before the model call that requested it has ended, and "
        "the next model call cannot start before the tool result is back. When nothing runs in parallel its length "
        "is the agent time. <b>model</b> is the model-call time on that chain, agent-observed (for OpenCode from the "
        "send to the first response part). <b>tool</b> is the tool execution time on it. <b>other</b> is the rest "
        "of the chain that known harness events account for, such as OpenCode's step processing between calls, "
        "sandbox waits and subagent handoffs; it is cp_other_time minus the unexplained share. <b>unexplained</b> "
        "is agent time covered by no logged model, tool or harness event at all (unexplained_no_model_tool_time), "
        "flagged per rollout when it exceeds 5 % of its agent time.</p>"
    )
    body += (
        '<p class="note"><b>How unexplained time is obtained.</b> Measured per rollout: the agent window (from the '
        "agent's first to its last persisted event), the model_call spans, the tool_call spans and the known "
        "harness events between them (compaction, subagent handoffs, step tails, observation assembly, shutdown). "
        "The union of the model and tool spans is taken out of the window, then the known events; unexplained is "
        "the remainder, agent time during which nothing the harness logs was running. It is not measured directly, "
        "so a missing or truncated span inflates it.</p>"
    )
    agent_values = [r["e2e_agent_time"] for r in summary["per_task"]]
    agent = (summary["time"].get("e2e_agent_time") or {}).get("all") or {}
    body += "<h3>Agent time per rollout (s)</h3>" + histogram(
        agent_values, {"p50": agent.get("median"), "p90": agent.get("p90"), "p99": agent.get("p99")}
    )

    def model_call_rows(stat: str) -> list[tuple[str, dict[str, Any]]]:
        return [
            (
                "per call",
                {
                    "queue": _entry(where.get("model_engine_queue_time"), "pooled", stat),
                    "prefill": _entry(where.get("model_engine_prefill_time"), "pooled", stat),
                    "decode": _entry(where.get("model_engine_decode_time"), "pooled", stat),
                    "gym_hop": _entry(where.get("gym_hop_overhead_time"), "pooled", stat),
                },
            )
        ]

    body += _scope(summary, "calls")
    body += "<h3>Model call, mean seconds per call</h3>" + stacked_bar(model_call_rows("mean"), ENGINE_COLORS)
    body += '<h3>Model call, median seconds per call</h3><div id="model-call-median">'
    body += stacked_bar(model_call_rows("median"), ENGINE_COLORS) + "</div>"
    body += (
        '<p class="note"><b>queue</b>, <b>prefill</b> and <b>decode</b> are vLLM\'s own accounting per request. '
        "<b>gym_hop</b> is the Gym model server's share of a call: the server-side duration of the request minus "
        "vLLM's end-to-end time, that is vLLM's HTTP layer and response streaming plus the Gym proxy and its "
        "capture, which cannot be separated further.</p>"
    )
    body += _per_step_tax(where.get("per_step_tax") or {})
    body += _engine_vs_agent(where.get("engine_vs_agent") or {})
    dist_names = (
        "model_call_time",
        "model_engine_queue_time",
        "model_engine_prefill_time",
        "model_engine_decode_time",
        "gym_hop_overhead_time",
        "tool_execution_time",
        "tool_dispatch_delay",
        "tool_observation_delay",
    )
    dist_rows = [stat_row(n, _entry(where.get(n), "pooled"), title=definitions.get(n, "")) for n in dist_names]
    body += table(STAT_HEADERS, dist_rows, table_id="dists")
    by_operation = where.get("tool_time_by_operation") or {}
    body += "<h3>Tool time by operation, total seconds</h3>" + bar_chart(
        [(op, d.get("total_time") or 0.0, SPLIT_COLORS["tool"]) for op, d in by_operation.items()]
    )
    body += '<h3>Tool time by operation, mean seconds per call</h3><div id="tool-mean">'
    body += bar_chart([(op, d.get("mean"), SPLIT_COLORS["tool"]) for op, d in by_operation.items()]) + "</div>"
    body += '<h3>Tool time by operation, median seconds per call</h3><div id="tool-median">'
    body += bar_chart([(op, d.get("median"), SPLIT_COLORS["tool"]) for op, d in by_operation.items()]) + "</div>"
    body += (
        '<p class="note"><b>How tool time is measured.</b> OpenCode: a tool span starts when OpenCode persisted the '
        "tool-call part (the request) and ends when the tool's state reached its end with the result stored, so it "
        "includes any queueing before the tool actually ran. Terminus: a tool span is one command; it starts when "
        "the keystrokes are sent to the terminal and ends when the wait the model asked for has elapsed (or the "
        "batch timed out), so it measures the requested wait, not the command's own run time; reading the screen "
        "afterwards is tool_observation_delay. Operations are the first word of the command (Terminus) or the "
        "tool name (OpenCode).</p>"
    )
    agent_total = agent.get("sum")
    sub = where.get("subagent_invocations") or {}
    unexplained_total = _total(where.get("unexplained_no_model_tool_time"))
    compaction_total = _total(where.get("compaction_time"))
    subagent_total = _total(where.get("subagent_time"))
    retry_total = _total(where.get("model_retry_overhead_time"))
    contributions = [
        [
            "compactions",
            num(_total(where.get("compactions"))),
            num(compaction_total, 1),
            _share_text(compaction_total, agent_total),
        ],
        [
            "subagents",
            num(sub.get("total") if isinstance(sub, dict) else sub),
            num(subagent_total, 1),
            _share_text(subagent_total, agent_total),
        ],
        [
            "retries (failed attempts)",
            num(_total(where.get("failed_model_attempts"))),
            num(retry_total, 1),
            _share_text(retry_total, agent_total),
        ],
        ["timeouts", num(summary["outcomes"]["agent_status"].get("timeout", 0)), "–", "–"],
        ["unexplained", "–", num(unexplained_total, 1), _share_text(unexplained_total, agent_total)],
    ]
    body += "<h3>Contributions</h3>" + table(
        ("source", "count", "total s", "share of agent time"), contributions, table_id="contrib", sortable=False
    )
    body += (
        '<p class="note"><b>compactions</b>: OpenCode summarising the conversation when the context nears the '
        "model's limit; the count of compaction spans and their total duration. <b>subagents</b>: child sessions "
        "the agent delegated to; their time overlaps the agent's own time. <b>retries (failed attempts)</b>: "
        "model-call attempts that ended in an error, a timeout or a cancellation and were retried by the harness; "
        "the time is the overhead spent in those failed attempts before the one that succeeded. Only attempts that "
        "reached the Gym model server are visible, so this is a lower bound. <b>timeouts</b>: rollouts whose agent "
        "was stopped at the sandbox time limit; their time is inside agent time, so no share is given. "
        "<b>unexplained</b>: agent time covered by no logged model, tool or harness event. Shares are of total "
        "agent time; a share printed as &lt;0.1 % is above zero but rounds to it.</p>"
    )
    body += _behaviour_block(summary)
    return body


def _per_step_tax(tax: dict[str, Any]) -> str:
    if not tax.get("steps"):
        return ""
    rows = [
        stat_row(name, tax.get(name))
        for name in ("model_per_step", "tool_per_step", "known_per_step", "unexplained_per_step")
    ]
    return (
        f'<h3>Per-step tax</h3><p class="note">Seconds per main-agent step, one value per rollout '
        f"({num(tax['steps'])} steps over {num(tax['rollouts'])} rollouts): the rollout's model, tool, known-harness "
        "and unexplained time divided by its number of steps. A policy that solves a task in fewer steps pays each "
        "of these fewer times; the unexplained row is the part no event accounts for.</p>"
        + table(STAT_HEADERS, rows, table_id="step-tax")
    )


def _engine_vs_agent(view: dict[str, Any]) -> str:
    buckets = view.get("buckets") or []
    if not buckets:
        return ""
    colors = {"agent_s": "#4c78a8", "server_s": "#72b7b2", "engine_e2e_s": "#e45756", "ttft_s": "#bab0ac"}
    names = {"agent_s": "agent-observed", "server_s": "server", "engine_e2e_s": "engine e2e", "ttft_s": "first token"}
    bars = [
        (f"{b['bucket']} · {names[key]}", b[key].get("median"), colors[key])
        for b in buckets
        for key in ("agent_s", "server_s", "engine_e2e_s", "ttft_s")
        if _is_number(b[key].get("median"))
    ]
    rows = [
        [
            esc(b["bucket"]),
            num(b["calls"]),
            num(b["agent_s"].get("median"), 2),
            num(b["server_s"].get("median"), 2),
            num(b["engine_e2e_s"].get("median"), 2),
            num(b["ttft_s"].get("median"), 2),
        ]
        for b in buckets
    ]
    return (
        f'<h3>Engine versus agent</h3><p class="note">Median seconds per model call by generated tokens '
        f"({num(view.get('calls'))} calls with an engine record): what the agent waited, what the model server "
        "measured for the attempt that answered, and vLLM's own end-to-end and first-token durations. The gap "
        "between agent-observed and server is network, retries and the harness; between server and engine it is "
        "vLLM's HTTP layer, the Gym proxy and its capture.</p>"
        + bar_chart(bars)
        + table(
            ("generated tokens", "calls", "agent s", "server s", "engine s", "first token s"),
            rows,
            table_id="eva",
            sortable=False,
        )
    )


def box_plot(groups: dict[str, dict[str, Any]], *, svg_id: str, width: int = 640, unit: str = "s") -> str:
    """One box per group: min, p90-less whiskers at min and max, box from the lower to the upper quartile as the
    median's neighbours (p25 is not in the stats, so the box spans median to p90 and the whiskers min to max)."""
    rows = [(g, st) for g, st in groups.items() if _is_number(st.get("count")) and st["count"]]
    if not rows:
        return '<p class="note">no values</p>'
    top = max(float(st["max"]) for _, st in rows) or 1.0
    label_w, bar_h, pad = 90, 22, 8
    height = len(rows) * (bar_h + pad) + 24

    def px(v: float) -> float:
        return label_w + v / top * (width - label_w - 10)

    out = [f'<svg id="{svg_id}" xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">']
    for i, (group, st) in enumerate(rows):
        y = 8 + i * (bar_h + pad)
        color = OUTCOME_COLORS.get(group, "#4c78a8")
        out.append(f'<text x="4" y="{y + bar_h / 2 + 4:.1f}">{esc(group)} (n={num(st["count"])})</text>')
        out.append(
            f'<line x1="{px(st["min"]):.1f}" y1="{y + bar_h / 2:.1f}" x2="{px(st["max"]):.1f}" y2="{y + bar_h / 2:.1f}" stroke="#444"/>'
        )
        out.append(
            f'<rect x="{px(st["median"]):.1f}" y="{y}" width="{max(1.0, px(st["p90"]) - px(st["median"])):.1f}" height="{bar_h}" fill="{color}" fill-opacity="0.5" stroke="{color}"/>'
        )
        out.append(
            f'<line x1="{px(st["median"]):.1f}" y1="{y}" x2="{px(st["median"]):.1f}" y2="{y + bar_h}" stroke="#000" stroke-width="2"/>'
        )
        for v in st.get("values") or []:
            out.append(f'<circle cx="{px(v):.1f}" cy="{y + bar_h / 2:.1f}" r="2" fill="#000" fill-opacity="0.35"/>')
    out.append(
        f'<text x="{label_w}" y="{height - 4}" fill="#555">0</text><text x="{width - 60}" y="{height - 4}" fill="#555">{num(top, 1)} {esc(unit)}</text></svg>'
    )
    return "".join(out)


def ecdf_chart(points: list[list[float]], *, svg_id: str, width: int = 640, height: int = 170) -> str:
    """The empirical distribution: x is the value, y the share of values at or below it."""
    if not points:
        return '<p class="note">no values</p>'
    top = points[-1][0] or 1.0
    x0, y0, w, h = 40, 10, width - 50, height - 30
    path = " ".join(f"{x0 + p[0] / top * w:.1f},{y0 + h - p[1] * h:.1f}" for p in points)
    return (
        f'<svg id="{svg_id}" xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'<polyline points="{path}" fill="none" stroke="#4c78a8" stroke-width="1.5"/>'
        f'<line x1="{x0}" y1="{y0 + h}" x2="{x0 + w}" y2="{y0 + h}" stroke="#999"/>'
        f'<text x="{x0}" y="{height - 4}" fill="#555">0</text><text x="{x0 + w - 50}" y="{height - 4}" fill="#555">{num(top, 1)} s</text>'
        f'<text x="2" y="{y0 + 10}" fill="#555">1.0</text><text x="2" y="{y0 + h}" fill="#555">0</text></svg>'
    )


def _distributions_section(summary: dict[str, Any]) -> str:
    dist = summary.get("distributions") or {}
    body = _scope(summary, "agent")
    body += (
        '<p class="note">Per-rollout values by outcome: the thick line is the median, the box runs from the median '
        "to the 90th percentile, the whiskers from the minimum to the maximum, and each dot is one rollout.</p>"
    )
    for key, unit in (
        ("e2e_agent_time", "s"),
        ("unexplained_no_model_tool_time", "s"),
        ("model_calls", "calls"),
        ("tool_calls", "calls"),
    ):
        groups = dist.get(key) or {}
        if groups:
            body += f"<h3>{esc(key)} by outcome</h3>" + box_plot(groups, svg_id=f"box-{key}", unit=unit)
    body += _scope(summary, "calls")
    for key in ("model_call_time", "tool_execution_time"):
        pooled = (dist.get(key) or {}).get("all")
        if pooled:
            body += f"<h3>{esc(key)}, all calls</h3>" + ecdf_chart(pooled["ecdf"], svg_id=f"ecdf-{key}")
    return body


def _limitations_section(summary: dict[str, Any]) -> str:
    items = "".join(
        f"<li><b>{esc(item['title'])}.</b> {esc(item['what'])} <i>{esc(item['implication'])}</i></li>"
        for item in summary.get("limitations") or []
    )
    return (
        '<p class="note">What these measurements cannot see and what that does to the numbers. These hold for every '
        "rollout of every run; per-rollout conditions are the flags in the data-quality section and the per-task "
        f"table.</p><ul>{items}</ul>"
    )


def _ratio_text(value: Any) -> str:
    return f"{value:.2f}×" if _is_number(value) else "–"


def _behaviour_block(summary: dict[str, Any]) -> str:
    where = summary["where_time_goes"]
    definitions = summary.get("definitions") or {}
    body = ""
    beh = where.get("behaviour_by_outcome") or {}
    if beh:

        def compare(v: dict[str, Any]) -> str:
            if _is_number(v.get("ratio")):
                return _ratio_text(v["ratio"])
            difference = v.get("difference")
            return f"{difference:+.1f} difference" if _is_number(difference) else "–"

        rows = [
            [
                f'<span title="{esc(definitions.get(k, ""))}">{esc(k)}</span>',
                num(v["success"], 2),
                num(v["failure"], 2),
                compare(v),
                esc(definitions.get(k, "")),
            ]
            for k, v in beh.items()
        ]
        body += _scope(summary, "outcome")
        body += "<h3>Behaviour by outcome</h3>" + table(
            ("metric", "resolved", "failed", "failed / resolved", "definition"),
            rows,
            table_id="behaviour",
            sortable=False,
        )
        body += (
            '<p class="note">Per-rollout means for resolved and failed rollouts. A ratio well above 1 marks a behaviour '
            "the policy shows when it is losing: more calls, repeated actions, rising reasoning, a longer silence "
            "before the end. For a signed metric (reasoning_per_step_drift) a ratio means nothing, so the failed "
            "minus resolved difference is shown instead; a negative drift means reasoning per step shrinks over the "
            "rollout, a positive one that it grows.</p>"
        )
    tail = where.get("generation_tail")
    if tail:
        body += _scope(summary, "calls")
        rows = [
            [
                "calls above the threshold",
                f"{num(tail['calls'])} of {num(tail['calls_total'])}",
                f"rollouts touched: {num(tail['rollouts'])}",
            ],
            ["generated tokens in them", num(tail["tokens"]), f"{pct(tail['token_share'])} of all generated tokens"],
            [
                "decode time in them",
                f"{num(tail['decode_time'], 1)} s",
                f"{pct(tail['decode_share'])} of all vLLM decode time",
            ],
            [
                "calls stopped by the length cap",
                num(tail["length_stops"]),
                "generation cut off; the response is unusable",
            ],
        ]
        body += f"<h3>Generation tail (calls above {num(tail['threshold_tokens'])} generated tokens)</h3>" + table(
            ("", "value", "note"), rows, table_id="gen-tail", sortable=False
        )
        hist = where.get("generated_tokens_histogram") or {}
        edges = hist.get("edges") or []
        items = [
            (f"{num(edges[i])}–{num(edges[i + 1])}", c, "#4c78a8") for i, c in enumerate(hist.get("counts") or [])
        ]
        body += _figure(
            "tok-hist-generated",
            "generated tokens per model call (reasoning + output), calls per decade",
            bar_chart(items, width=560),
        )
    for key, title, note in (
        (
            "reasoning_by_step",
            "Reasoning tokens per step",
            "median reasoning tokens of the main agent's model calls, by step index bucket; rising reasoning in failed rollouts is the agent getting lost",
        ),
        (
            "context_by_step",
            "Context per step",
            "median input tokens of the main agent's model calls, by step index bucket",
        ),
    ):
        view = where.get(key) or {}
        if view.get("buckets"):
            body += _scope(summary, "outcome")
            body += f"<h3>{esc(title)}</h3>" + line_chart(
                view["buckets"],
                {"success": view.get("success") or [], "failure": view.get("failure") or []},
                OUTCOME_COLORS,
                svg_id=key.replace("_by_step", "-step").replace("_", "-"),
                rollouts=view.get("rollouts"),
            )
            body += (
                f'<p class="note">{esc(note)}; x is the first step index of each bucket of {num(view.get("bucket"))} '
                f"steps. Buckets that fewer than {num(view.get('min_rollouts'))} rollouts of a group reach are not "
                "drawn, so a line ends where that group's long rollouts run out; hover a point for the number of "
                "rollouts behind it.</p>"
            )
    clusters = where.get("reasoning_clusters")
    if clusters:
        body += _scope(summary, "steps")
        baseline = f"{(clusters.get('baseline') or 0.05) * 100:g} %"
        body += "<h3>Where long reasoning clusters</h3>" + (
            f'<p class="note">{num(clusters["steps"])} main-agent steps; reasoning per step median '
            f"{num(clusters['median'], 0)}, p90 {num(clusters['p90'], 0)}, p99 {num(clusters['p99'], 0)} tokens. A "
            f"<b>stall</b> is a step in the top {baseline} of reasoning length (at least {num(clusters['threshold'])} "
            f"tokens), so every group's stall rate is {baseline} by construction (baseline {baseline}, the dashed "
            "line); a group above it is where the long thoughts cluster. If stalls concentrate in a few conditions a reasoning budget can be "
            "conditional; if they are spread evenly only a uniform budget or a sampling change reaches them.</p>"
        )
        for key, fig_id, caption in (
            ("by_action", "stall-action", "stall rate by the action the step emitted (most frequent actions)"),
            ("by_position", "stall-position", "stall rate by position in the rollout (decile of the step index)"),
            ("by_context", "stall-context", "stall rate by context size at the step"),
            ("by_preceded", "stall-preceded", "stall rate by what preceded the step, and by the rollout's outcome"),
        ):
            items = [
                (
                    f"{g['label']} · {num(g['steps'])} steps · median {num(g['median'], 0)} tok",
                    (g["stall_rate"] or 0.0) * 100,
                    "#4c78a8",
                )
                for g in clusters.get(key) or []
            ]
            body += _figure(
                fig_id,
                caption,
                bar_chart(items, width=820, label_w=400, marker=(clusters.get("baseline") or 0.05) * 100, suffix=" %"),
            )
    waits = where.get("waits")
    if waits:
        body += _scope(summary, "agent")
        share = waits.get("wait_share") or {}
        rows = [
            [
                "all commands",
                num(waits.get("waits")),
                f"{num(waits.get('wait_time'), 1)} s",
                "every Terminus command waits the model-requested time, then the screen is read",
            ],
            [
                "bare waits (no command)",
                num(waits.get("bare_waits")),
                f"{num(waits.get('bare_wait_time'), 1)} s",
                "the model asked to wait without doing anything",
            ],
            [
                f"trivial commands waited ≥ {num(waits.get('trivial_wait_s'), 0)} s",
                num(waits.get("trivial_waits")),
                f"{num(waits.get('trivial_wait_time'), 1)} s",
                f"after {esc(', '.join(waits.get('trivial_commands') or []))}; these never need a long wait",
            ],
            [
                "wait share of agent time",
                f"resolved {pct(share.get('success'))}, failed {pct(share.get('failure'))}",
                "",
                "Σ command waits / agent time",
            ],
        ]
        body += "<h3>Terminus waits</h3>" + table(
            ("", "count", "time", "note"), rows, table_id="waits", sortable=False
        )
    eff = where.get("subagent_effect")
    if eff and (eff.get("with") or {}).get("rollouts"):
        body += _scope(summary, "agent")
        rows = [
            [esc(k), num(v["rollouts"]), num(v["reward_mean"], 3), num(v["agent_time_mean"], 1)]
            for k, v in (("with subagents", eff["with"]), ("without", eff["without"]))
        ]
        for group, parts in (eff.get("by_group") or {}).items():
            w, wo = parts["with"], parts["without"]
            rows.append(
                [
                    esc(f"  {group}: with / without"),
                    f"{num(w['rollouts'])} / {num(wo['rollouts'])}",
                    f"{num(w['reward_mean'], 3)} / {num(wo['reward_mean'], 3)}",
                    f"{num(w['agent_time_mean'], 1)} / {num(wo['agent_time_mean'], 1)}",
                ]
            )
        body += "<h3>Subagents</h3>" + table(
            ("rollouts", "n", "mean reward", "mean agent s"), rows, table_id="subagent-effect", sortable=False
        )
        body += '<p class="note">Rollouts that delegated to a subagent against those that did not, overall and per task group (task difficulty confounds the overall row).</p>'
    return body


def _range_label(edges: list[Any], i: int, *, prefix: str = "", plus: bool = False) -> str:
    if i == len(edges) - 1:
        return f"≥{prefix}{edges[i]:g}" if not plus else f"{prefix}{edges[i]:g}+"
    return f"{prefix}{edges[i]:g}–{edges[i + 1]:g}" if not plus else f"{prefix}{edges[i]:g}–{edges[i + 1] - 1:g}"


def _pair(fast: Any, slow: Any, ratio: Any, *, digits: int = 0, unit: str = "") -> str:
    text = f"{num(fast, digits)}{unit} → {num(slow, digits)}{unit}"
    return f"{text} ({num(ratio, 1)}×)" if _is_number(ratio) else text


def _repeats_section(summary: dict[str, Any]) -> str:
    repeats = summary.get("repeats") or {}
    repeatability = repeats.get("repeatability") or {}
    body = _scope(summary, "solved")
    if not repeatability.get("n_tasks"):
        body += (
            '<p class="note">no always-solved tasks: wall-time repeatability, excess tool calls and the fast-vs-slow '
            "comparison need a task resolved in at least two repeats.</p>"
        )
    else:
        tasks = repeatability["tasks"]
        body += "<h3>Wall-time repeatability</h3>" + dot_strip(tasks, svg_id="repeat-strip")
        body += (
            f'<p class="note">{num(repeatability["n_tasks"])} tasks were solved in every repeat; one row per task, '
            "a dot per repeat, the tick at the task median. How repeatable is the wall time when the model solves the "
            f"task every time? Spread = slowest ÷ fastest repeat of the same task: median ×{num(repeatability['spread_median'], 1)}, "
            f"coefficient of variation median {pct(repeatability['cv_median'])}. Model-side sampling variance and "
            "sandbox variance are not separable here.</p>"
        )
        hist = repeatability["spread_histogram"]
        body += _figure(
            "spread-hist",
            "spread within a task: slowest ÷ fastest repeat, tasks per bucket",
            bar_chart(
                [(_range_label(hist["edges"], i, prefix="×"), c, "#4c78a8") for i, c in enumerate(hist["counts"])],
                width=560,
            ),
        )
        excess = repeats.get("excess_tool_calls") or {}
        hist = excess.get("histogram") or {"edges": [], "counts": []}
        body += "<h3>Excess tool calls</h3>" + _figure(
            "excess-hist",
            "extra tool calls over the leanest successful repeat of the same task, repeats per bucket",
            bar_chart(
                [
                    (_range_label(hist["edges"], i, prefix="+", plus=True), c, "#f58518")
                    for i, c in enumerate(hist["counts"])
                ],
                width=560,
            ),
        )
        per_step = excess.get("tools_per_step") or {}
        body += (
            f'<p class="note">Over {num(excess.get("rollouts"))} repeats of always-solved tasks: median +{num(excess.get("median"), 0)}, '
            f"p90 +{num(excess.get('p90'), 0)} tool calls. There is no absolute threshold for too many calls; the usable "
            "yardstick is the same task solved with fewer: the leanest successful repeat sets the floor and every call "
            "above it is overhead the policy could have avoided. Tools per step (median of tool calls ÷ steps over all "
            f"rollouts): resolved {num(per_step.get('success'), 2)}, failed {num(per_step.get('failure'), 2)}; near 1 means "
            "the model issues one tool at a time.</p>"
        )
        fast_slow = repeats.get("fast_vs_slow") or {}
        medians = fast_slow.get("ratio_medians") or {}
        body += "<h3>Fast vs slow repeat</h3>" + (
            f'<p class="note">Across {num(fast_slow.get("n_tasks"))} always-solved tasks the slow repeat had, as the median '
            f"ratio slow / fast: steps ×{num(medians.get('steps'), 1)}, tool calls ×{num(medians.get('tool_calls'), 1)}, "
            f"reasoning tokens ×{num(medians.get('reasoning_tokens'), 1)}, output tokens ×{num(medians.get('output_tokens'), 1)}, "
            f"input tokens ×{num(medians.get('input_tokens'), 1)}, model time ×{num(medians.get('model_time'), 1)}, "
            f"tool time ×{num(medians.get('tool_time'), 1)}, other time ×{num(medians.get('other_time'), 1)}.</p>"
        )
        shown = (fast_slow.get("tasks") or [])[:20]
        body += (
            '<div id="fast-slow-extra">'
            + stacked_bar(
                [
                    (t["task_id"], {k: v for k, v in (t.get("extra") or {}).items() if _is_number(v) and v > 0})
                    for t in shown
                ],
                SPLIT_COLORS,
                width=900,
                label_w=260,
            )
            + "</div>"
        )
        body += '<p class="note">Where the extra seconds of the slow repeat went (slow − fast), the 20 most variable tasks.</p>'
        rows = [
            [
                esc(t["task_id"]),
                f"×{num(t['spread'], 1)}" if _is_number(t.get("spread")) else "–",
                f"{_short(t['fast']['agent_time'])} → {_short(t['slow']['agent_time'])}",
                _pair(t["fast"]["steps"], t["slow"]["steps"], t["ratios"]["steps"]),
                _pair(t["fast"]["tool_calls"], t["slow"]["tool_calls"], t["ratios"]["tool_calls"]),
                _pair(t["fast"]["reasoning_tokens"], t["slow"]["reasoning_tokens"], t["ratios"]["reasoning_tokens"]),
                _pair(t["fast"]["output_tokens"], t["slow"]["output_tokens"], t["ratios"]["output_tokens"]),
                _pair(t["fast"]["input_tokens"], t["slow"]["input_tokens"], t["ratios"]["input_tokens"]),
                _pair(t["fast"]["subagents"], t["slow"]["subagents"], None),
                f"{_short(t['fast']['longest_model_call'])} → {_short(t['slow']['longest_model_call'])}",
            ]
            for t in fast_slow.get("tasks") or []
        ]
        body += table(
            (
                "task",
                "spread",
                "agent time fast → slow",
                "steps",
                "tool calls",
                "reasoning tokens",
                "output tokens",
                "input tokens",
                "sub-agents",
                "longest model call",
            ),
            rows,
            table_id="fast-slow",
        )
        body += (
            '<p class="note">Every always-solved task, most variable first; for each the fastest and the slowest repeat '
            "are compared. Read the ratios together: if steps and tool calls grow about as much as time, the slow "
            "repeat took a longer route (more exploration or trial and error); if reasoning tokens grow more than "
            "steps, the model thought longer per step; if tool time grows alone, it ran longer commands.</p>"
        )
    budgets = repeats.get("budgets") or {}
    body += "<h3>Budgets: what if long rollouts were stopped?</h3>"
    steps = budgets.get("step_caps") or []
    rows = [
        [
            num(r["cap"]),
            f"{num(r['passes_lost'])} of {num(r['passes'])} ({pct(r['passes_lost_share'])})",
            f"{_short(r['time_saved_s'])} of {_short(r['time_total_s'])} ({pct(r['time_saved_share'])})",
        ]
        for r in steps
    ]
    body += "<h4>Cap on main-agent steps</h4>" + table(
        ("cap (steps)", "passes lost", "agent time saved"), rows, table_id="step-caps", sortable=False
    )
    if steps:
        body += line_chart(
            [r["cap"] for r in steps],
            {
                "passes lost": [(r["passes_lost_share"] or 0) * 100 for r in steps],
                "saved": [(r["time_saved_share"] or 0) * 100 for r in steps],
            },
            BUDGET_COLORS,
            svg_id="step-caps-chart",
        )
    generation = budgets.get("generation_caps") or []
    rows = [
        [
            num(r["cap"]),
            f"{num(r['passes_lost'])} of {num(r['passes'])} ({pct(r['passes_lost_share'])})",
            f"{num(r['tokens_saved'])} of {num(r['tokens_total'])} ({pct(r['tokens_saved_share'])})",
        ]
        for r in generation
    ]
    body += "<h4>Cap on generated tokens per model call</h4>" + table(
        ("cap (tokens per call)", "passes lost", "generated tokens saved"),
        rows,
        table_id="generation-caps",
        sortable=False,
    )
    if generation:
        body += line_chart(
            [r["cap"] for r in generation],
            {
                "passes lost": [(r["passes_lost_share"] or 0) * 100 for r in generation],
                "saved": [(r["tokens_saved_share"] or 0) * 100 for r in generation],
            },
            BUDGET_COLORS,
            svg_id="generation-caps-chart",
        )
    body += (
        '<p class="note">What-ifs on the recorded data, in percent of all passes and of all agent time or generated '
        "tokens. Step cap: a resolved rollout with more main-agent steps than the cap counts as a lost pass; a rollout "
        "over the cap saves (1 − cap ÷ steps) of its agent time, time spread evenly over its steps. Generation cap: a "
        "resolved rollout with any model call above the cap counts as a lost pass; the tokens above the cap are the "
        "tokens saved. Neither simulates the agent under a cap, which might finish earlier or give up differently: "
        "read them as an upper bound on savings and a lower bound on losses.</p>"
    )
    return body


def _tokens_section(summary: dict[str, Any]) -> str:
    tokens = summary["tokens"]
    definitions = summary.get("definitions") or {}
    cache_share = pct(_entry(tokens.get("prompt_cache_hit_share"), "all", "mean"))
    cards = _cards([(k, num(v)) for k, v in tokens["totals"].items()] + [("cache hit share", cache_share)])
    rows = [stat_row(k, st) for k, st in tokens["per_rollout"].items()]
    purpose_rows = [
        [esc(p), *[num(t.get(k, 0)) for k in ("input", "output", "reasoning", "cached")]]
        for p, t in tokens["by_purpose"].items()
    ]
    context_rows = [
        stat_row(n, _entry(tokens.get(n), "all"), title=definitions.get(n, ""))
        for n in ("max_context", "context_growth", "cumulative_input_tokens")
    ]
    per_task = summary.get("per_task") or []
    kinds = ("input", "output", "reasoning", "cached")
    kind_figs = []
    for kind in kinds:
        st = tokens["per_rollout"].get(kind) or {}
        chart = histogram(
            [r.get(f"tokens_{kind}") for r in per_task],
            {"p50": st.get("median"), "p90": st.get("p90")},
            width=560,
            height=120,
        )
        kind_figs.append(_figure(f"tok-hist-{kind}", f"{kind} tokens per rollout", chart))
    purposes = list(tokens["by_purpose"])
    colors = {p: PURPOSE_PALETTE[i % len(PURPOSE_PALETTE)] for i, p in enumerate(purposes)}
    purpose_bars = [(kind, {p: tokens["by_purpose"][p].get(kind) or 0 for p in purposes}) for kind in kinds]
    context_figs = []
    for name in ("max_context", "context_growth", "cumulative_input_tokens"):
        st = _entry(tokens.get(name), "all") or {}
        chart = histogram(
            [r.get(name) for r in per_task], {"p50": st.get("median"), "p90": st.get("p90")}, width=560, height=120
        )
        context_figs.append(_figure(f"tok-hist-{name}", f"{name}: {definitions.get(name, '')}", chart))
    return (
        _scope(summary, "agent")
        + cards
        + "<h3>Per rollout</h3>"
        + table(STAT_HEADERS, rows, table_id="tok-rollout")
        + f'<div class="grid">{"".join(kind_figs)}</div>'
        + _scope(summary, "calls")
        + "<h3>By purpose</h3>"
        + table(("purpose", "input", "output", "reasoning", "cached"), purpose_rows, table_id="tok-purpose")
        + '<p class="note">Share of each token kind by the purpose of the call that used it.</p>'
        + f'<div id="tok-purpose-chart">{stacked_bar(purpose_bars, colors)}</div>'
        + "<h3>Context</h3>"
        + table(STAT_HEADERS, context_rows, table_id="tok-ctx")
        + f'<div class="grid">{"".join(context_figs)}</div>'
    )


def _figure(fig_id: str, caption: str, svg: str) -> str:
    return f'<figure id="{esc(fig_id)}"><figcaption>{esc(caption)}</figcaption>{svg}</figure>'


def _link_cell(entry: dict[str, Any]) -> str:
    text = esc(entry["rollout_id"])
    return f'<a href="{esc(entry["page"])}">{text}</a>' if entry.get("page") else text


def _tail_section(summary: dict[str, Any]) -> str:
    out = [_scope(summary, "agent")]
    for title, key, unit in (
        ("Slowest rollouts", "slowest", "agent s"),
        ("Most unexplained time", "most_unexplained", "unexplained s"),
        ("Most model calls", "most_calls", "calls"),
    ):
        rows = [
            [
                esc(e["replica"]),
                _link_cell(e),
                esc(e["task_id"]),
                num(e["reward"], 2),
                esc(e["status"]),
                num(e["value"], 1),
            ]
            for e in summary["tail"][key]
        ]
        headers = ("replica", "rollout", "task", "reward", "status", unit)
        out.append(f"<h3>{esc(title)}</h3>" + table(headers, rows, table_id=f"tail-{key}"))
    return "".join(out)


def _task_consistency_table(summary: dict[str, Any]) -> str:
    tasks = summary.get("tasks") or []
    if not tasks:
        return ""
    counts: dict[str, int] = {}
    for t in tasks:
        counts[t["verdict"]] = counts.get(t["verdict"], 0) + 1
    rows = [
        [
            esc(t["task_id"]),
            num(t["repeats"]),
            num(t["reward_mean"], 2),
            esc(t["verdict"]),
            num(t["agent_time_mean"], 0),
            num(t["model_calls_mean"], 1),
            num(t["repeated_actions_mean"], 1),
            num(t["timeouts"]),
            num(t.get("agent_time_pass_median"), 0),
            num(t.get("agent_time_fail_median"), 0),
            num(t.get("steps_median"), 0),
        ]
        for t in tasks
    ]
    summary_text = ", ".join(f"{n} {v}" for v, n in sorted(counts.items()))
    return (
        f'<h3>Task consistency</h3><p class="note">{len(tasks)} tasks over their repeats: {esc(summary_text)}. Mixed tasks '
        "are where the outcome varies between repeats; always-fail tasks with many repeated actions are loops, "
        "with few calls they die at setup. pass s and fail s are the median agent time of the passing and the failing "
        "repeats, steps the median main-agent steps: a never-pass task that gives up fast with few steps points at the "
        "environment, one that grinds to the cap is hard.</p>"
        + table(
            (
                "task",
                "repeats",
                "mean reward",
                "verdict",
                "agent s",
                "calls",
                "repeated",
                "timeouts",
                "pass s",
                "fail s",
                "steps",
            ),
            rows,
            table_id="task-consistency",
        )
    )


def _per_task_section(summary: dict[str, Any]) -> str:
    rows = [
        [
            esc(r["replica"]),
            _link_cell(r),
            esc(r["task_id"]),
            num(r["reward"], 2),
            esc(r["status"]),
            num(r["e2e_agent_time"], 1),
            num(r["model_calls"]),
            num(r["tool_calls"]),
            num(r["tokens_total"]),
            num(r["compactions"]),
            num(r["subagents"]),
            num(r.get("repeated_actions")),
            num(r.get("stall_s"), 0),
            esc(", ".join(r["flags"])),
        ]
        for r in summary["per_task"]
    ]
    headers = (
        "replica",
        "rollout",
        "task",
        "reward",
        "status",
        "agent s",
        "calls",
        "tools",
        "tokens",
        "compactions",
        "subagents",
        "repeated",
        "stall s",
        "flags",
    )
    return (
        _scope(summary, "recorded")
        + _task_consistency_table(summary)
        + f'<h3>Rollouts</h3><p class="note">{len(rows)} rollouts. Click a header to sort.</p>'
        + table(headers, rows, table_id="tasks")
    )


def _quality_section(summary: dict[str, Any]) -> str:
    quality = summary["data_quality"]
    gaps = [[esc(c), num(n)] for c, n in quality["gaps"].items()]
    flags = [[esc(c), num(d["count"]), esc(d.get("text", ""))] for c, d in quality["flags"].items()]
    notes = "".join(f"<li>{esc(n)}</li>" for n in quality["notes"]) or "<li>none</li>"
    return (
        _scope(summary, "recorded")
        + f"<p>Rollouts without a timeline: {num(quality['trials_without_timeline'])}</p><h3>Notes</h3><ul>{notes}</ul>"
        + "<h3>Flags</h3>"
        + '<p class="note">A flag marks a rollout whose value needs the stated reading; the limitations section at '
        "the top says what holds for every rollout.</p>"
        + table(("flag", "rollouts", "meaning"), flags, table_id="flags")
        + "<h3>Gaps</h3>"
        + '<p class="note">A gap says what could not be observed at all for a rollout.</p>'
        + table(("code", "rollouts"), gaps, table_id="gaps")
    )


def render_report_html(summary: dict[str, Any]) -> str:
    run = summary["run"]
    title = f"Benchmark report: {run.get('benchmark')} {run.get('experiment_name')} job {run.get('slurm_job_id')}"
    bodies = (
        _limitations_section(summary),
        _run_section(summary),
        _outcome_section(summary),
        _native_section(summary),
        _timeline_section(summary),
        _where_section(summary),
        _distributions_section(summary),
        _repeats_section(summary),
        _tokens_section(summary),
        _tail_section(summary),
        _per_task_section(summary),
        _quality_section(summary),
    )
    sections = "".join(_section(name, body) for name, body in zip(SECTIONS, bodies))
    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8"><title>'
        + esc(title)
        + "</title><style>"
        + STYLE
        + "</style></head><body><h1>"
        + esc(title)
        + "</h1>"
        + f'<p class="note">generated {esc(summary["generated_at"])}, schema {esc(summary["schema_version"])}</p>'
        + '<div id="tip" class="tip"></div>'
        + sections
        + "<script>"
        + SORT_SCRIPT
        + TIP_SCRIPT
        + "</script></body></html>"
    )
