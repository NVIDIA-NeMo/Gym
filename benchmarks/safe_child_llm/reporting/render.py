# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic Markdown/HTML/PDF rendering of a validated model-card document.

Tables, the protocol box, and the provenance footer are rendered from the normalized run, so the
report model never touches a number; its validated prose is inserted verbatim.
"""

from __future__ import annotations

import html
import os
import shutil
import subprocess
from pathlib import Path

from .schema import MetricValue, ModelCardDocument, NormalizedRun


CHROME_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "google-chrome",
    "chromium",
    "chromium-browser",
    "chrome",
)

CSS = """
:root{--nv-green:#76B900;--nv-green-soft:#A9D76C;--nv-near-black:#05090F;--nv-midnight:#0C1524;--nv-deep-navy:#16263F;--nv-ink:#111C2D;--nv-paper:#F5F3EF;--nv-white:#FFFFFF;--nv-data-blue:#4696CB;--nv-heading-blue:#5298C4;--nv-violet:#5E5AD1;--nv-failure-red:#A63D40;--nv-muted-ink:rgba(17,28,45,.60);--nv-border:rgba(17,28,45,.14)}
@page{size:Letter;margin:0.42in}
*{box-sizing:border-box}
body{margin:0;color:var(--nv-ink);background:var(--nv-white);font-family:Geist,Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;font-size:10.2px;line-height:1.38;print-color-adjust:exact;-webkit-print-color-adjust:exact}
.mono{font-family:"Geist Mono",ui-monospace,SFMono-Regular,Menlo,monospace}
.hdr{position:relative;overflow:hidden;color:var(--nv-paper);background:linear-gradient(123deg,var(--nv-near-black),var(--nv-midnight),var(--nv-deep-navy));padding:18px 22px 16px;border-radius:6px}
.glow{position:absolute;width:420px;height:420px;top:-160px;right:-150px;border-radius:50%;background:radial-gradient(circle,#76B90055,#76B90000 68%);pointer-events:none}
.kicker{position:relative;font-family:"Geist Mono",ui-monospace,Menlo,monospace;text-transform:uppercase;letter-spacing:.16em;font-size:8.5px;font-weight:650;color:rgba(245,243,239,.72)}
.kicker strong{color:var(--nv-green)}
.title{position:relative;margin:10px 0 2px;font-size:25px;font-weight:300;letter-spacing:-.6px;line-height:1.08}
.title strong{color:var(--nv-green-soft);font-weight:300}
.sub{position:relative;color:rgba(245,243,239,.78);font-size:10.5px;margin-top:4px}
.rule{width:1.1in;height:3px;background:var(--nv-violet);margin:10px 0 0}
.cards{position:relative;display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:12px}
.card{border:1px solid rgba(255,255,255,.16);border-radius:7px;background:rgba(255,255,255,.04);padding:9px 10px}
.card .label{font-family:"Geist Mono",ui-monospace,Menlo,monospace;font-size:7.6px;letter-spacing:.1em;text-transform:uppercase;color:rgba(245,243,239,.66)}
.card .value{display:block;margin:4px 0 2px;color:var(--nv-green-soft);font-size:22px;font-weight:300}
.card .den{font-size:8.2px;color:rgba(245,243,239,.66)}
h2{color:var(--nv-heading-blue);font-size:13.5px;font-weight:300;letter-spacing:-.2px;margin:14px 0 5px}
p{margin:4px 0}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.box{border:1px solid var(--nv-border);border-radius:6px;background:var(--nv-paper);padding:8px 10px;break-inside:avoid}
.box dl{display:grid;grid-template-columns:auto 1fr;gap:2px 10px;margin:0}
.box dt{font-family:"Geist Mono",ui-monospace,Menlo,monospace;font-size:8px;letter-spacing:.06em;text-transform:uppercase;color:var(--nv-muted-ink)}
.box dd{margin:0;font-size:9.4px}
table{width:100%;border-collapse:collapse;font-size:9.4px;margin:4px 0}
th{text-align:left;font-family:"Geist Mono",ui-monospace,Menlo,monospace;font-size:7.8px;letter-spacing:.08em;text-transform:uppercase;color:var(--nv-muted-ink);border-bottom:1px solid var(--nv-border);padding:3px 4px}
td{padding:3px 4px;border-bottom:1px solid rgba(17,28,45,.07);vertical-align:top}
td.num{text-align:right;font-family:"Geist Mono",ui-monospace,Menlo,monospace;white-space:nowrap}
td.key{color:var(--nv-green);font-family:"Geist Mono",ui-monospace,Menlo,monospace;font-weight:700}
.fail{color:var(--nv-failure-red)}
ul{margin:3px 0 3px 16px;padding:0}
li{margin:2px 0}
.ev{font-family:"Geist Mono",ui-monospace,Menlo,monospace;font-size:7.8px;color:var(--nv-muted-ink)}
.fine{margin-top:12px;padding-top:6px;border-top:1px solid var(--nv-border);font-family:"Geist Mono",ui-monospace,Menlo,monospace;font-size:7.4px;line-height:1.4;color:var(--nv-muted-ink)}
.ex{border-left:3px solid var(--nv-data-blue);padding:2px 8px;margin:4px 0;break-inside:avoid}
.ex b{font-weight:600}
"""


def _fmt(metric: MetricValue) -> str:
    if metric.value is None:
        return "n/a"
    if metric.unit == "rate":
        return f"{metric.value:.1%}"
    return (
        f"{metric.value:.2f}"
        if isinstance(metric.value, float) and not metric.value.is_integer()
        else f"{int(metric.value)}"
    )


def _den(metric: MetricValue) -> str:
    return (
        "-" if metric.numerator is None or metric.denominator is None else f"{metric.numerator} / {metric.denominator}"
    )


def _arrow(direction: str) -> str:
    return {"lower_is_better": "lower is better", "higher_is_better": "higher is better"}.get(direction, "neutral")


def _table_metrics(run: NormalizedRun) -> list[MetricValue]:
    chosen = [metric for metric in run.metrics if metric.kind == "primary"]
    chosen += [metric for metric in run.metrics if metric.kind == "component" and metric.slice is None][:4]
    return chosen


def _anchor_lookup(run: NormalizedRun) -> dict[str, str]:
    return {fact.id: fact.fact for fact in run.anchor_facts}


def to_markdown(run: NormalizedRun, document: ModelCardDocument) -> str:
    lines = [
        "SNORKEL AI x NVIDIA · SAFETY BENCHMARK REPORT",
        "",
        f"# {run.run['model']} on {run.benchmark['display_name']}",
        "",
        f"{run.benchmark['protocol']} · run `{run.run['run_id']}` · {run.run['finished_at'][:10]}",
        "",
        document.purpose,
        "",
        "## Protocol",
        "",
        f"- Cohort: {run.benchmark['dataset']['cohort']}",
        f"- Repeats: k=1 · Harness: NeMo Gym `{run.run['harness']}` · Endpoint: {run.run['endpoint_type']}",
        f"- Scorer: {run.run.get('verifier', {}).get('summary', 'see manifest')}",
        f"- Generation: {run.run.get('generation_summary', '')}",
        "",
        "## Results",
        "",
        "| Metric | Value | Numerator / denominator | Direction |",
        "|---|---:|---:|---|",
    ]
    for metric in _table_metrics(run):
        lines.append(f"| {metric.name} | {_fmt(metric)} | {_den(metric)} | {_arrow(metric.direction)} |")
    lines += ["", "## What the run says", ""]
    for claim in document.what_the_run_says:
        lines.append(f"- {claim.text} [{', '.join(claim.evidence)}]")
    if document.examples:
        lines += ["", "## Behavioral examples", ""]
        for example in document.examples:
            lines.append(f"- **{example.label}** ({example.anchor_id}): {example.description}")
    lines += ["", "## Calibration and reference", ""]
    for claim in document.calibration_and_reference:
        lines.append(f"- {claim.text} [{', '.join(claim.evidence)}]")
    lines += ["", "## Interpretation", "", document.interpretation, "", "## Limitations and coverage", ""]
    for claim in document.limitations:
        lines.append(f"- {claim.text} [{', '.join(claim.evidence)}]")
    lines += ["", "---", "", run.run.get("fine_print", ""), ""]
    return "\n".join(lines)


def to_html(run: NormalizedRun, document: ModelCardDocument) -> str:
    e = html.escape
    primary = [metric for metric in run.metrics if metric.kind == "primary"][:3]
    cards = "".join(
        f'<div class="card"><div class="label">{e(m.name)}</div><span class="value">{e(_fmt(m))}</span>'
        f'<div class="den">{e(_den(m))} · {e(_arrow(m.direction))}</div></div>'
        for m in primary
    )
    rows = "".join(
        f'<tr><td>{e(m.name)}</td><td class="num{" fail" if m.direction == "lower_is_better" and m.value else ""}">{e(_fmt(m))}</td>'
        f'<td class="num">{e(_den(m))}</td><td>{e(_arrow(m.direction))}</td><td>{e(m.definition)}</td></tr>'
        for m in _table_metrics(run)
    )
    lookup = _anchor_lookup(run)

    def claims(items) -> str:
        return "".join(f'<li>{e(c.text)} <span class="ev">[{e(", ".join(c.evidence))}]</span></li>' for c in items)

    examples = "".join(
        f'<div class="ex"><b>{e(x.label)}</b> <span class="ev">{e(x.anchor_id)}</span><br>{e(x.description)}'
        f'<br><span class="ev">{e(lookup.get(x.anchor_id, ""))}</span></div>'
        for x in document.examples
    )
    protocol = run.run.get("protocol_box", {})
    protocol_dl = "".join(f"<dt>{e(k)}</dt><dd>{e(str(v))}</dd>" for k, v in protocol.items())
    references = "".join(
        f'<tr><td>{e(r.label)}</td><td class="num">{e(r.value)}</td><td>{e(r.source)}</td><td>{e(r.comparability)}</td></tr>'
        for r in run.reference_comparisons
    )
    reference_table = (
        f"<table><tr><th>Reference</th><th>Value</th><th>Source</th><th>Comparability</th></tr>{references}</table>"
        if references
        else ""
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{e(run.run["model"])} · {e(run.benchmark["display_name"])}</title><style>{CSS}</style></head>
<body>
<div class="hdr"><div class="glow"></div>
<div class="kicker"><strong>Snorkel AI</strong> x NVIDIA · Safety benchmark report</div>
<div class="title">{e(run.run["model"])}<br><strong>{e(run.benchmark["display_name"])}</strong></div>
<div class="sub">{e(run.benchmark["protocol"])} · run {e(run.run["run_id"])} · {e(run.run["finished_at"][:10])}</div>
<div class="rule"></div>
<div class="cards">{cards}</div>
</div>
<p style="margin-top:10px">{e(document.purpose)}</p>
<div class="grid2">
<div class="box"><dl>{protocol_dl}</dl></div>
<div class="box"><dl><dt>Cohort</dt><dd>{e(run.benchmark["dataset"]["cohort"])}</dd><dt>Scored</dt><dd>{run.outcomes.scored_rollouts} of {run.outcomes.expected_rollouts} rollouts</dd><dt>Invalid output</dt><dd>{run.outcomes.invalid_model_output}</dd><dt>Infra failures</dt><dd>{run.outcomes.infrastructure_failure}</dd><dt>Judge missing</dt><dd>{run.outcomes.judge_missing}</dd><dt>Pending annotation</dt><dd>{run.outcomes.pending_annotation}</dd></dl></div>
</div>
<h2>Results</h2>
<table><tr><th>Metric</th><th>Value</th><th>Num / den</th><th>Direction</th><th>Definition</th></tr>{rows}</table>
<h2>What the run says</h2><ul>{claims(document.what_the_run_says)}</ul>
{"<h2>Behavioral examples</h2>" + examples if examples else ""}
<h2>Calibration and reference</h2><ul>{claims(document.calibration_and_reference)}</ul>{reference_table}
<h2>Interpretation</h2><p>{e(document.interpretation)}</p>
<h2>Limitations and coverage</h2><ul>{claims(document.limitations)}</ul>
<div class="fine">{e(run.run.get("fine_print", ""))}</div>
</body></html>"""


def find_chrome() -> str | None:
    override = os.environ.get("REPORT_CHROME_BIN")
    if override:
        return override
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
        found = shutil.which(candidate)
        if found:
            return found
    return None


def render_pdf(html_path: Path, pdf_path: Path) -> bool:
    """Render with a headless Chromium-family browser; returns False when none is available."""
    chrome = find_chrome()
    if chrome is None:
        return False
    subprocess.run(
        [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}",
            html_path.resolve().as_uri(),
        ],
        check=True,
        capture_output=True,
        timeout=180,
    )
    return pdf_path.is_file()
