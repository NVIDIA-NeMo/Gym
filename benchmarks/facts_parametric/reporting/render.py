# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic Markdown/HTML/PDF rendering of a validated model-card document.

Every number, table, and provenance line comes from the normalized run; the writer's validated prose is inserted
verbatim. The visual system is the Snorkel AI x NVIDIA report theme (typography-only kicker, dark gradient header
with a single green glow, warm-paper panels, green/blue data colours, red only for failure states).
"""

from __future__ import annotations

import html
import os
import shutil
import subprocess
from pathlib import Path

from .blade import _den, _fmt
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


def _table_metrics(run: NormalizedRun) -> list[MetricValue]:
    chosen = [metric for metric in run.metrics if metric.kind == "primary"]
    chosen += [metric for metric in run.metrics if metric.kind == "component" and metric.slice is None][:5]
    return chosen


def _cards(run: NormalizedRun) -> list[MetricValue]:
    return _table_metrics(run)[:3]


def _anchor_lookup(run: NormalizedRun) -> dict[str, str]:
    return {fact.id: fact.fact for fact in run.anchor_facts}


def _direction(direction: str) -> str:
    return {"lower_is_better": "lower is better", "higher_is_better": "higher is better"}.get(direction, "neutral")


def _footer_text(run: NormalizedRun) -> str:
    bench = run.benchmark
    r = run.run
    parts = [
        f"Model: {r['model']} via {r.get('endpoint_type', 'OpenAI-compatible endpoint')}; harness NeMo Gym {r.get('harness', '')} at {r.get('gym_revision', 'n/a')}.",
        f"Benchmark: {bench['display_name']} — {bench['protocol']}; data {bench['dataset'].get('source', '')} ({bench['dataset'].get('revision', '')}, {bench['dataset'].get('count', '')} rows, sha256 {str(bench['dataset'].get('sha256', ''))[:16]}).",
        f"Judge/verifier: {r.get('verifier', {}).get('summary', '')}.",
        f"Sampling: {r.get('sampling', '')}; limits: {r.get('generation_limits', '')}.",
        f"Exclusions and invalids: scored {run.outcomes.scored_rollouts}/{run.outcomes.expected_rollouts}; judge failed {run.outcomes.judge_failed}; simulator failed {run.outcomes.simulation_failed}; infrastructure failed {run.outcomes.infrastructure_failed}; missing {run.outcomes.missing_rollouts}.",
        f"Paper: {bench.get('paper', {}).get('citation', '')} ({bench.get('paper', {}).get('url', '')}).",
        f"Run id {r['run_id']}; finished {r.get('finished_at', '')}; package hash in checksums.sha256.",
    ]
    return " ".join(parts)


def to_markdown(run: NormalizedRun, document: ModelCardDocument) -> str:
    anchors = _anchor_lookup(run)
    lines = [
        "SNORKEL AI x NVIDIA · SAFETY AND FACTUALITY EVALUATION",
        "",
        f"# {run.run['model']} on {run.benchmark['display_name']}",
        "",
        f"{run.benchmark['protocol']} · run `{run.run['run_id']}` · {str(run.run.get('finished_at', ''))[:10]}",
        "",
        document.purpose,
        "",
        "## Protocol",
        "",
        f"- Cohort: {run.benchmark['dataset'].get('cohort', '')}",
        f"- Tasks / rollouts: {run.outcomes.expected_tasks} / {run.outcomes.expected_rollouts} · repeats: {run.run.get('repeats', 1)}",
        f"- Model settings: {run.run.get('sampling', '')}; {run.run.get('generation_limits', '')}",
        f"- Judge / verifier: {run.run.get('verifier', {}).get('summary', '')}",
        "",
        "## Results",
        "",
        "| Metric | Value | Numerator / denominator | Direction |",
        "|---|---:|---:|---|",
    ]
    for metric in _table_metrics(run):
        lines.append(f"| {metric.name} | {_fmt(metric)} | {_den(metric)} | {_direction(metric.direction)} |")
    lines += ["", "## What the run says", ""]
    for claim in document.what_the_run_says:
        lines.append(f"- {claim.text} `[{', '.join(claim.evidence)}]`")
    if document.examples:
        lines += ["", "## Examples", ""]
        for example in document.examples:
            lines.append(f"- **{example.label}** — {example.description} `[{example.anchor_id}]`")
            excerpt = next((fact.excerpt for fact in run.anchor_facts if fact.id == example.anchor_id), None)
            if excerpt:
                lines.append(f"  > {excerpt.replace(chr(10), ' ')}")
    lines += ["", "## Calibration and reference", ""]
    for claim in document.calibration_and_reference:
        lines.append(f"- {claim.text} `[{', '.join(claim.evidence)}]`")
    for reference in run.reference_comparisons:
        lines.append(
            f"- Reference: {reference.label}: {reference.value} ({reference.source}; {reference.comparability})"
        )
    lines += ["", "## Limitations", ""]
    for claim in document.limitations:
        lines.append(f"- {claim.text} `[{', '.join(claim.evidence)}]`")
    lines += ["", "## Interpretation", "", document.interpretation, "", "---", "", _footer_text(run), ""]
    _ = anchors
    return "\n".join(lines)


def _esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def to_html(run: NormalizedRun, document: ModelCardDocument) -> str:
    cards = "".join(
        f'<div class="card"><div class="label">{_esc(m.name)}</div><span class="value">{_esc(_fmt(m))}</span>'
        f'<div class="den">{_esc(_den(m))} · {_esc(_direction(m.direction))}</div></div>'
        for m in _cards(run)
    )
    rows = "".join(
        f"<tr><td>{_esc(m.name)}</td><td class='num'>{_esc(_fmt(m))}</td><td class='num'>{_esc(_den(m))}</td>"
        f"<td>{_esc(_direction(m.direction))}</td><td>{_esc(m.kind)}</td></tr>"
        for m in _table_metrics(run)
    )
    claims = "".join(
        f"<li>{_esc(c.text)} <span class='ev'>[{_esc(', '.join(c.evidence))}]</span></li>"
        for c in document.what_the_run_says
    )
    examples = ""
    for example in document.examples:
        excerpt = next((fact.excerpt for fact in run.anchor_facts if fact.id == example.anchor_id), None)
        examples += (
            f"<div class='ex'><b>{_esc(example.label)}</b> — {_esc(example.description)} "
            f"<span class='ev'>[{_esc(example.anchor_id)}]</span>"
            + (f"<div class='ev' style='margin-top:2px;white-space:pre-wrap'>{_esc(excerpt)}</div>" if excerpt else "")
            + "</div>"
        )
    calibration = "".join(
        f"<li>{_esc(c.text)} <span class='ev'>[{_esc(', '.join(c.evidence))}]</span></li>"
        for c in document.calibration_and_reference
    )
    references = "".join(
        f"<li>Reference — {_esc(r.label)}: {_esc(r.value)} <span class='ev'>({_esc(r.source)}; {_esc(r.comparability)})</span></li>"
        for r in run.reference_comparisons
    )
    limitations = "".join(
        f"<li>{_esc(c.text)} <span class='ev'>[{_esc(', '.join(c.evidence))}]</span></li>"
        for c in document.limitations
    )
    verifier = run.run.get("verifier", {}).get("summary", "")
    protocol_box = (
        "<div class='box'><dl>"
        f"<dt>Cohort</dt><dd>{_esc(run.benchmark['dataset'].get('cohort', ''))}</dd>"
        f"<dt>Tasks / rollouts</dt><dd>{run.outcomes.expected_tasks} / {run.outcomes.expected_rollouts} · repeats {_esc(run.run.get('repeats', 1))}</dd>"
        f"<dt>Model settings</dt><dd>{_esc(run.run.get('sampling', ''))}; {_esc(run.run.get('generation_limits', ''))}</dd>"
        f"<dt>Judge / verifier</dt><dd>{_esc(verifier)}</dd>"
        f"<dt>Scored</dt><dd>{run.outcomes.scored_rollouts} of {run.outcomes.expected_rollouts} (judge failed {run.outcomes.judge_failed}, simulator failed {run.outcomes.simulation_failed}, infrastructure failed {run.outcomes.infrastructure_failed}, missing {run.outcomes.missing_rollouts})</dd>"
        "</dl></div>"
    )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>{_esc(run.run["model"])} on {_esc(run.benchmark["display_name"])}</title>
<style>{CSS}</style></head>
<body>
<div class="hdr"><div class="glow"></div>
<div class="kicker"><strong>Snorkel AI</strong> x NVIDIA · Safety and factuality evaluation</div>
<div class="title">{_esc(run.run["model"])} on <strong>{_esc(run.benchmark["display_name"])}</strong></div>
<div class="sub">{_esc(run.benchmark["protocol"])} · run {_esc(run.run["run_id"])} · {_esc(str(run.run.get("finished_at", ""))[:10])}</div>
<div class="rule"></div>
<div class="cards">{cards}</div>
</div>
<p style="margin-top:10px">{_esc(document.purpose)}</p>
<div class="grid2">
<div>{protocol_box}</div>
<div><table><thead><tr><th>Metric</th><th>Value</th><th>Num / den</th><th>Direction</th><th>Role</th></tr></thead><tbody>{rows}</tbody></table></div>
</div>
<h2>What the run says</h2><ul>{claims}</ul>
{"<h2>Examples</h2>" + examples if examples else ""}
<h2>Calibration and reference</h2><ul>{calibration}{references}</ul>
<h2>Limitations</h2><ul>{limitations}</ul>
<h2>Interpretation</h2><p>{_esc(document.interpretation)}</p>
<div class="fine">{_esc(_footer_text(run))}</div>
</body></html>
"""


def find_chrome() -> str | None:
    for candidate in CHROME_CANDIDATES:
        if os.path.isabs(candidate) and os.path.exists(candidate):
            return candidate
        found = shutil.which(candidate)
        if found:
            return found
    return None


def render_pdf(html_path: Path, pdf_path: Path, *, timeout: float = 120.0) -> bool:
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
        timeout=timeout,
    )
    return pdf_path.exists()
