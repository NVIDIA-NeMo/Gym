# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render visual_agent rollouts as a browsable HTML report.

One index page (scores per run, category and task) and one page per task with the prompt,
reference images and rubric, and for every rollout: scores, the grader's renders, the artifact
itself (live in an iframe, or as a video), per-item verdicts with the judge's evidence, the
screenshots the judge looked at, the agent's last previews and a condensed agent trace.
Several rollout files (e.g. two models) are shown side by side.

    python benchmarks/visual_agent/report.py results/<a>.jsonl [results/<b>.jsonl ...] \\
        [--labels qwen super] --out results/visual_agent_report
    python -m http.server -d results/visual_agent_report 8000

Serve it over HTTP rather than opening index.html as a file: browsers refuse ES-module imports
(three.js artifacts) from file:// pages. The output folder is self-contained and relocatable.
"""

import argparse
import base64
import hashlib
import html
import io
import json
import os
import re
import tarfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "resources_servers" / "visual_agent" / "data"
CATEGORIES = ["website", "interactive_app", "game", "3d_scene", "slides", "svg", "video", "figma"]
PRIMARY_RENDERS = ["desktop.png", "slides_overview.png", "frames_overview.png", "svg.png"]
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024

CSS = """
body{font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0;color:#1d2330;background:#f5f6f8}
main{max-width:1500px;margin:0 auto;padding:24px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:19px;margin:28px 0 10px}h3{font-size:16px;margin:18px 0 8px}
a{color:#2457c5;text-decoration:none}a:hover{text-decoration:underline}
table{border-collapse:collapse;background:#fff;margin:8px 0}
th,td{border:1px solid #dde1e8;padding:5px 8px;text-align:left;vertical-align:top}
th{background:#eef1f6;font-weight:600}td.num{text-align:right;font-variant-numeric:tabular-nums}
.muted{color:#6a7385}.card{background:#fff;border:1px solid #dde1e8;border-radius:8px;padding:14px 16px;margin:14px 0}
.chip{display:inline-block;min-width:34px;padding:1px 6px;margin:1px;border-radius:10px;font-size:12px;
 text-align:center;color:#fff;font-variant-numeric:tabular-nums}
.s4{background:#1f9d55}.s3{background:#7cb342}.s2{background:#f0a202}.s1{background:#e4572e}.s0{background:#b3261e}
.sm{background:#9aa1ad}
.pass{color:#1f9d55;font-weight:600}.fail{color:#b3261e;font-weight:600}
.gallery{display:flex;flex-wrap:wrap;gap:10px}.gallery figure{margin:0;max-width:420px}
.gallery img{max-width:420px;max-height:320px;border:1px solid #dde1e8;background:#fff;display:block}
.gallery figcaption{font-size:12px;color:#6a7385;word-break:break-all}
.thumb{max-width:220px;max-height:140px;border:1px solid #dde1e8;background:#fff}
pre{white-space:pre-wrap;word-break:break-word;background:#f0f2f5;padding:8px 10px;border-radius:6px;margin:6px 0;
 font-size:12.5px;max-height:420px;overflow:auto}
.frame{border:1px solid #dde1e8;background:#fff;overflow:hidden;position:relative}
.frame iframe{border:0;transform-origin:0 0;position:absolute;top:0;left:0}
.runs{display:grid;gap:14px}.kv span{margin-right:14px;white-space:nowrap}
details>summary{cursor:pointer;font-weight:600;margin:6px 0}.step{border-left:3px solid #dde1e8;padding-left:8px;margin:6px 0}
.tag{display:inline-block;padding:0 6px;border-radius:4px;background:#eef1f6;font-size:12px;margin-right:4px}
.tabs{display:flex;gap:4px;border-bottom:2px solid #dde1e8;margin:14px 0 4px;position:sticky;top:0;background:#f5f6f8;
 z-index:5;padding-top:6px}
.tabs button{font:inherit;font-weight:600;border:1px solid #dde1e8;border-bottom:0;background:#eef1f6;color:#3a4252;
 padding:6px 14px;border-radius:6px 6px 0 0;cursor:pointer}
.tabs button.active{background:#fff;color:#2457c5;border-color:#2457c5;box-shadow:0 2px 0 #fff}
.hidden{display:none!important}
"""

# One tab per run plus "compare". Elements with data-panel show only under their tab; elements with
# data-run show under "compare" and their own run. The choice is remembered across pages, and a link
# to a rollout (#<run>-r<i>) opens that run's tab.
TAB_SCRIPT = """<script>
(function(){
const tabs=[...document.querySelectorAll('.tabs button')];if(!tabs.length)return;
function show(key){
 if(!tabs.some(b=>b.dataset.key===key))key=tabs[0].dataset.key;
 tabs.forEach(b=>b.classList.toggle('active',b.dataset.key===key));
 document.querySelectorAll('[data-panel]').forEach(e=>e.classList.toggle('hidden',e.dataset.panel!==key));
 document.querySelectorAll('[data-run]').forEach(e=>e.classList.toggle('hidden',key!=='compare'&&e.dataset.run!==key));
 try{localStorage.setItem('visual-agent-report-tab',key)}catch(e){}
}
tabs.forEach(b=>b.addEventListener('click',()=>show(b.dataset.key)));
let key=null;const h=decodeURIComponent(location.hash.slice(1));
const card=h&&document.getElementById(h);if(card&&card.dataset.run)key=card.dataset.run;
if(!key){try{key=localStorage.getItem('visual-agent-report-tab')}catch(e){}}
show(key||tabs[0].dataset.key);if(card)card.scrollIntoView();
})();
</script>"""


@dataclass
class Run:
    label: str
    rows: List[Dict[str, Any]]
    by_task: Dict[str, List[Dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))

    @property
    def slug(self) -> str:
        """URL- and id-safe form of the label (used for media paths, anchors and tabs)."""
        return re.sub(r"[^a-z0-9.]+", "-", self.label.lower()).strip("-")


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def fmt(value: Optional[float], digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def avg(values: Iterable[Optional[float]]) -> Optional[float]:
    kept = [v for v in values if v is not None]
    return mean(kept) if kept else None


def chip(row: Dict[str, Any], href: str = "") -> str:
    if row.get("mask_sample"):
        cls, text = "sm", "mask"
    else:
        score = row.get("pointwise_reward") or 0.0
        cls = (
            "s4" if score >= 0.9 else "s3" if score >= 0.75 else "s2" if score >= 0.5 else "s1" if score > 0 else "s0"
        )
        text = f"{score:.2f}"
    body = f'<span class="chip {cls}" title="rollout {row.get("rollout_index")}">{text}</span>'
    return f'<a href="{href}">{body}</a>' if href else body


def page(title: str, body: str, depth: int = 0) -> str:
    home = f'<p class="muted"><a href="{"../" * depth}index.html">Visual agent report</a></p>' if depth else ""
    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{esc(title)}</title>'
        f'<meta name="viewport" content="width=device-width,initial-scale=1"><style>{CSS}</style></head>'
        f"<body><main>{home}{body}</main>{TAB_SCRIPT}</body></html>"
    )


class ReportBuilder:
    def __init__(self, runs: List[Run], out: Path, path_maps: List[Tuple[str, str]], args: argparse.Namespace):
        self.runs = runs
        self.out = out
        self.path_maps = path_maps
        self.args = args
        self.pool = ThreadPoolExecutor(max_workers=args.workers)
        self.jobs = []
        self.task_sources: Dict[str, Optional[Path]] = {}

    # ---------------------------------------------------------------- paths and media
    def local(self, path: Optional[str]) -> Optional[Path]:
        if not path:
            return None
        for src, dst in self.path_maps:
            if path.startswith(src):
                return Path(dst) / path[len(src) :].lstrip("/")
        return Path(path)

    def image(self, source: Any, rel_dest: str, max_width: int) -> str:
        """Queue a JPEG copy of `source` (path or PNG bytes) under the report; return its relative path."""
        rel = rel_dest.rsplit(".", 1)[0] + ".jpg"
        dest = self.out / rel
        if not dest.exists():
            self.jobs.append(self.pool.submit(_write_jpeg, source, dest, max_width))
        return rel

    def figure(self, source: Any, rel_dest: str, caption: str, depth: int, max_width: int = 1600) -> str:
        rel = "../" * depth + self.image(source, rel_dest, max_width)
        return f'<figure><a href="{rel}"><img loading="lazy" src="{rel}"></a><figcaption>{esc(caption)}</figcaption></figure>'

    def task_data_dir(self, row: Dict[str, Any]) -> Optional[Path]:
        """The data folder (current or an archived version) whose task matches this row's prompt."""
        task_id = row["task_id"]
        if task_id not in self.task_sources:
            self.task_sources[task_id] = None
            candidates = [DATA_DIR] + sorted((DATA_DIR / "archive").glob("*"))
            for data_dir in candidates:
                for tasks_file in sorted((data_dir / "tasks").glob("*.json")):
                    for task in json.loads(tasks_file.read_text()):
                        if task.get("task_id") == task_id and task.get("prompt") == row.get("prompt"):
                            self.task_sources[task_id] = data_dir
                            break
                if self.task_sources[task_id]:
                    break
        return self.task_sources[task_id]

    def extract_artifact(self, row: Dict[str, Any], rel_dir: str) -> Optional[str]:
        grading_dir = self.local(row.get("grading_dir"))
        tgz = grading_dir / "artifact.tar.gz" if grading_dir else None
        if not tgz or not tgz.exists():
            return None
        dest = self.out / rel_dir
        if not dest.exists():
            dest.mkdir(parents=True)
            vendored = set(row.get("vendor") or [])
            with tarfile.open(tgz) as tar:
                members = []
                total = 0
                for member in tar.getmembers():
                    name = member.name.lstrip("./")
                    parts = name.split("/")
                    # Vendored libraries are shared by every artifact: link one copy instead.
                    if len(parts) >= 2 and parts[0] == "vendor" and parts[1] in vendored:
                        continue
                    total += member.size
                    if total > MAX_ARTIFACT_BYTES:
                        break
                    members.append(member)
                tar.extractall(dest, members=members, filter="data")
            for lib in vendored:
                shared = self.out / "_vendor" / lib
                if not shared.exists():
                    _copy_tree(DATA_DIR / "vendor" / lib, shared)
                (dest / "vendor").mkdir(exist_ok=True)
                link = dest / "vendor" / lib
                if not link.exists():
                    link.symlink_to(os.path.relpath(shared, link.parent))
        return rel_dir

    # ---------------------------------------------------------------- page sections
    def reference_section(self, row: Dict[str, Any]) -> str:
        data_dir = self.task_data_dir(row)
        names = row.get("reference_images") or []
        if not names:
            return ""
        if data_dir is None:
            return '<p class="muted">Reference images: task definition changed since this run; not shown.</p>'
        figures = []
        for name in names:
            path = data_dir / "assets" / row["task_id"] / name
            if path.exists():
                figures.append(self.figure(path, f"references/{row['task_id']}/{name}", name, depth=1))
        return f'<h3>Reference images</h3><div class="gallery">{"".join(figures)}</div>'

    def rubric_section(self, row: Dict[str, Any]) -> str:
        items = "".join(
            f"<tr><td>{esc(i.get('id'))}</td><td>{esc(i.get('type'))}</td>"
            f"<td class='num'>{esc(i.get('weight', 1))}</td><td>{esc(i.get('criterion'))}</td></tr>"
            for i in row.get("rubric") or []
        )
        return (
            "<details><summary>Rubric</summary><table><tr><th>id</th><th>type</th><th>weight</th>"
            f"<th>criterion</th></tr>{items}</table></details>"
        )

    def live_artifact(self, row: Dict[str, Any], rel_dir: Optional[str]) -> str:
        if not rel_dir:
            return '<p class="muted">No artifact was collected.</p>'
        artifact = row.get("artifact") or {}
        entry = artifact.get("entry") or "index.html"
        src = f"../{rel_dir}/{entry}"
        if not (self.out / rel_dir / entry).exists():
            return (
                f'<p class="muted">Artifact folder has no {esc(entry)}. <a href="../{rel_dir}/">Browse files</a></p>'
            )
        if artifact.get("kind") == "video":
            return f'<video controls preload="none" style="max-width:720px;width:100%" src="{src}"></video>'
        viewport = row.get("viewport") or {"width": 1280, "height": 800}
        width, height = int(viewport.get("width", 1280)), int(viewport.get("height", 800))
        if artifact.get("kind") == "slides":
            height = 720
            width = 1280
        scale = min(1.0, 720 / width)
        return (
            f'<details><summary>Live artifact (<a href="{src}" target="_blank">open in a new tab</a>)</summary>'
            f'<div class="frame" style="width:{int(width * scale)}px;height:{int(height * scale)}px">'
            f'<iframe loading="lazy" data-src="{src}" width="{width}" height="{height}" '
            f'style="transform:scale({scale:.4f})" sandbox="allow-scripts allow-same-origin allow-pointer-lock">'
            "</iframe></div></details>"
        )

    def verdict_table(self, row: Dict[str, Any]) -> str:
        results = row.get("rubric_results") or []
        if not results:
            return ""
        criteria = {i.get("id"): i.get("criterion") for i in row.get("rubric") or []}
        lines = []
        for item in results:
            passed = item.get("passed")
            mark = '<span class="pass">pass</span>' if passed else '<span class="fail">fail</span>'
            lines.append(
                f"<tr><td>{esc(item.get('id'))}</td><td>{mark}</td><td>{esc(item.get('source'))}</td>"
                f"<td>{esc(criteria.get(item.get('id'), ''))}</td><td>{esc(item.get('evidence'))}</td></tr>"
            )
        return (
            "<table><tr><th>item</th><th>verdict</th><th>source</th><th>criterion</th><th>evidence</th></tr>"
            + "".join(lines)
            + "</table>"
        )

    def export_images(self, export_path: Optional[Path], limit: int, last: bool) -> List[Tuple[str, bytes]]:
        """Images the OpenCode session read (tool attachments), as (file path, PNG/JPEG bytes)."""
        if limit <= 0 or not export_path or not export_path.exists():
            return []
        try:
            export = json.loads(export_path.read_text())
        except (OSError, json.JSONDecodeError):
            return []
        images = []
        for message in export.get("messages") or []:
            for part in message.get("parts") or []:
                state = part.get("state") or {}
                for attachment in state.get("attachments") or []:
                    url = attachment.get("url") or ""
                    if url.startswith("data:image/") and ";base64," in url:
                        caption = (state.get("input") or {}).get("filePath") or attachment.get("filename") or ""
                        images.append((caption, base64.b64decode(url.split(",", 1)[1])))
        return images[-limit:] if last else images[:limit]

    def trace(self, row: Dict[str, Any]) -> str:
        output = (row.get("response") or {}).get("output") or []
        steps = []
        calls = 0
        for item in output:
            kind = item.get("type")
            if kind == "function_call":
                calls += 1
                steps.append(
                    f'<div class="step"><span class="tag">{esc(item.get("name"))}</span>{_call_summary(item)}</div>'
                )
            elif kind == "function_call_output":
                text = str(item.get("output") or "")
                steps.append(f'<div class="step"><pre>{esc(_clip(text, 600))}</pre></div>')
            elif kind == "message":
                texts = [
                    p.get("text") or ""
                    for p in item.get("content") or []
                    if p.get("type") in ("output_text", "input_text")
                ]
                images = sum(1 for p in item.get("content") or [] if p.get("type") == "input_image")
                if item.get("role") == "assistant" and any(texts):
                    steps.append(
                        f'<div class="step"><b>assistant</b><pre>{esc(_clip(" ".join(texts), 2000))}</pre></div>'
                    )
                elif images:
                    steps.append(f'<div class="step muted">{images} image(s) attached to the agent context</div>')
        if not steps:
            return ""
        return f"<details><summary>Agent trace ({calls} tool calls)</summary>{''.join(steps)}</details>"

    def rollout_card(self, run: Run, row: Dict[str, Any]) -> str:
        task_id, index = row["task_id"], row.get("rollout_index")
        slot = f"{run.slug}/{task_id}/r{index}"
        grading_dir = self.local(row.get("grading_dir"))
        usage = (row.get("response") or {}).get("usage") or {}
        tier = row.get("groupwise_tier")
        facts = [
            ("pointwise", fmt(row.get("pointwise_reward"))),
            ("reward", fmt(row.get("reward"))),
            ("rubric", fmt(row.get("rubric_score"))),
            ("similarity", fmt(row.get("similarity"))),
            ("gate", "passed" if row.get("gate_passed") else "failed"),
            ("judge", row.get("judge_status")),
            ("groupwise", f"{row.get('groupwise_status')}" + (f" (tier {tier})" if tier is not None else "")),
            ("input tokens", f"{usage.get('input_tokens', 0) / 1e6:.2f}M"),
            ("output tokens", f"{usage.get('output_tokens', 0) / 1e3:.1f}K"),
        ]
        if row.get("mask_sample"):
            facts.append(("masked", row.get("failure_kind") or "yes"))
        parts = [
            f'<div class="card" id="{run.slug}-r{index}" data-run="{run.slug}">'
            f"<h3>{esc(run.label)} · rollout {index} {chip(row)}</h3>",
            '<div class="kv">' + "".join(f"<span><b>{k}</b> {esc(v)}</span>" for k, v in facts) + "</div>",
        ]
        problems = (row.get("gate_reasons") or []) + (row.get("hack_reasons") or [])
        if row.get("failure_reason"):
            problems.append(row["failure_reason"])
        if problems:
            parts.append("<pre>" + esc("\n".join(problems)) + "</pre>")

        renders = sorted((grading_dir / "renders").glob("*.png")) if grading_dir else []
        if renders:
            figures = [self.figure(p, f"media/{slot}/renders/{p.name}", p.name, depth=1) for p in renders]
            parts.append(f'<h3>Grader renders</h3><div class="gallery">{"".join(figures)}</div>')
        parts.append(self.live_artifact(row, self.extract_artifact(row, f"artifacts/{slot}")))

        if row.get("judge_summary"):
            scores = ", ".join(f"{k} {v}" for k, v in (row.get("judge_scores") or {}).items())
            parts.append(
                f"<h3>Judge</h3><p>{esc(row['judge_summary'])}</p>"
                + (f'<p class="muted">{esc(scores)}</p>' if scores else "")
            )
        parts.append(self.verdict_table(row))

        attempts = int(row.get("judge_attempts") or 0)
        if grading_dir and attempts:
            shots = self.export_images(
                grading_dir / f"judge_export_{attempts}.json", self.args.judge_images, last=False
            )
            if shots:
                figures = [
                    self.figure(
                        data, f"media/{slot}/judge/{i:02d}_{_slug(caption)}.png", caption, depth=1, max_width=960
                    )
                    for i, (caption, data) in enumerate(shots)
                ]
                parts.append(
                    f"<details><summary>Screenshots the judge inspected ({len(shots)})</summary>"
                    f'<div class="gallery">{"".join(figures)}</div></details>'
                )
        previews = self.export_images(self.local(row.get("opencode_results_fpath")), self.args.agent_images, last=True)
        if previews:
            figures = [
                self.figure(data, f"media/{slot}/agent/{i:02d}_{_slug(caption)}.png", caption, depth=1, max_width=960)
                for i, (caption, data) in enumerate(previews)
            ]
            parts.append(
                f"<details><summary>Agent's last previews ({len(previews)})</summary>"
                f'<div class="gallery">{"".join(figures)}</div></details>'
            )
        parts.append(self.trace(row))
        parts.append("</div>")
        return "".join(parts)

    def task_page(self, task_id: str) -> str:
        first = next(run.by_task[task_id][0] for run in self.runs if run.by_task.get(task_id))
        header = [
            f"<h1>{esc(first.get('title') or task_id)}</h1>",
            f'<p class="muted">{esc(task_id)} · {esc(first.get("category"))} · {esc(first.get("mode"))} · '
            f"artifact {esc((first.get('artifact') or {}).get('kind'))}</p>",
            f"<details open><summary>Prompt</summary><pre>{esc(first.get('prompt'))}</pre></details>",
            self.reference_section(first),
            self.rubric_section(first),
        ]
        summary = ["<table><tr><th>run</th><th>mean pointwise</th><th>rollouts</th></tr>"]
        cards = []
        for run in self.runs:
            rows = sorted(run.by_task.get(task_id, []), key=lambda r: r.get("rollout_index") or 0)
            if not rows:
                continue
            chips = "".join(chip(r, f"#{run.slug}-r{r.get('rollout_index')}") for r in rows)
            scored = [r.get("pointwise_reward") for r in rows if not r.get("mask_sample")]
            summary.append(
                f"<tr data-run='{run.slug}'><td>{esc(run.label)}</td><td class='num'>{fmt(avg(scored))}</td>"
                f"<td>{chips}</td></tr>"
            )
            cards.extend(self.rollout_card(run, r) for r in rows)
        summary.append("</table>")
        # Live iframes load only when their <details> opens, so a page with 8 artifacts stays light.
        script = (
            "<script>document.querySelectorAll('details').forEach(d=>d.addEventListener('toggle',()=>{"
            "if(d.open)d.querySelectorAll('iframe[data-src]').forEach(f=>{f.src=f.dataset.src;f.removeAttribute('data-src')})}))"
            "</script>"
        )
        return page(
            task_id,
            "".join(header) + "<h2>Rollouts</h2>" + self.tab_bar() + "".join(summary) + "".join(cards) + script,
            depth=1,
        )

    def tab_bar(self) -> str:
        """Compare (with more than one run) plus one tab per run."""
        keys = ([("compare", "Compare")] if len(self.runs) > 1 else []) + [(r.slug, r.label) for r in self.runs]
        buttons = "".join(f'<button data-key="{key}">{esc(label)}</button>' for key, label in keys)
        return f'<div class="tabs">{buttons}</div>'

    def index_page(self) -> str:
        body = [
            "<h1>Visual agent report</h1>",
            '<p class="muted">MiMo-V2.6 §4.2.3 visual agent tasks. <b>pointwise</b> = absolute score '
            "(rubric, and 0.6·similarity + 0.4·rubric for replication); <b>reward</b> adds the groupwise "
            "adjustment. Chips link to rollouts.</p>",
            self.tab_bar(),
        ]
        if len(self.runs) > 1:
            body.append(f'<section data-panel="compare">{self._index_panel(self.runs)}</section>')
        for run in self.runs:
            body.append(f'<section data-panel="{run.slug}">{self._index_panel([run])}</section>')
        return page("Visual agent report", "".join(body))

    def _index_panel(self, runs: List[Run]) -> str:
        """Run summary, per-category means and the task table for `runs`."""
        body = [
            "<h2>Runs</h2><table><tr><th>run</th><th>rollouts</th><th>pointwise</th><th>reward</th><th>rubric</th>"
            "<th>similarity</th><th>gate fail</th><th>judge fail</th><th>masked</th></tr>"
        ]
        for run in runs:
            rows = run.rows
            scored = [r for r in rows if not r.get("mask_sample")]
            body.append(
                f"<tr><td>{esc(run.label)}</td><td class='num'>{len(rows)}</td>"
                f"<td class='num'>{fmt(avg(r.get('pointwise_reward') for r in scored))}</td>"
                f"<td class='num'>{fmt(avg(r.get('reward') for r in scored))}</td>"
                f"<td class='num'>{fmt(avg(r.get('rubric_score') for r in scored))}</td>"
                f"<td class='num'>{fmt(avg(r.get('similarity') for r in scored))}</td>"
                f"<td class='num'>{fmt(avg(0.0 if r.get('gate_passed') else 1.0 for r in rows), 2)}</td>"
                f"<td class='num'>{fmt(avg(1.0 if r.get('judge_status') == 'failed' else 0.0 for r in rows), 2)}</td>"
                f"<td class='num'>{fmt(avg(1.0 if r.get('mask_sample') else 0.0 for r in rows), 2)}</td></tr>"
            )
        body.append("</table>")
        body.append(
            "<h2>By category</h2><table><tr><th>category</th><th>mode</th>"
            + "".join(f"<th>{esc(run.label)}</th>" for run in runs)
            + "</tr>"
        )
        for category in CATEGORIES:
            for mode in ("open_ended", "replication"):
                cells = []
                seen = False
                for run in runs:
                    rows = [r for r in run.rows if r.get("category") == category and r.get("mode") == mode]
                    seen |= bool(rows)
                    value = avg(r.get("pointwise_reward") for r in rows if not r.get("mask_sample"))
                    cells.append(f"<td class='num'>{fmt(value)} <span class='muted'>({len(rows)})</span></td>")
                if seen:
                    body.append(f"<tr><td>{category}</td><td>{mode}</td>{''.join(cells)}</tr>")
        body.append("</table>")
        task_ids = sorted(
            {t for run in runs for t in run.by_task},
            key=lambda t: (_category_order(self._any_row(t)), self._any_row(t).get("mode"), t),
        )
        body.append(
            "<h2>Tasks</h2><table><tr><th>preview</th><th>task</th><th>category</th><th>mode</th>"
            + "".join(f"<th>{esc(run.label)}</th>" for run in runs)
            + "</tr>"
        )
        for task_id in task_ids:
            row = self._any_row(task_id)
            cells = []
            for run in runs:
                rows = sorted(run.by_task.get(task_id, []), key=lambda r: r.get("rollout_index") or 0)
                scored = [r.get("pointwise_reward") for r in rows if not r.get("mask_sample")]
                chips = "".join(chip(r, f"tasks/{task_id}.html#{run.slug}-r{r.get('rollout_index')}") for r in rows)
                cells.append(f"<td><b>{fmt(avg(scored), 2)}</b> {chips}</td>")
            body.append(
                f"<tr><td>{self._thumbnail(task_id, runs)}</td><td><a href='tasks/{task_id}.html'>{esc(task_id)}</a>"
                f"<br><span class='muted'>{esc(row.get('title'))}</span></td><td>{esc(row.get('category'))}</td>"
                f"<td>{esc(row.get('mode'))}</td>{''.join(cells)}</tr>"
            )
        body.append("</table>")
        return "".join(body)

    def _any_row(self, task_id: str) -> Dict[str, Any]:
        return next(run.by_task[task_id][0] for run in self.runs if run.by_task.get(task_id))

    def _thumbnail(self, task_id: str, runs: List[Run]) -> str:
        """Primary render of the best rollout of the first of `runs` that has this task."""
        for run in runs:
            rows = [r for r in run.by_task.get(task_id, []) if not r.get("mask_sample")]
            rows.sort(key=lambda r: r.get("pointwise_reward") or 0.0, reverse=True)
            for row in rows:
                grading_dir = self.local(row.get("grading_dir"))
                if not grading_dir:
                    continue
                renders = [grading_dir / "renders" / n for n in PRIMARY_RENDERS]
                renders += sorted((grading_dir / "renders").glob("*.png"))
                for path in renders:
                    if path.exists():
                        rel = self.image(path, f"media/thumbs/{run.slug}/{task_id}.png", 440)
                        return f'<a href="tasks/{task_id}.html"><img class="thumb" loading="lazy" src="{rel}"></a>'
        return ""

    def build(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "tasks").mkdir(exist_ok=True)
        task_ids = {t for run in self.runs for t in run.by_task}
        for task_id in sorted(task_ids):
            (self.out / "tasks" / f"{task_id}.html").write_text(self.task_page(task_id))
        (self.out / "index.html").write_text(self.index_page())
        for job in self.jobs:
            job.result()
        self.pool.shutdown()


def _write_jpeg(source: Any, dest: Path, max_width: int) -> None:
    image = Image.open(io.BytesIO(source) if isinstance(source, bytes) else source)
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGBA")
        background = Image.new("RGB", image.size, "white")
        background.paste(image, mask=image.getchannel("A"))
        image = background
    else:
        image = image.convert("RGB")
    if image.width > max_width:
        image = image.resize((max_width, round(image.height * max_width / image.width)), Image.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest, "JPEG", quality=85, optimize=True)


def _copy_tree(src: Path, dest: Path) -> None:
    for path in src.rglob("*"):
        target = dest / path.relative_to(src)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"… [{len(text) - limit} more chars]"


def _slug(text: str) -> str:
    name = Path(text).name or "image"
    return (
        "".join(c if c.isalnum() or c in "-_." else "_" for c in name)[:60]
        + "_"
        + hashlib.sha1(text.encode()).hexdigest()[:6]
    )


def _category_order(row: Dict[str, Any]) -> int:
    category = row.get("category")
    return CATEGORIES.index(category) if category in CATEGORIES else len(CATEGORIES)


def _call_summary(item: Dict[str, Any]) -> str:
    try:
        arguments = json.loads(item.get("arguments") or "{}")
    except json.JSONDecodeError:
        return f"<pre>{esc(_clip(str(item.get('arguments')), 400))}</pre>"
    if not isinstance(arguments, dict):
        return f"<pre>{esc(_clip(json.dumps(arguments), 400))}</pre>"
    if "command" in arguments:
        return f"<pre>{esc(_clip(str(arguments['command']), 800))}</pre>"
    if "content" in arguments:
        return f"{esc(arguments.get('filePath'))} <span class='muted'>({len(str(arguments['content']))} chars)</span>"
    if "filePath" in arguments:
        return esc(arguments["filePath"]) + (
            f" <span class='muted'>(edit, {len(str(arguments.get('newString', '')))} chars)</span>"
            if "newString" in arguments
            else ""
        )
    return f"<pre>{esc(_clip(json.dumps(arguments), 400))}</pre>"


def load_run(path: Path, label: str) -> Run:
    run = Run(label=label, rows=[])
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            row.pop("responses_create_params", None)  # the full agent context; the prompt is in `prompt`
            run.rows.append(row)
            run.by_task[row["task_id"]].append(row)
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rollouts", type=Path, nargs="+", help="visual_agent rollouts JSONL file(s)")
    parser.add_argument("--labels", nargs="+", help="one label per rollouts file (default: file stem)")
    parser.add_argument("--out", type=Path, required=True, help="output folder")
    parser.add_argument(
        "--path-map",
        action="append",
        default=[],
        metavar="SRC=DST",
        help="rewrite path prefixes recorded in the rollouts (default /opt/Gym=<repo root>, the eval container mount)",
    )
    parser.add_argument("--judge-images", type=int, default=12, help="judge screenshots per rollout")
    parser.add_argument("--agent-images", type=int, default=4, help="last agent previews per rollout")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    labels = args.labels or [p.stem for p in args.rollouts]
    if len(labels) != len(args.rollouts):
        parser.error("--labels needs one label per rollouts file")
    path_maps = [tuple(m.split("=", 1)) for m in args.path_map] or [("/opt/Gym", str(REPO_ROOT))]
    runs = [load_run(path, label) for path, label in zip(args.rollouts, labels)]
    ReportBuilder(runs, args.out, path_maps, args).build()
    print(f"Wrote {args.out / 'index.html'} ({sum(len(r.rows) for r in runs)} rollouts, {len(runs)} run(s))")
    print(f"View: python -m http.server -d {args.out} 8000")


if __name__ == "__main__":
    main()
