# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One self-contained HTML page with every rollout's timeline: an index table, a selector and an SVG drawn by a
small inline script from embedded data. Nothing external is loaded, so the page opens from a Lustre path."""

from __future__ import annotations

import html
import json
from typing import Any, Optional

from ng_latency.derive import coverage_intervals


PHASE_SPANS = (
    "seed_session",
    "connect_sandbox",
    "start_sandbox",
    "agent_setup",
    "agent_execute",
    "agent_sandbox_exec",
    "agent_export",
    "agent_observations_snapshot",
    "verify",
    "sandbox_stop",
    "agent",
)
HEADLINE = (
    "e2e_rollout_time",
    "e2e_agent_time",
    "pre_agent_time",
    "post_agent_time",
    "model_union_time",
    "tool_union_time",
    "unexplained_no_model_tool_time",
    "unexplained_fraction",
    "per_step_unexplained_time",
    "critical_path_duration",
    "cp_model_fraction",
    "cp_tool_fraction",
    "parallel_execution_time",
    "agent_steps",
    "model_calls",
    "tool_calls",
    "compactions",
)
_TOKEN_KEYS = ("input", "output", "reasoning", "cached", "tool_args", "tool_output")
_WORK_SPANS = (
    "model_call",
    "tool_call",
    "tool_batch",
    "subagent",
    "model_step",
    "model_call_attempt",
    "model_call_server",
)
_CHILD_SPANS = ("model_call", "tool_call", "subagent", "model_step")
_INDEX_COLUMNS = (
    "replica",
    "rollout_id",
    "task_id",
    "agent_kind",
    "success",
    "e2e_agent_time",
    "model_calls",
    "tool_calls",
    "unexplained_fraction",
    "flags",
)


def _r(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(float(value), 6)


def _tokens_text(attrs: dict[str, Any]) -> str:
    tokens = attrs.get("tokens") or {}
    return " ".join(f"{k}={tokens[k]}" for k in _TOKEN_KEYS if tokens.get(k) is not None)


def _placement(record: dict[str, Any]) -> tuple[dict[str, float], str]:
    """Per clock domain: seconds to add to a wall time to land on the rollout-relative axis. Sandbox domains are
    placed with the probe's offset when the agent measured one (its uncertainty is stated), else with the
    offset-plus-minimum-latency estimate, else drawn from their own first span and labelled unplaced."""
    spans = record.get("spans") or []
    rollout = next((s for s in spans if s["name"] == "rollout"), None)
    origin = float(rollout["start_wall_s"]) if rollout else min(float(s["start_wall_s"]) for s in spans)
    node_domain = rollout["clock"]["domain"] if rollout else None
    shifts: dict[str, float] = {}
    notes: list[str] = []
    offsets = record.get("clock_offsets") or {}
    for domain in sorted({s["clock"]["domain"] for s in spans}):
        if domain == node_domain or domain.startswith("node:"):
            shifts[domain] = -origin
            continue
        entry = offsets.get(domain) or {}
        probe, estimate = entry.get("offset_s"), entry.get("offset_plus_min_latency_s")
        if isinstance(probe, (int, float)):
            shifts[domain] = -float(probe) - origin
            uncertainty = entry.get("uncertainty_s")
            tail = f" ± {float(uncertainty):.3f} s" if isinstance(uncertainty, (int, float)) else ""
            notes.append(f"{domain}: placed by the sandbox clock probe ({float(probe):+.3f} s{tail})")
        elif isinstance(estimate, (int, float)):
            shifts[domain] = float(estimate) - origin
            notes.append(
                f"{domain}: placed by offset estimate ({float(estimate):+.3f} s, includes one-way network latency)"
            )
        else:
            first = min(float(s["start_wall_s"]) for s in spans if s["clock"]["domain"] == domain)
            shifts[domain] = -first
            notes.append(f"{domain}: unplaced (no offset estimate); drawn from its own first span")
    return shifts, "; ".join(notes) or "all spans on the node clock"


def _critical_span_ids(spans: list[dict[str, Any]], trial: dict[str, Any]) -> set[str]:
    cp_nodes = set(trial.get("metrics", {}).get("critical_path_node_ids") or [])
    found: set[str] = set()
    for span in spans:
        if span["name"] not in ("model_call", "tool_call", "tool_batch"):
            continue
        if span["name"] == "model_call":
            node_id = f"model:{span['span_id']}"
        else:
            node_id = f"tool:{(span.get('ids') or {}).get('tool_call_id') or span['span_id']}"
        if node_id in cp_nodes:
            found.add(span["span_id"])
    return found


def timeline_items(record: dict[str, Any], trial: dict[str, Any]) -> dict[str, Any]:
    spans = sorted(
        record.get("spans") or [],
        key=lambda s: (s["start_wall_s"], s.get("end_wall_s") or s["start_wall_s"], s["span_id"]),
    )
    agent_domain = (trial.get("clock_domains") or {}).get("agent")
    shifts, note = _placement(record)
    main_session = next(((s.get("ids") or {}).get("invocation_id") for s in spans if s["name"] == "agent"), None)
    cp_span_ids = _critical_span_ids(spans, trial)
    lanes: list[dict[str, Any]] = [
        {"id": "phase", "label": "phases"},
        {"id": "model", "label": "model calls"},
        {"id": "attempts", "label": "request attempts (harness, server)"},
        {"id": "engine", "label": "engine queue/prefill/decode (approximate placement)"},
        {"id": "tool", "label": "tools"},
        {"id": "harness", "label": "harness processing (steps)"},
        {"id": "compaction", "label": "compaction"},
    ]
    items: list[dict[str, Any]] = []

    def place(span: dict[str, Any]) -> tuple[float, float]:
        shift = shifts.get(span["clock"]["domain"], 0.0)
        start = float(span["start_wall_s"]) + shift
        end = span.get("end_wall_s")
        return (start, float(end) + shift) if end is not None else (start, start)

    def add(
        span: dict[str, Any], lane: str, *, name: Optional[str] = None, bounds: Optional[tuple[float, float]] = None
    ) -> None:
        placed = bounds or place(span)
        attrs = span.get("attrs") or {}
        purpose = attrs.get("purpose") or attrs.get("tool") or attrs.get("operation") or attrs.get("reason") or ""
        items.append(
            {
                "n": name or span["name"],
                "l": lane,
                "r": 0,
                "s": _r(placed[0]),
                "e": _r(placed[1]),
                "st": span.get("status", "ok"),
                "p": str(purpose)[:120],
                "t": _tokens_text(attrs),
                "cp": span["span_id"] in cp_span_ids,
            }
        )

    resumed_sessions = {
        entry.get("session") for entry in (trial.get("series") or {}).get("subagents") or [] if entry.get("resumed")
    }
    child_sessions: list[str] = []
    for span in spans:
        ids = span.get("ids") or {}
        session = ids.get("invocation_id")
        is_child = bool(ids.get("parent_invocation_id")) or (
            main_session is not None and session not in (None, main_session) and span["name"] in _CHILD_SPANS
        )
        if span["name"] == "rollout" or span["name"] in PHASE_SPANS:
            add(span, "phase")
        elif span["name"] in _WORK_SPANS:
            if span["name"] in ("model_call_attempt", "model_call_server"):
                add(span, "attempts")  # every request attempt, harness-side and server-side, retries included
                if span["name"] == "model_call_server":
                    # The engine reports durations only: its phases are drawn backwards from the end of the
                    # server span, where the engine's response reached Gym.
                    vllm = (span.get("attrs") or {}).get("vllm") or {}
                    e2e = vllm.get("e2e_s")
                    if isinstance(e2e, (int, float)) and span.get("end_wall_s") is not None:
                        cursor = place(span)[1] - float(e2e)
                        for key in ("queue_s", "prefill_s", "decode_s"):
                            length = vllm.get(key)
                            if isinstance(length, (int, float)) and length > 0:
                                add(
                                    span, "engine", name=key.replace("_s", ""), bounds=(cursor, cursor + float(length))
                                )
                                cursor += float(length)
            elif is_child and session:
                lane = f"subagent:{session}"
                if session not in child_sessions:
                    child_sessions.append(session)
                    suffix = " (resumed)" if session in resumed_sessions else ""
                    lanes.append({"id": lane, "label": f"subagent {session}{suffix}"})
                if span["name"] in ("model_call", "tool_call", "subagent"):
                    add(span, lane)
            elif span["name"] == "model_call":
                add(span, "model")
            elif span["name"] in ("tool_call", "tool_batch"):
                add(span, "tool")
            elif span["name"] == "model_step":
                add(span, "harness")
        elif span["name"] == "compaction":
            add(span, "compaction")
    cov = coverage_intervals(record, str(trial.get("agent_kind")))
    lanes.append({"id": "gap", "label": "unexplained time"})
    shift = shifts.get(agent_domain or "", 0.0)
    for start, end in cov["unexplained"]:
        items.append(
            {
                "n": "unexplained",
                "l": "gap",
                "r": 0,
                "s": _r(start + shift),
                "e": _r(end + shift),
                "st": "unknown",
                "p": "",
                "t": "",
                "cp": False,
            }
        )
    _stack_rows(lanes, items)
    starts = [i["s"] for i in items] or [0.0]
    ends = [i["e"] for i in items] or [1.0]
    title = (
        f"{trial.get('replica') or '.'}/{trial.get('rollout_id')} · {trial.get('task_id')} · {trial.get('agent_kind')}"
    )
    return {
        "title": title,
        "t0": _r(min(starts)),
        "t1": _r(max(ends)),
        "placement_note": note,
        "lanes": lanes,
        "items": items,
        "kv": _kv_rows(trial),
    }


def _stack_rows(lanes: list[dict[str, Any]], items: list[dict[str, Any]]) -> None:
    """Concurrent items of one lane go on separate rows (greedy, by start) so none hides another."""
    for lane in lanes:
        row_ends: list[float] = []
        lane_items = sorted((i for i in items if i["l"] == lane["id"]), key=lambda i: (i["s"], i["e"]))
        for item in lane_items:
            row = next((k for k, end in enumerate(row_ends) if end <= item["s"]), None)
            if row is None:
                row = len(row_ends)
                row_ends.append(item["e"])
            else:
                row_ends[row] = item["e"]
            item["r"] = row
        lane["rows"] = max(1, len(row_ends))


def _kv_rows(trial: dict[str, Any]) -> list[list[str]]:
    metrics = trial.get("metrics") or {}
    rows: list[list[str]] = []
    for name in HEADLINE:
        value = metrics.get(name)
        if isinstance(value, dict):
            value = value.get("sum")
        rows.append([name, "" if value is None else str(value)])
    rows.append(["outcome", json.dumps(trial.get("outcome") or {}, sort_keys=True)])
    rows.append(["flags", ", ".join(trial.get("flags") or []) or "none"])
    return rows


def rollout_key(trial: dict[str, Any]) -> str:
    return f"{trial.get('replica') or '.'}/{trial.get('rollout_id')}"


_STYLE = (
    "body{font:13px/1.4 system-ui,sans-serif;margin:16px;color:#222}table.kv{border-collapse:collapse;margin:8px 0}"
    "table.kv td{padding:2px 10px;border-bottom:1px solid #eee}"
    "#chart{border:1px solid #ccc;overflow:hidden;position:relative}"
    "#tip{position:fixed;pointer-events:none;background:#fff;border:1px solid #888;padding:6px;font-size:12px;"
    "display:none;max-width:480px}"
    ".note{color:#555}.flag{color:#a00;font-weight:600}svg text{font-size:11px}"
    "table#trials{border-collapse:collapse}#trials th,#trials td{padding:3px 8px;border-bottom:1px solid #ddd;text-align:left}"
    "#trials th{cursor:pointer;background:#f5f5f5}#picker{font:13px system-ui,sans-serif;margin:8px 0;max-width:900px}"
)

_SCRIPT = r"""
(function(){
const ALL=JSON.parse(document.getElementById('data').textContent);
const C={phase:'#b0bec5',model:'#1976d2',attempts:'#5c6bc0',engine:'#90caf9',tool:'#43a047',harness:'#9e9d24',compaction:'#fb8c00',gap:'#e53935'};
const W=1400,L=190,H=24,TOP=28;let D=null,t0=0,t1=1,tops=[],bottom=TOP;
const chart=document.getElementById('chart'),tip=document.getElementById('tip'),picker=document.getElementById('picker');
const title=document.getElementById('rtitle'),note=document.getElementById('rnote'),kv=document.getElementById('kv');
function esc(v){return String(v).replace(/&/g,'&amp;').replace(/</g,'&lt;');}
function x(t){return L+(t-t0)/(t1-t0)*(W-L-10);}
function color(it){if(it.l.startsWith('subagent'))return it.n==='model_call'?'#7b1fa2':'#8e24aa';if(it.l==='engine'){return it.n==='queue'?'#ef9a9a':it.n==='prefill'?'#64b5f6':'#bbdefb';}return C[it.l]||'#999';}
function niceStep(raw){const p=Math.pow(10,Math.floor(Math.log10(raw)));const m=raw/p;return (m<1.5?1:m<3.5?2:m<7.5?5:10)*p;}
function render(){if(!D)return;let s='<svg xmlns="http://www.w3.org/2000/svg" width="'+W+'" height="'+(bottom+10)+'">';
 const step=niceStep((t1-t0)/8);for(let t=Math.ceil(t0/step)*step;t<=t1;t+=step){const px=x(t);s+='<line x1="'+px+'" y1="'+TOP+'" x2="'+px+'" y2="'+bottom+'" stroke="#eee"/><text x="'+(px+2)+'" y="'+(TOP-8)+'" fill="#555">'+t.toFixed(step<1?2:0)+'s</text>';}
 D.lanes.forEach(function(ln,i){const y=tops[i],h=H*(ln.rows||1);s+='<text x="4" y="'+(y+16)+'">'+esc(ln.label)+'</text><line x1="'+L+'" y1="'+(y+h)+'" x2="'+W+'" y2="'+(y+h)+'" stroke="#f3f3f3"/>';});
 D.items.forEach(function(it,idx){const li=D.lanes.findIndex(function(l){return l.id===it.l;});if(li<0)return;const y=tops[li]+(it.r||0)*H+3;const x0=x(it.s),x1=Math.max(x(it.e),x0+1.5);if(x1<L||x0>W)return;
  s+='<rect data-i="'+idx+'" x="'+Math.max(L,x0)+'" y="'+y+'" width="'+(Math.min(W,x1)-Math.max(L,x0))+'" height="'+(H-6)+'" fill="'+color(it)+'" fill-opacity="'+(it.st==='ok'||it.st==='unknown'?0.85:0.4)+'"'+(it.cp?' stroke="#000" stroke-width="1.5"':it.st!=='ok'&&it.st!=='unknown'?' stroke="#c62828" stroke-dasharray="3 2"':'')+'/>';});
 s+='</svg>';chart.innerHTML=s;}
function select(key){D=ALL[key];if(!D)return;t0=D.t0;t1=Math.max(D.t1,D.t0+1e-3);tops=[];bottom=TOP;D.lanes.forEach(function(ln){tops.push(bottom);bottom+=H*(ln.rows||1);});
 title.textContent=D.title;note.textContent='Clock placement: '+D.placement_note+'. Wheel to zoom, drag to pan, hover for details; black outline = critical path, dashed red = non-ok status.';
 kv.innerHTML=D.kv.map(function(r){return '<tr><td>'+esc(r[0])+'</td><td'+(r[0]==='flags'&&r[1]!=='none'?' class="flag"':'')+'>'+esc(r[1])+'</td></tr>';}).join('');
 picker.value=key;if(location.hash!=='#'+key)history.replaceState(null,'','#'+key);render();}
Object.keys(ALL).forEach(function(k){const o=document.createElement('option');o.value=k;o.textContent=ALL[k].title;picker.appendChild(o);});
picker.addEventListener('change',function(){select(picker.value);});
window.addEventListener('hashchange',function(){const k=decodeURIComponent(location.hash.slice(1));if(ALL[k])select(k);});
chart.addEventListener('wheel',function(ev){ev.preventDefault();const r=chart.getBoundingClientRect();const f=(ev.clientX-r.left-L)/(W-L-10);const at=t0+f*(t1-t0);const k=ev.deltaY>0?1.25:0.8;t0=at-(at-t0)*k;t1=at+(t1-at)*k;render();},{passive:false});
let drag=null;chart.addEventListener('mousedown',function(ev){drag={x:ev.clientX,t0:t0,t1:t1};});
window.addEventListener('mouseup',function(){drag=null;});
window.addEventListener('mousemove',function(ev){if(drag){const dt=(ev.clientX-drag.x)/(W-L-10)*(drag.t1-drag.t0);t0=drag.t0-dt;t1=drag.t1-dt;render();return;}
 const el=document.elementFromPoint(ev.clientX,ev.clientY);if(el&&el.tagName==='rect'&&el.dataset.i&&D){const it=D.items[+el.dataset.i];tip.style.display='block';tip.style.left=(ev.clientX+12)+'px';tip.style.top=(ev.clientY+12)+'px';
  tip.innerHTML='<b>'+esc(it.n)+'</b> ['+esc(it.l)+'] '+(it.e-it.s).toFixed(3)+'s<br>start '+it.s.toFixed(3)+'s · status '+esc(it.st)+(it.cp?' · critical path':'')+(it.p?'<br>'+esc(it.p):'')+(it.t?'<br>'+esc(it.t):'');}else{tip.style.display='none';}});
const first=decodeURIComponent(location.hash.slice(1));select(ALL[first]?first:Object.keys(ALL)[0]);})();
"""

_INDEX_SCRIPT = (
    "function sortBy(i){const t=document.getElementById('trials');const r=Array.from(t.tBodies[0].rows);"
    "const asc=t.dataset.col!==String(i)||t.dataset.dir!=='asc';t.dataset.col=i;t.dataset.dir=asc?'asc':'desc';"
    "r.sort(function(a,b){const x=a.cells[i].textContent,y=b.cells[i].textContent;"
    "const nx=parseFloat(x),ny=parseFloat(y);"
    "const c=(!isNaN(nx)&&!isNaN(ny))?nx-ny:x.localeCompare(y);return asc?c:-c;});"
    "r.forEach(function(row){t.tBodies[0].appendChild(row);});}"
)


def _index_cell(trial: dict[str, Any], column: str) -> str:
    metrics = trial.get("metrics") or {}
    if column in ("replica", "rollout_id", "task_id", "agent_kind"):
        value = trial.get(column)
    elif column == "success":
        value = (trial.get("outcome") or {}).get("success")
    elif column == "flags":
        value = ", ".join(trial.get("flags") or [])
    else:
        value = metrics.get(column)
    text = html.escape("" if value is None else str(value))
    if column == "rollout_id":
        text = f'<a href="#{html.escape(rollout_key(trial))}">{text}</a>'
    return f"<td>{text}</td>"


def render_timelines_html(entries: list[tuple[dict[str, Any], dict[str, Any]]]) -> str:
    """The page for a run: ``entries`` are (timeline record, derived trial) pairs; the selector and the index table
    pick a rollout, whose timeline is drawn from the embedded data."""
    data = {rollout_key(trial): timeline_items(record, trial) for record, trial in entries}
    # "<" is escaped so no span label can open a tag or comment inside the data script
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).replace("<", "\\u003c")
    trials = sorted((trial for _, trial in entries), key=lambda t: (str(t.get("replica")), str(t.get("rollout_id"))))
    rows = ["<tr>" + "".join(_index_cell(trial, column) for column in _INDEX_COLUMNS) + "</tr>" for trial in trials]
    header = "".join(f'<th onclick="sortBy({i})">{html.escape(c)}</th>' for i, c in enumerate(_INDEX_COLUMNS))
    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8"><title>Latency timelines</title><style>'
        + _STYLE
        + "</style></head><body><h1>Latency timelines</h1>"
        + f'<p class="note">{len(rows)} rollouts. Pick one below or click a rollout id in the table; the address bar '
        "keeps the selection so a link to one rollout can be shared.</p>"
        '<select id="picker"></select>'
        '<h2 id="rtitle"></h2><table class="kv"><tbody id="kv"></tbody></table>'
        '<p class="note" id="rnote"></p>'
        '<div id="chart"></div><div id="tip"></div>'
        "<h2>All rollouts</h2>"
        '<table id="trials"><thead><tr>' + header + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        '<script id="data" type="application/json">' + payload + "</script>"
        "<script>" + _INDEX_SCRIPT + _SCRIPT + "</script></body></html>"
    )
