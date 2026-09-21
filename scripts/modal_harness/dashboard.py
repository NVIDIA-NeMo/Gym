# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Web dashboard for NeMo Gym campaigns running on Modal.

Serves from the same app as the runner and reads the same Volume, so it reports what the
runner actually wrote rather than inferring state from process inspection. That distinction
is the reason this exists: a container that dies leaves its last status behind, and "stopped
at 6,298 rows at 20:01" is a different fact from "no process found", which is how a
deliberate stop gets misread as a crash.

Pages:
  /                         every campaign on the Volume, with coverage and state
  /run/{ns}/{slug}          one run: progress, status, recent rollouts
  /run/{ns}/{slug}/rollout/{i}   one trajectory -- the messages the model actually saw
  /run/{ns}/{slug}/logs     server log tail
  /run/{ns}/{slug}/export   download the raw rollouts JSONL
  /api/runs                 the same listing as JSON

Auth is HTTP Basic against a Modal secret. It is a real gate, not a decoration, but it is
the weakest useful one -- single shared credential, no sessions, no audit. Fine for a
private dashboard over HTTPS; not something to put user data behind.
"""

from __future__ import annotations

import html
import json
import os
import secrets
import time
from typing import Any, Optional

import modal
from scripts.modal_harness.campaign import (
    RESULTS_ROOT,
    RESULTS_VOLUME,
    app,
)


DASHBOARD_SECRET = "nemo-gym-dashboard-auth"

WEB_IMAGE = modal.Image.debian_slim(python_version="3.13").pip_install("fastapi[standard]==0.121.1", "jinja2==3.1.6")

CSS = """
:root{--bg:#0b0d10;--panel:#14171c;--line:#242a33;--fg:#e6e9ef;--dim:#98a2b3;--accent:#6ea8fe;
--ok:#3ddc97;--warn:#f0b429;--bad:#f2637e}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
header{padding:16px 20px;border-bottom:1px solid var(--line);display:flex;gap:16px;align-items:baseline;flex-wrap:wrap}
h1{font-size:16px;margin:0;font-weight:650;letter-spacing:.2px}
.muted{color:var(--dim);font-size:12px}
main{padding:20px;max-width:1180px;margin:0 auto}
table{width:100%;border-collapse:collapse;margin:12px 0 24px}
th,td{text-align:left;padding:9px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--dim);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.4px}
tr:hover td{background:#101318}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.stat b{display:block;font-size:21px;font-weight:650;margin-top:2px}
.bar{height:7px;background:#1d222a;border-radius:4px;overflow:hidden;margin-top:8px}
.bar>i{display:block;height:100%;background:var(--accent)}
.pill{display:inline-block;padding:2px 8px;border-radius:99px;font-size:11px;font-weight:650;letter-spacing:.3px}
.s-running{background:rgba(110,168,254,.15);color:var(--accent)}
.s-complete{background:rgba(61,220,151,.15);color:var(--ok)}
.s-settled_short{background:rgba(240,180,41,.15);color:var(--warn)}
.s-stopped,.s-unknown{background:rgba(242,99,126,.15);color:var(--bad)}
pre{background:#0e1116;border:1px solid var(--line);border-radius:8px;padding:12px;overflow:auto;
white-space:pre-wrap;word-break:break-word;font:12px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;max-height:460px}
.msg{border:1px solid var(--line);border-radius:8px;margin-bottom:10px;overflow:hidden}
.msg>.role{padding:6px 10px;background:#11151b;color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.5px}
.msg>.body{padding:10px 12px;white-space:pre-wrap;word-break:break-word;font:12px/1.6 ui-monospace,Menlo,monospace}
.r-system>.role{color:#c0a2ff}.r-user>.role{color:#7fd1ff}.r-assistant>.role{color:var(--ok)}
nav{margin-bottom:14px;font-size:12px}
.stub{border-left:3px solid var(--warn);padding-left:10px;color:var(--dim);font-size:12px;margin:8px 0}
"""


#: Seconds without a new row before a run is called stuck, per namespace. A constant
#: cannot serve both a fast benchmark and one whose cold first row legitimately takes
#: twenty minutes -- AgentDyn's DRIFT is ~55 policy calls per rollout.
STALL_THRESHOLDS: dict[str, float] = {"agentdyn": 2700.0}
DEFAULT_STALL_THRESHOLD = 900.0


def stall_threshold(namespace: str) -> float:
    return STALL_THRESHOLDS.get(namespace, DEFAULT_STALL_THRESHOLD)


def _page(title: str, body: str, *, subtitle: str = "") -> str:
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>"
        f"<header><h1><a href='/'>NeMo Gym · Modal harness</a></h1>"
        f"<span class='muted'>{html.escape(subtitle)}</span></header><main>{body}</main></body></html>"
    )


def _count_rows(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "rb") as handle:
        return sum(1 for _ in handle)


def _read_status(path: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def _discover() -> list[dict[str, Any]]:
    """Every run on the Volume.

    Driven by the rollout files rather than the status files: a run whose container died
    before writing status still has rows on disk, and omitting it would hide exactly the
    case a dashboard is for.
    """
    runs: list[dict[str, Any]] = []
    if not os.path.isdir(RESULTS_ROOT):
        return runs
    for namespace in sorted(os.listdir(RESULTS_ROOT)):
        ns_dir = os.path.join(RESULTS_ROOT, namespace)
        if not os.path.isdir(ns_dir):
            continue
        for name in sorted(os.listdir(ns_dir)):
            if not name.endswith(".jsonl") or name.endswith(".status.json"):
                continue
            if "_failures" in name or "_materialized_inputs" in name:
                continue
            slug = name[: -len(".jsonl")]
            path = os.path.join(ns_dir, name)
            status = _read_status(os.path.join(ns_dir, f"{slug}.status.json"))
            landed = _count_rows(path)
            expected = int(status.get("expected") or 0)
            runs.append(
                {
                    "namespace": namespace,
                    "slug": slug,
                    "path": path,
                    "landed": landed,
                    "expected": expected,
                    "missing": max(0, expected - landed) if expected else 0,
                    "state": status.get("state", "unknown"),
                    "model": status.get("model", ""),
                    "attempt": status.get("attempt"),
                    "concurrency": status.get("concurrency"),
                    "updated_at": status.get("updated_at") or os.path.getmtime(path),
                    "git_ref": status.get("git_ref", ""),
                    "log": status.get("log", os.path.join(ns_dir, f"{slug}.servers.log")),
                    # Seconds since rows last landed. Derived from the rollout file's mtime
                    # rather than the status heartbeat: a hung rollout keeps the container
                    # healthy and heartbeating, so "running" and a fresh timestamp say
                    # nothing about whether work is happening. Only new rows do.
                    # How long THIS container has been alive. A slug outlives its
                    # containers, so `stalled_for` alone cannot distinguish "stuck" from
                    # "respawned and still starting" -- the latter inherits the whole
                    # death-and-restart gap from its predecessor. The pair is unambiguous
                    # where either alone is not.
                    "started_at": status.get("started_at", 0.0),
                    "container_age": (
                        time.time() - float(status.get("started_at") or 0) if status.get("started_at") else 0.0
                    ),
                    "heartbeat_age": time.time() - float(status.get("updated_at") or 0),
                    "stalled_for": time.time()
                    - max(
                        float(status.get("last_progress_at") or 0),
                        os.path.getmtime(path),
                    ),
                }
            )
    runs.sort(key=lambda r: r["updated_at"], reverse=True)
    return runs


def _ago(ts: Optional[float]) -> str:
    if not ts:
        return "-"
    delta = max(0, int(time.time() - ts))
    if delta < 90:
        return f"{delta}s ago"
    if delta < 5400:
        return f"{delta // 60}m ago"
    return f"{delta // 3600}h {delta % 3600 // 60}m ago"


def _tail(path: str, limit: int = 400) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as handle:
        data = handle.read()[-262144:]
    return "\n".join(data.decode("utf-8", errors="replace").splitlines()[-limit:])


def _read_rollout(path: str, index: int) -> Optional[dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        for position, line in enumerate(handle):
            if position == index:
                try:
                    return json.loads(line)
                except ValueError:
                    return None
    return None


@app.function(
    image=WEB_IMAGE,
    volumes={RESULTS_ROOT: RESULTS_VOLUME},
    secrets=[modal.Secret.from_name(DASHBOARD_SECRET, environment_name="FDR")],
    min_containers=1,
    timeout=60 * 60,
)
@modal.asgi_app()
def dashboard():
    from fastapi import Depends, FastAPI, HTTPException
    from fastapi import status as http_status
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
    from fastapi.security import HTTPBasic, HTTPBasicCredentials

    api = FastAPI(title="NeMo Gym harness")
    basic = HTTPBasic()

    def require_auth(creds: HTTPBasicCredentials = Depends(basic)) -> str:
        expected_user = os.environ.get("DASHBOARD_USER", "")
        expected_pass = os.environ.get("DASHBOARD_PASSWORD", "")
        # compare_digest on both, always, so a wrong username and a wrong password take the
        # same time -- a short-circuit here leaks which half was right.
        user_ok = secrets.compare_digest(creds.username, expected_user)
        pass_ok = secrets.compare_digest(creds.password, expected_pass)
        if not (user_ok and pass_ok and expected_user and expected_pass):
            raise HTTPException(
                status_code=http_status.HTTP_401_UNAUTHORIZED,
                detail="unauthorized",
                headers={"WWW-Authenticate": "Basic"},
            )
        return creds.username

    def fresh() -> None:
        # Another container writes these files; without a reload this one serves whatever
        # it had at mount time, which on a live run is always stale.
        RESULTS_VOLUME.reload()

    @api.get("/", response_class=HTMLResponse)
    def index(_: str = Depends(require_auth)):
        fresh()
        runs = _discover()
        if not runs:
            return _page("NeMo Gym harness", "<div class='card'>No runs on the volume yet.</div>")

        total_landed = sum(r["landed"] for r in runs)
        total_expected = sum(r["expected"] for r in runs)
        active = sum(1 for r in runs if r["state"] == "running")
        stats = (
            "<div class='grid'>"
            f"<div class='stat'><span class='muted'>Runs</span><b>{len(runs)}</b></div>"
            f"<div class='stat'><span class='muted'>Active</span><b>{active}</b></div>"
            f"<div class='stat'><span class='muted'>Rollouts</span><b>{total_landed:,}</b></div>"
            f"<div class='stat'><span class='muted'>Expected</span><b>{total_expected:,}</b></div>"
            "</div>"
        )

        rows = []
        for run in runs:
            pct = (run["landed"] / run["expected"] * 100) if run["expected"] else 0
            missing = f"<span style='color:var(--bad)'>{run['missing']}</span>" if run["missing"] else "0"
            # A hung rollout keeps its container healthy and heartbeating, so rows-landed
            # is the only external signal that work has stopped.
            # Only flag when THIS container has itself been alive long enough to have
            # produced a row. Without that clause a cell respawned 300s ago inherits its
            # predecessor's silence and reads as stuck for an hour -- which produced two
            # false alarms out of three the first time this column was used in anger.
            age = run["container_age"]
            # Per-namespace: a constant cannot serve a 24s/it defense and one whose cold
            # first row legitimately takes twenty minutes.
            threshold = stall_threshold(run["namespace"])
            stuck = run["state"] == "running" and run["stalled_for"] > threshold and (age == 0 or age > threshold)
            # Heartbeat age separates two failures that look identical in rows-landed and
            # need opposite responses. The publisher is a 45s thread that survives a wedged
            # eval, so a fresh heartbeat with no rows means the eval is stuck (wait, or
            # --force) while a silent publisher means the container is gone (plain
            # relaunch -- the guard respawns it once the entry goes stale).
            beat = time.time() - float(run["updated_at"] or 0)
            kind = "dead" if beat > 300 else "wedged"
            colour = "var(--bad)" if kind == "dead" else "var(--warn)"
            stall = (
                f"<span style='color:{colour}'>{kind} {int(run['stalled_for'] // 60)}m</span>"
                if stuck
                else (
                    f"<span class='muted'>starting {int(age // 60)}m</span>"
                    if run["state"] == "running" and 0 < age <= threshold
                    else "-"
                )
            )
            rows.append(
                f"<tr><td><a href='/run/{html.escape(run['namespace'])}/{html.escape(run['slug'])}'>"
                f"{html.escape(run['slug'])}</a><div class='muted'>{html.escape(run['namespace'])}"
                f" · {html.escape(run['model'] or '-')}</div></td>"
                f"<td><span class='pill s-{html.escape(run['state'])}'>{html.escape(run['state'])}</span></td>"
                f"<td>{run['landed']:,}<span class='muted'> / {run['expected']:,}</span>"
                f"<div class='bar'><i style='width:{min(100, pct):.1f}%'></i></div></td>"
                f"<td>{missing}</td><td>{stall}</td>"
                f"<td class='muted'>{_ago(run['updated_at'])}</td></tr>"
            )
        table = (
            "<table><tr><th>Run</th><th>State</th><th>Progress</th><th>Missing</th>"
            "<th>No rows for</th><th>Updated</th></tr>" + "".join(rows) + "</table>"
        )
        note = (
            "<div class='stub'>Stubbed for now: cost/GPU-hour accounting, alerting, and "
            "cross-run comparison. Live rows, status, trajectories, logs and export are real.</div>"
        )
        return _page("NeMo Gym harness", stats + table + note, subtitle="all campaigns on this volume")

    @api.get("/run/{namespace}/{slug}", response_class=HTMLResponse)
    def run_detail(namespace: str, slug: str, _: str = Depends(require_auth)):
        fresh()
        run = next((r for r in _discover() if r["namespace"] == namespace and r["slug"] == slug), None)
        if run is None:
            raise HTTPException(status_code=404, detail="no such run")

        pct = (run["landed"] / run["expected"] * 100) if run["expected"] else 0
        head = (
            "<nav><a href='/'>&larr; all runs</a></nav>"
            "<div class='grid'>"
            f"<div class='stat'><span class='muted'>State</span><b>"
            f"<span class='pill s-{html.escape(run['state'])}'>{html.escape(run['state'])}</span></b></div>"
            f"<div class='stat'><span class='muted'>Landed</span><b>{run['landed']:,}</b></div>"
            f"<div class='stat'><span class='muted'>Expected</span><b>{run['expected']:,}</b></div>"
            f"<div class='stat'><span class='muted'>Missing</span><b>{run['missing']}</b></div>"
            f"<div class='stat'><span class='muted'>Attempt</span><b>{run['attempt'] or '-'}</b></div>"
            f"<div class='stat'><span class='muted'>Concurrency</span><b>{run['concurrency'] or '-'}</b></div>"
            "</div>"
            f"<div class='bar' style='margin:14px 0 18px'><i style='width:{min(100, pct):.1f}%'></i></div>"
            f"<div class='card'><div class='muted'>model</div>{html.escape(run['model'] or '-')}"
            f"<div class='muted' style='margin-top:8px'>git ref</div>{html.escape(run['git_ref'] or '-')}"
            f"<div class='muted' style='margin-top:8px'>updated</div>{_ago(run['updated_at'])}</div>"
            f"<p><a href='/run/{namespace}/{slug}/logs'>server logs</a> · "
            f"<a href='/run/{namespace}/{slug}/export'>export rollouts (.jsonl)</a> · "
            f"<a href='/run/{namespace}/{slug}/export?kind=log'>export logs</a></p>"
        )

        # Latest rollouts: what the run is doing right now, newest first.
        recent = []
        total = run["landed"]
        for offset in range(1, min(26, total + 1)):
            index = total - offset
            row = _read_rollout(run["path"], index)
            if not row:
                continue
            reward = row.get("reward")
            attack = row.get("attack_success")
            flag = (
                "<span style='color:var(--bad)'>attacked</span>"
                if attack
                else "<span style='color:var(--ok)'>resisted</span>"
                if attack is False
                else "<span class='muted'>-</span>"
            )
            recent.append(
                f"<tr><td><a href='/run/{namespace}/{slug}/rollout/{index}'>#{index}</a></td>"
                f"<td class='muted'>{html.escape(str(row.get('condition', ''))[:62])}</td>"
                f"<td>{flag}</td>"
                f"<td>{'' if reward is None else f'{float(reward):.2f}'}</td>"
                f"<td class='muted'>{html.escape(str(row.get('workflow_parse_path', '')))}</td></tr>"
            )
        table = (
            "<h3 style='font-size:13px;color:var(--dim);text-transform:uppercase;letter-spacing:.4px'>"
            "Latest rollouts</h3><table>"
            "<tr><th>#</th><th>Condition</th><th>Outcome</th><th>Reward</th><th>Plan parse</th></tr>"
            + ("".join(recent) or "<tr><td colspan=5 class='muted'>no rows yet</td></tr>")
            + "</table>"
        )
        return _page(f"{slug} · NeMo Gym", head + table, subtitle=f"{namespace}/{slug}")

    @api.get("/run/{namespace}/{slug}/rollout/{index}", response_class=HTMLResponse)
    def rollout(namespace: str, slug: str, index: int, _: str = Depends(require_auth)):
        fresh()
        path = os.path.join(RESULTS_ROOT, namespace, f"{slug}.jsonl")
        row = _read_rollout(path, index)
        if row is None:
            raise HTTPException(status_code=404, detail="no such rollout")

        meta_keys = (
            "asb_id",
            "condition",
            "reward",
            "attack_success",
            "original_task_success",
            "refused",
            "workflow_failure",
            "workflow_parse_path",
            "plan_attempts",
            "rounds",
            "tool_call_success",
            "attacker_tool_invoked",
        )
        meta = "".join(
            f"<tr><td class='muted'>{html.escape(key)}</td><td>{html.escape(str(row.get(key)))}</td></tr>"
            for key in meta_keys
            if key in row
        )

        # The transcript is the point of this page: ASB's trajectory is its message list,
        # and reading it is how you tell a model that resisted from one that never planned.
        blocks = []
        for message in row.get("messages") or []:
            role = str(message.get("role", "?"))
            content = str(message.get("content", ""))
            blocks.append(
                f"<div class='msg r-{html.escape(role)}'><div class='role'>{html.escape(role)}</div>"
                f"<div class='body'>{html.escape(content)}</div></div>"
            )
        body = (
            f"<nav><a href='/run/{namespace}/{slug}'>&larr; {html.escape(slug)}</a></nav>"
            f"<div class='card'><table>{meta}</table></div>"
            "<h3 style='font-size:13px;color:var(--dim);text-transform:uppercase;letter-spacing:.4px'>"
            f"Trajectory · {len(blocks)} messages</h3>"
            + ("".join(blocks) or "<div class='muted'>no messages recorded</div>")
        )
        return _page(f"rollout #{index}", body, subtitle=f"{namespace}/{slug} · #{index}")

    @api.get("/run/{namespace}/{slug}/logs", response_class=HTMLResponse)
    def logs(namespace: str, slug: str, _: str = Depends(require_auth)):
        fresh()
        path = os.path.join(RESULTS_ROOT, namespace, f"{slug}.servers.log")
        text = _tail(path) or "no log on the volume yet (the run may not have started here)"
        body = (
            f"<nav><a href='/run/{namespace}/{slug}'>&larr; {html.escape(slug)}</a></nav>"
            f"<pre>{html.escape(text)}</pre>"
        )
        return _page(f"logs · {slug}", body, subtitle="last 400 lines")

    @api.get("/run/{namespace}/{slug}/export")
    def export(namespace: str, slug: str, kind: str = "rollouts", _: str = Depends(require_auth)):
        fresh()
        name = f"{slug}.servers.log" if kind == "log" else f"{slug}.jsonl"
        path = os.path.join(RESULTS_ROOT, namespace, name)
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="nothing to export")
        return FileResponse(path, filename=f"{namespace}-{name}", media_type="application/octet-stream")

    @api.get("/api/runs")
    def api_runs(_: str = Depends(require_auth)):
        fresh()
        return JSONResponse([{k: v for k, v in r.items() if k != "path"} for r in _discover()])

    @api.get("/healthz", response_class=PlainTextResponse)
    def healthz():
        return "ok"

    return api
