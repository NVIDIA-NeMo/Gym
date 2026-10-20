"""Type 51 rootcause: root cause across microservices (the v2 rootcause track as a family-F type, K3 task text)."""
import hashlib
import json
import math
import os
import re
import threading
from datetime import datetime, timedelta
from pathlib import Path

from .. import common
from . import rc_world as wg

TYPE_ID = "51_rootcause"
TIERS = {"easy": (64, 1800), "medium": (512, 1200), "hard": (4096, 600)}
EPOCH = datetime(2026, 3, 2)
MAX_OUT = 8000
METRICS = ["rps", "error_rate", "latency_p99_ms", "memory_mb", "cpu_pct"]
TIERS_SVC = {0: "edge", 1: "api", 2: "core", 3: "data", 4: "datastore"}
# v2 weights: the chain, which needs the whole estate scanned, carries most of the score; pooled over all tickets.
WEIGHTS = {"root": 0.1, "change": 0.1, "chain_f1": 0.8}
FAIL_MSG = {"503": "503 Service Unavailable (retries=3)", "timeout": "timeout after 3000ms (retries=2)"}
T = wg.T  # K3-written text (text/k3.json)
FORMAT = """INC-1234 | root | <service>
INC-1234 | change | <change id, e.g. chg-0a1b2c>
INC-1234 | edge | <upstream service> -> <downstream service>"""


def build_prompt(n, start, end, k, tickets):
    tools = "\n".join(f"- {name}: {T['tools'][name].replace('{metrics}', ', '.join(METRICS))}"
                       for name in ["list_services", "get_service_info", "query_logs", "query_metrics",
                                    "list_changes", "read_ticket"])
    intro = T["intro"].replace("{n}", str(n)).replace("{start}", start).replace("{end}", end).replace("{k}", str(k))
    rules = "\n".join(f"- {r.lstrip('- ')}" for r in T["rules"])
    return (f"{intro}\n\nTools (offline, read-only):\n{tools}\n\n{rules}\n\n{T['answer_intro']}\n{FORMAT}\n\n"
            f"{T['score']}\n\nTickets: {', '.join(tickets)}\n")


def fmt(t, sec=True):
    return (EPOCH + timedelta(seconds=int(t))).strftime("%Y-%m-%d %H:%M:%S" if sec else "%Y-%m-%d %H:%M")


def parse_time(v, default):
    if v is None or str(v).strip() == "":
        return default
    if isinstance(v, (int, float)):
        return int(v)
    s = str(v).strip().replace("T", " ").rstrip("Z").strip()
    for f in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return int((datetime.strptime(s, f) - EPOCH).total_seconds())
        except ValueError:
            pass
    raise ValueError(f"bad time {v!r}; use 'YYYY-MM-DD HH:MM'")


def h64(*parts):
    return int.from_bytes(hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).digest(), "big")


def generate(n, seed, out_dir):
    g = wg.Gen(n, seed)
    incs = g.incidents()
    g.background_changes()
    g.finalize_changes()
    g.emit_incidents(incs)
    g.emit_noise(incs)
    rng, svcs = g.rng, g.svcs
    base = rng.randint(1000, 8000)
    tickets, gold = [], []
    for j, inc in enumerate(sorted(incs, key=lambda x: x["opened"])):
        tid = f"INC-{base + 37 * j + rng.randint(0, 30)}"
        ts = svcs[inc["ticket_svc"]]
        a, b = inc["window"]
        members = sorted(inc["onset"], key=lambda i: inc["onset"][i])
        loud = sorted((m for m in members if m != inc["ticket_svc"]), key=lambda m: svcs[m]["period"])[:2]
        noisy = [s["name"] for i, s in enumerate(svcs) if i not in inc["onset"] and any(
            e["kind"] == "self" and e["start"] <= inc["alert"] + 900 and e["end"] >= inc["alert"] - 900
            for e in s["ep"])][:3]
        also = [f"{svcs[m]['name']}: error_rate > 2% (fired {fmt(inc['onset'][m] + 180, False)})" for m in loud]
        also += [f"{x}: error budget burn (fired {fmt(inc['alert'] - rng.randint(-600, 900), False)})" for x in noisy]
        rng.shuffle(also)
        d = inc["decoy"]
        title = rng.choice(T["ticket_titles"]).replace("{s}", ts["name"])
        text = "\n".join([
            f"{tid} [SEV{rng.choice([1, 2, 2, 3])}] {title}",
            f"Opened: {fmt(inc['opened'], False)} by pagerduty   Resolved: {fmt(inc['resolved'], False)}",
            f"Alerting service: {ts['name']} (team {ts['team']}, on-call {ts['owners'][0]})",
            f"Alert: {ts['name']} error_rate > 5% since {fmt(inc['alert'], False)}",
            T["also_firing"], *[f"  - {x}" for x in also],
            "On-call note: " + T["oncall_note"].replace("{svc}", d["service"]).replace("{time}", fmt(d["t"], False))
            .replace("{chg}", d["id"]).replace("{summary}", d["summary"]),
            "Customer impact: " + rng.choice(T["impact"]),
        ])
        tickets.append({"id": tid, "text": text})
        assert inc["t_on"] >= a and inc["t_fix"] < b
        gold.append({"ticket": tid, "root": svcs[inc["root"]]["name"], "change": inc["change"]["id"],
                     "cause": inc["cause"], "alerting": ts["name"], "window": [a, b],
                     "decoy_change": d["id"],
                     "edges": [[svcs[inc["parent"][m]]["name"], svcs[m]["name"]] for m in members if m != inc["root"]],
                     "onset": {svcs[m]["name"]: inc["onset"][m] for m in members}})
    world = {"n": n, "seed": seed, "services": svcs, "changes": g.changes, "tickets": tickets}
    prompt = build_prompt(n, fmt(0, False), fmt(wg.SPAN, False), len(tickets), [t["id"] for t in tickets])
    return common.finish(TYPE_ID, n, seed, out_dir, prompt, {"n": n, "seed": seed, "incidents": gold}, TIERS,
                         files={"world.json": json.dumps(world, separators=(",", ":"))})


class World:
    def __init__(self, world_dir):
        with open(os.path.join(world_dir, "world.json")) as f:
            w = json.load(f)
        self.seed = w["seed"]
        self.svcs = w["services"]
        self.by_name = {s["name"]: s for s in self.svcs}
        self.sorted = sorted(self.svcs, key=lambda s: s["name"])
        self.changes = w["changes"]
        self.changes_by_svc = {}
        for c in self.changes:
            self.changes_by_svc.setdefault(c["service"], []).append(c)
        self.tickets = {t["id"]: t["text"] for t in w["tickets"]}

    def svc(self, name):
        s = self.by_name.get(str(name or "").strip())
        if s is None:
            raise KeyError(f"unknown service {name!r}; use list_services for exact names")
        return s

    def episode_lines(self, s, a, b, errors_only=False):
        out = []
        for ei, e in enumerate(s["ep"]):
            if e["end"] < a or e["start"] > b:
                continue
            p = e["period"]
            k0 = max(0, (a - e["start"]) // p - 1)
            k1 = (min(b, e["end"]) - e["start"]) // p
            if e["kind"] == "fail":
                msg = f"call to {e['target']} failed: {FAIL_MSG[e['style']]}"
            else:
                msg = e["msg"]
            for k in range(k0, k1 + 1):
                t = e["start"] + k * p + (h64(self.seed, s["name"], ei, k) % max(1, p // 3) if k else 0)
                if a <= t <= b and t <= e["end"]:
                    out.append((t, e["level"], msg, h64(s["name"], ei, k)))
        if errors_only:
            return out
        for c in self.changes_by_svc.get(s["name"], []):
            if a <= c["t"] <= b:
                if c["type"] == "deploy":
                    msg = f"deployment {c['id']}: rolling out {c['new']} (from {c['old']})"
                elif c["type"] == "config":
                    msg = f"config reloaded ({c['id']}): {c['key']} = {c['new']}"
                else:
                    msg = f"platform event {c['id']}: {c['summary']}"
                out.append((c["t"], "INFO", msg, h64(c["id"])))
        return out

    def background(self, s, a, b):
        # One optional line per minute, derived from a hash so any range is reproducible without storing logs.
        out = []
        p_line = min(0.9, 0.15 + s["rps"] / 1500)
        deps = s["calls"]
        res = s["team"].split("-")[0]
        for m in range(max(0, a // 60), b // 60 + 1):
            r = h64(self.seed, s["name"], m)
            if (r & 0xFFFF) / 65536 >= p_line:
                continue
            t = m * 60 + (r >> 16) % 60
            if not a <= t <= b:
                continue
            kind, x = (r >> 24) % 100, (r >> 32) % 100000
            if s["tier"] == 4:
                if kind < 90:
                    out.append((t, "INFO", f"query ok rows={x % 500} {1 + x % 40}ms", r))
                else:
                    out.append((t, "WARN", f"slow query {800 + x % 2000}ms on table {res}_{x % 7}", r))
            elif kind < 80:
                out.append((t, "INFO", f"GET /v1/{res}/{x} 200 {5 + x % 180}ms", r))
            elif kind < 90:
                out.append((t, "INFO", f"POST /v1/{res} 201 {12 + x % 300}ms", r))
            elif kind < 95 and deps:
                out.append((t, "WARN", f"call to {deps[x % len(deps)]} returned 404 Not Found", r))
            elif kind < 98 or not deps:
                out.append((t, "WARN", f"slow request /v1/{res}/{x} {900 + x % 2500}ms", r))
            else:
                out.append((t, "WARN", f"retry 1/3 for call to {deps[x % len(deps)]}: 429 Too Many Requests", r))
        return out

    def logs(self, s, a, b):
        lines = self.background(s, a, b) + self.episode_lines(s, a, b)
        lines.sort(key=lambda x: (x[0], x[3]))
        pod = h64(s["name"]) % 0xFFFF
        return [f"{fmt(t)} {lvl:<5} {s['name']} pod={s['name']}-{pod:04x}-{tag % 3} trace={tag % 0xFFFFFFFF:08x} {msg}"
                for t, lvl, msg, tag in lines]

    def error_counts(self, s, a, b):
        counts = {}
        for t, _, _, _ in self.episode_lines(s, a, b, errors_only=True):
            counts[t // 60] = counts.get(t // 60, 0) + 1
        return counts

    def metric(self, s, metric, a, b, step):
        m0, m1 = a // 60, b // 60
        errs = self.error_counts(s, m0 * 60, m1 * 60 + 59)
        lat0 = 15 + h64(s["name"], "lat") % 120
        rows = []
        for bs in range(m0, m1 + 1, step):
            vals = []
            for m in range(bs, min(m1 + 1, bs + step)):
                r = h64(self.seed, s["name"], metric, m)
                u = (r % 10000) / 10000
                e = errs.get(m, 0)
                if metric == "error_rate":
                    v = e
                elif metric == "rps":
                    v = s["rps"] * (1 + 0.35 * math.sin(2 * math.pi * (m / 1440 - 0.3))) * (0.95 + 0.1 * u)
                elif metric == "latency_p99_ms":
                    v = lat0 * (0.85 + 0.3 * u) + (1800 + 900 * u if e else 0)
                elif metric == "cpu_pct":
                    v = 20 + 25 * u + (20 if e else 0)
                else:
                    v = self.memory(s, m * 60) * (0.985 + 0.03 * u)
                vals.append(v)
            agg = max(vals) if metric in ("latency_p99_ms", "memory_mb") else sum(vals) / len(vals)
            rows.append((bs * 60, agg))
        return rows

    def memory(self, s, t):
        base, lim = s["mem_base"], s["mem_limit"]
        if "leak" in s:
            t_c, t_on, t_fix, per = s["leak"]
            if t_c <= t < t_on:
                return base + (lim - base) * (t - t_c) / (t_on - t_c)
            if t_on <= t < t_fix:
                return base + (lim - base) * ((t - t_on) % per) / per
        return base


_WORLDS, _LOCK = {}, threading.Lock()


def load(world_dir):
    key = os.path.abspath(world_dir)
    if key not in _WORLDS:
        with _LOCK:
            if key not in _WORLDS:
                _WORLDS[key] = World(key)
    return _WORLDS[key]


def _int(v, default, lo, hi):
    try:
        return max(lo, min(hi, int(v)))
    except (TypeError, ValueError):
        return default


def _cap(text):
    return text if len(text) <= MAX_OUT else text[:MAX_OUT - 60] + "\n...[truncated; narrow the query]"


def _safe(fn):
    def run(args):
        try:
            return _cap(fn(args if isinstance(args, dict) else {}))
        except (KeyError, ValueError) as e:
            return f"Error: {e.args[0] if e.args else e}"
    return run


def tools(world_dir):
    def list_services(args):
        w = load(world_dir)
        team = str(args.get("team") or "").strip()
        rows = [s for s in w.sorted if not team or s["team"] == team]
        off, lim = _int(args.get("offset"), 0, 0, len(rows)), _int(args.get("limit"), 150, 1, 150)
        page = rows[off:off + lim]
        more = f"\n[next: offset={off + lim}]" if off + lim < len(rows) else "\n[end]"
        return (f"{len(rows)} services; showing {off}-{off + len(page) - 1}\n" +
                "\n".join(f"{s['name']} | {TIERS_SVC[s['tier']]} | {s['team']}" for s in page) + more)

    def get_service_info(args):
        s = load(world_dir).svc(args.get("service") or args.get("id"))
        return "\n".join([
            f"service: {s['name']}", f"team: {s['team']}", f"tier: {TIERS_SVC[s['tier']]}", f"kind: {s['kind']}",
            f"language: {s['lang']}", f"owners: {', '.join(s['owners'])} (on-call: {s['owners'][0]})",
            f"declared dependencies: {', '.join(s['deps']) or '(none)'}",
            f"typical load: {s['rps']} rps; memory limit {s['mem_limit']}Mi",
            f"metrics: {', '.join(METRICS)}", f"runbook: https://runbooks.internal/{s['team']}/{s['name']}"])

    def query_logs(args):
        w = load(world_dir)
        s = w.svc(args.get("service"))
        a = max(0, parse_time(args.get("start"), 0))
        b = min(wg.SPAN, parse_time(args.get("end"), wg.SPAN))
        lim = _int(args.get("limit"), 50, 1, 200)
        pat = str(args.get("pattern") or "")
        try:
            rx = re.compile(pat, re.I)
        except re.error:
            rx = re.compile(re.escape(pat), re.I)
        hits = [x for x in w.logs(s, a, b) if rx.search(x)]
        head = f"{len(hits)} lines matched in {s['name']} [{fmt(a)} .. {fmt(b)}]; showing first {min(lim, len(hits))}"
        body = "\n".join(hits[:lim])
        if len(hits) > lim:
            body += "\n[more: re-query with start after the last line shown]"
        return head + ("\n" + body if body else "")

    def query_metrics(args):
        w = load(world_dir)
        s = w.svc(args.get("service"))
        metric = str(args.get("metric") or "").strip()
        if metric not in METRICS:
            raise ValueError(f"unknown metric {metric!r}; one of {METRICS}")
        a = max(0, parse_time(args.get("start"), 0))
        b = min(wg.SPAN - 1, parse_time(args.get("end"), wg.SPAN - 1))
        if b < a:
            raise ValueError("end is before start")
        span = b // 60 - a // 60 + 1
        step = max(_int(args.get("step_min"), 1, 1, 10080), math.ceil(span / 120))
        rows = w.metric(s, metric, a, b, step)
        agg = {"rps": "mean", "cpu_pct": "mean", "error_rate": "mean errors/min"}.get(metric, "max")
        return (f"{s['name']} {metric} step={step}min ({agg} per step)\n" +
                "\n".join(f"{fmt(t, False)} {v:.1f}" for t, v in rows))

    def list_changes(args):
        w = load(world_dir)
        name = str(args.get("service") or "").strip()
        rows = w.changes_by_svc.get(w.svc(name)["name"], []) if name else w.changes
        a, b = parse_time(args.get("start"), 0), parse_time(args.get("end"), wg.SPAN)
        rows = [c for c in rows if a <= c["t"] <= b]
        off, lim = _int(args.get("offset"), 0, 0, len(rows)), _int(args.get("limit"), 60, 1, 60)
        page = rows[off:off + lim]
        more = f"\n[next: offset={off + lim}]" if off + lim < len(rows) else "\n[end]"
        return (f"{len(rows)} changes; showing {off}-{off + len(page) - 1}\n" +
                "\n".join(f"{c['id']} | {fmt(c['t'], False)} | {c['service']} | {c['type']} | {c['summary']} | "
                          f"by {c['author']}" for c in page) + more)

    def read_ticket(args):
        w = load(world_dir)
        tid = str(args.get("ticket_id") or args.get("id") or "").strip().upper()
        if tid not in w.tickets:
            raise KeyError(f"unknown ticket {tid!r}; known: {', '.join(w.tickets)}")
        return w.tickets[tid]

    S, I = {"type": "string"}, {"type": "integer"}
    obj = lambda props, req: {"type": "object", "properties": props, "required": req}
    fns = {"list_services": (obj({"offset": I, "limit": I, "team": S}, []), list_services),
           "get_service_info": (obj({"service": S}, ["service"]), get_service_info),
           "query_logs": (obj({"service": S, "start": S, "end": S, "pattern": S, "limit": I}, ["service"]), query_logs),
           "query_metrics": (obj({"service": S, "metric": S, "start": S, "end": S, "step_min": I},
                                 ["service", "metric"]), query_metrics),
           "list_changes": (obj({"service": S, "start": S, "end": S, "offset": I, "limit": I}, []), list_changes),
           "read_ticket": (obj({"ticket_id": S}, ["ticket_id"]), read_ticket)}
    spec = [(n, T["tools"][n].replace("{metrics}", ", ".join(METRICS)), p, f) for n, (p, f) in fns.items()]
    return [{"name": n, "description": d, "parameters": p, "fn": _safe(f)} for n, d, p, f in spec]


LINE_RE = re.compile(r"^[\s|*`>-]*(INC-\d+)[\s*`]*\|\s*[*`]*(root|change|edge)[*`]*\s*\|\s*(.+?)[\s|*`]*$", re.I)
EDGE_RE = re.compile(r"^[`*]*([\w.-]+?)[`*]*\s*(?:->|→|=>|>)\s*[`*]*([\w.-]+?)[`*]*$")


def parse_answer(answer):
    out = {}
    for line in (answer or "").splitlines():
        m = LINE_RE.match(line)
        if not m:
            continue
        tid, kind, val = m.group(1).upper(), m.group(2).lower(), m.group(3).strip().strip("`*")
        d = out.setdefault(tid, {"root": None, "change": None, "edges": set()})
        if kind == "edge":
            e = EDGE_RE.match(val)
            if e:
                d["edges"].add((e.group(1), e.group(2)))
        else:
            d[kind] = val.split()[0] if val else None
    return out


def score(task, answer):
    with open(task["gold_path"]) as f:
        gold = json.load(f)["incidents"]
    pred = parse_answer(answer)
    w = WEIGHTS
    per = []
    for g in gold:
        p = pred.get(g["ticket"], {"root": None, "change": None, "edges": set()})
        ge = {tuple(e) for e in g["edges"]}
        tp = len(ge & p["edges"])
        prec = tp / len(p["edges"]) if p["edges"] else 0.0
        rec = tp / len(ge) if ge else 1.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        root_ok = p["root"] == g["root"]
        change_ok = (p["change"] or "").lower() == g["change"]
        s = w["root"] * root_ok + w["change"] * change_ok + w["chain_f1"] * f1
        per.append({"ticket": g["ticket"], "score": round(s, 4), "root_ok": root_ok, "change_ok": change_ok,
                    "chain_precision": round(prec, 4), "chain_recall": round(rec, 4), "chain_f1": round(f1, 4),
                    "n_edges": len(ge)})
    k = len(gold)
    # chain F1 pooled over every edge of every ticket (each member of every incident counts once)
    g_all = {(x["ticket"], *e) for x in gold for e in x["edges"]}
    p_all = {(t, *e) for t, p in pred.items() for e in p["edges"]}
    pooled = common.f1(p_all, g_all)[0]
    total = k * (w["root"] * sum(x["root_ok"] for x in per) / k + w["change"] * sum(x["change_ok"] for x in per) / k
                 + w["chain_f1"] * pooled)
    return {"score": total / k, "score_partial": total / k, "score_exact": float(total / k > 1 - 1e-9), "n": k, "n_correct": sum(x["root_ok"] and x["change_ok"] for x in per),
            "root_acc": sum(x["root_ok"] for x in per) / k, "change_acc": sum(x["change_ok"] for x in per) / k,
            "chain_f1": sum(x["chain_f1"] for x in per) / k,
            "precision": sum(x["chain_precision"] for x in per) / k, "recall": sum(x["chain_recall"] for x in per) / k,
            "per_ticket": per}


UNIT_RE = re.compile(r"UNIT[ \t*`]*\|[ \t*`]*(MEMBER|SELF|NO)\b[ \t*`]*\|?[ \t*`]*([\w-]*)", re.I)


def unit_label(task, ticket, service):
    # Locally decidable label of one (service, ticket) unit: MEMBER <upstream>, SELF <culprit change or none>, NO.
    with open(task["gold_path"]) as f:
        g = next(x for x in json.load(f)["incidents"] if x["ticket"] == ticket)
    if service == g["root"]:
        return ("SELF", g["change"])
    for up, down in g["edges"]:
        if down == service:
            return ("MEMBER", up)
    w = load(task["world_dir"])
    a, b = g["window"]
    own = [m for _, _, m, _ in w.episode_lines(w.svc(service), a, b, errors_only=True) if not m.startswith("call to")]
    return ("SELF", "none") if own else ("NO", "")


def parse_unit(text):
    # The last "UNIT | LABEL | value" line wins; None if there is none.
    found = None
    for m in UNIT_RE.finditer(text or ""):
        label = m.group(1).upper()
        found = (label, "" if label == "NO" else m.group(2))
    return found
