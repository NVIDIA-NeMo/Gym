import json
import math
import os
import random
import re
import threading
import time
from fractions import Fraction

from pathlib import Path

from .. import common

TYPE_ID = "60_dag"
TIERS = {"easy": (64, 1800), "medium": (256, 1200), "hard": (512, 1200)}
MAX_CHARS = 8000
UNITS = ("req/s", "req/min")
REL_TOL = 0.005  # absorbs off-by-one floors from float slop; every trap moves a value by >= 2%
# Task-facing prose is written by Kimi K3 (text/k3.json, see README). Names and status markers stay in code.
T = common.load_text(Path(__file__).parent)
STATUS_RE = re.compile(r"^\((adopted|proposed, not adopted|reverted)\)\s*")
for _k in ("rerun", "vendor", "amend_factor", "amend_weight", "amend_threshold", "amend_replace"):
    T[_k] = STATUS_RE.sub("", T[_k])
T["benign_log"] = [STATUS_RE.sub("", x) for x in T["benign_log"]]
COMPANY = T["company"]
TEAMS = ["Payments", "Identity", "Search", "Storage", "Messaging", "Billing", "Ledger", "Catalog", "Routing", "Edge",
         "Analytics", "Notifications", "Fraud", "Inventory", "Pricing", "Media", "Telemetry", "Scheduling"]
WORDS_A = ["ledger", "token", "quota", "event", "index", "session", "invoice", "route", "cache", "policy", "profile",
           "media", "ingest", "audit", "metric", "order", "refund", "pricing", "search", "geo", "notify", "batch",
           "stream", "vault", "shard", "replica", "queue", "feature", "rating", "catalog", "fraud", "consent"]
WORDS_B = ["sync", "gateway", "broker", "worker", "relay", "planner", "store", "resolver", "fanout", "compactor",
           "scorer", "router", "ledger", "proxy", "indexer", "collector", "dispatcher", "validator", "mirror", "agent"]
REGIONS = ["us-east", "us-west", "eu-central", "eu-west", "ap-south", "ap-northeast"]
CONVENTIONS = T["conventions"]
PAD = T["pad"]
INCIDENTS = T["incidents"]
BENIGN_LOG = T["benign_log"]


def fill(tmpl, **kw):
    for k, v in kw.items():
        tmpl = tmpl.replace("{" + k + "}", f"{v:,}" if isinstance(v, int) else str(v))
    return tmpl


def _nice(x):
    # Three significant digits, so thresholds and constants read like human-chosen numbers.
    x = float(x)
    if x < 1:
        return 1
    p = 10 ** max(0, int(math.floor(math.log10(x))) - 2)
    return max(1, int(round(x / p)) * p)


def _conv(v, frm, to):
    if frm == to:
        return Fraction(v)
    return Fraction(v) * 60 if (frm, to) == ("req/s", "req/min") else Fraction(v) / 60


def _act(x, act):
    kind, c = act
    return x + c if kind == "add" else x - c if kind == "sub" else x * Fraction(c, 100)


def evaluate(node, vals):
    # node: one entry of gold["services"] (or the parsed equivalent); vals: published values of its inputs.
    if not node["inputs"]:
        return node["measured"]
    xs = [_conv(vals[i["id"]], i["unit"], node["unit"]) for i in node["inputs"]]
    if node["op"] == "sum":
        x = sum(Fraction(i["w"]) * v for i, v in zip(node["inputs"], xs))
    else:
        x = max(xs) if node["op"] == "max" else min(xs)
    c = node["cond"]
    if c:
        ref = next(i for i in node["inputs"] if i["id"] == c["ref"])
        r = Fraction(vals[c["ref"]]) if c["on"] == "own" else _conv(vals[c["ref"]], ref["unit"], node["unit"])
        x = _act(x, c["then"] if r >= c["thr"] else c["else"])
    v = math.floor(x * Fraction(node["factor"], 100))
    if node["cap"] is not None and v > node["cap"]:
        v = node["cap"]
    return max(v, 1)


def _depth(n):
    return max(4, min(12, 6 + round(math.log2(n / 64)))) if n >= 16 else 3


def _level_sizes(n, d):
    # Non-decreasing sizes so round-robin first parents give every non-sink node at least one child.
    base, extra = divmod(n, d)
    return [base + (1 if k >= d - extra else 0) for k in range(d)]


def _act_text(act, unit):
    kind, c = act
    return fill(T["act_" + kind], c=c, unit=unit)


def _make_world(n, seed):
    rng = random.Random(f"dag:{seed}:{n}")
    d = _depth(n)
    sizes = _level_sizes(n, d)
    perm = list(range(n))
    rng.shuffle(perm)  # ids are not in topological order
    levels, k = [], 0
    for s in sizes:
        levels.append([f"S{perm[k + j]:04d}" for j in range(s)])
        k += s
    names, used = {}, set()
    for lev in levels:
        for sid in lev:
            while True:
                nm = f"{rng.choice(WORDS_A)}-{rng.choice(WORDS_B)}"
                nm = nm if nm not in used else f"{nm}-{rng.randint(2, 99)}"
                if nm not in used:
                    used.add(nm)
                    names[sid] = nm
                    break
    svc, vals = {}, {}
    for L, lev in enumerate(levels):
        prev = levels[L - 1][:] if L else []
        rng.shuffle(prev)
        below = [s for lv in levels[:L] for s in lv]
        recent = [s for lv in levels[max(0, L - 2):L] for s in lv]
        for j, sid in enumerate(lev):
            unit = rng.choice(UNITS)
            target = math.exp(rng.uniform(math.log(200), math.log(20000))) * (60 if unit == "req/min" else 1)
            node = {"id": sid, "name": names[sid], "team": rng.choice(TEAMS), "level": L, "unit": unit,
                    "inputs": [], "op": None, "cond": None, "factor": 100, "cap": None, "measured": None,
                    "amend": None}
            if L == 0:
                # The load test may be quoted in the other unit; req/s quotes of a req/min service stay exact.
                node["measured_unit"] = rng.choice(UNITS)
                node["measured"] = _nice(target)
                if unit == "req/min" and node["measured_unit"] == "req/s":
                    node["measured"] = _nice(target / 60) * 60
                _source_amend(node, rng)
            else:
                k_in = rng.choices([1, 2, 3, 4], weights=[2, 4, 3, 2])[0]
                ins = [prev[j % len(prev)]]
                pool = recent if rng.random() < 0.7 else below
                while len(ins) < k_in and set(pool) - set(ins):
                    c = rng.choice(pool)
                    if c not in ins:
                        ins.append(c)
                rng.shuffle(ins)
                node["op"] = "sum" if len(ins) == 1 or rng.random() < 0.6 else rng.choice(["max", "min"])
                node["inputs"] = [{"id": p, "unit": svc[p]["unit"],
                                   "w": rng.choices([1, 2, 3], weights=[5, 3, 2])[0] if node["op"] == "sum" else 1}
                                  for p in ins]
                _derived_params(node, svc, vals, rng, target)
                _derived_amend(node, svc, vals, below, rng)
            v = evaluate(node, vals)
            vals[sid] = v
            node["value"] = v
            node["default"] = _nice(v * (rng.uniform(0.4, 0.75) if rng.random() < 0.5 else rng.uniform(1.3, 2.0)))
            node["snapshot"] = v if rng.random() < 0.08 else _nice(
                v * (rng.uniform(0.6, 0.93) if rng.random() < 0.5 else rng.uniform(1.07, 1.5)))
            svc[sid] = node
    return svc, levels


def _source_amend(node, rng):
    if rng.random() < 0.3:
        f = rng.uniform(0.6, 0.9) if rng.random() < 0.5 else rng.uniform(1.1, 1.5)
        old = _nice(node["measured"] * f)
        if node["unit"] == "req/min" and node["measured_unit"] == "req/s":
            old = _nice(node["measured"] / 60 * f) * 60
        if abs(old - node["measured"]) > 0.02 * node["measured"]:
            node["amend"] = {"type": "measure", "old": old}


def _derived_params(node, svc, vals, rng, target):
    unit = node["unit"]
    xs = [_conv(vals[i["id"]], i["unit"], unit) for i in node["inputs"]]
    if node["op"] == "sum":
        x = sum(Fraction(i["w"]) * v for i, v in zip(node["inputs"], xs))
    else:
        x = max(xs) if node["op"] == "max" else min(xs)
    if rng.random() < 0.6:
        ref = rng.choice(node["inputs"])
        on = "own" if ref["unit"] == unit or rng.random() < 0.6 else "converted"
        r = vals[ref["id"]] if on == "own" else _conv(vals[ref["id"]], ref["unit"], unit)
        holds = rng.random() < 0.5
        thr = _nice(r * (rng.uniform(0.7, 0.92) if holds else rng.uniform(1.09, 1.4)))
        acts = []
        for kind in rng.sample(["add", "sub", "mul"], 2):
            if kind == "mul":
                acts.append(("mul", rng.choice([rng.randint(55, 90), rng.randint(112, 160)])))
            else:
                acts.append((kind, _nice(x * Fraction(rng.randint(10, 40), 100))))
        node["cond"] = {"ref": ref["id"], "on": on, "thr": thr, "then": acts[0], "else": acts[1]}
        x = _act(x, acts[0] if r >= thr else acts[1])
    node["factor"] = max(5, min(300, round(target / float(x) * 100)))
    if rng.random() < 0.25:
        v = math.floor(x * Fraction(node["factor"], 100))
        node["cap"] = _nice(v * (rng.uniform(0.75, 0.95) if rng.random() < 0.6 else rng.uniform(1.1, 1.5)))


def _derived_amend(node, svc, vals, below, rng):
    if rng.random() >= 0.3:
        return
    true = evaluate(node, vals)
    kinds = ["factor", "replace"] + (["weight"] if node["op"] == "sum" else []) + (["threshold"] if node["cond"] else [])
    rng.shuffle(kinds)
    for kind in kinds:
        old = json.loads(json.dumps(node))
        if kind == "factor":
            amend = {"type": "factor", "old": max(5, node["factor"] + rng.choice([-1, 1]) * rng.randint(10, 25))}
            old["factor"] = amend["old"]
        elif kind == "weight":
            i = rng.randrange(len(node["inputs"]))
            w0 = rng.choice([w for w in (1, 2, 3) if w != node["inputs"][i]["w"]])
            amend = {"type": "weight", "id": node["inputs"][i]["id"], "old": w0}
            old["inputs"][i]["w"] = w0
        elif kind == "threshold":
            c = node["cond"]
            ref = next(i for i in node["inputs"] if i["id"] == c["ref"])
            r = vals[c["ref"]] if c["on"] == "own" else _conv(vals[c["ref"]], ref["unit"], node["unit"])
            t0 = _nice(r * (rng.uniform(1.09, 1.4) if r >= c["thr"] else rng.uniform(0.7, 0.92)))
            amend = {"type": "threshold", "old": t0}
            old["cond"]["thr"] = t0
        else:
            ids = {i["id"] for i in node["inputs"]}
            cands = [s for s in below if s not in ids and s != node["id"]]
            if not cands:
                continue
            i = rng.randrange(len(node["inputs"]))
            decoy = rng.choice(cands)
            if node["cond"] and node["cond"]["ref"] == node["inputs"][i]["id"]:
                continue
            amend = {"type": "replace", "id": decoy, "unit": svc[decoy]["unit"], "new": node["inputs"][i]["id"]}
            old["inputs"][i] = {"id": decoy, "unit": svc[decoy]["unit"], "w": node["inputs"][i]["w"]}
        if abs(evaluate(old, vals) - true) > 0.02 * true:
            node["amend"] = amend
            return


# ---------- spec text ----------

def _pad(rng, node, k, all_ids):
    others = [s for s in all_ids if s != node["id"] and s not in {i["id"] for i in node["inputs"]}]
    out = []
    for t in rng.sample(PAD, k):
        out.append(fill(t, name=node["name"], team=node["team"], region=rng.choice(REGIONS),
                        other=rng.choice(others) if others else "S9999"))
    return " ".join(out)


def _date(rng, lo, hi):
    m = rng.randint(lo, hi)
    return f"2026-{m:02d}-{rng.randint(1, 28):02d}"


def render_spec(node, seed, all_ids):
    rng = random.Random(f"dag-text:{seed}:{node['id']}")
    u, a, H = node["unit"], node["amend"], T["headings"]
    s = [f"# {node['id']} {node['name']}", "",
         f"Owner: {node['team']} team | Tier: {rng.randint(1, 3)} | Published unit: {u}", "", CONVENTIONS, "",
         f"## 1. {H['overview']}", "", _pad(rng, node, 13, all_ids), ""]
    log = []
    if not node["inputs"]:
        meas = node["measured"] if not (a and a["type"] == "measure") else a["old"]
        mu = node["measured_unit"]
        shown = meas if mu == u else (meas // 60 if mu == "req/s" else meas * 60)
        s += [f"## 2. {H['inputs']}", "", T["edge_inputs"], "", f"## 3. {H['computation']}", "",
              fill(T["edge_capacity"], shown=shown, mu=mu, u=u) + (" " + T["conv_note"] if mu != u else ""), ""]
        if a:
            new = node["measured"] if mu == u else (node["measured"] // 60 if mu == "req/s" else node["measured"] * 60)
            log.append((_date(rng, 7, 9), "adopted", fill(T["rerun"], new=new, mu=mu)))
        if rng.random() < 0.5:
            fake = _nice(shown * rng.uniform(1.15, 1.6))
            log.append((_date(rng, 3, 6), "proposed, not adopted", fill(T["vendor"], fake=fake, mu=mu)))
    else:
        shown_inputs = [dict(i) for i in node["inputs"]]
        w_shown = {i["id"]: i["w"] for i in node["inputs"]}
        if a and a["type"] == "replace":
            for i in shown_inputs:
                if i["id"] == a["new"]:
                    i["id"], i["unit"] = a["id"], a["unit"]
        if a and a["type"] == "weight":
            for i in shown_inputs:
                if i["id"] == a["id"]:
                    i["w"] = a["old"]
        s += [f"## 2. {H['inputs']}", "", "| # | Upstream service | Upstream publishes in | Weight |", "|---|---|---|---|"]
        for k, i in enumerate(shown_inputs, 1):
            s.append(f"| {k} | {i['id']} | {i['unit']} | {i['w'] if node['op'] == 'sum' else 'n/a'} |")
        s += ["", f"## 3. {H['computation']}", "", fill(T["step1"], u=u), "", T["step2_" + node["op"]], ""]
        c = node["cond"]
        if c:
            thr = a["old"] if a and a["type"] == "threshold" else c["thr"]
            ref_unit = next(i["unit"] for i in node["inputs"] if i["id"] == c["ref"])
            acts = {"then": _act_text(c["then"], u), "else": _act_text(c["else"], u)}
            s.append(fill(T["cond_own"], ref=c["ref"], ref_unit=ref_unit, thr=thr, **acts) if c["on"] == "own" else
                     fill(T["cond_conv"], ref=c["ref"], u=u, u2=u, thr=thr, **acts))
        else:
            s.append(T["cond_none"])
        s.append("")
        fac = a["old"] if a and a["type"] == "factor" else node["factor"]
        s += [fill(T["step4"], fac=fac), ""]
        s.append(fill(T["step5_cap"], cap=node["cap"], cap2=node["cap"], u=u) if node["cap"] is not None
                 else T["step5_none"])
        s.append("")
        if a:
            when = _date(rng, 7, 9)
            if a["type"] == "factor":
                log.append((when, "adopted", fill(T["amend_factor"], old=a["old"], new=node["factor"])))
            elif a["type"] == "weight":
                log.append((when, "adopted", fill(T["amend_weight"], id=a["id"], old=a["old"], new=w_shown[a["id"]])))
            elif a["type"] == "threshold":
                log.append((when, "adopted", fill(T["amend_threshold"], old=a["old"], new=c["thr"])))
            else:
                new = next(i for i in node["inputs"] if i["id"] == a["new"])
                log.append((when, "adopted", fill(T["amend_replace"], old_id=a["id"], old_id2=a["id"], new_id=a["new"],
                                                  unit=new["unit"], w=new["w"] if node["op"] == "sum" else "n/a")))
        for _ in range(rng.choice([0, 1, 1, 2])):
            status = rng.choice(["proposed, not adopted", "reverted"])
            when = _date(rng, 2, 6)
            if rng.random() < 0.5 or node["op"] != "sum":
                log.append((when, status, fill(T["amend_factor"], old=fac,
                                               new=max(5, fac + rng.choice([-1, 1]) * rng.randint(8, 30)))))
            else:
                i = rng.choice(shown_inputs)
                log.append((when, status, fill(T["amend_weight"], id=i["id"], old=i["w"],
                                               new=rng.choice([w for w in (1, 2, 3, 4) if w != i["w"]]))))
    s += [f"## 4. {H['operations']}", "", _pad(rng, node, 12, all_ids), "",
          fill(T["bootstrap"], id=node["id"], default=node["default"], u=u), "",
          fill(T["snapshot"], snapshot=node["snapshot"], u=u), ""]
    for _ in range(rng.randint(1, 2)):
        log.append((_date(rng, 1, 9), "adopted", rng.choice(BENIGN_LOG)))
    log.sort()
    s += [f"## 5. {H['changelog']}", ""] + [f"- {d} ({st}): {t}" for d, st, t in log] + [""]
    s += [f"## 6. {H['runbook']}", "", _pad(rng, node, 10, all_ids), ""]
    hist = [_nice(node["snapshot"] * rng.uniform(0.6, 1.1)) for _ in range(3)] + [node["snapshot"]]
    s += [f"## 7. {H['history']}", "", f"| Quarter | {u} |", "|---|---|"]
    s += [f"| {q} | {v:,} |" for q, v in zip(["2025-Q3", "2025-Q4", "2026-Q1", "2026-Q2"], hist)] + [""]
    s += [f"## 8. {H['incidents']}", ""] + [
        "- " + fill(t, d=_date(rng, 1, 9), region=rng.choice(REGIONS), u=u,
                    x=_nice(node["value"] * rng.uniform(0.3, 1.6))) for t in rng.sample(INCIDENTS, 3)] + [""]
    return "\n".join(s)


# ---------- generate / tools / score ----------

def generate(n, seed, out_dir):
    svc, levels = _make_world(n, seed)
    ids = sorted(svc)
    targets = sorted(levels[-1])
    services = "".join(json.dumps({"id": sid, "name": svc[sid]["name"], "team": svc[sid]["team"],
                                   "unit": svc[sid]["unit"], "inputs": [i["id"] for i in svc[sid]["inputs"]],
                                   "spec": render_spec(svc[sid], seed, ids)}) + "\n" for sid in ids)
    srng = random.Random(f"dag-stale:{seed}:{n}")
    scratch = "".join(json.dumps({"seq": k + 1, "id": sid, "value": svc[sid]["snapshot"], "agent": "legacy-sync",
                                  "note": "Q2 snapshot import (not recomputed)", "t": 0}) + "\n"
                      for k, sid in enumerate(srng.sample(ids, max(1, n // 10))))
    gold = {"n": n, "seed": seed, "depth": len(levels), "targets": targets,
            "values": {s: svc[s]["value"] for s in ids}, "levels": levels, "services": [svc[s] for s in ids]}
    tools_txt = "\n".join(f"- {k}: {v}" for k, v in T["tools"].items())
    prompt = "\n\n".join([
        fill(T["intro"], company=COMPANY, n=str(n), first=ids[0], last=ids[-1], depth=str(len(levels))),
        "Tools:\n" + tools_txt, T["store_note"],
        fill(T["task_line"], k=str(len(targets)), targets=", ".join(targets)),
        fill(T["answer_line"], example=f'{{"{targets[0]}": 12345, ...}}')])
    return common.finish(TYPE_ID, n, seed, out_dir, prompt, gold, TIERS,
                         files={"services.jsonl": services, "scratch.jsonl": scratch})


class _Store:
    def __init__(self, path):
        self.path, self.lock, self.posts = path, threading.Lock(), {}
        self.seq = 0
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    if line.strip():
                        self._add(json.loads(line))

    def _add(self, p):
        self.posts.setdefault(p["id"], []).append(p)
        self.seq = max(self.seq, p["seq"])

    def post(self, sid, value, note, agent):
        with self.lock:
            p = {"seq": self.seq + 1, "id": sid, "value": value, "agent": agent, "note": note, "t": round(time.time(), 3)}
            with open(self.path, "a") as f:
                f.write(json.dumps(p) + "\n")
            self._add(p)
            return p["seq"]

    def get(self, sid):
        with self.lock:
            return list(self.posts.get(sid, []))


_STORES, _STORES_LOCK = {}, threading.Lock()


def _store(world_dir):
    key = os.path.realpath(world_dir)
    with _STORES_LOCK:
        if key not in _STORES:
            _STORES[key] = _Store(os.path.join(key, "scratch.jsonl"))
        return _STORES[key]


def _cap(s):
    return s if len(s) <= MAX_CHARS else s[:MAX_CHARS - 60] + "\n[truncated; narrow the request]"


def _num(x):
    if isinstance(x, bool):
        raise ValueError("not a number")
    if isinstance(x, (int, float)):
        return x
    s = str(x).strip().replace(",", "").replace("_", "")
    s = re.sub(r"\s*req/(s|min)$", "", s)
    v = float(s)
    return int(v) if v.is_integer() else v


def tools(world_dir):
    with open(os.path.join(world_dir, "services.jsonl")) as f:
        rows = [json.loads(l) for l in f]
    by_id = {r["id"]: r for r in rows}
    store = _store(world_dir)

    def sid_of(args, key="service_id"):
        sid = str(args.get(key, "")).strip().upper()
        if sid not in by_id:
            raise KeyError(f"unknown service_id {args.get(key)!r}; use list_services")
        return sid

    def list_services(args):
        off, lim = max(0, int(args.get("offset", 0))), max(1, min(int(args.get("limit", 100)), 100))
        out = [f"{r['id']} | {r['name']} | {r['team']} | {r['unit']} | inputs: "
               f"{','.join(r['inputs']) if r['inputs'] else 'none (edge service)'}" for r in rows[off:off + lim]]
        tail = f"\n[next: offset={off + lim}]" if off + lim < len(rows) else "\n[end]"
        return _cap(f"{len(rows)} services; showing {off}-{off + len(out) - 1}\n" + "\n".join(out) + tail)

    def read_spec(args):
        text, off = by_id[sid_of(args)]["spec"], max(0, int(args.get("offset", 0)))
        chunk = text[off:off + MAX_CHARS - 80]
        end = off + len(chunk)
        return chunk + (f"\n[chars {off}-{end} of {len(text)}; continue with offset={end}]" if end < len(text) else "")

    def post_value(args):
        sid = sid_of(args)
        v = _num(args.get("value"))
        seq = store.post(sid, v, str(args.get("note", "") or "")[:300], str(args.get("_agent", "") or "anonymous")[:64])
        return f"ok: posted {sid} = {v} (post #{seq})"

    def get_value(args):
        ids = [s.strip().upper() for s in str(args.get("service_id", "")).split(",") if s.strip()][:25]
        if not ids:
            raise KeyError("service_id is required (one id or up to 25 comma-separated ids)")
        out = []
        for sid in ids:
            if sid not in by_id:
                out.append(f"{sid}: unknown service id")
                continue
            ps = store.get(sid)
            if not ps:
                out.append(f"{sid}: no posts")
                continue
            hist = "; ".join(f"{p['value']} by {p['agent']} (#{p['seq']}{', ' + p['note'] if p['note'] else ''})"
                             for p in reversed(ps[-3:]))
            out.append(f"{sid}: latest {ps[-1]['value']} | {len(ps)} post(s), newest first: {hist}")
        return _cap("\n".join(out))

    def safe(fn):
        def run(args):
            try:
                return fn(args or {})
            except Exception as e:  # noqa: BLE001 - tool errors go back to the agent as text
                return f"error: {e}"
        return run

    sid = {"service_id": {"type": "string", "description": "e.g. S0012"}}
    return [
        {"name": "list_services", "description": T["tools"]["list_services"],
         "parameters": {"type": "object", "properties": {"offset": {"type": "integer"}, "limit": {"type": "integer"}}},
         "fn": safe(list_services)},
        {"name": "read_spec", "description": T["tools"]["read_spec"],
         "parameters": {"type": "object", "properties": {**sid, "offset": {"type": "integer"}}, "required": ["service_id"]},
         "fn": safe(read_spec)},
        {"name": "post_value", "description": T["tools"]["post_value"],
         "parameters": {"type": "object", "properties": {**sid, "value": {"type": "number"}, "note": {"type": "string"}},
                        "required": ["service_id", "value"]},
         "fn": safe(post_value)},
        {"name": "get_value", "description": T["tools"]["get_value"],
         "parameters": {"type": "object", "properties": {"service_id": {"type": "string"}}, "required": ["service_id"]},
         "fn": safe(get_value)},
    ]


def parse_answer(answer):
    # The last JSON object keyed by service ids wins (a draft block before the final one does not count).
    text, last = str(answer or ""), None
    for m in re.finditer(r"\{", text):
        depth = 0
        for j in range(m.start(), len(text)):
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            if depth == 0:
                try:
                    obj = json.loads(text[m.start():j + 1])
                except ValueError:
                    break
                if isinstance(obj, dict) and any(re.fullmatch(r"S\d{4}", str(k).strip().upper()) for k in obj):
                    last = {str(k).strip().upper(): v for k, v in obj.items()}
                break
    if last is not None:
        return last
    return {m.group(1): m.group(2) for m in re.finditer(r"\b(S\d{4})\b\W{0,6}?(-?[\d,]+(?:\.\d+)?)", text)}


def score(task, answer):
    with open(task["gold_path"]) as f:
        gold = json.load(f)
    got = parse_answer(answer)
    n_ok = n_exact = n_missing = 0
    for t in gold["targets"]:
        g = gold["values"][t]
        try:
            v = _num(got[t])
        except (KeyError, ValueError, TypeError):
            n_missing += 1
            continue
        n_exact += v == g
        n_ok += abs(v - g) <= max(1, REL_TOL * abs(g))
    k = len(gold["targets"])
    return {"score": n_ok / k, "n_correct": n_ok, "n_exact": n_exact, "n_targets": k, "n_missing": n_missing,
            "exact": n_exact == k}
