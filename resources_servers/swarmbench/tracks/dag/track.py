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
import json
import math
import os
import random
import re
import threading
import time
from fractions import Fraction

MAX_CHARS = 8000
UNITS = ("req/s", "req/min")
REL_TOL = 0.005  # absorbs off-by-one floors from float slop; every trap moves a value by >= 2%
COMPANY = "Corvane Systems"
TEAMS = ["Payments", "Identity", "Search", "Storage", "Messaging", "Billing", "Ledger", "Catalog", "Routing", "Edge",
         "Analytics", "Notifications", "Fraud", "Inventory", "Pricing", "Media", "Telemetry", "Scheduling"]
WORDS_A = ["ledger", "token", "quota", "event", "index", "session", "invoice", "route", "cache", "policy", "profile",
           "media", "ingest", "audit", "metric", "order", "refund", "pricing", "search", "geo", "notify", "batch",
           "stream", "vault", "shard", "replica", "queue", "feature", "rating", "catalog", "fraud", "consent"]
WORDS_B = ["sync", "gateway", "broker", "worker", "relay", "planner", "store", "resolver", "fanout", "compactor",
           "scorer", "router", "ledger", "proxy", "indexer", "collector", "dispatcher", "validator", "mirror", "agent"]
REGIONS = ["us-east", "us-west", "eu-central", "eu-west", "ap-south", "ap-northeast"]
CONVENTIONS = (
    "Catalog conventions: every service publishes one number, its sustained capacity, as a whole number in its "
    "published unit (req/s or req/min; 1 req/s = 60 req/min). Work in exact arithmetic and round down only where the "
    "spec says. Only change-log entries marked (adopted) are in force; proposed or reverted entries are not. The "
    "bootstrap default and the Q2 dashboard snapshot are not the computed capacity.")
PAD = [
    "{name} is owned by the {team} team and runs active-active in {region} and two other regions.",
    "Traffic to {name} is shaped by the shared ingress tier, so short bursts above capacity are queued rather than dropped.",
    "Historically {name} was carved out of {other}, but the two services no longer share code or capacity.",
    "Do not confuse {name} with {other}, which has a similar name and is owned by a different team.",
    "The {team} team reviews this spec every quarter together with the platform capacity council.",
    "Capacity figures in this catalog drive autoscaling floors, rate limits and the quarterly hardware forecast.",
    "During the 2025 migration, {name} briefly read from {other}; that dependency was removed and is not an input.",
    "Latency SLO: p99 under 250 ms at the published capacity, measured at the regional load balancer.",
    "Error budget: 0.1% of requests per 28-day window, tracked on the {team} reliability dashboard.",
    "On-call: the {team} primary rotation, with escalation to the platform duty manager after 30 minutes.",
    "Deploys go out through the standard canary pipeline: 1% of traffic for 20 minutes, then 25%, then 100%.",
    "Dashboards for {name} live in the {team} folder; the capacity panel shows the last published value.",
    "Runbook: if {name} saturates, first check the upstream inputs listed in section 2, then the regional ingress.",
    "Runbook: a sudden drop in capacity usually means a dependency was degraded; compare with the inputs' own pages.",
    "Runbook: never raise the published capacity by hand; change the inputs or file an amendment instead.",
    "The service emits structured logs to the central pipeline with a 14-day retention period.",
    "Load tests are run in the staging region against a production-shaped dataset at least twice a year.",
    "{name} has no persistent state of its own; everything it needs is read from upstream services at request time.",
    "Feature flags for {name} are managed in the shared flag service and audited weekly.",
    "Rate limits enforced at the gateway are derived from the published capacity with a 10% safety margin.",
    "Security review: last completed in 2026-Q1 with no open high-severity findings.",
    "The capacity model was simplified in 2025; older documents that mention {other} as an input are out of date.",
    "Consumers of {name} should read its published value from the catalog rather than hard-coding a number.",
    "The {team} team keeps a backlog of capacity improvements, none of which affects the current published value.",
    "Regional failover moves traffic to the nearest healthy region within about 90 seconds.",
    "Disaster recovery drills run every six months; the most recent one finished within the target window.",
    "Synthetic probes call {name} every 30 seconds from each region and page on three consecutive failures.",
    "Cost allocation for {name} is charged to the {team} cost center by request volume.",
    "Capacity reviews compare the published value with observed peaks; the gap is reported but not applied.",
    "Upstream services listed as inputs may themselves depend on further services; see their own specs.",
    "Client libraries for {name} retry idempotent calls twice with jittered exponential backoff.",
    "The {team} team runs a game day for {name} each quarter, injecting dependency failures one at a time.",
    "Connection pools to {name} are sized from the published capacity, not from observed traffic.",
    "Schema changes to {name} require a two-phase rollout so that old and new readers coexist for one release.",
    "Audit logs for administrative actions on {name} are retained for 400 days.",
    "Partners do not call {name} directly; all external traffic goes through the public API gateway.",
    "A read-only mirror of {name} serves internal analytics jobs and is excluded from the capacity figure.",
    "Alert thresholds for {name} are set at 80% and 95% of the published capacity.",
    "Hardware for {name} is provisioned on the shared compute pool with a reserved baseline per region.",
    "The {team} team publishes a weekly reliability digest that includes {name}'s saturation minutes.",
    "Backfills and batch replays run against {name} only during the nightly low-traffic window.",
    "Any spec change for {name} must be approved by a second engineer from the {team} team.",
    "A shadow deployment of {name} receives mirrored traffic for validation and is not counted as capacity.",
    "Capacity planning for {other} once referenced {name}, but the two are planned separately now.",
    "TLS certificates for {name} rotate automatically every 60 days.",
    "The dependency graph view in the catalog is regenerated nightly from the input tables of every spec.",
    "Throttled requests return HTTP 429 with a retry-after header computed from the current queue depth.",
    "{name} exports per-tenant usage counters, which feed the monthly chargeback report.",
    "Cold starts for {name} take about four minutes, during which the bootstrap default is published.",
    "Long-running requests are cut off after 30 seconds and counted as errors against the budget.",
]
INCIDENTS = [
    "{d}: a bad config push in {region} cut throughput to about {x:,} {u} for 40 minutes; rolled back.",
    "{d}: an upstream outage caused retries to spike; observed peak load was {x:,} {u}.",
    "{d}: a noisy tenant pushed load to {x:,} {u}; a per-tenant limit was added.",
    "{d}: a regional failover drill briefly routed {x:,} {u} to a single region without errors.",
    "{d}: a dependency deploy raised p99 latency; the service shed load down to {x:,} {u} until it recovered.",
    "{d}: a capacity test in staging reached {x:,} {u} before saturating the shared cache.",
]
BENIGN_LOG = ["Runbook links refreshed.", "On-call rotation moved to the follow-the-sun schedule.",
              "Dashboard panels renamed; no change to the computation.", "Spec reformatted into the catalog template.",
              "Owner contact updated after a team reorganisation.", "Added the regional failover note."]


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


def _act_text(act, unit, what="the combined input"):
    kind, c = act
    if kind == "add":
        return f"add {c:,} {unit} to {what}"
    if kind == "sub":
        return f"subtract {c:,} {unit} from {what}"
    return f"multiply {what} by {c}%"


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
        out.append(t.format(name=node["name"], team=node["team"], region=rng.choice(REGIONS),
                            other=rng.choice(others) if others else "the legacy monolith"))
    return " ".join(out)


def _date(rng, lo, hi):
    m = rng.randint(lo, hi)
    return f"2026-{m:02d}-{rng.randint(1, 28):02d}"


def render_spec(node, seed, all_ids):
    rng = random.Random(f"dag-text:{seed}:{node['id']}")
    u, a = node["unit"], node["amend"]
    s = [f"# {node['id']} {node['name']}", "",
         f"Owner: {node['team']} team | Tier: {rng.randint(1, 3)} | Published unit: {u}", "", CONVENTIONS, "",
         "## 1. Overview", "", _pad(rng, node, 13, all_ids), ""]
    log = []
    if not node["inputs"]:
        meas = node["measured"] if not (a and a["type"] == "measure") else a["old"]
        mu = node["measured_unit"]
        shown = meas if mu == u else (meas // 60 if mu == "req/s" else meas * 60)
        s += ["## 2. Inputs", "", "This is an edge service with no upstream inputs.", "",
              "## 3. Capacity computation", "",
              f"The capacity is set by the latest load test, which sustained {shown:,} {mu}. "
              f"Publish that figure in {u}." + (" (1 req/s = 60 req/min.)" if mu != u else ""), ""]
        if a:
            new = node["measured"] if mu == u else (node["measured"] // 60 if mu == "req/s" else node["measured"] * 60)
            log.append((_date(rng, 7, 9), "adopted", f"Load test rerun; the latest load test now sustained {new:,} {mu}."))
        if rng.random() < 0.5:
            fake = _nice(shown * rng.uniform(1.15, 1.6))
            log.append((_date(rng, 3, 6), "proposed, not adopted",
                        f"Use the vendor benchmark of {fake:,} {mu} instead of the load test."))
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
        s += ["## 2. Inputs", "", "| # | Upstream service | Upstream publishes in | Weight |", "|---|---|---|---|"]
        for k, i in enumerate(shown_inputs, 1):
            s.append(f"| {k} | {i['id']} | {i['unit']} | {i['w'] if node['op'] == 'sum' else 'n/a'} |")
        s += ["", "## 3. Capacity computation", "",
              f"Step 1 - Convert. Convert every input to {u}, using the unit its upstream publishes in "
              f"(1 req/s = 60 req/min).", ""]
        s.append("Step 2 - Combine. " + {
            "sum": "The combined input is the weighted sum of the converted inputs, using the weights in the table.",
            "max": "The combined input is the largest of the converted inputs; weights do not apply.",
            "min": "The combined input is the smallest of the converted inputs; weights do not apply."}[node["op"]])
        s.append("")
        c = node["cond"]
        if c:
            thr = a["old"] if a and a["type"] == "threshold" else c["thr"]
            ref_unit = next(i["unit"] for i in node["inputs"] if i["id"] == c["ref"])
            what = (f"{c['ref']}'s published value, in its own unit ({ref_unit}), is at least {thr:,}" if c["on"] == "own"
                    else f"{c['ref']}'s input, after conversion to {u}, is at least {thr:,} {u}")
            s.append(f"Step 3 - Condition. If {what}, then {_act_text(c['then'], u)}; otherwise "
                     f"{_act_text(c['else'], u)}.")
        else:
            s.append("Step 3 - Condition. No conditional adjustment applies to this service.")
        s.append("")
        fac = a["old"] if a and a["type"] == "factor" else node["factor"]
        s += [f"Step 4 - Capacity factor. Multiply by the capacity factor of {fac}% and round down to a whole number.", ""]
        s.append(f"Step 5 - Limit. The published value never exceeds {node['cap']:,} {u}; if the result is higher, "
                 f"publish {node['cap']:,}." if node["cap"] is not None else "Step 5 - Limit. There is no hard limit.")
        s.append("")
        if a:
            when = _date(rng, 7, 9)
            if a["type"] == "factor":
                log.append((when, "adopted", f"The capacity factor changed from {a['old']}% to {node['factor']}%."))
            elif a["type"] == "weight":
                log.append((when, "adopted", f"The weight on {a['id']} changed from {a['old']} to "
                                             f"{w_shown[a['id']]}."))
            elif a["type"] == "threshold":
                log.append((when, "adopted", f"The condition threshold changed from {a['old']:,} to {c['thr']:,}."))
            else:
                new = next(i for i in node["inputs"] if i["id"] == a["new"])
                log.append((when, "adopted", f"Input {a['id']} was replaced by {a['new']} (publishes in {new['unit']}, "
                                             f"weight {new['w'] if node['op'] == 'sum' else 'n/a'}); {a['id']} is no "
                                             f"longer an input."))
        for _ in range(rng.choice([0, 1, 1, 2])):
            status = rng.choice(["proposed, not adopted", "reverted"])
            when = _date(rng, 2, 6)
            if rng.random() < 0.5 or node["op"] != "sum":
                log.append((when, status, f"The capacity factor changed from {fac}% to "
                                          f"{max(5, fac + rng.choice([-1, 1]) * rng.randint(8, 30))}%."))
            else:
                i = rng.choice(shown_inputs)
                log.append((when, status, f"The weight on {i['id']} changed from {i['w']} to "
                                          f"{rng.choice([w for w in (1, 2, 3, 4) if w != i['w']])}."))
    s += ["## 4. Operations", "", _pad(rng, node, 12, all_ids), "",
          f"Bootstrap default: during a cold start, before its inputs are available, {node['id']} temporarily "
          f"publishes {node['default']:,} {u}. This is a placeholder, not its computed capacity.", "",
          f"Q2 dashboard snapshot: {node['snapshot']:,} {u} (recorded last quarter; may be stale).", ""]
    for _ in range(rng.randint(1, 2)):
        log.append((_date(rng, 1, 9), "adopted", rng.choice(BENIGN_LOG)))
    log.sort()
    s += ["## 5. Change log", ""] + [f"- {d} ({st}): {t}" for d, st, t in log] + [""]
    s += ["## 6. Runbook", "", _pad(rng, node, 10, all_ids), ""]
    hist = [_nice(node["snapshot"] * rng.uniform(0.6, 1.1)) for _ in range(3)] + [node["snapshot"]]
    s += ["## 7. Capacity history (published values in past quarters; not current)", "", f"| Quarter | {u} |", "|---|---|"]
    s += [f"| {q} | {v:,} |" for q, v in zip(["2025-Q3", "2025-Q4", "2026-Q1", "2026-Q2"], hist)] + [""]
    s += ["## 8. Recent incidents", ""] + [
        "- " + t.format(d=_date(rng, 1, 9), region=rng.choice(REGIONS), u=u,
                        x=_nice(node["value"] * rng.uniform(0.3, 1.6))) for t in rng.sample(INCIDENTS, 3)] + [""]
    return "\n".join(s)


# ---------- generate / tools / score ----------

def generate(n, seed, out_dir):
    svc, levels = _make_world(n, seed)
    os.makedirs(out_dir, exist_ok=True)
    ids = sorted(svc)
    targets = sorted(levels[-1])
    with open(os.path.join(out_dir, "services.jsonl"), "w") as f:
        for sid in ids:
            x = svc[sid]
            f.write(json.dumps({"id": sid, "name": x["name"], "team": x["team"], "unit": x["unit"],
                                "inputs": [i["id"] for i in x["inputs"]], "spec": render_spec(x, seed, ids)}) + "\n")
    srng = random.Random(f"dag-stale:{seed}:{n}")
    with open(os.path.join(out_dir, "scratch.jsonl"), "w") as f:
        for k, sid in enumerate(srng.sample(ids, max(1, n // 10))):
            f.write(json.dumps({"seq": k + 1, "id": sid, "value": svc[sid]["snapshot"], "agent": "legacy-sync",
                                "note": "Q2 snapshot import (not recomputed)", "t": 0}) + "\n")
    gold = {"n": n, "seed": seed, "depth": len(levels), "targets": targets,
            "values": {s: svc[s]["value"] for s in ids}, "levels": levels, "services": [svc[s] for s in ids]}
    gold_path = os.path.join(out_dir, "gold.json")
    with open(gold_path, "w") as f:
        json.dump(gold, f)
    prompt = (
        f"You are given the internal service catalog of {COMPANY}: {n} services with ids {ids[0]}-{ids[-1]}. Each "
        f"service publishes one number, its sustained capacity, as a whole number in its own published unit (req/s "
        f"or req/min; 1 req/s = 60 req/min). Most services compute their capacity from the published values of "
        f"their upstream input services, following the steps in their spec (conversions, a weighted sum or max/min, "
        f"conditions, a capacity factor with rounding down, limits, and adopted change-log amendments). Edge services "
        f"have no inputs. The dependencies form a DAG about {len(levels)} levels deep, and service ids are not in "
        f"dependency order, so a service can only be computed once all its inputs are known, and an error in any "
        f"upstream value propagates downstream.\n\n"
        f"Tools: list_services (id, name, unit, inputs; paginated), read_spec(service_id), and a shared scratch "
        f"store: post_value(service_id, value, note) and get_value(service_id) let agents publish and reuse "
        f"intermediate results. The store is not checked by anyone: it was pre-loaded with stale Q2 snapshot values "
        f"by 'legacy-sync', and other posts may be wrong. Only your final answer is graded.\n\n"
        f"Task: report the published capacity of each of these {len(targets)} target services, in each service's own "
        f"published unit: {', '.join(targets)}.\n\n"
        f"Final answer: one JSON object mapping every target id to its whole-number value, e.g. "
        f'{{"{targets[0]}": 12345, ...}}. Each target is graded separately (within 0.5% counts as correct).')
    return {"id": f"dag-n{n}-s{seed}", "prompt": prompt, "world_dir": out_dir, "n_units": n, "gold_path": gold_path}


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
        {"name": "list_services", "description": "List services: id | name | team | published unit | input service ids. "
                                                 "Up to 100 rows per call.",
         "parameters": {"type": "object", "properties": {"offset": {"type": "integer"}, "limit": {"type": "integer"}}},
         "fn": safe(list_services)},
        {"name": "read_spec", "description": "Read one service's spec document (~7k chars; optional char offset).",
         "parameters": {"type": "object", "properties": {**sid, "offset": {"type": "integer"}}, "required": ["service_id"]},
         "fn": safe(read_spec)},
        {"name": "post_value", "description": "Publish your computed value for a service to the shared scratch store "
                                              "so other agents can reuse it. Not graded; add a short note.",
         "parameters": {"type": "object", "properties": {**sid, "value": {"type": "number"}, "note": {"type": "string"}},
                        "required": ["service_id", "value"]},
         "fn": safe(post_value)},
        {"name": "get_value", "description": "Read scratch-store posts for a service id (or up to 25 comma-separated "
                                             "ids): latest value, author and note. Posts are unverified.",
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
