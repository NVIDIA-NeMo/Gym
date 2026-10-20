"""Type 35 solutions: tool-only oracle (parses the K3 clause templates back with common.tre and recomputes every
contract minute by minute, independently of type.py's interval sweep), plausible-wrong baselines including a
grep/regex baseline, partial readers that skip one trap kind each, and single units (check 5)."""
import datetime as dt
import json
import re

from .. import common
from . import type as track

T = track.T
NUM = r"\d+(?:\.\d+)?"
INT = r"\d+"
WIN = r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday) \d\d:\d\d-\d\d:\d\d UTC"
RX = {"c_target": [common.tre(v, target=NUM) for v in T["c_target"]],
      "c_minlen": [common.tre(v, minutes=INT) for v in T["c_minlen"]],
      "c_maint": [common.tre(v, hours=INT, window=WIN) for v in T["c_maint"]],
      "c_support": [common.tre(v, p1=INT, p2=INT, p3=INT) for v in T["c_support"]],
      "c_support_credit": [common.tre(v, support_credit=INT) for v in T["c_support_credit"]],
      "c_tiers": [common.tre(v, t1=NUM, c1=INT, t2=NUM, c2=INT, t3=NUM, c3=INT) for v in T["c_tiers"]],
      "c_cap": [common.tre(v, cap=INT) for v in T["c_cap"]],
      "c_claim": [common.tre(v, days=INT) for v in T["c_claim"]]}
EXACT = {"c_deg_full": 1.0, "c_deg_half": 0.5, "c_deg_none": 0.0}
M0, M1, TOTAL = track.M0, track.M1, track.TOTAL


def _t(s):
    p = s.strip().split(" ")
    return dt.datetime.fromisoformat(f"{p[0]} {p[-1]}")


def parse_contract(text):
    c = {"tz": None, "minlen": 0, "upstream_counts": None}
    m = re.search(r"^Customer timezone: UTC([+-])(\d\d):(\d\d)$", text, re.M)
    c["tz"] = (1 if m.group(1) == "+" else -1) * (int(m.group(2)) * 60 + int(m.group(3)))
    for cl in re.findall(r"^\d+\. (.*)$", text, re.M):
        cl = cl.strip()
        for key, w in EXACT.items():
            if cl in T[key]:
                c["weight"] = w
        if cl in T["c_minlen_none"]:
            c["minlen"] = 0
        if cl in T["c_upstream_counts"]:
            c["upstream_counts"] = True
        if cl in T["c_upstream_excluded"]:
            c["upstream_counts"] = False
        for key, rxs in RX.items():
            for rx in rxs:
                mm = rx.fullmatch(cl)
                if not mm:
                    continue
                g = mm.groupdict()
                if key == "c_target":
                    c["target"] = float(g["target"])
                elif key == "c_minlen":
                    c["minlen"] = int(g["minutes"])
                elif key == "c_maint":
                    day, rest = g["window"].split(" ", 1)
                    h0, h1 = rest.split(" ")[0].split("-")
                    c["notice_h"], c["win"] = int(g["hours"]), (track.DAYS.index(day), int(h0[:2]), int(h1[:2]))
                elif key == "c_support":
                    c["p1"], c["p2"], c["p3"] = int(g["p1"]), int(g["p2"]), int(g["p3"])
                elif key == "c_support_credit":
                    c["support_credit"] = int(g["support_credit"])
                elif key == "c_tiers":
                    c["tiers"] = [(float(g["t1"]), int(g["c1"])), (float(g["t2"]), int(g["c2"])),
                                  (float(g["t3"]), int(g["c3"]))]
                elif key == "c_cap":
                    c["cap"] = int(g["cap"])
                elif key == "c_claim":
                    c["claim_days"] = int(g["days"])
    need = ["target", "weight", "notice_h", "p1", "support_credit", "tiers", "cap", "claim_days"]
    assert all(k in c for k in need) and c["upstream_counts"] is not None, (c, text[:300])
    return c


def _rows(out, prefix):
    return [r for r in out.splitlines() if r.startswith(prefix)]


def read_world(task):
    fns = {t["name"]: t["fn"] for t in track.tools(task["world_dir"])}
    ids, off = [], 0
    while True:
        out = fns["list_customers"]({"offset": off, "limit": 100})
        ids += [r.split(" | ")[0] for r in _rows(out, "CUS")]
        m = re.search(r"next offset (\d+)\)$", out)
        if not m:
            break
        off = int(m.group(1))
    incs, off = [], 0
    while True:
        out = fns["search_incidents"]({"offset": off})
        got = _rows(out, "INC-")
        for r in got:
            f = r.split(" | ")
            incs.append({"id": f[0], "start": _t(f[1]), "end": _t(f[2]), "service": f[4], "region": f[5],
                         "scope": f[6], "impact": f[7], "cause": f[8], "by": f[9], "notice": f[10]})
        m = re.search(r"next offset (\d+)\)$", out)
        if not m:
            break
        off = int(m.group(1))
    notices = {}
    for nid in sorted({x["notice"] for x in incs if x["notice"] != "-"}):
        f = _rows(fns["read_notice"]({"notice_id": nid}), "NTC-")[0].split(" | ")
        notices[nid] = {"sent": _t(f[1]), "ps": _t(f[5]), "pe": _t(f[6])}
    custs = []
    for cid in ids:
        c = parse_contract(fns["read_contract"]({"customer_id": cid}))
        c["id"] = cid
        c["inst"] = [tuple(r.split(" | ")) for r in _rows(fns["get_subscriptions"]({"customer_id": cid}), "INS-")]
        c["tickets"] = [r.split(" | ") for r in _rows(fns["list_tickets"]({"customer_id": cid}), "TKT-")]
        c["claims"] = [r.split(" | ") for r in _rows(fns["list_claims"]({"customer_id": cid}), "CLM-")]
        custs.append(c)
    return custs, incs, notices


def _idx(t):
    return int((t - M0).total_seconds() // 60)


def availability(c, incs, notices, skip):
    mine = {x[0] for x in c["inst"]}
    pairs = {(x[1], x[2]) for x in c["inst"]}
    w, extra = [0.0] * TOTAL, 0.0
    for x in incs:
        if x["scope"] == "region-wide":
            if "regionwide" in skip or (x["service"], x["region"]) not in pairs:
                continue
        elif not (mine & set(x["scope"].split(", "))) and not ("decoy" in skip and (x["service"], x["region"]) in pairs):
            continue
        if "cause" not in skip:
            if x["cause"] == "force_majeure" or (x["cause"] == "upstream_provider" and not c["upstream_counts"]):
                continue
            if x["cause"] == "customer_action" and (x["by"] == c["id"] or "own_only" in skip):
                continue
        full = int((x["end"] - x["start"]).total_seconds() // 60)
        if "minlen" not in skip and full < c["minlen"]:
            continue
        wt = 1.0 if x["impact"] == "outage" or "weight" in skip else c["weight"]
        lo, hi = _idx(x["start"]), _idx(x["end"])
        if x["cause"] == "maintenance" and "maint_none" not in skip:
            n = notices[x["notice"]]
            d, h0, h1 = c["win"]
            day = dt.datetime(n["ps"].year, n["ps"].month, n["ps"].day)
            ok = (n["ps"] - n["sent"]).total_seconds() >= c["notice_h"] * 3600 and n["ps"].weekday() == d and \
                n["ps"] >= day + dt.timedelta(hours=h0) and n["pe"] <= day + dt.timedelta(hours=h1)
            if "maint_all" in skip:
                continue
            if ok:
                plo, phi = _idx(n["ps"]), _idx(n["pe"])
                for k in range(max(0, lo), min(TOTAL, hi)):
                    if not plo <= k < phi:
                        w[k] = max(w[k], wt)
                continue
        if "clip" in skip:
            extra += (max(0, -lo) + max(0, hi - TOTAL)) * wt
        for k in range(max(0, lo), min(TOTAL, hi)):
            if wt > w[k]:
                w[k] = wt
    down = sum(w) + extra
    return 100 * (TOTAL - down) / TOTAL


def _bmin(a, b, tz):
    n, t, step = 0, a + dt.timedelta(minutes=tz), dt.timedelta(minutes=1)
    end = b + dt.timedelta(minutes=tz)
    while t < end:
        if t.weekday() < 5 and 9 <= t.hour < 17:
            n += 1
        t += step
    return n


def support(c, skip):
    for _, _, prio, op, rs in c["tickets"]:
        op, rs = _t(op), _t(rs)
        if op < M0 and "prev" not in skip or op >= M1:
            continue
        if prio == "P1":
            late = (rs - op).total_seconds() / 60 > c["p1"]
        else:
            tgt = (c["p2"] if prio == "P2" else c["p3"]) * 60
            el = (rs - op).total_seconds() / 60 if "bh" in skip else _bmin(op, rs, 0 if "tz" in skip else c["tz"])
            late = el > tgt
        if late:
            return True
    return False


def audit(c, incs, notices, skip=frozenset()):
    av = availability(c, incs, notices, skip)
    sup = support(c, skip)
    breaches = (["availability"] if av < c["target"] else []) + (["support"] if sup else [])
    tier = 0
    for th, cr in c["tiers"]:
        if av < th:
            tier = cr
    total = tier + (c["support_credit"] if sup else 0)
    if "cap" not in skip:
        total = min(total, c["cap"])
    last = track.M1.date() - dt.timedelta(days=1)
    valid = "claims" in skip or any(m == track.MONTH and dt.date.fromisoformat(d) <= last + dt.timedelta(
        days=c["claim_days"]) for _, _, m, d in c["claims"])
    return {"breaches": breaches, "availability": round(av, 3), "credit": total if valid else 0}


def _answer(res):
    return "Final answer:\n" + json.dumps(res)


def solve(task, skip=frozenset()):
    custs, incs, notices = read_world(task)
    return {c["id"]: audit(c, incs, notices, skip) for c in custs}


def oracle(task):
    return _answer(solve(task))


def regex_baseline(task):
    """grep: incidents whose row mentions one of the customer's instance ids, full logged length, no rules; target =
    the largest percentage >= 90 in the contract; support late if any ticket took > 4 calendar hours; credit 10."""
    w = task["world_dir"]
    cs = json.loads(open(w + "/customers.json").read())
    rows = json.loads(open(w + "/incidents.json").read())
    res = {}
    for c in cs:
        ids = [s.split(" | ")[0] for s in c["subs"]]
        down = 0
        for r in rows:
            if any(i in r for i in ids):
                f = r.split(" | ")
                down += (_t(f[2]) - _t(f[1])).total_seconds() / 60
        pcts = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)%", c["contract"]) if float(x) >= 90]
        tgt = max(pcts) if pcts else 99.9
        av = round(100 * (TOTAL - down) / TOTAL, 3)
        sup = any((_t(t.split(" | ")[4]) - _t(t.split(" | ")[3])).total_seconds() > 4 * 3600 for t in c["tickets"])
        br = (["availability"] if av < tgt else []) + (["support"] if sup else [])
        res[c["id"]] = {"breaches": br, "availability": av, "credit": 10 if br else 0}
    return _answer(res)


SKIPS = ["regionwide", "maint_all", "maint_none", "weight", "minlen", "clip", "decoy", "cause", "own_only", "bh",
         "tz", "prev", "claims", "cap"]


def wrong(task):
    out = {"regex": regex_baseline(task),
           "naive_reader": _answer(solve(task, frozenset({"maint_none", "minlen", "cause", "bh", "claims", "cap",
                                                          "decoy"}))),
           "service_region_calendar": _answer(solve(task, frozenset({"decoy", "weight", "bh", "prev", "clip",
                                                                     "maint_all"})))}
    for k in SKIPS:
        out["skip_" + k] = _answer(solve(task, frozenset({k})))
    return out


def units(task, k, rng):
    gold = common.gold_of(task)["customers"]
    ids = sorted(gold, key=lambda a: -len(gold[a]["kinds"]))[:max(2 * k, k)]
    rng.shuffle(ids)
    out = []
    for cid in ids[:k]:
        prompt = (common.CANARY + "\n\n" + common.fill(T["unit_prompt"], customer=cid, month=track.MONTH) + "\n\n" +
                  track.rules_text(track.example(cid)))

        def check(text, cid=cid):
            p = track.parse_answer(text or "")
            if cid not in p:
                return None
            return all(track.unit_fields(p[cid], gold[cid]))
        out.append({"id": f"{task['id']}:{cid}", "kind": "+".join(gold[cid]["kinds"]), "prompt": prompt,
                    "gold": {x: gold[cid][x] for x in ("breaches", "availability", "credit")}, "check": check})
    return out
