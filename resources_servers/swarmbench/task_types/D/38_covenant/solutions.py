"""Type 38 solutions: tool-only oracle (parses the K3 templates back with common.tre), plausible-wrong baselines,
partial readers that skip one rule each, and single units (check 5)."""
import datetime as dt
import json
import re

from .. import common
from . import type as track

T = track.T
DATE = r"[A-Z][a-z]+ \d{1,2}, \d{4}"
AMT = r"[\d,]+"
RAT = r"\d+\.\d{2}"
INT = r"\d+"
tre = common.tre
R = {
    "title": tre(T["agreement_title"], name=r".+?", date=DATE),
    "net_cap": tre(T["def_net_cap"], cap=AMT),
    "ebitda_cap": tre(T["def_ebitda_cap"], addbacks=r".+?", pct=INT),
    "ebitda_nocap": tre(T["def_ebitda_nocap"], addbacks=r".+?"),
    "cov_leverage": tre(T["cov_leverage"], max=RAT),
    "cov_step": tre(T["cov_leverage_step"], max1=RAT, date=DATE, max2=RAT),
    "cov_coverage": tre(T["cov_coverage"], min=RAT),
    "cov_current": tre(T["cov_current"], min=RAT),
    "cure_right": tre(T["cure_right"], days=INT),
    "amend_leverage": tre(T["amend_leverage"], k=INT, signed=DATE, **{"from": DATE}, max=RAT),
    "amend_coverage": tre(T["amend_coverage"], k=INT, signed=DATE, **{"from": DATE}, min=RAT),
    "amend_current": tre(T["amend_current"], k=INT, signed=DATE, **{"from": DATE}, min=RAT),
    "amend_cap": tre(T["amend_cap"], k=INT, signed=DATE, **{"from": DATE}, pct=INT),
    "amend_fl_excl": tre(T["amend_fl_excl"], k=INT, signed=DATE, **{"from": DATE}),
    "amend_fl_incl": tre(T["amend_fl_incl"], k=INT, signed=DATE, **{"from": DATE}),
    "amend_draft": tre(T["amend_draft"], k=INT, signed=DATE, max=RAT),
    "request": tre(T["request"], date=DATE, end=DATE),
    "cure_notice": tre(T["cure_notice"], date=DATE, amount=AMT, end=DATE),
}
LABEL2KEY = {v: k for k, v in T["labels"].items()}
SUBJ = {"amend_leverage": "net_leverage", "amend_coverage": "interest_coverage", "amend_current": "current_ratio"}


def _n(s):
    return int(s.replace(",", ""))


def _fm(rx, line):
    return rx.fullmatch(line.strip())


def parse_borrower(fns, bid):
    b = {"id": bid, "fl": None, "cash_cap": None, "addbacks": [], "pct": None, "thr": {}, "step": None,
         "cure_days": None, "amends": [], "drafts": [], "notices": [], "end": None, "figs": {}}
    for line in fns["read_agreement"]({"borrower_id": bid}).splitlines():
        line = line.strip()
        if line == T["def_debt_incl"].strip():
            b["fl"] = True
        elif line == T["def_debt_excl"].strip():
            b["fl"] = False
        elif (m := _fm(R["net_cap"], line)):
            b["cash_cap"] = _n(m["cap"])
        elif (m := _fm(R["ebitda_cap"], line)):
            b["addbacks"], b["pct"] = [LABEL2KEY[x] for x in m["addbacks"].split("; ")], int(m["pct"])
        elif (m := _fm(R["ebitda_nocap"], line)):
            b["addbacks"] = [LABEL2KEY[x] for x in m["addbacks"].split("; ")]
        elif (m := _fm(R["cov_step"], line)):
            b["step"] = (float(m["max1"]), track.pdate(m["date"]), float(m["max2"]))
            b["thr"]["net_leverage"] = None
        elif (m := _fm(R["cov_leverage"], line)):
            b["thr"]["net_leverage"] = float(m["max"])
        elif (m := _fm(R["cov_coverage"], line)):
            b["thr"]["interest_coverage"] = float(m["min"])
        elif (m := _fm(R["cov_current"], line)):
            b["thr"]["current_ratio"] = float(m["min"])
        elif (m := _fm(R["cure_right"], line)):
            b["cure_days"] = int(m["days"])
    for line in fns["read_amendments"]({"borrower_id": bid}).splitlines():
        for key in ("amend_leverage", "amend_coverage", "amend_current", "amend_cap", "amend_fl_excl", "amend_fl_incl"):
            if (m := _fm(R[key], line)):
                b["amends"].append({"key": key, "from": track.pdate(m["from"]), **{k: v for k, v in m.groupdict().items()
                                                                                  if k in ("max", "min", "pct")}})
                break
        else:
            if (m := _fm(R["amend_draft"], line)):
                b["drafts"].append(float(m["max"]))
    for line in fns["read_correspondence"]({"borrower_id": bid}).splitlines():
        if (m := _fm(R["request"], line)):
            b["end"] = track.pdate(m["end"])
        elif (m := _fm(R["cure_notice"], line)):
            b["notices"].append((track.pdate(m["date"]), _n(m["amount"]), track.pdate(m["end"])))
    st = fns["read_statements"]({"borrower_id": bid}).splitlines()
    cols = [c.strip() for c in st[1].strip("|").split("|")][1:]
    for row in st[3:]:
        cells = [c.strip() for c in row.strip("|").split("|")]
        key = LABEL2KEY[cells[0]]
        for col, v in zip(cols, cells[1:]):
            if v != "-":
                b["figs"].setdefault(col, {})[key] = _n(v)
    b["cols"] = cols
    return b


def solve(b, skip=()):
    """Answer for one parsed borrower; skip names rules a careless reader ignores (for the wrong baselines)."""
    end = b["end"]
    col = b["cols"][-1] if "period" in skip else end.isoformat()
    if "prior_year" in skip:
        col = b["cols"][max(0, b["cols"].index(end.isoformat()) - 1)] if end.isoformat() != b["cols"][0] else b["cols"][1]
    fig = b["figs"].get(col, {})
    amends = [] if "amend" in skip else [a for a in b["amends"] if "amend_dates" in skip or a["from"] <= end]
    eff = {}
    for a in sorted(amends, key=lambda a: a["from"]):
        eff[a["key"]] = a
    fl = b["fl"]
    fls = [a for a in amends if a["key"] in ("amend_fl_excl", "amend_fl_incl")]
    if fls:
        fl = max(fls, key=lambda a: a["from"])["key"] == "amend_fl_incl"
    if "fl" in skip:
        fl = True
    pct = int(eff["amend_cap"]["pct"]) if "amend_cap" in eff else b["pct"]
    if "addback_cap" in skip:
        pct = None
    addbacks = track.ADDBACKS if "addbacks" in skip else b["addbacks"]
    cure = 0
    if "cure" in skip:
        cure = sum(n[1] for n in b["notices"])
    elif b["cure_days"]:
        cure = sum(a for d, a, e in b["notices"] if e == end and 0 < (d - end).days <= b["cure_days"])
    tested = list(b["thr"])
    d = {"fl": fl, "cash_cap": 0 if "gross" in skip else None if "cash_cap" in skip else b["cash_cap"], "addbacks": addbacks, "pct": pct,
         "cure": cure, "tested": tested}
    ratios = track.compute(fig, d)
    out = {}
    for c in tested:
        if c == "net_leverage":
            if "amend_leverage" in eff:
                t = float(eff["amend_leverage"]["max"])
            elif b["step"]:
                t = b["step"][0] if ("step" in skip or end <= b["step"][1]) else b["step"][2]
            else:
                t = b["thr"][c]
            if "draft" in skip and b["drafts"]:
                t = b["drafts"][-1]
            ok = ratios[c] <= t
        else:
            k = {"interest_coverage": "amend_coverage", "current_ratio": "amend_current"}[c]
            t = float(eff[k]["min"]) if k in eff else b["thr"][c]
            ok = ratios[c] >= t
        out[c] = {"ratio": round(ratios[c], 2), "pass": ok}
    out["compliant"] = all(out[c]["pass"] for c in tested)
    return out


def read_world(task):
    fns = {t["name"]: t["fn"] for t in track.tools(task["world_dir"])}
    ids, off = [], 0
    while True:
        out = fns["list_borrowers"]({"offset": off, "limit": 100})
        ids += re.findall(r"^(BR-\d{4}) \|", out, re.M)
        m = re.search(r"next offset (\d+)\)$", out)
        if not m:
            break
        off = int(m.group(1))
    return [parse_borrower(fns, x) for x in ids]


def answer(task, skip=()):
    return "```json\n" + json.dumps({b["id"]: solve(b, skip) for b in read_world(task)}) + "\n```"


def oracle(task):
    return answer(task)


PARTIAL = {"latest_year": ("period",), "no_amendments": ("amend",), "amendment_dates_ignored": ("amend_dates",),
           "fl_always_in": ("fl",), "no_cash_cap": ("cash_cap",), "no_addback_cap": ("addback_cap",),
           "all_addbacks": ("addbacks",), "any_cure_counts": ("cure",), "no_step_down": ("step",),
           "draft_counts": ("draft",)}


def wrong(task):
    bs = read_world(task)
    out = {"regex": ("period", "amend", "fl", "cash_cap", "addback_cap", "addbacks", "cure", "step", "draft"),
           "gross_leverage": ("gross",), "adjacent_year_column": ("prior_year",),
           "naive_reader": ("period", "amend", "cure", "draft"),
           "headline_figures": ("amend", "fl", "cash_cap", "addback_cap", "addbacks", "cure", "step"),
           **PARTIAL}
    res = {k: "```json\n" + json.dumps({b["id"]: solve(b, v) for b in bs}) + "\n```" for k, v in out.items()}
    # a shifted answer: right numbers attached to the next borrower
    ans = {b["id"]: solve(b) for b in bs}
    ids = list(ans)
    res["shifted_ids"] = json.dumps({ids[(i + 1) % len(ids)]: ans[x] for i, x in enumerate(ids)})
    return res


def unit_prompt(task, bid):
    g = common.gold_of(task)["borrowers"][bid]
    ex = json.dumps({bid: {"net_leverage": {"ratio": 2.5, "pass": True}, "compliant": True}})
    return "\n\n".join([common.fill(T["unit_prompt"], id=bid, example=ex), T["rule_general"],
                        common.fill(T["rule_answer"], example=ex)]), g


def units(task, k, rng):
    g = common.gold_of(task)["borrowers"]
    out = []
    for bid in rng.sample(sorted(g), min(k, len(g))):
        prompt, gb = unit_prompt(task, bid)

        def check(text, bid=bid, gb=gb):
            p = track.parse_answer(text).get(bid)
            if p is None:
                return None
            return track.unit_ok(gb, p)[0]
        out.append({"id": f"{task['id']}:{bid}", "kind": "+".join(sorted(set(gb["kinds"]))) or "plain",
                    "prompt": prompt, "gold": {c: v for c, v in gb.items() if c in track.COVS or c == "compliant"},
                    "check": check})
    return out
