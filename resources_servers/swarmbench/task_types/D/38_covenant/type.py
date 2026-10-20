"""Type 38 covenant: financial covenant compliance review over N borrowers with real SEC EDGAR figures.

Unit = one borrower. Each borrower is a real filer (fictional alias) whose reported FY2022-FY2024 figures (USD
thousands, XBRL frames, see data/SOURCES.json) sit next to a covenant excerpt, an amendment file and a correspondence
file built from K3-written clause templates with per-borrower parameters. Gold is computed by code from the same
numbers. Task-facing prose is K3's (text/k3.json, see README); ids, names, dates and numbers are code."""
import datetime as dt
import json
import random
import re
import threading
from pathlib import Path

from .. import common

TYPE_ID = "38_covenant"
TIERS = {"easy": (8, 1800), "medium": (24, 1200), "hard": (90, 1200)}
HERE = Path(__file__).parent
T = common.load_text(HERE)
# 2026-10-14 audit fix: K3 rewrote cure_notice so {date} is explicitly the receipt date (text/request_cure.json)
T.update(json.loads((HERE / "text" / "k3_cure.json").read_text()))
POOL = json.loads((HERE / "data" / "companies.json").read_text())
YEARS = ["2022", "2023", "2024"]
LINES = ["revenue", "op_income", "da", "sbc", "restructuring", "impairment", "interest", "net_income", "cash", "ca",
         "cl", "ltd_c", "ltd_nc", "stb", "fl_c", "fl_nc", "ol_nc"]
ADDBACKS = ["sbc", "restructuring", "impairment"]
COVS = ["net_leverage", "interest_coverage", "current_ratio"]
TOL = 0.0101
W1 = ["Alder", "Brookline", "Cedarview", "Dunmore", "Elmstead", "Foxhollow", "Granby", "Hawthorne", "Inverness",
      "Kestrel", "Larchmont", "Meridian", "Northcote", "Oakhurst", "Penrose", "Quillon", "Ravenscroft", "Stanmore",
      "Thornfield", "Upton", "Vale", "Westbury", "Yarrow", "Ashgrove", "Belmont", "Carrick", "Dovecote", "Easton",
      "Fairlight", "Glenmore", "Holloway", "Ironbridge", "Juniper", "Kingsmere", "Lockwood", "Millbrook"]
W2 = ["Industries", "Holdings", "Technologies", "Brands", "Systems", "Materials", "Health", "Logistics", "Energy",
      "Manufacturing", "Services", "Products", "Networks", "Foods", "Instruments"]
W3 = ["Inc.", "Corp.", "LLC", "Group, Inc.", "Company"]


def fdate(d):
    return f"{d:%B} {d.day}, {d.year}"


def pdate(s):
    return dt.datetime.strptime(s.strip(), "%B %d, %Y").date()


def amt(x):
    return f"{x:,}"


def r2(x):
    return f"{x:.2f}"


def _thr(r, kind, passes, rng):
    """A threshold with margin >= 0.03 from ratio r: kind 'max' or 'min'; passes = the covenant is met."""
    for _ in range(100):
        f = rng.uniform(1.05, 1.6) if (kind == "max") == passes else rng.uniform(0.6, 0.95)
        t = round(max(0.25, r * f + (0.1 if f > 1 else -0.1)), 2)
        if abs(t - r) >= 0.03 and t > 0 and ((t >= r) if kind == "max" else (t <= r)) == passes:
            return t
    return round(r + 0.5, 2) if (kind == "max") == passes else round(max(0.05, r - 0.5), 2)


def compute(fig, d):
    """Exact covenant ratios for one borrower from its test-year figures fig and effective definitions d."""
    g = lambda k: fig.get(k, 0)  # noqa: E731
    debt = g("ltd_nc") + g("ltd_c") + g("stb") + ((g("fl_c") + g("fl_nc")) if d["fl"] else 0)
    cash = min(g("cash"), d["cash_cap"]) if d["cash_cap"] is not None else g("cash")
    base = g("op_income") + g("da")
    add = sum(g(k) for k in d["addbacks"])
    if d["pct"] is not None:
        add = min(add, base * d["pct"] / 100)
    ebitda = base + add
    out = {"net_leverage": (debt - cash) / (ebitda + d["cure"]), "interest_coverage": ebitda / g("interest"),
           "current_ratio": g("ca") / g("cl")}
    return {c: out[c] for c in d["tested"]}


def _borrower(i, seed, used):
    rng = random.Random(f"covenant:{seed}:{i}")
    perm = random.Random(f"covenant:{seed}").sample(range(len(POOL)), len(POOL))
    co = POOL[perm[i % len(POOL)]]
    while True:
        name = f"{rng.choice(W1)} {rng.choice(W2)} {rng.choice(W3)}"
        if name not in used:
            used.add(name)
            break
    b = {"id": f"BR-{i + 1:04d}", "name": name, "cik": co["cik"], "kinds": []}
    year = rng.choice(YEARS)
    fig = co["years"][year]
    end = dt.date.fromisoformat(fig["end"])
    b["year"], b["end"] = year, end
    kinds = b["kinds"]
    if year != "2024":
        kinds.append("old_period")
    has_fl = fig.get("fl_c", 0) + fig.get("fl_nc", 0) > 0
    # original definitions
    fl0 = rng.random() < 0.5
    cash_cap = None
    if rng.random() < 0.5:
        c = fig["cash"]
        cash_cap = max(1000, int(round(c * rng.uniform(0.15, 0.8), -3))) if rng.random() < 0.75 else \
            int(round(c * rng.uniform(1.3, 3.0), -3)) + 1000
        if cash_cap < c:
            kinds.append("cash_cap")
    reported = [k for k in ADDBACKS if fig.get(k, 0) > 0]
    addbacks = sorted(rng.sample(ADDBACKS, rng.choice([1, 2, 2, 3])), key=ADDBACKS.index)
    if set(reported) - set(addbacks):
        kinds.append("addback_subset")
    pct0 = rng.choice([3, 5, 5, 8, 10, 15]) if rng.random() < 0.6 else None
    tested = ["net_leverage"] + [c for c, p in (("interest_coverage", 0.65), ("current_ratio", 0.5)) if rng.random() < p]
    step = None
    if rng.random() < 0.35:
        step = dt.date(*rng.choice([(2022, 12, 31), (2023, 6, 30), (2023, 12, 31)]))
        kinds.append("step_down")
    cure_days = rng.choice([30, 45, 60, 90]) if rng.random() < 0.4 else None
    # amendments: (subject, from-date, value placeholder); value filled after ratios are known
    subjects = ["leverage", "cap", "fl"] + [{"interest_coverage": "coverage", "current_ratio": "current"}[c]
                                            for c in tested[1:]]
    amends = []
    for subj in rng.sample(subjects, rng.choice([0, 1, 1, 2, 2, 3])):
        for _ in range(rng.choice([1, 1, 2])):
            applies = rng.random() < 0.6
            frm = end - dt.timedelta(days=rng.randint(0, 500)) if applies else end + dt.timedelta(days=rng.randint(1, 400))
            frm = dt.date(frm.year, frm.month, 1) - dt.timedelta(days=1)  # month ends
            if frm > end:
                applies = False
            amends.append({"subj": subj, "from": frm, "applies": frm <= end})
    if rng.random() < 0.25:
        amends.append({"subj": "draft", "from": None, "applies": False})
        kinds.append("draft")
    seen = set()
    amends = [a for a in amends if (a["subj"], a["from"]) not in seen and not seen.add((a["subj"], a["from"]))]
    rng.shuffle(amends)
    eff = {}  # subject -> governing amendment
    for a in amends:
        if a["applies"] and (a["subj"] not in eff or a["from"] > eff[a["subj"]]["from"]):
            eff[a["subj"]] = a
    for a in amends:
        if a["subj"] in ("cap", "fl", "leverage", "coverage", "current"):
            kinds.append("amend_applies" if a["applies"] else "amend_future")
        if a["subj"] == "cap":
            a["pct"] = rng.choice([3, 5, 8, 10, 15, 20])
        if a["subj"] == "fl":
            a["incl"] = rng.random() < 0.5
    request = end + dt.timedelta(days=rng.randint(45, 100))
    for a in amends:
        if a["from"] is not None and a["applies"]:
            a["signed"] = min(a["from"] + dt.timedelta(days=rng.randint(-60, 40)), request - dt.timedelta(days=5))
        else:
            a["signed"] = end + dt.timedelta(days=rng.randint(-200, (request - end).days - 5))
    amends.sort(key=lambda a: a["signed"])
    fl = eff["fl"]["incl"] if "fl" in eff else fl0
    if fl and has_fl:
        kinds.append("fl_included")
    if "cap" in eff and pct0 is None:  # a cap amendment needs a capped definition to amend
        pct0 = rng.choice([10, 20])
    pct = eff["cap"]["pct"] if "cap" in eff else pct0
    # cure notices
    notices = []
    if cure_days or rng.random() < 0.25:
        base = fig["op_income"] + fig["da"]
        amount = int(round(base * rng.uniform(0.04, 0.25), -2)) or 100
        r = rng.random()
        if cure_days and r < 0.6:
            notices.append((end + dt.timedelta(days=rng.randint(5, cure_days)), amount, end, True))
        elif cure_days and r < 0.8:
            notices.append((end + dt.timedelta(days=cure_days + rng.randint(1, 40)), amount, end, False))
        elif cure_days:
            other = dt.date.fromisoformat(co["years"][rng.choice([y for y in YEARS if y != year])]["end"])
            notices.append((other + dt.timedelta(days=rng.randint(5, cure_days)), amount, other, False))
        else:
            notices.append((end + dt.timedelta(days=rng.randint(5, 40)), amount, end, False))
    if not set(kinds) - {"old_period"}:  # every borrower carries at least one definitional or file trap
        cash_cap = max(1000, int(round(fig["cash"] * rng.uniform(0.15, 0.6), -3)))
        if cash_cap < fig["cash"]:
            kinds.append("cash_cap")
    cure = sum(n[1] for n in notices if n[3])
    if notices:
        kinds.append("cure_valid" if cure else "cure_invalid")
    d = {"fl": fl, "cash_cap": cash_cap, "addbacks": addbacks, "pct": pct, "cure": cure, "tested": tested}
    ratios = compute(fig, d)
    base_add = sum(fig.get(k, 0) for k in addbacks)
    if pct is not None and base_add > (fig["op_income"] + fig["da"]) * pct / 100:
        kinds.append("addback_cap")
    # thresholds: the governing one gets pass/breach with margin; superseded ones are random around the ratio
    passes = {c: rng.random() < 0.72 for c in tested}
    thr = {}
    lev_kind = "max"
    r = ratios["net_leverage"]
    if notices and r > 0.3:
        alt = compute(fig, dict(d, cure=sum(n[1] for n in notices)) if not cure else dict(d, cure=0))["net_leverage"]
        lo, hi = sorted([r, alt])
        if hi - lo >= 0.08:  # a threshold between the cured and uncured ratio makes the cure decision count
            t = round(rng.uniform(lo + 0.03, hi - 0.03), 2)
            thr["net_leverage"] = t
            passes["net_leverage"] = r <= t
    if "net_leverage" not in thr:
        thr["net_leverage"] = _thr(r, lev_kind, passes["net_leverage"], rng) if r > 0.3 else round(rng.uniform(2, 5), 2)
        passes["net_leverage"] = r <= thr["net_leverage"]
    for c in tested[1:]:
        thr[c] = _thr(ratios[c], "min", passes[c], rng)
    other = lambda c, t: (_thr(ratios[c], "max" if c == "net_leverage" else "min",  # noqa: E731
                                not ((ratios[c] <= t) if c == "net_leverage" else (ratios[c] >= t)), rng)
                          if c == "net_leverage" and ratios[c] > 0.3 or c != "net_leverage"
                          else round(rng.uniform(0.2, 0.3), 2))
    subj_cov = {"leverage": "net_leverage", "coverage": "interest_coverage", "current": "current_ratio"}
    orig = {}
    for c in tested:
        s = next(k for k, v in subj_cov.items() if v == c)
        orig[c] = thr[c] if s not in eff else other(c, thr[c])
    lev_step = None
    if step:
        gov_first = end <= step and "leverage" not in eff
        a1, a2 = (orig["net_leverage"], other("net_leverage", orig["net_leverage"]))
        lev_step = (a1, step, a2) if gov_first else (a2, step, a1)
    for a in amends:
        if a["subj"] in subj_cov:
            c = subj_cov[a["subj"]]
            a["val"] = thr[c] if eff.get(a["subj"]) is a else other(c, thr[c])
        if a["subj"] == "draft":
            a["val"] = other("net_leverage", thr["net_leverage"]) if r > 0.3 else round(r + 1, 2)
    b.update(request=request, fig=fig, d=d, ratios=ratios, thr=thr, orig=orig, step=lev_step, fl0=fl0, pct0=pct0, cash_cap=cash_cap,
             addbacks=addbacks, cure_days=cure_days, amends=amends, notices=notices, tested=tested, co=co)
    b["gold"] = {**{c: {"ratio": round(ratios[c], 4),
                        "pass": (ratios[c] <= thr[c]) if c == "net_leverage" else (ratios[c] >= thr[c])}
                    for c in tested}}
    b["gold"]["compliant"] = all(b["gold"][c]["pass"] for c in tested)
    return b


def render(b, rng):
    f = common.fill
    L = [f(T["agreement_title"], name=b["name"], date=fdate(dt.date(2020, 1, 1) + dt.timedelta(days=rng.randint(0, 700))))]
    L.append(T["def_debt_incl"] if b["fl0"] else T["def_debt_excl"])
    L.append(f(T["def_net_cap"], cap=amt(b["cash_cap"])) if b["cash_cap"] is not None else T["def_net_full"])
    labels = "; ".join(T["labels"][k] for k in b["addbacks"])
    L.append(f(T["def_ebitda_cap"], addbacks=labels, pct=b["pct0"]) if b["pct0"] is not None
             else f(T["def_ebitda_nocap"], addbacks=labels))
    if "interest_coverage" in b["tested"]:
        L.append(T["def_coverage"])
    if "current_ratio" in b["tested"]:
        L.append(T["def_current"])
    if b["step"]:
        L.append(f(T["cov_leverage_step"], max1=r2(b["step"][0]), date=fdate(b["step"][1]), max2=r2(b["step"][2])))
    else:
        L.append(f(T["cov_leverage"], max=r2(b["orig"]["net_leverage"])))
    if "interest_coverage" in b["tested"]:
        L.append(f(T["cov_coverage"], min=r2(b["orig"]["interest_coverage"])))
    if "current_ratio" in b["tested"]:
        L.append(f(T["cov_current"], min=r2(b["orig"]["current_ratio"])))
    if b["cure_days"]:
        L.append(f(T["cure_right"], days=b["cure_days"]))
    agreement = "\n".join(L)
    A = []
    for k, a in enumerate(b["amends"], 1):
        kw = {"k": k, "signed": fdate(a["signed"])}
        if a["subj"] != "draft":
            kw["from"] = fdate(a["from"])
        if a["subj"] == "leverage":
            A.append(f(T["amend_leverage"], max=r2(a["val"]), **kw))
        elif a["subj"] == "coverage":
            A.append(f(T["amend_coverage"], min=r2(a["val"]), **kw))
        elif a["subj"] == "current":
            A.append(f(T["amend_current"], min=r2(a["val"]), **kw))
        elif a["subj"] == "cap":
            A.append(f(T["amend_cap"], pct=a["pct"], **kw))
        elif a["subj"] == "fl":
            A.append(f(T["amend_fl_incl" if a["incl"] else "amend_fl_excl"], **kw))
        else:
            A.append(f(T["amend_draft"], max=r2(a["val"]), **kw))
    amendments = "\n".join(A) if A else T["no_amendments"]
    C = [(b["request"], f(T["request"], date="{date}", end=fdate(b["end"])))]
    for day, amount, end, _ in b["notices"]:
        C.append((day, f(T["cure_notice"], date="{date}", amount=amt(amount), end=fdate(end))))
    for _ in range(rng.randint(2, 4)):
        C.append((b["end"] + dt.timedelta(days=rng.randint(-300, 200)), rng.choice(T["pad_notes"])))
    C.sort(key=lambda x: x[0])
    corr = "\n".join(f(t, date=fdate(day)) for day, t in C)
    co = b["co"]
    head = "| item | " + " | ".join(co["years"][y]["end"] for y in YEARS) + " |\n|---|---|---|---|"
    rows = [f"| {T['labels'][k]} | " + " | ".join(amt(co["years"][y][k]) if k in co["years"][y] else "-" for y in YEARS)
            + " |" for k in LINES]
    stmts = f(T["statement_title"], name=b["name"]) + "\n" + head + "\n" + "\n".join(rows)
    return {"id": b["id"], "name": b["name"], "agreement": agreement, "amendments": amendments,
            "correspondence": corr, "statements": stmts}


def example(b):
    return json.dumps({b["id"]: {"net_leverage": {"ratio": 2.5, "pass": True},
                                 "interest_coverage": {"ratio": 6.25, "pass": True}, "compliant": True}})


def generate(n, seed, out_dir):
    used = set()
    bs = [_borrower(i, seed, used) for i in range(n)]
    world = [render(b, random.Random(f"covenant-render:{seed}:{i}")) for i, b in enumerate(bs)]
    gold = {"borrowers": {b["id"]: {**b["gold"], "kinds": b["kinds"], "cik": b["cik"], "year": b["year"]} for b in bs}}
    prompt = "\n\n".join([common.fill(T["intro"], bank=T["bank"], n=n, first=bs[0]["id"], last=bs[-1]["id"]),
                          T["rule_general"], common.fill(T["rule_answer"], example=example(bs[0]))])
    return common.finish(TYPE_ID, n, seed, out_dir, prompt, gold, TIERS, files={"borrowers.json": world})


# ---------- tools ----------
_cache, _lock = {}, threading.Lock()


def _world(world_dir):
    with _lock:
        if world_dir not in _cache:
            bs = json.loads((Path(world_dir) / "borrowers.json").read_text())
            _cache[world_dir] = ({b["id"]: b for b in bs}, [b["id"] for b in bs])
        return _cache[world_dir]


def tools(world_dir):
    def list_borrowers(a):
        by, order = _world(world_dir)
        off, lim = common.page_args(a, 100, 100)
        rows = [f"{x} | {by[x]['name']}" for x in order[off:off + lim]]
        nxt = off + len(rows)
        more = f"\n(borrowers {off}-{nxt - 1} of {len(order)}; next offset {nxt})" if nxt < len(order) else ""
        return "borrower | name\n" + "\n".join(rows) + more

    def reader(field):
        def fn(a):
            by, _ = _world(world_dir)
            x = str(a["borrower_id"]).strip().upper()
            if x not in by:
                raise KeyError(f"unknown borrower {x}")
            return by[x][field]
        return fn

    D = T["tools"]
    S = {"type": "string"}
    I = {"type": "integer"}
    one = {"borrower_id": S}
    return [common.tool("list_borrowers", D["list_borrowers"], {"offset": I, "limit": I}, [], list_borrowers)] + \
        [common.tool(f"read_{k}", D[f"read_{k}"], one, ["borrower_id"], reader(k))
         for k in ("agreement", "amendments", "correspondence", "statements")] + \
        [common.calculator_tool(D["calculator"])]


# ---------- scoring ----------
_BID = re.compile(r"BR\W?0*(\d{1,4})", re.I)


def _bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "yes", "pass", "passed", "met", "compliant", "in compliance"):
        return True
    if s in ("false", "no", "fail", "failed", "breach", "breached", "not met", "non-compliant", "noncompliant"):
        return False
    return None


def parse_answer(answer):
    obj = common.last_json(answer, lambda o: isinstance(o, dict) and any(_BID.fullmatch(str(k).strip()) for k in o))
    out = {}
    for k, v in (obj or {}).items():
        m = _BID.fullmatch(str(k).strip())
        if not m or not isinstance(v, dict):
            continue
        e = {"compliant": _bool(v.get("compliant"))}
        for c in COVS:
            x = v.get(c)
            if isinstance(x, dict):
                e[c] = {"ratio": common.num(x.get("ratio")), "pass": _bool(x.get("pass"))}
        out[f"BR-{int(m.group(1)):04d}"] = e
    return out


def unit_ok(g, p):
    """(all-exact, items right, items) for one borrower's gold g and parsed answer p (None if missing)."""
    tested = [c for c in COVS if c in g]
    items = 2 * len(tested) + 1
    if not p:
        return False, 0, items
    right = int(p.get("compliant") is not None and p["compliant"] == g["compliant"])
    for c in tested:
        x = p.get(c) or {}
        right += x.get("ratio") is not None and abs(x["ratio"] - g[c]["ratio"]) <= TOL
        right += x.get("pass") is not None and x["pass"] == g[c]["pass"]
    return right == items, right, items


def score(task, answer):
    g = common.gold_of(task)["borrowers"]
    p = parse_answer(answer)
    ok, part = 0, 0.0
    for bid, gb in g.items():
        exact, right, items = unit_ok(gb, p.get(bid))
        ok += exact
        part += right / items
    n = len(g)
    return {"score": ok / n, "score_exact": float(ok == n), "score_partial": part / n, "n_correct": ok, "n_units": n}
