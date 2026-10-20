"""Type 26 harmonize: calendar-year energy (MWh) and cost (USD) per facility from two fiscal-year reports in mixed
units and currencies, with notes that change figures. One facility = one unit. Task text: K3 templates (text/)."""
import json
import math
import random
import re
from pathlib import Path

from .. import common

TYPE_ID = "26_harmonize"
TIERS = {"easy": (8, 1800), "medium": (40, 1200), "hard": (80, 1200)}
YEAR = 2025
REL_TOL = 0.001  # every trap moves energy or cost by >= 1% (checked in _facility)
T = common.load_text(Path(__file__).parent)
MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
# MWh per unit (stated exactly in the conventions document)
FACTORS = {"kWh": 0.001, "MWh": 1.0, "GJ": 0.2777778, "therms": 0.0293071, "MMBtu": 0.293071, "m3": 0.0105,
           "litres": 0.01, "US gallons": 0.0378541, "tonnes": 6.9, "short tons": 6.2596}
KINDS = {"electricity": ["kWh", "MWh"], "gas": ["therms", "MMBtu", "m3", "GJ"], "diesel": ["litres", "US gallons"],
         "coal": ["tonnes", "short tons"], "steam": ["GJ", "MMBtu"]}
PRICE = {"electricity": 120, "gas": 40, "diesel": 110, "coal": 25, "steam": 55}  # USD per MWh, roughly
FX0 = {"EUR": 1.08, "GBP": 1.27, "JPY": 0.0068, "CAD": 0.73, "CHF": 1.12, "AUD": 0.66, "SEK": 0.095, "USD": 1.0}
CURS = ["EUR", "GBP", "JPY", "CAD", "CHF", "AUD", "SEK", "USD"]
CUR_W = [4, 3, 3, 3, 2, 2, 2, 1]
CITIES = ["Aldmere", "Brackton", "Corvale", "Dunhollow", "Eastfen", "Farrowby", "Glenmarsh", "Harlowe", "Ivelton",
          "Jarrow Bay", "Kestrel Point", "Lindqvist", "Marrowgate", "Northwick", "Oakhurst", "Pellham", "Quenby",
          "Ravensgill", "Saltmoor", "Thornbury", "Ulverby", "Vantmoor", "Westerholm", "Yarrowdale"]
PLANTS = ["Press Works", "Casting Plant", "Assembly Hall", "Polymer Plant", "Packaging Works", "Machining Centre",
          "Finishing Plant", "Forge", "Coating Line", "Distribution Hub"]
TRAPS = ["scale", "mult", "corr", "sub", "est"]
BIG_UNITS = {"kWh", "m3", "litres", "therms"}


def mname(ym):
    return f"{MON[ym[1] - 1]} {ym[0]}"


def months_of(fy, end):
    """Months of the fiscal year ending in month `end` of year fy, oldest first."""
    out, y, m = [], fy, end
    for _ in range(12):
        out.append((y, m))
        y, m = (y, m - 1) if m > 1 else (y - 1, 12)
    return out[::-1]


def fx_table(seed):
    rng = random.Random(f"harm-fx:{seed}")
    out = {}
    for cur in CURS:
        r, rates = FX0[cur], {}
        for y in (2024, 2025, 2026):
            for m in range(1, 13):  # volatile enough that converting at the annual average misses most facilities
                r = r if cur == "USD" else r * math.exp(rng.gauss(0, 0.045))
                rates[(y, m)] = float(f"{r:.5g}") if cur != "USD" else 1.0
        out[cur] = rates
    return out


def fmt_q(v, scaled):
    return f"{v / 1000:,.3f}" if scaled else f"{v:,}"


def fmt_c(c, cur):
    return f"{c:,.0f}" if cur == "JPY" else f"{c:,.2f}"


def rc(c, cur):
    return float(round(c)) if cur == "JPY" else round(c, 2)


def compute(f, fx, ignore=(), fiscal=False, fx_mode="month"):
    """Energy (MWh) and cost (USD) of one facility from its facts (what the reports say). ignore: trap kinds to
    skip; fiscal: use the fiscal year ending in YEAR instead of the calendar year; fx_mode: month|annual|none."""
    want = set(months_of(YEAR, f["fy_end"])) if fiscal else {(YEAR, m) for m in range(1, 13)}
    rates = fx[f["currency"]]
    avg = sum(rates[(YEAR, m)] for m in range(1, 13)) / 12
    e = c = 0.0
    for fy, rep in f["reports"].items():
        for mid, met in rep["meters"].items():
            if mid in f["sub"] and "sub" not in ignore:
                continue
            for ym, (q, cost) in met["rows"].items():
                if ym not in want:
                    continue
                if (fy, mid) in f["scale"] and "scale" not in ignore:
                    q *= 1000
                mu = f["mult"].get((fy, mid))
                if mu and ym >= mu[0] and "mult" not in ignore:
                    q *= mu[1]
                if (fy, mid, ym) in f["corr"] and "corr" not in ignore:
                    q = f["corr"][(fy, mid, ym)]
                if (mid, ym) in f["est"] and "est" not in ignore:
                    q, cost = f["est"][(mid, ym)]
                e += q * FACTORS[met["unit"]]
                rate = rates[ym] if fx_mode == "month" else avg if fx_mode == "annual" else 1.0
                c += cost * rate
    return e, c


def _close(a, b, tol=REL_TOL):
    return abs(a - b) <= tol * max(abs(b), 1e-9)


def _facility(seed, i, fx):
    """One facility: truth, rendered facts and render plan. Resampled until every trap matters (>= 1%)."""
    rng = random.Random(f"harm:{seed}:{i}")
    for _ in range(200):
        f = _try(rng, i, fx)
        if f:
            return f
    raise RuntimeError("could not build facility")


def _try(rng, i, fx):
    fid = f"FAC{i + 1:04d}"
    cur = rng.choices(CURS, CUR_W)[0]
    end = rng.randint(2, 10)
    fys = [YEAR, YEAR + 1]
    allm = months_of(YEAR, end) + months_of(YEAR + 1, end)
    ntr = rng.choice([2, 2, 3, 3, 4])
    traps = set(rng.sample(TRAPS, ntr))
    # At most 2 meters per facility (check 5 was 16/20 from arithmetic slips with 3-4): two billed meters, or one
    # billed meter plus its sub-meter (listed but excluded).
    kinds = ["electricity"] + ([] if "sub" in traps else [rng.choice(["gas", "diesel", "coal", "steam"])])
    meters = []  # truth
    for k, kind in enumerate(kinds):
        base = math.exp(rng.uniform(math.log(150), math.log(900)))
        peak = rng.randint(1, 12) if kind not in ("electricity", "gas") else (7 if kind == "electricity" else 1)
        amp, yoy = rng.uniform(0.1, 0.35), rng.choice([-1, 1]) * rng.uniform(0.06, 0.25)
        units = {fy: rng.choice(KINDS[kind]) for fy in fys}
        if kind != "gas":
            units[fys[1]] = units[fys[0]] if rng.random() < 0.7 else units[fys[1]]
        rows = {}
        for ym in allm:
            mwh = base * (1 + amp * math.cos(2 * math.pi * (ym[1] - peak) / 12)) * (1 + yoy * (ym[0] - YEAR + ym[1] / 12))
            mwh *= rng.uniform(0.93, 1.07)
            rows[ym] = mwh
        meters.append({"id": f"M{k + 1}", "kind": kind, "units": units, "mwh": rows, "desc": rng.choice(T["meter_desc"][kind])})
    # per report table: integer quantities in that report's unit
    tabs = {}
    for fy in fys:
        for met in meters:
            u = met["units"][fy]
            q = {ym: max(1, round(met["mwh"][ym] / FACTORS[u])) for ym in months_of(fy, end)}
            cost = {ym: rc(q[ym] * FACTORS[u] * PRICE[met["kind"]] * rng.uniform(0.9, 1.1) / fx[cur][ym], cur) for ym in q}
            tabs[(fy, met["id"])] = {"unit": u, "q": q, "cost": cost, "shown": dict(q), "shown_cost": dict(cost),
                                     "rt": {ym: "E" if rng.random() < 0.08 else "A" for ym in q}, "mult": None}
    facts = {"currency": cur, "fy_end": end, "scale": set(), "mult": {}, "corr": {}, "sub": set(), "est": {}}
    notes = {fy: [] for fy in fys}
    used = set()
    cy = lambda fy: [ym for ym in months_of(fy, end) if ym[0] == YEAR]
    free = lambda pred: [t for t in tabs if t not in used and pred(t)]
    sub_of = None
    if "sub" in traps:
        main = meters[0]
        sid = f"M{len(meters) + 1}"
        share = rng.uniform(0.25, 0.45)
        for fy in fys:
            m0 = tabs[(fy, main["id"])]
            q = {ym: max(1, round(m0["q"][ym] * share * rng.uniform(0.85, 1.15))) for ym in m0["q"]}
            cost = {ym: rc(m0["cost"][ym] * q[ym] / m0["q"][ym], cur) for ym in q}
            tabs[(fy, sid)] = {"unit": m0["unit"], "q": q, "cost": cost, "shown": dict(q), "shown_cost": dict(cost),
                               "rt": {ym: "A" for ym in q}, "mult": None, "sub": True}
            used.add((fy, sid))
            notes[fy].append(("sub", {"sub": sid, "main": main["id"], "main2": main["id"]}))
        facts["sub"].add(sid)
        sub_of = (sid, main)
    if "est" in traps:
        c = free(lambda t: t[0] == YEAR)
        if not c:
            return None
        t = rng.choice(c)
        used.add(t)
        tab = tabs[t]
        ym = rng.choice(cy(YEAR))
        fac = rng.choice([rng.uniform(0.35, 0.65), rng.uniform(1.5, 2.2)])
        tab["shown"][ym] = max(1, round(tab["q"][ym] * fac))
        tab["shown_cost"][ym] = rc(tab["cost"][ym] * fac, cur)
        tab["rt"][ym] = "E"
        facts["est"][(t[1], ym)] = (float(tab["q"][ym]), tab["cost"][ym])
        notes[YEAR + 1].append(("est", {"meter": t[1], "month": mname(ym), "value": f"{tab['q'][ym]:,}",
                                        "unit": tab["unit"], "cost": fmt_c(tab["cost"][ym], cur), "cur": cur}))
    if "corr" in traps:
        c = free(lambda t: True)
        if not c:
            return None
        t = rng.choice(c)
        used.add(t)
        tab = tabs[t]
        ym = rng.choice(cy(t[0]))
        tab["shown"][ym] = max(1, round(tab["q"][ym] * rng.choice([0.1, 0.4, 2.5, 10])))
        facts["corr"][(t[0], t[1], ym)] = float(tab["q"][ym])
        notes[t[0]].append(("corr", {"meter": t[1], "month": mname(ym), "value": f"{tab['q'][ym]:,}", "unit": tab["unit"]}))
    if "mult" in traps:
        c = free(lambda t: True)
        if not c:
            return None
        t = rng.choice(c)
        used.add(t)
        tab = tabs[t]
        ms = months_of(t[0], end)
        cym = cy(t[0])
        start = rng.choice(cym[:-1] if t[0] == YEAR + 1 and len(cym) > 1 else cym)
        mult = rng.choice([2, 4, 5, 10, 20])
        for ym in ms:
            if ym >= start:
                r = max(1, round(tab["q"][ym] / mult))
                tab["q"][ym] = r * mult
                tab["shown"][ym] = r
        tab["mult"] = (start, mult)
        facts["mult"][t] = (start, float(mult))
        notes[t[0]].append(("mult", {"meter": t[1], "month": mname(start), "month2": mname(start), "mult": mult}))
    if "scale" in traps:
        c = [t for t in tabs if (t not in used or tabs[t]["mult"]) and not tabs[t].get("sub")
             and tabs[t]["unit"] in BIG_UNITS and min(tabs[t]["q"].values()) >= 5000
             and not any(k[:2] == t for k in facts["corr"]) and not any(k[0] == t[1] and t[0] == YEAR for k in facts["est"])]
        if not c:
            return None
        t = rng.choice(c)
        used.add(t)
        facts["scale"].add(t)
    # decoy and benign notes
    for fy in fys:
        met = rng.choice(meters)
        ym = rng.choice(months_of(fy, end))
        tab = tabs[(fy, met["id"])]
        notes[fy].append(("decoy_budget", {"meter": met["id"], "month": mname(ym),
                                           "value": fmt_c(tab["cost"][ym] * rng.uniform(0.85, 1.15), cur), "cur": cur}))
        met = rng.choice(meters)
        tab = tabs[(fy, met["id"])]
        ym = rng.choice(months_of(fy, end))
        notes[fy].append(("decoy_prior", {"month": mname(ym), "meter": met["id"],
                                          "value": f"{round(tab['q'][ym] * rng.uniform(0.8, 1.2)):,}", "unit": tab["unit"]}))
        notes[fy] += [("benign", {"text": x}) for x in rng.sample(T["benign_notes"], 2)]
        rng.shuffle(notes[fy])
    # facts as the reports state them (shown numbers parsed back from their rendering)
    facts["reports"] = {}
    for fy in fys:
        facts["reports"][fy] = {"meters": {}}
        for (tfy, mid), tab in tabs.items():
            if tfy == fy:
                sc = (fy, mid) in facts["scale"]
                rows = {ym: (float(fmt_q(tab["shown"][ym], sc).replace(",", "")), tab["shown_cost"][ym]) for ym in tab["q"]}
                facts["reports"][fy]["meters"][mid] = {"unit": tab["unit"], "rows": rows}
    # truth: all real meters, calendar months, true quantities and costs
    te = tc = 0.0
    for (fy, mid), tab in tabs.items():
        if tab.get("sub"):
            continue
        for ym in tab["q"]:
            if ym[0] == YEAR:
                te += tab["q"][ym] * FACTORS[tab["unit"]]
                tc += tab["cost"][ym] * fx[cur][ym]
    e, c = compute(facts, fx)
    assert _close(e, te, 1e-9) and _close(c, tc, 1e-9), (e, te, c, tc)
    for k in traps:  # every trap must matter
        e2, c2 = compute(facts, fx, ignore=(k,))
        if _close(e2, e, 0.01) and _close(c2, c, 0.01):
            return None
    city = CITIES[rng.randrange(len(CITIES))]
    return {"id": fid, "name": f"{city} {rng.choice(PLANTS)}", "city": city, "currency": cur, "fy_end": end,
            "meters": meters, "tabs": tabs, "notes": notes, "facts": facts, "traps": sorted(traps),
            "gold": {"energy_mwh": round(e, 6), "cost_usd": round(c, 6)}, "sub": sub_of[0] if sub_of else None,
            "pad": rng.sample(T["pad"], 7), "other_i": rng.randrange(10 ** 6)}


def _report(f, fy, n_fac):
    h, col = T["headings"], T["columns"]
    cur, end = f["currency"], f["fy_end"]
    ms = months_of(fy, end)
    other = f"FAC{(f['other_i'] % max(1, n_fac)) + 1:04d}"
    pad = [common.fill(p, name=f["name"], site=f["city"], other=other) for p in f["pad"]]
    rid = f"{f['id']}-FY{fy}"
    out = [f"# {T['company']}: {f['name']} ({f['id']}), {T['title']} FY{fy} [{rid}]", "",
           common.fill(T["period"], start=mname(ms[0]), end=mname(ms[-1])), common.fill(T["currency_line"], cur=cur), "",
           f"## {h['overview']}", " ".join(pad[:3]), "", f"## {h['meters']}",
           f"| {col['meter']} | {col['description']} | {col['unit']} |", "|---|---|---|"]
    mids = sorted({mid for (tfy, mid) in f["tabs"] if tfy == fy})
    desc = {m["id"]: m["desc"] for m in f["meters"]}
    if f["sub"]:
        desc[f["sub"]] = T["meter_desc"]["electricity"][(f["other_i"] + 1) % len(T["meter_desc"]["electricity"])]
    for mid in mids:
        out.append(f"| {mid} | {desc[mid]} | {f['tabs'][(fy, mid)]['unit']} |")
    out += ["", f"## {h['monthly']}"]
    for mid in mids:
        tab = f["tabs"][(fy, mid)]
        sc = (fy, mid) in f["facts"]["scale"]
        u = tab["unit"]
        out += ["", f"### {mid}"]
        if sc:
            out.append(common.fill(T["scale_note"], unit=u))
        out += [f"| {col['month']} | {col['consumption']} ({u}) | {col['read_type']} | {col['cost']} ({cur}) | "
                f"{col['budget']} ({cur}) | {col['prior']} ({u}) |", "|---|---|---|---|---|---|"]
        r = random.Random(f"{rid}:{mid}")
        for ym in ms:
            out.append(f"| {mname(ym)} | {fmt_q(tab['shown'][ym], sc)} | {tab['rt'][ym]} | "
                       f"{fmt_c(tab['shown_cost'][ym], cur)} | {fmt_c(tab['cost'][ym] * r.uniform(0.85, 1.15), cur)} | "
                       f"{fmt_q(round(tab['q'][ym] * r.uniform(0.75, 1.25)), sc)} |")
    out += ["", f"## {h['notes']}"]
    adj = []
    for kind, kw in f["notes"][fy]:
        line = kw["text"] if kind == "benign" else common.fill(T[kind + "_note" if kind in ("sub", "corr", "mult", "est") else kind], **kw)
        (adj if kind == "est" else out).append("- " + line)
    out += ["", f"## {h['adjustments']}"] + (adj or [T["no_adjustments"]])
    out += ["", f"## {h['background']}", " ".join(pad[3:]), ""]
    return rid, "\n".join(out)


def conventions_doc():
    rules = [common.fill(r, year=YEAR, year2=YEAR) for r in T["conventions_rules"]]
    tab = ["| Unit | MWh per unit |", "|---|---|"] + [f"| {u} | {v} |" for u, v in FACTORS.items()]
    return "\n".join([T["conventions_intro"], ""] + [f"{k + 1}. {r}" for k, r in enumerate(rules)] + [""] + tab) + "\n"


def generate(n, seed, out_dir):
    fx = fx_table(seed)
    facs = [_facility(seed, i, fx) for i in range(n)]
    files, listing = {"conventions.md": conventions_doc()}, []
    for f in facs:
        rids = []
        for fy in (YEAR, YEAR + 1):
            rid, text = _report(f, fy, n)
            files[f"reports/{rid}.md"] = text
            rids.append(rid)
        listing.append({"id": f["id"], "name": f["name"], "city": f["city"], "currency": f["currency"],
                        "fy_end": MON[f["fy_end"] - 1], "reports": rids})
    files["facilities.json"] = listing
    files["fx.json"] = {cur: {f"{y}-{m:02d}": r for (y, m), r in rates.items()} for cur, rates in fx.items()}
    gold = {"year": YEAR, "facilities": {f["id"]: f["gold"] for f in facs}, "traps": {f["id"]: f["traps"] for f in facs}}
    ex = json.dumps({facs[0]["id"]: {"energy_mwh": 1234.567, "cost_usd": 98765.43}})
    prompt = (common.fill(T["intro"], company=T["company"], n=n, first=facs[0]["id"], last=facs[-1]["id"], year=YEAR)
              + "\n\n" + common.fill(T["answer_line"], example=ex))
    return common.finish(TYPE_ID, n, seed, out_dir, prompt, gold, TIERS, files)


_cache, _lock = {}, __import__("threading").Lock()


def _world(world_dir):
    with _lock:
        if world_dir not in _cache:
            w = Path(world_dir)
            fac = json.loads((w / "facilities.json").read_text())
            reps = {p.stem: p.read_text() for p in (w / "reports").glob("*.md")}
            _cache[world_dir] = (fac, reps, (w / "conventions.md").read_text(), json.loads((w / "fx.json").read_text()))
        return _cache[world_dir]


def _ym(s):
    s = str(s).strip()
    for k, m in enumerate(MON):
        if s[:3].lower() == m.lower() and s[-4:].isdigit():
            return f"{s[-4:]}-{k + 1:02d}"
    if len(s) == 7 and s[4] == "-":
        return s
    raise ValueError(f"month must look like 'Mar 2025' or '2025-03', got {s!r}")


def tools(world_dir):
    def list_facilities(a):
        fac = _world(world_dir)[0]
        off, lim = common.page_args(a, 50, 60)  # 60 rows stay under the 8k output cap
        rows = [f"{x['id']} | {x['name']} | {x['city']} | {x['currency']} | FY ends {x['fy_end']} | {', '.join(x['reports'])}"
                for x in fac[off:off + lim]]
        more = f"\n... {len(fac) - off - lim} more; use offset={off + lim}" if off + lim < len(fac) else ""
        return f"{len(fac)} facilities\n" + "\n".join(rows) + more

    def read_report(a):
        reps = _world(world_dir)[1]
        rid = str(a["report_id"]).strip()
        if rid not in reps:
            raise KeyError(f"unknown report {rid!r}; report ids look like FAC0001-FY{YEAR}")
        off = int(a.get("offset") or 0)
        if off < 0:
            raise ValueError("offset must be >= 0")
        text = reps[rid]
        chunk = text[off:off + 7800]
        tail = f"\n[chars {off}-{off + len(chunk)} of {len(text)}; continue with offset={off + len(chunk)}]" \
            if off + len(chunk) < len(text) else ""
        return chunk + tail

    def read_fx(a):
        fx = _world(world_dir)[3]
        cur = str(a["currency"]).strip().upper()
        if cur not in fx:
            raise KeyError(f"unknown currency {cur!r}; known: {', '.join(sorted(fx))}")
        lo = _ym(a["start_month"]) if a.get("start_month") else "0000-00"
        hi = _ym(a["end_month"]) if a.get("end_month") else "9999-99"
        rows = [f"| {k} | {v} |" for k, v in sorted(fx[cur].items()) if lo <= k <= hi]
        avg = sum(v for k, v in fx[cur].items() if k.startswith(str(YEAR))) / 12
        return "\n".join([common.fill(T["fx_note"], cur=cur), "| Month | USD per unit |", "|---|---|"] + rows +
                         [f"| {YEAR} annual average | {avg:.5g} |"])

    def search_report(a):
        fac_id, q = str(a["facility_id"]).strip(), str(a["text"]).strip().lower()
        if not q:
            raise ValueError("text must not be empty")
        reps = _world(world_dir)[1]
        hits = []
        for rid in sorted(r for r in reps if r.startswith(fac_id + "-")):
            t = reps[rid]
            i = t.lower().find(q)
            while i >= 0 and len(hits) < 12:
                hits.append(f"[{rid} @{i}] ...{t[max(0, i - 150):i + 200]}...")
                i = t.lower().find(q, i + 1)
        if not any(r.startswith(fac_id + "-") for r in reps):
            raise KeyError(f"unknown facility {fac_id!r}")
        return "\n\n".join(hits) or "no matches"

    tt = T["tools"]
    s, o = {"type": "string"}, {"type": "integer"}
    return [
        common.tool("list_facilities", tt["list_facilities"], {"offset": o, "limit": o}, [], list_facilities),
        common.tool("read_report", tt["read_report"], {"report_id": s, "offset": o}, ["report_id"], read_report),
        common.tool("read_conventions", tt["read_conventions"], {}, [], lambda a: _world(world_dir)[2]),
        common.tool("read_fx", tt["read_fx"], {"currency": s, "start_month": s, "end_month": s}, ["currency"], read_fx),
        common.tool("search_report", tt["search_report"], {"facility_id": s, "text": s}, ["facility_id", "text"],
                    search_report),
        common.calculator_tool(tt["calculator"]),
    ]


_ROW = re.compile(r'"(FAC\d{4})"\s*:\s*(?=\{)')


def parse_answer(answer):
    """The last JSON object keyed by facility ids. If none parses (e.g. a truncated answer), salvage every complete
    "FACnnnn": {...} entry; malformed or missing facilities simply count as wrong."""
    obj = common.last_json(answer, lambda o: isinstance(o, dict) and any(str(k).startswith("FAC") for k in o))
    if obj is None:
        obj, dec = {}, json.JSONDecoder()
        for m in _ROW.finditer(answer or ""):
            try:
                obj[m.group(1)] = dec.raw_decode(answer, m.end())[0]
            except ValueError:
                pass
    out = {}
    for k, v in (obj or {}).items():
        if isinstance(v, dict):
            out[str(k).strip()] = (common.num(v.get("energy_mwh")), common.num(v.get("cost_usd")))
    return out


def score(task, answer):
    gold = common.gold_of(task)["facilities"]
    pred = parse_answer(answer)
    ok_e = ok_c = n_ok = 0
    for fid, g in gold.items():
        e, c = pred.get(fid, (None, None))
        a = e is not None and _close(e, g["energy_mwh"])
        b = c is not None and _close(c, g["cost_usd"])
        ok_e += a
        ok_c += b
        n_ok += a and b
    n = len(gold)
    return {"score": n_ok / n, "score_exact": float(n_ok == n), "score_partial": (ok_e + ok_c) / (2 * n),
            "n_correct": n_ok, "n_units": n, "energy_correct": ok_e, "cost_correct": ok_c}
