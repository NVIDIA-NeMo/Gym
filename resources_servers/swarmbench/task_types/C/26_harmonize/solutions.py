"""Type 26 solutions: tool-only oracle (parses reports with regexes built from the K3 templates), plausible-wrong
baselines, single units (check 5)."""
import json
import re

from .. import common
from . import type as track

T = track.T
NUM = r"-?[\d,]+(?:\.\d+)?"
MONTH = r"[A-Z][a-z]{2} \d{4}"
METER = r"M\d+"
UNITRX = "|".join(re.escape(u) for u in sorted(track.FACTORS, key=len, reverse=True))
CUR = r"[A-Z]{3}"
SCALE = common.tre(T["scale_note"], _unit=UNITRX)
MULT = common.tre(T["mult_note"], meter=METER, month=MONTH, _month2=MONTH, mult=NUM)
CORR = common.tre(T["corr_note"], meter=METER, month=MONTH, value=NUM, _unit=UNITRX)
SUB = common.tre(T["sub_note"], sub=METER, _main=METER, _main2=METER)
EST = common.tre(T["est_note"], meter=METER, month=MONTH, value=NUM, _unit=UNITRX, cost=NUM, _cur=CUR)
PERIOD = common.tre(T["period"], start=MONTH, end=MONTH)
CURLINE = common.tre(T["currency_line"], cur=CUR)


def _f(s):
    return float(s.replace(",", ""))


def _ym(s):
    mon, y = s.split()
    return int(y), track.MON.index(mon) + 1


def _fns(task):
    return {t["name"]: t["fn"] for t in track.tools(task["world_dir"])}


def _read(fns, rid):
    text, off = "", 0
    while True:
        chunk = fns["read_report"]({"report_id": rid, "offset": off})
        m = re.search(r"\n\[chars \d+-(\d+) of \d+; continue with offset=\d+\]$", chunk)
        text += chunk[:m.start()] if m else chunk
        if not m:
            return text
        off = int(m.group(1))


def parse_report(text, facts):
    fy = int(re.search(r"FY(\d{4}) \[", text).group(1))
    facts["currency"] = CURLINE.search(text).group("cur")
    facts["fy_end"] = _ym(PERIOD.search(text).group("end"))[1]
    rep = facts["reports"].setdefault(fy, {"meters": {}})
    block = None
    for line in text.splitlines():
        line = line.strip()
        m = re.fullmatch(r"### (M\d+)", line)
        if m:
            block = m.group(1)
            continue
        if line.startswith("## "):
            block = None
        if block and SCALE.fullmatch(line):
            facts["scale"].add((fy, block))
        cells = [c.strip() for c in line.strip("|").split("|")] if line.startswith("|") else []
        if block and len(cells) == 6:
            if re.fullmatch(MONTH, cells[0]):
                rep["meters"][block]["rows"][_ym(cells[0])] = (_f(cells[1]), _f(cells[3]))
            elif not cells[0].startswith("---"):
                rep["meters"][block] = {"unit": re.search(r"\(([^()]*)\)\s*$", cells[1]).group(1), "rows": {}}
        if line.startswith("- "):
            note = line[2:]
            if (m := MULT.fullmatch(note)):
                facts["mult"][(fy, m.group("meter"))] = (_ym(m.group("month")), _f(m.group("mult")))
            elif (m := CORR.fullmatch(note)):
                facts["corr"][(fy, m.group("meter"), _ym(m.group("month")))] = _f(m.group("value"))
            elif (m := SUB.fullmatch(note)):
                facts["sub"].add(m.group("sub"))
            elif (m := EST.fullmatch(note)):
                facts["est"][(m.group("meter"), _ym(m.group("month")))] = (_f(m.group("value")), _f(m.group("cost")))


def read_fx(fns, cur):
    out = {}
    for line in fns["read_fx"]({"currency": cur}).splitlines():
        m = re.fullmatch(r"\| (\d{4})-(\d{2}) \| ([\d.eE+-]+) \|", line.strip())
        if m:
            out[(int(m.group(1)), int(m.group(2)))] = float(m.group(3))
    return out


def facts_all(task):
    fns = _fns(task)
    conv = fns["read_conventions"]({})
    for line in conv.splitlines():  # factors from the conventions table (must equal the code's)
        m = re.fullmatch(r"\| (.+?) \| ([\d.]+) \|", line.strip())
        if m and m.group(1) in track.FACTORS:
            assert float(m.group(2)) == track.FACTORS[m.group(1)]
    listing, off = [], 0
    while True:
        out = fns["list_facilities"]({"offset": off, "limit": 60})
        rows = [ln.split(" | ") for ln in out.splitlines() if ln.startswith("FAC")]
        listing += rows
        if "use offset=" not in out:
            break
        off += len(rows)
    fxs, res = {}, {}
    for row in listing:
        facts = {"reports": {}, "scale": set(), "mult": {}, "corr": {}, "sub": set(), "est": {}}
        for rid in row[5].split(", "):
            parse_report(_read(fns, rid), facts)
        cur = facts["currency"]
        if cur not in fxs:
            fxs[cur] = read_fx(fns, cur)
        res[row[0]] = facts
    return res, fxs


def _answer(task, **kw):
    res, fxs = facts_all(task)
    out = {}
    for fid, f in res.items():
        e, c = track.compute(f, {f["currency"]: fxs[f["currency"]]}, **kw)
        out[fid] = {"energy_mwh": round(e, 3), "cost_usd": round(c, 2)}
    return json.dumps(out)


def oracle(task):
    return _answer(task)


def wrong(task):
    res, fxs = facts_all(task)
    naive = {}
    for fid, f in res.items():  # every consumption figure read as kWh, raw local costs, calendar months, all meters
        e = c = 0.0
        for rep in f["reports"].values():
            for met in rep["meters"].values():
                for ym, (q, cost) in met["rows"].items():
                    if ym[0] == track.YEAR:
                        e, c = e + q / 1000, c + cost
        naive[fid] = {"energy_mwh": round(e, 3), "cost_usd": round(c, 2)}
    out = {"naive_kwh_rawcost": json.dumps(naive),
           "ignore_notes": _answer(task, ignore=tuple(track.TRAPS)),
           "fiscal_year": _answer(task, fiscal=True),
           "no_fx (partial)": _answer(task, fx_mode="none"),
           "annual_avg_fx (partial)": _answer(task, fx_mode="annual")}
    for k in track.TRAPS:
        out[f"ignore_{k} (partial)"] = _answer(task, ignore=(k,))
    return out


def units(task, k, rng):
    gold = common.gold_of(task)
    fids = sorted(gold["facilities"], key=lambda f: -len(gold["traps"][f]))
    pick = fids[:max(2 * k, k)]
    rng.shuffle(pick)
    out = []
    for fid in pick[:k]:
        g = gold["facilities"][fid]
        prompt = common.fill(T["unit_prompt"], year=track.YEAR, fid2=fid, fid=fid)

        def check(text, g=g, fid=fid):
            m = re.findall(rf"UNIT\W*\|\W*{fid}\W*\|\W*({NUM})\W*\|\W*({NUM})", text or "")
            if not m:
                return None
            e, c = _f(m[-1][0]), _f(m[-1][1])
            return track._close(e, g["energy_mwh"]) and track._close(c, g["cost_usd"])
        out.append({"id": f"{task['id']}:{fid}", "kind": "+".join(gold["traps"][fid]), "prompt": prompt, "gold": g,
                    "check": check})
    return out
