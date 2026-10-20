"""Type 30 solutions: tool-only oracle (parses journals, item master and notes through the tools; note sentences via
regexes built from the K3 templates), plausible-wrong baselines, single units (check 5)."""
import datetime as dt
import json
import re

from .. import common
from . import type as track

T = track.T
LID, SKU = r"J\d{3}-\d{3}", r"SKU-\d{4}"
WRONG = common.tre(T["note_wrong_sku"], line=LID, _wrong=SKU, right=SKU, _right2=SKU)
CONSIGN = common.tre(T["note_consignment"], line=LID, _vendor=r".+?")
DUP = common.tre(T["note_duplicate"], line=LID, _orig=LID)


def _fns(task):
    return {t["name"]: t["fn"] for t in track.tools(task["world_dir"])}


def _date(s):
    return dt.date.fromisoformat(s.strip())


def read_world(task):
    """Everything the oracle needs, read through the tools only."""
    f = _fns(task)
    whs = re.findall(r"^\| (WH\d{3}) \|", f["list_warehouses"]({"limit": 200}), re.M)
    packs = {}
    off = 0
    while True:
        out = f["list_items"]({"offset": off, "limit": 40})
        for s, p, d in re.findall(rf"^\| ({SKU}) \| .*? \| (\d+) \| ([\d-]+) \|$", out, re.M):
            packs.setdefault(s, []).append((_date(d), int(p)))
        if "(next: offset=" not in out:
            break
        off += 40
    data = {}
    for w in whs:
        lines, opening, off = [], {}, 0
        while True:
            out = f["read_journal"]({"warehouse": w, "offset": off, "limit": 50})
            if off == 0:
                opening = {s: int(q) for s, q in re.findall(rf"^\| ({SKU}) \| (-?\d+) \|$", out, re.M)}
            for raw in out.splitlines():
                if re.match(rf"^\| {LID} \|", raw):
                    c = [x.strip() for x in raw.strip().strip("|").split("|")]
                    lines.append({"id": c[0], "posted": _date(c[1]), "eff": _date(c[2]), "type": c[3], "sku": c[4],
                                  "qty": int(c[5]) if c[5] else 0, "uom": c[6], "ref": c[7], "memo": c[8]})
            if "(next: offset=" not in out:
                break
            off += 50
        notes = f["read_notes"]({"warehouse": w})
        ns = {"wrong": {m["line"]: m["right"] for m in WRONG.finditer(notes)},
              "excluded": {m["line"] for m in CONSIGN.finditer(notes)} | {m["line"] for m in DUP.finditer(notes)}}
        data[w] = (lines, opening, ns)
    return data, packs


def solve(task, flags=()):
    data, packs = read_world(task)
    xfer_in = {l["ref"]: l["eff"] for lines, _, _ in data.values() for l in lines if l["type"] == "XFER-IN"}
    restock = set(T["restock_reasons"])
    return {w: track.roll(lines, opening, packs, ns, xfer_in, lambda m: m in restock, flags)
            for w, (lines, opening, ns) in data.items()}


def oracle(task):
    return json.dumps(solve(task))


def naive(task):
    """Opening plus the signed sum of every line's raw quantity (no conversion, cutoff, voids, counts or notes)."""
    data, _ = read_world(task)
    sign = {"RECEIPT": 1, "SHIPMENT": -1, "ADJUST": 1, "RETURN": 1, "XFER-IN": 1, "XFER-OUT": -1, "COUNT": 0, "VOID": 0}
    out = {}
    for w, (lines, opening, _) in data.items():
        out[w] = {s: q + sum(sign[l["type"]] * l["qty"] for l in lines if l["sku"] == s) for s, q in opening.items()}
    return json.dumps(out)


def wrong(task):
    return {
        "naive_signed_sum": naive(task),
        "spreadsheet_posted_date": json.dumps(solve(task, ("posted_cutoff", "no_notes", "no_count", "xfer_at_ship"))),
        "careful_but_no_notes_posted": json.dumps(solve(task, ("posted_cutoff", "no_notes"))),
        "skip_posted_cutoff": json.dumps(solve(task, ("posted_cutoff",))),
        "skip_notes": json.dumps(solve(task, ("no_notes",))),
        "skip_count": json.dumps(solve(task, ("no_count",))),
        "skip_void": json.dumps(solve(task, ("no_void",))),
        "skip_conversion": json.dumps(solve(task, ("no_conv",))),
        "first_pack_size": json.dumps(solve(task, ("pack_first",))),
        "transfer_at_ship": json.dumps(solve(task, ("xfer_at_ship",))),
        "restock_all_returns": json.dumps(solve(task, ("all_restock",))),
    }


def units(task, k, rng):
    gold = common.gold_of(task)
    whs = sorted(gold["ending"], key=lambda w: -len(gold["traps"][w]))
    pick = whs[:max(2 * k, k)]
    rng.shuffle(pick)
    out = []
    for w in pick[:k]:
        g = gold["ending"][w]
        prompt = common.fill(T["unit_prompt"], wh=w, cutoff=track.CUTOFF.isoformat(), wh2=w)

        def check(text, g=g, w=w):
            m = re.findall(rf"UNIT\W*\|\W*{w}\W*\|([^\n]*)", text or "")
            if not m:
                return None
            got = {s.upper(): float(v.replace(",", "")) for s, v in re.findall(r"(SKU-\d{4})\W*=\W*(-?[\d,]+)", m[-1])}
            return got == {s: float(q) for s, q in g.items()}
        out.append({"id": f"{task['id']}:{w}", "kind": "+".join(sorted(set(gold["traps"][w])))[:60],
                    "prompt": prompt, "gold": g, "check": check})
    return out
