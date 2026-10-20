"""Type 30 inventory roll-forward (family C, procedural). Unit = one warehouse: the ending on-hand in eaches of every
SKU it stocks, from an opening count, a transaction journal and warehouse notes, under a written policy (case packs
that change over time, effective-date cutoff, cycle counts, voids, returns by condition, transfers in transit,
note corrections). Transfers link warehouses within a block of four. Task-facing prose is K3's (text/, README)."""
import datetime as dt
import json
import random
import re
import threading
from pathlib import Path

from .. import common

TYPE_ID = "30_inventory"
TIERS = {"easy": (8, 1800), "medium": (40, 1200), "hard": (120, 1200)}
T = common.load_text(Path(__file__).parent)
START, CUTOFF, EXTRACT, OPEN_DATE = dt.date(2026, 1, 1), dt.date(2026, 3, 31), dt.date(2026, 4, 10), dt.date(2025, 12, 31)
BLOCK = 4  # warehouses that may transfer to each other
CITIES = ["Reno", "Joliet", "Laredo", "Allentown", "Ontario", "Savannah", "Memphis", "Tacoma", "Columbus", "Fresno",
          "Lakeland", "Dayton", "Wichita", "Albany", "Boise", "Tulsa"]
ADJ = ["Stout", "Compact", "Heavy", "Clear", "Steel", "Mini", "Dual", "Coated", "Folding", "Insulated", "Round", "Flat"]
NOUN = ["Bracket", "Hinge", "Clamp", "Filter", "Gasket", "Tray", "Valve", "Spool", "Coupler", "Bin", "Strap", "Lens",
        "Cable", "Sleeve", "Nozzle", "Tote"]
VENDORS = ["Halvern Supply", "Brightwell Parts", "Oakmere Industrial", "Castor & Lyle", "Northgate Components",
           "Pellam Trading"]
TRAPS = ["future", "pack", "count", "void", "ret_scrap", "ret_restock", "xfer", "wrong_sku", "consign", "dup"]
COLS = "| line | posted | effective | type | sku | qty | uom | ref | memo |"


def _d(rng, a, b):
    return a + dt.timedelta(days=rng.randint(0, (b - a).days))


def pack_on(hist, day):
    """hist: [(from_date, pack)] ascending; pack in effect on day."""
    return [p for f, p in hist if f <= day][-1]


# ---------------------------------------------------------------- simulation (shared by gold, oracle, baselines)
def roll(wh_lines, opening, packs, notes, xfer_in, restock, flags=()):
    """Ending on-hand per SKU. wh_lines: dicts id/posted/eff/type/sku/qty/uom/ref/memo (dates as date); opening:
    {sku: qty}; packs: {sku: hist}; notes: {"wrong": {line: sku}, "excluded": set}; xfer_in: {transfer id: eff of
    its XFER-IN}; restock(memo) -> bool. flags switch rules off for the baselines."""
    flags = set(flags)
    dropped = {l["id"] for l in wh_lines if l["type"] == "VOID"}
    if "no_void" not in flags:
        dropped |= {l["ref"] for l in wh_lines if l["type"] == "VOID"}
    wrong = {} if "no_notes" in flags else notes["wrong"]
    if "no_notes" not in flags:
        dropped |= notes["excluded"]
    when = (lambda l: l["posted"]) if "posted_cutoff" in flags else (lambda l: l["eff"])
    active = [l for l in wh_lines if l["id"] not in dropped and when(l) <= CUTOFF]

    def conv(l):
        if l["uom"] == "EA" or "no_conv" in flags:
            return l["qty"]
        hist = packs[l["sku"]]
        return l["qty"] * (hist[0][1] if "pack_first" in flags else pack_on(hist, l["eff"]))

    out = {}
    for sku in opening:
        ls = [l for l in active if wrong.get(l["id"], l["sku"]) == sku]
        counts = [l for l in ls if l["type"] == "COUNT"] if "no_count" not in flags else []
        bal = opening[sku]
        if counts:
            c = max(counts, key=lambda l: l["eff"])
            bal, ls = c["qty"], [l for l in ls if l["eff"] > c["eff"]]
        for l in ls:
            t = l["type"]
            if t == "RECEIPT":
                bal += conv(l)
            elif t == "SHIPMENT":
                bal -= conv(l)
            elif t == "ADJUST" or t == "XFER-IN":
                bal += l["qty"]
            elif t == "RETURN":
                bal += l["qty"] if ("all_restock" in flags or restock(l["memo"])) else 0
            elif t == "XFER-OUT":
                got = xfer_in.get(l["ref"])
                if "xfer_at_ship" in flags or (got is not None and got <= CUTOFF):
                    bal -= conv(l)
        out[sku] = bal
    return out


# ---------------------------------------------------------------- generation
def _block(seed, b, m):
    """Items and warehouses b*BLOCK .. b*BLOCK+m-1 (structured, before rendering)."""
    rng = random.Random(f"inv:{seed}:{b}:{m}")
    items = {}
    for k in range(12):
        sku = f"SKU-{1000 + b * 12 + k}"
        base = rng.choice([6, 8, 10, 12, 16, 20, 24, 30])
        hist = [(dt.date(2024, 1, 1), base)]
        if rng.random() < 0.5:
            hist.append((_d(rng, dt.date(2026, 1, 20), dt.date(2026, 3, 10)),
                         rng.choice([p for p in (6, 8, 10, 12, 16, 20, 24, 30) if p != base])))
        items[sku] = {"sku": sku, "desc": f"{rng.choice(ADJ)} {rng.choice(NOUN)} {rng.choice('ABCDEFGHJK')}"
                                         f"{rng.randint(10, 99)}", "packs": hist}
    whs = []
    for j in range(m):
        skus = sorted(rng.sample(sorted(items), rng.randint(4, 8)))
        whs.append({"id": f"WH{b * BLOCK + j + 1:03d}", "city": rng.choice(CITIES), "skus": skus,
                    "opening": {s: rng.randint(4, 120) * 10 for s in skus}, "lines": [], "notes": [], "traps": []})
    seq = [0]

    def add(w, typ, sku, qty, uom, eff, posted=None, ref="", memo=""):
        seq[0] += 1
        line = {"type": typ, "sku": sku, "qty": qty, "uom": uom, "eff": eff, "posted": posted or eff, "ref": ref,
                "memo": memo, "seq": seq[0]}
        w["lines"].append(line)
        return line

    def plain_day():
        return _d(rng, START + dt.timedelta(days=1), CUTOFF - dt.timedelta(days=3))

    for w in whs:  # ordinary activity
        for s in w["skus"]:
            for _ in range(rng.randint(2, 5)):
                e = plain_day()
                p = min(e + dt.timedelta(days=rng.choice([0, 0, 1, 2])), CUTOFF)
                r = rng.random()
                if r < 0.45:
                    cs = rng.random() < 0.6
                    add(w, "RECEIPT", s, rng.randint(2, 30) if cs else rng.randint(5, 60) * 5, "CS" if cs else "EA",
                        e, p, f"PO-{rng.randint(40000, 99999)}")
                elif r < 0.85:
                    cs = rng.random() < 0.25
                    add(w, "SHIPMENT", s, rng.randint(1, 8) if cs else rng.randint(1, 40) * 3, "CS" if cs else "EA",
                        e, p, f"SO-{rng.randint(100000, 999999)}")
                else:
                    add(w, "ADJUST", s, rng.choice([-1, 1]) * rng.randint(1, 25), "EA", e, p,
                        f"ADJ-{rng.randint(1000, 9999)}")
    # transfers inside the block
    xn = 0
    for w in whs:
        partners = [v for v in whs if v is not w and set(v["skus"]) & set(w["skus"])]
        if not partners or rng.random() < 0.25:
            continue
        w["traps"].append("xfer")
        for _ in range(rng.randint(1, 2)):
            v = rng.choice(partners)
            s = rng.choice(sorted(set(v["skus"]) & set(w["skus"])))
            xn += 1
            tr = f"TR-{b + 1:03d}{xn:02d}"
            kind = rng.choice(["transit", "short", "done", "unreceived"])
            ship = _d(rng, dt.date(2026, 3, 22), CUTOFF) if kind in ("transit", "unreceived") else plain_day()
            cs = rng.random() < 0.6
            qty = rng.randint(2, 15) if cs else rng.randint(4, 40) * 5
            out = add(w, "XFER-OUT", s, qty, "CS" if cs else "EA", ship, ship, tr, f"to {v['id']}")
            ea = qty * pack_on(items[s]["packs"], ship) if cs else qty
            if kind == "unreceived":
                continue
            rec = _d(rng, dt.date(2026, 4, 2), dt.date(2026, 4, 9)) if kind == "transit" else \
                ship + dt.timedelta(days=rng.randint(1, 5))
            got = ea - rng.randint(1, max(1, ea // 10)) if kind == "short" else ea
            add(v, "XFER-IN", s, got, "EA", rec, rec, tr, f"from {w['id']}")
            out["kind"] = kind
    xskus = {(w["id"], l["sku"]) for w in whs for l in w["lines"] if l["type"].startswith("XFER")}
    for w in whs:  # cutoff traps in every warehouse, then a sample of the other kinds
        s = rng.choice(w["skus"])
        add(w, rng.choice(["RECEIPT", "SHIPMENT"]), s, rng.randint(3, 40) * 5, "EA",
            _d(rng, dt.date(2026, 3, 22), CUTOFF), _d(rng, dt.date(2026, 4, 1), dt.date(2026, 4, 8)),
            f"PO-{rng.randint(40000, 99999)}")
        w["traps"].append("backdate")
        changed = [s for s in w["skus"] if len(items[s]["packs"]) > 1]
        countable = [s for s in w["skus"] if (w["id"], s) not in xskus]
        kinds = [k for k in TRAPS if k != "xfer" and not (k == "pack" and not changed)
                 and not (k == "count" and not countable)]
        pick = rng.sample(kinds, min(len(kinds), rng.randint(4, 6)))
        if not {"wrong_sku", "consign", "dup"} & set(pick):
            pick.append(rng.choice(["wrong_sku", "consign", "dup"]))
        for k in pick:
            w["traps"].append(k)
            s = rng.choice(w["skus"])
            if k == "future":
                e = _d(rng, dt.date(2026, 4, 1), dt.date(2026, 4, 6))
                add(w, "RECEIPT", s, rng.randint(3, 20), "CS", e, _d(rng, dt.date(2026, 3, 26), CUTOFF),
                    f"PO-{rng.randint(40000, 99999)}")
            elif k == "pack":
                s = rng.choice(changed)
                ch = items[s]["packs"][1][0]
                add(w, "RECEIPT", s, rng.randint(3, 20), "CS", _d(rng, START + dt.timedelta(days=3),
                                                                     ch - dt.timedelta(days=1)), None,
                    f"PO-{rng.randint(40000, 99999)}")
                add(w, "RECEIPT", s, rng.randint(3, 20), "CS", _d(rng, ch, CUTOFF - dt.timedelta(days=2)), None,
                    f"PO-{rng.randint(40000, 99999)}")
            elif k == "count":
                s = rng.choice(countable)
                c = _d(rng, dt.date(2026, 2, 5), dt.date(2026, 2, 25))
                add(w, "COUNT", s, rng.randint(5, 150) * 10 + rng.randint(1, 9), "EA", c, c,
                    f"CC-{rng.randint(1000, 9999)}")
                add(w, "RECEIPT", s, rng.randint(4, 40) * 5, "EA", c - dt.timedelta(days=rng.randint(1, 10)),
                    c + dt.timedelta(days=rng.randint(2, 9)), f"PO-{rng.randint(40000, 99999)}")
            elif k == "void":
                e = plain_day()
                tgt = add(w, rng.choice(["RECEIPT", "SHIPMENT"]), s, rng.randint(4, 40) * 5, "EA", e, e,
                          f"PO-{rng.randint(40000, 99999)}")
                p = min(e + dt.timedelta(days=rng.randint(1, 6)), CUTOFF)  # voids fall inside the period
                v = add(w, "VOID", s, 0, "", p, p)
                v["ref_obj"] = tgt
            elif k in ("ret_scrap", "ret_restock"):
                bank = T["restock_reasons"] if k == "ret_restock" else T["scrap_reasons"]
                add(w, "RETURN", s, rng.randint(1, 30), "EA", plain_day(), None, f"RMA-{rng.randint(1000, 9999)}",
                    rng.choice(bank))
            elif k == "wrong_sku":
                s2 = rng.choice([x for x in w["skus"] if x != s])
                line = add(w, "RECEIPT", s, rng.randint(4, 40) * 5, "EA", plain_day(), None,
                           f"PO-{rng.randint(40000, 99999)}")
                w["notes"].append(("wrong", line, s2))
            elif k == "consign":
                cs = rng.random() < 0.5
                line = add(w, "RECEIPT", s, rng.randint(2, 20) if cs else rng.randint(4, 40) * 5, "CS" if cs else "EA",
                           plain_day(), None, f"PO-{rng.randint(40000, 99999)}")
                w["notes"].append(("consign", line, rng.choice(VENDORS)))
            elif k == "dup":
                e = plain_day()
                typ = rng.choice(["RECEIPT", "SHIPMENT"])
                ref = f"{'PO' if typ == 'RECEIPT' else 'SO'}-{rng.randint(40000, 99999)}"
                q = rng.randint(4, 40) * 5
                orig = add(w, typ, s, q, "EA", e, e, ref)
                dup = add(w, typ, s, q, "EA", e, e + dt.timedelta(days=1), ref)
                w["notes"].append(("dup", dup, orig))
    for w in whs:  # ids in posting order, then references and notes
        w["lines"].sort(key=lambda l: (l["posted"], l["seq"]))
        num = int(w["id"][2:])
        for k, l in enumerate(w["lines"]):
            l["id"] = f"J{num:03d}-{k + 1:03d}"
        for l in w["lines"]:
            if "ref_obj" in l:
                l["ref"] = l.pop("ref_obj")["id"]
        notes = []
        for kind, line, x in w["notes"]:
            if kind == "wrong":
                txt = common.fill(T["note_wrong_sku"], line=line["id"], wrong=line["sku"], right=x, right2=x)
            elif kind == "consign":
                txt = common.fill(T["note_consignment"], line=line["id"], vendor=x)
            else:
                txt = common.fill(T["note_duplicate"], line=line["id"], orig=x["id"])
            notes.append((line["posted"] + dt.timedelta(days=rng.randint(1, 12)), txt))
        for t in rng.sample(T["note_benign"], rng.randint(1, 3)):
            notes.append((plain_day(), t))
        notes.sort(key=lambda x: x[0])
        w["note_text"] = [f"- {d.isoformat()}: {t}" for d, t in notes]
        w["note_struct"] = {"wrong": {line["id"]: x for kind, line, x in w["notes"] if kind == "wrong"},
                            "excluded": {line["id"] for kind, line, _ in w["notes"] if kind != "wrong"}}
    return items, whs


def row(l):
    q = "" if l["type"] == "VOID" else str(l["qty"])
    memo = l["memo"].replace("|", "/")
    return (f"| {l['id']} | {l['posted'].isoformat()} | {l['eff'].isoformat()} | {l['type']} | {l['sku']} | {q} | "
            f"{l['uom']} | {l['ref']} | {memo} |")


def item_rows(it):
    return [f"| {it['sku']} | {it['desc']} | {p} | {f.isoformat()} |" for f, p in it["packs"]]


def policy_text():
    rules = [common.fill(T[k], cutoff=CUTOFF.isoformat()) for k in
             ("rule_units", "rule_cutoff", "rule_count", "rule_void", "rule_lines", "rule_returns", "rule_transfers",
              "rule_notes")]
    return T["policy_title"] + "\n\n" + "\n\n".join(f"{i + 1}. {r}" for i, r in enumerate(rules))


def generate(n, seed, out_dir):
    items, whs = {}, []
    for b in range((n + BLOCK - 1) // BLOCK):
        it, ws = _block(seed, b, min(BLOCK, n - b * BLOCK))
        items.update(it)
        whs += ws
    restock = set(T["restock_reasons"])
    xfer_in = {l["ref"]: l["eff"] for w in whs for l in w["lines"] if l["type"] == "XFER-IN"}
    packs = {s: it["packs"] for s, it in items.items()}
    ending = {}
    for w in whs:  # no negative ending stock: raise the opening count (or the cycle count that resets it)
        while True:
            end = roll(w["lines"], w["opening"], packs, w["note_struct"], xfer_in, lambda m: m in restock)
            neg = [s for s, q in end.items() if q < 0]
            if not neg:
                break
            for s in neg:
                bump = (-end[s] // 100 + 1) * 100
                counts = [l for l in w["lines"] if l["type"] == "COUNT" and l["sku"] == s]
                if counts:
                    counts[-1]["qty"] += bump
                else:
                    w["opening"][s] += bump
        ending[w["id"]] = end
    gold = {"ending": ending, "traps": {w["id"]: w["traps"] for w in whs}}
    used = sorted({s for w in whs for s in w["skus"]})
    world = {"policy": policy_text(),
             "items": {s: item_rows(items[s]) for s in used},
             "warehouses": [{"id": w["id"], "city": w["city"], "skus": w["skus"],
                             "heading": common.fill(T["journal_heading"], wh=w["id"], start=START.isoformat(),
                                                    cutoff=CUTOFF.isoformat(), extract=EXTRACT.isoformat()),
                             "opening": [common.fill(T["opening_heading"], date=OPEN_DATE.isoformat())] +
                                        [f"| {s} | {q} |" for s, q in w["opening"].items()],
                             "rows": [row(l) for l in w["lines"]],
                             "notes": [common.fill(T["notes_heading"], wh=w["id"])] + w["note_text"]} for w in whs]}
    ex = whs[0]
    example = json.dumps({ex["id"]: {s: 0 for s in ex["skus"][:2]}})
    prompt = (common.fill(T["intro"], company=T["company"], n=n, first=whs[0]["id"], last=whs[-1]["id"],
                          start=START.isoformat(), cutoff=CUTOFF.isoformat()) + "\n\n" +
              common.fill(T["answer_line"], example=example))
    return common.finish(TYPE_ID, n, seed, out_dir, prompt, gold, TIERS, files={"inventory.json": world})


# ---------------------------------------------------------------- tools
_cache, _lock = {}, threading.Lock()


def _world(world_dir):
    with _lock:
        if world_dir not in _cache:
            _cache[world_dir] = json.loads((Path(world_dir) / "inventory.json").read_text())
        return _cache[world_dir]


def tools(world_dir):
    W = lambda: _world(world_dir)  # noqa: E731
    D = T["tools"]

    def wh(a):
        w = next((x for x in W()["warehouses"] if x["id"] == str(a.get("warehouse", "")).strip().upper()), None)
        if w is None:
            raise KeyError(f"unknown warehouse {a.get('warehouse')!r}; see list_warehouses")
        return w

    def list_warehouses(a):
        off, lim = common.page_args(a, 100, 200)
        ws = W()["warehouses"]
        rows = [f"| {w['id']} | {w['city']} | {len(w['skus'])} | {len(w['rows'])} |" for w in ws[off:off + lim]]
        more = f"\n(next: offset={off + lim})" if off + lim < len(ws) else ""
        return f"{len(ws)} warehouses\n| id | city | SKUs | journal lines |\n" + "\n".join(rows) + more

    def list_items(a):
        off, lim = common.page_args(a, 40, 40)
        its = sorted(W()["items"])
        rows = [r for s in its[off:off + lim] for r in W()["items"][s]]
        more = f"\n(next: offset={off + lim})" if off + lim < len(its) else ""
        return f"{len(its)} SKUs\n| sku | description | EA per CS | effective from |\n" + "\n".join(rows) + more

    def read_item(a):
        s = str(a.get("sku", "")).strip().upper()
        if s not in W()["items"]:
            raise KeyError(f"unknown sku {a.get('sku')!r}")
        return "| sku | description | EA per CS | effective from |\n" + "\n".join(W()["items"][s])

    def read_journal(a):
        w = wh(a)
        off, lim = common.page_args(a, 50, 50)
        rows = w["rows"][off:off + lim]
        head = w["heading"] + "\n\n" + "\n".join(w["opening"]) + "\n\n" if off == 0 else ""
        more = f"\n(next: offset={off + lim} of {len(w['rows'])})" if off + lim < len(w["rows"]) else ""
        return head + f"lines {off + 1}-{off + len(rows)} of {len(w['rows'])}\n{COLS}\n" + "\n".join(rows) + more

    def search_journal(a):
        q = str(a.get("text", "")).strip().lower()
        if len(q) < 3:
            raise ValueError("text must have at least 3 characters")
        hits = [f"{w['id']} {r}" for w in W()["warehouses"] for r in w["rows"] if q in r.lower()]
        return (f"{len(hits)} hits" + (" (first 20)" if len(hits) > 20 else "") + f"\nwarehouse {COLS}\n" +
                "\n".join(hits[:20]))

    S = {"type": "string"}
    I = {"type": "integer"}
    return [
        common.tool("list_warehouses", D["list_warehouses"], {"offset": I, "limit": I}, [], list_warehouses),
        common.tool("read_policy", D["read_policy"], {}, [], lambda a: W()["policy"]),
        common.tool("list_items", D["list_items"], {"offset": I, "limit": I}, [], list_items),
        common.tool("read_item", D["read_item"], {"sku": S}, ["sku"], read_item),
        common.tool("read_journal", D["read_journal"], {"warehouse": S, "offset": I, "limit": I}, ["warehouse"],
                    read_journal),
        common.tool("search_journal", D["search_journal"], {"text": S}, ["text"], search_journal),
        common.tool("read_notes", D["read_notes"], {"warehouse": S}, ["warehouse"], lambda a: "\n".join(wh(a)["notes"])),
        common.calculator_tool(D["calculator"]),
    ]


# ---------------------------------------------------------------- scoring
def parse_answer(answer):
    obj = common.last_json(answer, lambda o: isinstance(o, dict) and any(
        isinstance(k, str) and k.strip().upper().startswith("WH") for k in o))
    if obj is None:  # row salvage (e.g. a truncated answer): grade every complete "WHnnn": {...} entry
        obj, dec = {}, json.JSONDecoder()
        for m in re.finditer(r'"(WH\d{3})"\s*:\s*(?=\{)', answer or ""):
            try:
                v, _ = dec.raw_decode(answer, m.end())
            except ValueError:
                continue
            obj[m.group(1)] = v
    out = {}
    for k, v in obj.items():
        if isinstance(v, dict):
            out[k.strip().upper()] = {str(s).strip().upper(): common.num(q) for s, q in v.items()}
    return out


def score(task, answer):
    gold = common.gold_of(task)["ending"]
    pred = parse_answer(answer)
    ok = cells = cells_ok = 0
    for w, skus in gold.items():
        p = pred.get(w, {})
        good = [p.get(s) is not None and abs(p[s] - q) < 1e-6 for s, q in skus.items()]
        cells += len(good)
        cells_ok += sum(good)
        ok += all(good) and len(p) == len(skus)
    n = len(gold)
    return {"score": ok / n, "score_exact": float(ok == n), "score_partial": cells_ok / cells, "n_correct": ok,
            "n_units": n}
