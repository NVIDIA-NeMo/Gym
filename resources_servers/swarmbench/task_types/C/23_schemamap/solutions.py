"""Type 23 solutions: tool-only oracle (parses dictionaries with regexes built from the K3 templates and reads the
values), plausible-wrong baselines, partial readers (each ignores one trap kind), single units (check 5)."""
import json
import re

from .. import common
from . import type as track

T = track.T
COL = r"[A-Za-z0-9_]+"
UNITS = {v: k for k, v in track.UNIT_WORDS.items()}
UNIT_RX = "|".join(re.escape(v) for v in sorted(UNITS, key=len, reverse=True))
PAIR = common.tre(T["date_pair_note"], a=COL, b=COL, f1=r"[a-z_]+", f2=r"[a-z_]+")
SWAP = common.tre(T["errata_swap"], a=COL, b=COL)
UNIT = common.tre(T["errata_unit"], _release=r"[\d.]+", col=COL, unit=UNIT_RX)
RENAME = common.tre(T["errata_rename"], old=COL, new=COL)
COSTS = [(c, common.tre(t, unit=UNIT_RX)) for c in track.COSTS for t in T[c + "_desc"]]
LETTERS = [common.tre(t, codes=r".+") for t in T["type_letter_desc"]]
STATIC = {T["pair_col_desc"]: ("PAIR", None)}
for f, ds in T["field_desc"].items():
    STATIC.update({d: (f, None) for d in ds})
for d, ds in T["decoy_desc"].items():
    STATIC.update({x: (d, None) for x in ds})
STATIC.update({d: ("permit_type", "code") for d in T["type_code_desc"]})
STATIC.update({d: ("permit_type", "name") for d in T["type_name_desc"]})
STATIC.update({d: ("site_address", None) for d in T["site_address_desc"]})
ROW = re.compile(r"^\| (.+?) \| (.*) \|$")
UNIT_RE = re.compile(r"UNIT\s*\|\s*(SRC\d{3})\s*\|", re.I)


def fns(task):
    return {t["name"]: t["fn"] for t in track.tools(task["world_dir"])}


def classify(desc):
    if desc in STATIC:
        return STATIC[desc]
    for c, rx in COSTS:
        m = rx.fullmatch(desc)
        if m:
            return (c, UNITS[m.group("unit")])
    for rx in LETTERS:
        m = rx.fullmatch(desc)
        if m:
            codes = dict(x.split(" = ", 1) for x in m.group("codes").split("; "))
            return ("permit_type", ("letter", codes))
    raise ValueError(f"unclassified description {desc!r}")


def read_source(f, sid):
    d = f["read_dictionary"]({"source": sid})
    rows, off = [], 0
    while True:
        txt = f["read_rows"]({"source": sid, "offset": off, "limit": 15})
        lines = txt.splitlines()
        body = list(__import__("csv").reader(lines[:-1]))
        header, page = body[0], body[1:]
        rows += page
        if len(page) < 15:
            break
        off += 15
    return d, header, rows


def _style(vals, careless):
    vals = [v for v in vals if v]
    if not vals or not all(re.fullmatch(r"\d{4}-\d\d-\d\d|\d{8}|\d\d/\d\d/\d{4}", v) for v in vals):
        return "identity"
    if all(re.fullmatch(r"\d{4}-\d\d-\d\d", v) for v in vals):
        return "identity"
    if all(re.fullmatch(r"\d{8}", v) for v in vals):
        return "date_yyyymmdd"
    parts = [v.split("/") for v in vals]
    if "style" in careless or any(int(p[1]) > 12 for p in parts):
        return "date_mdy"
    return "date_dmy" if any(int(p[0]) > 12 for p in parts) else "date_mdy"


def _iso(v, tf):
    if tf == "identity":
        return v
    if tf == "date_yyyymmdd":
        return f"{v[:4]}-{v[4:6]}-{v[6:]}"
    a, b, y = v.split("/")
    return f"{y}-{a}-{b}" if tf == "date_mdy" else f"{y}-{b}-{a}"


def map_source(d, header, rows, careless=frozenset()):
    """careless: subset of {swap, unit, rename, pair, style} to ignore (partial readers)."""
    cls, notes = {}, []
    for line in d.splitlines():
        m = ROW.match(line)
        if m and m.group(1) not in ("column", "---"):
            cls[m.group(1)] = classify(m.group(2))
        elif line.startswith("- "):
            notes.append(line[2:])
    for x in notes:
        m = SWAP.fullmatch(x)
        if m and "swap" not in careless:
            a, b = m.group("a"), m.group("b")
            cls[a], cls[b] = cls[b], cls[a]
        m = UNIT.fullmatch(x)
        if m and "unit" not in careless:
            cls[m.group("col")] = (cls[m.group("col")][0], UNITS[m.group("unit")])
    for x in notes:
        m = RENAME.fullmatch(x)
        if m and "rename" not in careless:
            cls[m.group("new")] = cls.pop(m.group("old"))
    col = {h: [r[k] for r in rows] for k, h in enumerate(header)}
    for x in notes:
        m = PAIR.fullmatch(x)
        if m:
            a, b, f1, f2 = m.group("a", "b", "f1", "f2")
            if "pair" in careless:
                cls[a], cls[b] = (f1, None), (f2, None)
                continue
            tf = _style(col.get(a, []) + col.get(b, []), careless)
            both = [(_iso(x, tf), _iso(y, tf)) for x, y in zip(col.get(a, []), col.get(b, [])) if x and y]
            first, second = sorted([f1, f2], key=track.DATES.index)
            a_first = all(x < y for x, y in both)
            cls[a], cls[b] = ((first, None), (second, None)) if a_first else ((second, None), (first, None))
    dates = [v for name, (meaning, _) in cls.items() if meaning in track.DATES for v in col.get(name, [])]
    style = _style(dates, careless)
    out = {f: None for f in track.FIELDS}
    for name, (meaning, extra) in cls.items():
        if meaning == "site_address":
            out["street_number"], out["street_name"] = [name, "address_number"], [name, "address_street"]
        elif meaning in track.DATES:
            out[meaning] = [name, style]
        elif meaning in track.COSTS:
            out[meaning] = [name, track.UNIT_TF[extra]]
        elif meaning == "permit_type":
            mode = extra[0] if isinstance(extra, tuple) else extra
            out[meaning] = [name, {"code": "identity", "letter": "decode_codelist", "name": "type_name_to_code"}[mode]]
        elif meaning in track.FIELDS:
            out[meaning] = [name, "identity"]
    return out


def source_ids(f):
    ids, off = [], 0
    while True:
        txt = f["list_sources"]({"offset": off, "limit": 100})
        page = re.findall(r"^(SRC\d{3}) \|", txt, re.M)
        if not page:
            return ids
        ids += page
        off += len(page)


def solve(task, careless=frozenset()):
    f = fns(task)
    return json.dumps({sid: map_source(*read_source(f, sid), careless=careless) for sid in source_ids(f)})


def oracle(task):
    return solve(task)


def name_similarity(task):
    f = fns(task)
    ans = {}
    for sid in source_ids(f):
        _, header, rows = read_source(f, sid)
        m = track.name_similarity_mapping(header)
        out = {}
        for fld, c in m.items():
            if c is None:
                out[fld] = None
            elif fld in track.DATES:
                out[fld] = [c, _style([r[header.index(c)] for r in rows], frozenset()) if all(
                    re.fullmatch(r"[\d/-]+", r[header.index(c)] or "0") for r in rows) else "identity"]
            else:
                out[fld] = [c, "identity"]
        ans[sid] = out
    return json.dumps(ans)


def wrong(task):
    return {"name_similarity": name_similarity(task),
            "dictionary_only": solve(task, frozenset({"swap", "unit", "rename", "pair", "style"})),
            "ignore_swaps": solve(task, frozenset({"swap"})),
            "ignore_unit_errata": solve(task, frozenset({"unit"})),
            "ignore_renames": solve(task, frozenset({"rename"})),
            "pair_by_order": solve(task, frozenset({"pair"})),
            "assume_mdy": solve(task, frozenset({"style"}))}


def units(task, k, rng):
    g = common.gold_of(task)
    ids = sorted(g["mapping"], key=lambda s: -(len(g["traps"][s]["errata"]) + g["traps"][s]["pair"]
                                              + g["traps"][s]["combined"] + (g["traps"][s]["type_mode"] != "code")))
    pick = ids[:max(2 * k, k)]
    rng.shuffle(pick)
    out = []
    for sid in pick[:k]:
        gold = g["mapping"][sid]
        prompt = common.fill(T["unit_prompt"], src=sid, src2=sid)

        def check(text, sid=sid, gold=gold):
            ms = [m for m in UNIT_RE.finditer(text or "") if m.group(1).upper() == sid]
            if not ms:
                return None
            obj = next((o for o in common.json_objects(text[ms[-1].end():]) if isinstance(o, dict)), None)
            if obj is None:
                return False
            pred = {str(k).strip(): track._norm_entry(v) for k, v in obj.items()}
            return bool(track.unit_correct(pred, gold)[0])
        t = g["traps"][sid]
        kind = "pair" if t["pair"] else "errata" if t["errata"] else "plain"
        out.append({"id": f"{task['id']}:{sid}", "kind": kind, "prompt": prompt, "gold": gold, "check": check})
    return out
