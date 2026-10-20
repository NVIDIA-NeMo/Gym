"""Type 23 schemamap: map the columns of many partner exports (real SF building-permit rows) onto one target schema.

One unit = one source table. Every target field must be mapped to (column, transform) or null; the mapping is fixed
by the source's data dictionary plus its notes and errata, and some parts only by the values (date order, which of
two undocumented dates is which). Task-facing prose is K3's (text/k3.json); ids, tables and real values are code/data.
"""
import csv
import threading
import difflib
import io
import json
import random
import re
from pathlib import Path

from .. import common

TYPE_ID = "23_schemamap"
TIERS = {"easy": (12, 1800), "medium": (40, 1200), "hard": (80, 1200)}  # coordinator after calibration A (see README)
HERE = Path(__file__).parent
T = common.load_text(HERE)
FIELDS = ["permit_id", "permit_type", "status", "filed_date", "issued_date", "completed_date", "street_number",
          "street_name", "zipcode", "estimated_cost", "revised_cost", "existing_stories", "proposed_stories",
          "existing_use", "proposed_use", "description"]
DECOYS = ["created_date", "status_date", "approved_date", "existing_units", "proposed_units", "plansets", "block",
          "lot", "record_id", "district", "neighborhood", "construction_type", "mailing_address"]
DATES = ["filed_date", "issued_date", "completed_date"]
COSTS = ["estimated_cost", "revised_cost"]
SRC_COL = {"permit_id": "permit_number", "status": "status", "filed_date": "filed_date", "issued_date": "issued_date",
           "completed_date": "completed_date", "street_number": "street_number", "zipcode": "zipcode",
           "estimated_cost": "estimated_cost", "revised_cost": "revised_cost",
           "existing_stories": "number_of_existing_stories", "proposed_stories": "number_of_proposed_stories",
           "existing_use": "existing_use", "proposed_use": "proposed_use", "description": "description",
           "created_date": "permit_creation_date", "status_date": "status_date", "approved_date": "approved_date",
           "existing_units": "existing_units", "proposed_units": "proposed_units", "plansets": "plansets",
           "block": "block", "lot": "lot", "record_id": "record_id", "district": "supervisor_district",
           "neighborhood": "neighborhoods_analysis_boundaries",
           "construction_type": "existing_construction_type_description"}
TYPE_NAMES = {1: "new construction", 2: "new construction wood frame", 3: "additions alterations or repairs",
              4: "sign - erect", 5: "grade or quarry or fill or excavate", 6: "demolitions",
              7: "wall or painted sign", 8: "otc alterations permit"}  # SF DBI permit type definitions (real)
UNIT_WORDS = {"dollars": "whole US dollars", "cents": "cents", "thousands": "thousands of US dollars"}
UNIT_TF = {"dollars": "identity", "cents": "cents_to_dollars", "thousands": "thousands_to_dollars"}
STYLE_TF = {"iso": "identity", "mdy": "date_mdy", "dmy": "date_dmy", "compact": "date_yyyymmdd"}
TRANSFORMS = list(T["transforms"])
SWAP_PAIRS = [("estimated_cost", "revised_cost"), ("existing_stories", "proposed_stories"),
              ("existing_use", "proposed_use"), ("filed_date", "issued_date"), ("issued_date", "completed_date"),
              ("issued_date", "approved_date"), ("completed_date", "status_date"), ("filed_date", "created_date"),
              ("existing_stories", "existing_units"), ("proposed_stories", "proposed_units"),
              ("permit_id", "record_id")]
ROWS_PER_SOURCE = 30
_rows_cache = []
_perms = {}
_load_lock = threading.Lock()


def snapshot():
    # Locked whole-list load: the runner generates instances in concurrent threads, and a lazily extended list let
    # early threads see a partial snapshot (family A found this race; see RESULTS.md).
    if not _rows_cache:
        with _load_lock:
            if not _rows_cache:
                with open(HERE / "data" / "permits.csv", newline="") as fh:
                    rows = list(csv.DictReader(fh))
                _rows_cache[:] = rows
    return _rows_cache


def rows_for(seed, i):
    """Source i's rows: slice i of a per-seed permutation of the snapshot (7200 rows = 240 x 30, so no row repeats
    within an instance up to n=240; beyond that slices wrap). Mailing-address decoys come from slice i + 120."""
    snap = snapshot()
    with _load_lock:
        if seed not in _perms:
            perm = list(range(len(snap)))
            random.Random(f"schemamap-rows:{seed}").shuffle(perm)
            _perms[seed] = perm
    perm, k, m = _perms[seed], ROWS_PER_SOURCE, len(snap) // ROWS_PER_SOURCE

    def sl(j):
        j %= m
        return [snap[x] for x in perm[j * k:(j + 1) * k]]
    return sl(i), sl(i + m // 2)


def fmt_date(iso, style):
    if not iso:
        return ""
    y, m, d = iso.split("-")
    return {"iso": iso, "mdy": f"{m}/{d}/{y}", "dmy": f"{d}/{m}/{y}", "compact": y + m + d}[style]


def fmt_cost(v, unit):
    v = int(v)
    return {"dollars": str(v), "cents": str(v * 100), "thousands": f"{v / 1000:.3f}"}[unit]


def _cell(r, kind, s, other):
    """Value of one column. kind: a target field, a decoy, or 'site_address'."""
    if kind == "permit_type":
        code = int(r["permit_type"])
        return {"code": str(code), "letter": s["letters"].get(code, ""), "name": TYPE_NAMES[code]}[s["type_mode"]]
    if kind in ("street_name",):
        return f"{r['street_name']} {r['street_suffix']}"
    if kind in ("site_address", "mailing_address"):
        a = other if kind == "mailing_address" else r
        return f"{a['street_number']} {a['street_name']} {a['street_suffix']}"
    if kind in DATES or kind in ("created_date", "status_date", "approved_date"):
        return fmt_date(r[SRC_COL[kind]], s["style"])
    if kind in COSTS:
        return fmt_cost(r[SRC_COL[kind]], s["units"][kind])
    return r[SRC_COL[kind]]


def _decidable(s, rows):
    """mdy/dmy must be settled by the values: some target date has a day > 12."""
    days = [int(r[f].split("-")[2]) for r in rows for f in DATES if f in s["present"] and r[f]]
    return any(d > 12 for d in days)


def make_source(seed, i):
    rng = random.Random(f"schemamap:{seed}:{i}")
    rows, others = rows_for(seed, i)
    present = [f for f in FIELDS if rng.random() < 0.8]
    while len(present) < 9:
        present = sorted(set(present) | {rng.choice(FIELDS)}, key=FIELDS.index)
    s = {"present": present, "style": rng.choice(["iso", "mdy", "dmy", "compact"]),
         "type_mode": rng.choices(["code", "letter", "name"], [3, 4, 3])[0],
         "units": {c: rng.choices(["dollars", "cents", "thousands"], [4, 3, 3])[0] for c in COSTS}}
    if s["style"] in ("mdy", "dmy") and not _decidable(s, rows):
        s["style"] = "compact"
    letters = list("ABCDEFGH")
    rng.shuffle(letters)
    s["letters"] = {code: letters[k] for k, code in enumerate(sorted(TYPE_NAMES))}
    combined = "street_number" in present and "street_name" in present and rng.random() < 0.35
    decoys = rng.sample([d for d in DECOYS if d != "mailing_address" or "street_name" in present], rng.randint(3, 5))
    # columns: (meaning, kind); kind = what the values hold
    cols = [(f, f) for f in present if not (combined and f in ("street_number", "street_name"))]
    if combined:
        cols.append(("site_address", "site_address"))
    cols += [(d, d) for d in decoys]
    rng.shuffle(cols)
    used, names = set(), {}
    for meaning, _ in cols:
        bank = T["aliases"].get(meaning) or T["decoy_aliases"].get(meaning) or T["aliases"]["street_name"]
        if meaning == "site_address":
            bank = ["site_address", "address", "location_addr", "addr_full"]
        opts = [a for a in bank if a not in used]
        name = f"c{len(used) + 1:02d}" if rng.random() < 0.25 or not opts else rng.choice(opts)
        while name in used:
            name += "_x"
        used.add(name)
        names[meaning] = name
    # true descriptions
    desc = {}
    for meaning, _ in cols:
        if meaning in COSTS:
            desc[meaning] = common.fill(rng.choice(T[meaning + "_desc"]), unit=UNIT_WORDS[s["units"][meaning]])
        elif meaning == "permit_type":
            key = {"code": "type_code_desc", "letter": "type_letter_desc", "name": "type_name_desc"}[s["type_mode"]]
            codes = "; ".join(f"{s['letters'][c]} = {TYPE_NAMES[c]}" for c in sorted(TYPE_NAMES, key=lambda c: s["letters"][c]))
            desc[meaning] = common.fill(rng.choice(T[key]), codes=codes)
        elif meaning == "site_address":
            desc[meaning] = rng.choice(T["site_address_desc"])
        elif meaning in FIELDS:
            desc[meaning] = rng.choice(T["field_desc"][meaning])
        else:
            desc[meaning] = rng.choice(T["decoy_desc"][meaning])
    shown, notes, errata, involved = dict(desc), [], [], set()
    dname = dict(names)  # names as written in the dictionary (renames change only the export header)
    dates_here = [f for f in DATES if f in present]
    if len(dates_here) >= 2 and rng.random() < 0.45:
        a, b = rng.sample(dates_here, 2)
        for f in (a, b):
            shown[f] = T["pair_col_desc"]
        x, y = sorted([a, b], key=lambda f: names[f])
        notes.append(common.fill(T["date_pair_note"], a=names[x], b=names[y], f1=a, f2=b))
        involved |= {a, b}
    k = rng.choices([0, 1, 2], [2, 10, 5])[0]
    if not notes and k == 0:
        k = 1
    simple = [m for m, _ in cols if m not in ("site_address", "permit_type")]
    for _ in range(k):
        kinds = ["swap", "rename"] + (["unit"] if any(c in names and c not in involved for c in COSTS) else [])
        kind = rng.choice(kinds)
        if kind == "swap":
            pairs = [p for p in SWAP_PAIRS if all(x in names and x not in involved for x in p)]
            free = [m for m in simple if m not in involved]
            if not pairs and len(free) < 2:
                continue
            a, b = rng.choice(pairs) if pairs and rng.random() < 0.8 else rng.sample(free, 2)
            shown[a], shown[b] = desc[b], desc[a]
            errata.append(common.fill(T["errata_swap"], a=names[a], b=names[b]))
            involved |= {a, b}
        elif kind == "unit":
            c = rng.choice([c for c in COSTS if c in names and c not in involved])
            stale = rng.choice([u for u in UNIT_WORDS if u != s["units"][c]])
            shown[c] = common.fill(rng.choice(T[c + "_desc"]), unit=UNIT_WORDS[stale])
            errata.append(common.fill(T["errata_unit"], release=f"{2019 + rng.randint(0, 5)}.{rng.randint(1, 4)}",
                                      col=names[c], unit=UNIT_WORDS[s["units"][c]]))
            involved.add(c)
        else:
            free = [m for m, _ in cols if m not in involved]
            m = rng.choice(free)
            old = rng.choice([a for a in (T["aliases"].get(m) or T["decoy_aliases"].get(m) or ["addr_line"])
                              if a not in used] or [f"col_{m}"])
            used.add(old)
            dname[m] = old
            errata.append(common.fill(T["errata_rename"], old=old, new=names[m]))
            involved.add(m)
    header = [names[m] for m, _ in cols]
    data = [[_cell(r, kind, s, o) for _, kind in cols] for r, o in zip(rows, others)]
    order = sorted(cols, key=lambda c: rng.random())
    partner = T["partners"][rng.randrange(len(T["partners"]))]
    lines = [common.fill(T["dict_header"], src=f"SRC{i + 1:03d}", partner=partner, ncols=len(cols)),
             "", "| column | description |", "|---|---|"]
    lines += [f"| {dname[m]} | {shown[m]} |" for m, _ in order]
    if notes:
        lines += ["", T["notes_header"]] + [f"- {x}" for x in notes]
    if errata:
        lines += ["", T["errata_header"]] + [f"- {x}" for x in errata]
    gold = {}
    for f in FIELDS:
        if f not in present:
            gold[f] = None
        elif combined and f in ("street_number", "street_name"):
            gold[f] = [names["site_address"], "address_number" if f == "street_number" else "address_street"]
        elif f in DATES:
            gold[f] = [names[f], STYLE_TF[s["style"]]]
        elif f in COSTS:
            gold[f] = [names[f], UNIT_TF[s["units"][f]]]
        elif f == "permit_type":
            gold[f] = [names[f], {"code": "identity", "letter": "decode_codelist", "name": "type_name_to_code"}[s["type_mode"]]]
        else:
            gold[f] = [names[f], "identity"]
    traps = {"pair": bool(notes), "errata": [e for e in errata], "combined": combined, "type_mode": s["type_mode"],
             "style": s["style"], "opaque": sum(1 for v in names.values() if re.fullmatch(r"c\d\d(_x)*", v))}
    src = {"id": f"SRC{i + 1:03d}", "partner": partner, "header": header, "rows": data,
           "dictionary": "\n".join(lines)}
    return src, gold, traps


def schema_text():
    out = [T["schema_header"], ""]
    out += [f"- {f}: {T['target_fields'][f]}" for f in FIELDS]
    out += ["", "| code | permit type |", "|---|---|"] + [f"| {c} | {n} |" for c, n in TYPE_NAMES.items()]
    out += [""] + [f"- {k}: {v}" for k, v in T["transforms"].items()]
    return "\n".join(out)


def generate(n, seed, out_dir):
    srcs, gold, traps = [], {}, {}
    for i in range(n):
        s, g, t = make_source(seed, i)
        srcs.append(s)
        gold[s["id"]] = g
        traps[s["id"]] = t
    ex = {"SRC001": {"permit_id": ["application_no", "identity"], "zipcode": None, "...": "..."}}
    prompt = (common.fill(T["intro"], agency=T["agency"], n=n, first=srcs[0]["id"], last=srcs[-1]["id"]) + "\n\n"
              + common.fill(T["answer_line"], example=json.dumps(ex)))
    return common.finish(TYPE_ID, n, seed, out_dir, prompt, {"mapping": gold, "traps": traps}, TIERS,
                         files={"sources.json": json.dumps(srcs), "schema.txt": schema_text()})


_worlds = {}


def _world(world_dir):
    w = _worlds.get(world_dir)
    if w is None:
        srcs = json.loads((Path(world_dir) / "sources.json").read_text())
        w = _worlds.setdefault(world_dir, ({s["id"]: s for s in srcs}, (Path(world_dir) / "schema.txt").read_text()))
    return w


def tools(world_dir):
    def src(a):
        sid = str(a.get("source", "")).strip().upper()
        srcs = _world(world_dir)[0]
        if sid not in srcs:
            raise KeyError(f"unknown source {sid!r}; use list_sources")
        return srcs[sid]

    def list_sources(a):
        srcs = list(_world(world_dir)[0].values())
        off, lim = common.page_args(a, 50, 100)
        out = [f"{s['id']} | {s['partner']} | {len(s['header'])} columns | {len(s['rows'])} rows" for s in srcs[off:off + lim]]
        more = f"\n(showing {off}-{off + len(out) - 1} of {len(srcs)}; next offset {off + lim})" if off + lim < len(srcs) else ""
        return "\n".join(out) + more

    def read_rows(a):
        s = src(a)
        off, lim = common.page_args(a, 15, 15)
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(s["header"])
        w.writerows(s["rows"][off:off + lim])
        tail = f"(rows {off}-{min(off + lim, len(s['rows'])) - 1} of {len(s['rows'])})"
        return buf.getvalue() + tail

    def column_profile(a):
        s = src(a)
        col = str(a.get("column", "")).strip()
        if col not in s["header"]:
            raise KeyError(f"no column {col!r} in {s['id']}; columns: {', '.join(s['header'])}")
        vals = [r[s["header"].index(col)] for r in s["rows"]]
        nb = [v for v in vals if v != ""]
        try:
            key = (lambda v: float(v)) if all(re.fullmatch(r"-?\d+(\.\d+)?", v) for v in nb) else str
        except ValueError:
            key = str
        ex = list(dict.fromkeys(nb))[:8]
        return (f"{s['id']}.{col}: {len(vals)} rows, {len(vals) - len(nb)} blank, {len(set(nb))} distinct\n"
                f"min: {min(nb, key=key) if nb else ''}\nmax: {max(nb, key=key) if nb else ''}\nexamples: " + " | ".join(ex))

    tt = T["tools"]
    return [
        common.tool("read_target_schema", tt["read_target_schema"], {}, [], lambda a: _world(world_dir)[1]),
        common.tool("list_sources", tt["list_sources"], {"offset": {"type": "integer"}, "limit": {"type": "integer"}},
                    [], list_sources),
        common.tool("read_dictionary", tt["read_dictionary"], {"source": {"type": "string"}}, ["source"],
                    lambda a: src(a)["dictionary"]),
        common.tool("read_rows", tt["read_rows"], {"source": {"type": "string"}, "offset": {"type": "integer"},
                                                   "limit": {"type": "integer"}}, ["source"], read_rows),
        common.tool("column_profile", tt["column_profile"], {"source": {"type": "string"}, "column": {"type": "string"}},
                    ["source", "column"], column_profile),
    ]


SRC_RE = re.compile(r"^SRC\d{3}$")


def _norm_entry(v):
    if v is None or v == "" or (isinstance(v, str) and v.strip().lower() in ("null", "none")):
        return None
    if isinstance(v, str):
        return (v.strip().lower(), "identity")
    if isinstance(v, (list, tuple)) and v and all(isinstance(x, str) for x in v):
        if v[0].strip().lower() in ("null", "none", ""):
            return None
        return (v[0].strip().lower(), (v[1] if len(v) > 1 else "identity").strip().lower())
    if isinstance(v, dict) and isinstance(v.get("column"), str):
        return (v["column"].strip().lower(), str(v.get("transform") or "identity").strip().lower())
    return ("<invalid>", "")


SALVAGE = re.compile(r'"(SRC\d{3})"\s*:\s*(?=\{)', re.I)


def parse_answer(answer):
    """The last JSON object keyed by source ids. If none parses (one malformed key, truncated output), salvage every
    complete '"SRCnnn": {...}' entry on its own; malformed or missing sources then count as wrong. A later entry for
    the same source replaces an earlier one, as a later answer object would."""
    obj = common.last_json(answer, lambda o: isinstance(o, dict) and any(SRC_RE.match(str(k).strip().upper()) for k in o))
    if obj is None:
        obj, dec = {}, json.JSONDecoder()
        for m in SALVAGE.finditer(answer or ""):
            try:
                v, _ = dec.raw_decode(answer, m.end())
            except ValueError:
                continue
            obj[m.group(1)] = v
    out = {}
    for k, v in obj.items():
        k = str(k).strip().upper()
        if SRC_RE.match(k) and isinstance(v, dict):
            out[k] = {str(f).strip(): _norm_entry(x) for f, x in v.items()}
    return out


def unit_correct(pred, gold):
    if pred is None:
        return 0, 0
    ok = 0
    for f in FIELDS:
        g = gold[f]
        p = pred.get(f)
        ok += (p is None) if g is None else (p == (g[0].lower(), g[1]))
    return ok == len(FIELDS), ok


def score(task, answer):
    gold = common.gold_of(task)["mapping"]
    pred = parse_answer(answer)
    n_ok, fields_ok = 0, 0
    for sid, g in gold.items():
        u, f = unit_correct(pred.get(sid), g)
        n_ok += u
        fields_ok += f
    n = len(gold)
    return {"score": n_ok / n, "score_exact": float(n_ok == n), "score_partial": fields_ok / (n * len(FIELDS)),
            "n_correct": n_ok, "n_units": n, "n_answered": sum(1 for s in gold if s in pred)}


def name_similarity_mapping(header):
    """Baseline helper: each field -> most similar header name (difflib), for the name-similarity heuristic."""
    out = {}
    for f in FIELDS:
        best = max(header, key=lambda h: difflib.SequenceMatcher(None, f, h).ratio())
        out[f] = best if difflib.SequenceMatcher(None, f, best).ratio() >= 0.5 else None
    return out
