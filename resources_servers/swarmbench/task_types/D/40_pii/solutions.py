"""Type 40 solutions: tool-only oracle (parses the K3 passage templates back with common.tre and looks people up in
the directory and register through the tools), plausible-wrong baselines (regex scan, careless reader), partial
readers that skip one rule each, and single units (check 5)."""
import json
import re

from .. import common
from . import type as track

T = track.T
NAME = r"[A-Z][a-z]+ [A-Z][a-z]+"
PHONE = r"\(\d{3}\) 555-\d{4}"
EMAIL = r"[a-z0-9.]+@[a-z]+\.example"
ADDR = r"\d+ [A-Za-z]+ [A-Za-z]+, [A-Za-z ]+?, [A-Z]{2} \d{5}"
DOB = r"[A-Z][a-z]+ \d{1,2}, \d{4}"
IDN = r"\d{3}-\d{2}-\d{4}"
CARD = r"\d{4} \d{4} \d{4} \d{4}"
ISBN = r"\d{3}-\d-\d{4}-\d{4}-\d"
TITLE = "|".join(map(re.escape, track.TITLES))
AGENCY = "|".join(map(re.escape, track.AGENCIES))
COMPANY = (f"(?:{'|'.join(map(re.escape, track.CO_A))}) (?:{'|'.join(map(re.escape, track.CO_B))}) "
           f"(?:{'|'.join(map(re.escape, track.CO_C))})")
PATS = {
    "contact": [common.tre(t, name=NAME, phone=PHONE, email=EMAIL) for t in T["p_contact"]],
    "home": [common.tre(t, name=NAME, address=ADDR, dob=DOB) for t in T["p_home"]],
    "id": [common.tre(t, name=NAME, id=IDN) for t in T["p_id"]],
    "card": [common.tre(t, name=NAME, card=CARD) for t in T["p_card"]],
    "via": [common.tre(t, name=NAME, phone=PHONE, company=COMPANY) for t in T["p_via_business"]],
    "offpriv": [common.tre(t, title=TITLE, name=NAME, agency=AGENCY, phone=PHONE) for t in T["p_official_private"]],
    "sample": [common.tre(t, name=NAME, id=IDN) for t in T["d_sample"]],
    "official": [common.tre(t, title=TITLE, name=NAME, agency=AGENCY, phone=PHONE) for t in T["d_official"]],
    "business": [common.tre(t, company=COMPANY, phone=PHONE, email=EMAIL, address=ADDR) for t in T["d_business"]],
    "case": [common.tre(t, case=IDN) for t in T["d_case"]],
    "part": [common.tre(t, part=IDN) for t in T["d_part"]],
    "isbn": [common.tre(t, isbn=ISBN) for t in T["d_isbn"]],
}


def _fns(task):
    return {t["name"]: t["fn"] for t in track.tools(task["world_dir"])}


def doc_ids(fns):
    ids, off = [], 0
    while True:
        out = fns["list_documents"]({"offset": off, "limit": 200})
        ids += re.findall(r"^(DOC-\d{4}) \|", out, re.M)
        m = re.search(r"next offset (\d+)\)$", out)
        if not m:
            return ids
        off = int(m.group(1))


def read_doc(fns, d):
    paras, start = [], 1
    while True:
        out = fns["read_document"]({"doc_id": d, "start": start})
        paras += [m.group(1) for m in re.finditer(r"^\[\d+\] (.*)$", out, re.M)]
        m = re.search(r"next start (\d+)\)$", out)
        if not m:
            return paras
        start = int(m.group(1))


def passages(paras):
    out = []
    for p in paras:
        for kind, rxs in PATS.items():
            m = next((m for rx in rxs for m in [rx.search(p)] if m), None)
            if m:
                out.append((kind, m.groupdict()))
                break
    return out


def lookup(fns, tool, name):
    out = fns[tool]({"query": name})
    for row in out.splitlines()[1:]:
        cells = [c.strip() for c in row.split(" | ")]
        if cells[0] == name:
            return cells
    return None


def decide(ps, status, skip_register=False, skip_luhn=False, officials=False, business=False, numbers=False):
    items = set()
    for kind, g in ps:
        if kind == "offpriv":
            items |= {("name", g["name"]), ("phone", g["phone"])}
        elif kind in ("contact", "home", "id", "card", "via"):
            if status.get(g["name"]) == "granted" and not skip_register:
                continue
            items.add(("name", g["name"]))
            if kind == "contact":
                items |= {("phone", g["phone"]), ("email", g["email"])}
            elif kind == "home":
                items |= {("address", g["address"]), ("dob", track.norm_value("dob", g["dob"]))}
            elif kind == "via":
                pass
            elif kind == "id":
                items.add(("gov_id", g["id"]))
            elif skip_luhn or track.luhn_ok(re.sub(r"\D", "", g["card"])):
                items.add(("card", g["card"]))
        elif kind == "official" and officials:
            items |= {("name", g["name"]), ("phone", g["phone"])}
        elif kind == "business" and business:
            items |= {("phone", g["phone"]), ("email", g["email"]), ("address", g["address"])}
        elif kind in ("case", "part") and numbers:
            items.add(("gov_id", g.get("case") or g.get("part")))
    return items


def read_all(task):
    fns = _fns(task)
    docs = {}
    for d in doc_ids(fns):
        ps = passages(read_doc(fns, d))
        status = {}
        for kind, g in ps:
            if kind in ("contact", "home", "id", "card", "via") and g["name"] not in status:
                row = lookup(fns, "search_register", g["name"])
                status[g["name"]] = row[1] if row else None
        docs[d] = (ps, status)
    return docs


def answer_of(sets):
    return json.dumps({d: [{"type": t, "value": v} for t, v in sorted(s)] for d, s in sets.items()})


def oracle(task):
    return answer_of({d: decide(ps, st) for d, (ps, st) in read_all(task).items()})


RX = {"phone": PHONE, "email": EMAIL, "gov_id": IDN, "card": CARD, "address": ADDR, "dob": DOB}


def regex_answer(task):
    """Pattern scan with no context rules: every phone/email/id/card/address/date and every capitalized pair in a
    paragraph that holds one of them."""
    fns = _fns(task)
    out = {}
    for d in doc_ids(fns):
        items = set()
        for p in read_doc(fns, d):
            hits = {(t, m) for t, rx in RX.items() for m in re.findall(rx, p)}
            if hits:
                items |= {(t, track.norm_value(t, v) if t == "dob" else v) for t, v in hits}
                items |= {("name", m) for m in re.findall(NAME, p)}
        out[d] = items
    return answer_of(out)


KEYWORDS = re.compile(r"\b(docket|case|part|catalog|catalogue|isbn|reprint)\b", re.I)


def smart_regex_answer(task):
    """The best pattern reader we could write without understanding passages: patterns plus role-mailbox, Luhn,
    keyword (docket/part/ISBN) and company filters, and exact-name greps of the directory and the register."""
    fns = _fns(task)
    out = {}
    for d in doc_ids(fns):
        items = set()
        for p in read_doc(fns, d):
            hits = {(t, m) for t, rx in RX.items() for m in re.findall(rx, p)}
            if not hits or (KEYWORDS.search(p) and not re.search(PHONE + "|" + EMAIL, p)):
                continue
            # a name gazetteer (as a NER library would have) instead of bare capitalized pairs
            names = [m for m in re.findall(NAME, p) if m.split()[0] in track.FIRST and m.split()[1] in track.LAST]
            official = [n for n in names if lookup(fns, "search_directory", n)]
            granted = [n for n in names if (lookup(fns, "search_register", n) or [0, 0])[1] == "granted"]
            if granted:
                continue
            items |= {("name", n) for n in names if n not in official}
            for t, v in hits:
                if t == "email" and v.split("@")[0] in track.ROLES:
                    continue
                if t == "card" and not track.luhn_ok(re.sub(r"\D", "", v)):
                    continue
                if t == "phone" and official:
                    continue
                if t == "address" and re.search(COMPANY, p):
                    continue
                items.add((t, track.norm_value(t, v) if t == "dob" else v))
        out[d] = items
    return answer_of(out)


def wrong(task):
    docs = read_all(task)
    sets = lambda **kw: answer_of({d: decide(ps, st, **kw) for d, (ps, st) in docs.items()})  # noqa: E731
    return {"regex": regex_answer(task), "smart_regex": smart_regex_answer(task),
            "careless_no_lookups": sets(skip_register=True, skip_luhn=True, officials=True, business=True,
                                        numbers=True),
            "skip_register": sets(skip_register=True),
            "skip_luhn": sets(skip_luhn=True),
            "report_officials": sets(officials=True),
            "report_business": sets(business=True),
            "numbers_as_ids": sets(numbers=True)}


def units(task, k, rng):
    g = common.gold_of(task)["docs"]
    ids = rng.sample(sorted(g), min(k, len(g)))
    out = []
    for d in ids:
        gold = {track.norm_item(t, v) for t, v in g[d]["items"]}
        ex = track.example(d)

        def check(text, d=d, gold=gold):
            p = track.parse_answer(text)
            return None if d not in p else p[d] == gold
        out.append({"id": f"{task['id']}:{d}", "kind": ",".join(sorted(set(g[d]["kinds"]))) or "none",
                    "prompt": common.fill(T["unit_prompt"], doc=d, example=ex) + "\n\n" + track.rules_text(ex),
                    "gold": g[d]["items"], "check": check})
    return out
