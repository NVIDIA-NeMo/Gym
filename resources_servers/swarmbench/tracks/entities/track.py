# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import fcntl
import json
import os
import random
import re
import threading
import time
from collections import Counter

from . import index
from .render import SOURCES, gold_profile, plan_records, render
from .world import build_entities

MAX_OUT = 8000
SOURCE_DESC = {
    "reg": "official registry filings (civil registry for people, companies registry). Every entity has at least one. "
           "An entity that changed its name or moved may have several filings, all with the same Personal ID (people) "
           "or Reg. No. (companies)",
    "crm": "CRM export rows. Names may be misspelled, nicknames or 'Last, First'. Only name and phone are kept current "
           "(as of last_updated); email, website, street, city and company are as entered at account creation",
    "sig": "emails ending in a personal signature (people) or a company footer (companies)",
    "ltr": "OCR text of scanned letters, with OCR errors in the prose (addresses and phone numbers are legible)",
    "evt": "event log entries: ADDRESS_CHANGE / RELOCATION, NAME_CHANGE, PHONE_CHANGE",
}
PERSON_KEYS = ["name", "born", "email", "phone", "city", "employer"]
COMPANY_KEYS = ["name", "founded", "website", "phone", "city"]
W_PAIR, W_B3, W_ATTR = 0.25, 0.25, 0.5

PROMPT = """You are resolving entities across {total} records from 5 offline sources. The records describe about {n} \
real-world entities (people and companies); each entity has 4 to 8 records spread over 3 to 5 sources. Group the \
records into one cluster per entity and write a canonical profile for each cluster.

Sources (record ids look like reg-00012):
{sources}

Linking evidence: different entities can share exactly the same name, and one entity can appear under a nickname, a \
typo, an initial, a maiden name or a former company name. Personal IDs, Reg. Nos., phone numbers, email addresses, \
websites and street addresses (number + street) each belong to exactly one entity, but their formats vary \
("+1 (415) 555-0134" = "415.555.0134"; "12 Oak St" = "12 Oak Street"). Names, cities and employers are not unique. \
Every record of an entity is connected to its registry filing through a chain of shared IDs, phones, emails, \
websites or street addresses (old ones count too: event "from" values, stale crm streets, letters that only give an \
address, so search every street address and phone you collect, not just names).
People and companies are separate entities. A person's crm contact row, email signature or letter that names a \
company as employer belongs to the person, not the company, and an email domain is not a company's website key. \
Company records are: Companies Registry filings, crm rows with "type: account", emails from a company footer \
(info@...), letters on company letterhead and RELOCATION / company NAME_CHANGE events.

Profile rules (apply exactly; "latest" means by the record's own date: filed / last_updated / Date / event date):
- name: the full legal (registered) name on the entity's most recently filed reg record.
- born (people, YYYY-MM-DD) / founded (companies, year): from the reg record.
- email (people): the sender address of the latest sig email; if the entity has no sig, the email of the most \
recently updated crm row.
- employer (people): the company named in the signature of the latest sig email; if no sig, the company field of \
the most recently updated crm row.
- website (companies): the website of the most recently updated crm row; if no crm row, the web address in the latest \
sig footer.
- phone: the phone of the most recently updated crm row; if no crm row, the phone in the latest sig.
- city: the destination city ("to") of the latest ADDRESS_CHANGE / RELOCATION event in evt; if there is none, the \
city of the registered address on the most recently filed reg record.
Values elsewhere (letters, registry contact phones, older or stale fields) are distractors when they conflict.

Tools: list_records(source, offset, limit) pages through a source with one-line summaries; read_record(ids) returns \
full records; search_records(query, source, k) is an offline keyword index that normalises phone numbers and \
tolerates misspelled names; scratch_write / scratch_read / scratch_list are a shared key-value store every agent in \
this run can see (use it for proposed clusters, claims and links; each write records who wrote it).

Answer: one line per entity, record ids first, then the profile fields:
reg-00012 crm-00401 sig-00977 evt-00020 | name=Katherine Ann Moreau | born=1984-03-12 | email=kate.moreau@varn.com \
| phone=415-555-0134 | city=Port Aster | employer=Varn Analytics
reg-00013 crm-00018 sig-00002 ltr-00310 | name=Varn Analytics Inc. | founded=1998 | website=varnanalytics.com | \
phone=415-555-0199 | city=Velda
Each record id belongs in exactly one line (a repeated id keeps its first line; unlisted records count as \
singletons). Score = {wp} x pairwise F1 + {wb} x B-cubed F1 over the clusters + {wa} x profile attribute accuracy \
(an entity's profile only counts when a single line holds most of its records and those records are mostly its own).
"""


def generate(n, seed, out_dir):
    rng = random.Random(seed)
    ents, _ = build_entities(n, random.Random(rng.randrange(2**31)))
    prng = random.Random(rng.randrange(2**31))
    plans = [plan_records(prng, e) for e in ents]
    allrecs = [(ei, r) for ei, recs in enumerate(plans) for r in recs]
    rng.shuffle(allrecs)
    counts = Counter()
    for _, r in allrecs:
        r["id"] = f"{r['src']}-{counts[r['src']]:05d}"
        counts[r["src"]] += 1
    trng = random.Random(rng.randrange(2**31))
    rows = sorted(({"id": r["id"], "text": render(trng, ents[ei], r)} for ei, r in allrecs), key=lambda x: x["id"])
    world_dir = os.path.join(out_dir, "world")
    os.makedirs(world_dir, exist_ok=True)
    with open(os.path.join(world_dir, "records.jsonl"), "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    index.build((row["text"] for row in rows), os.path.join(world_dir, "index.pkl"))
    gold = [{"kind": e["kind"], "records": sorted(r["id"] for r in recs), "profile": gold_profile(e, recs)}
            for e, recs in zip(ents, plans)]
    gold_path = os.path.join(out_dir, "gold.json")
    with open(gold_path, "w") as f:
        json.dump({"n": n, "seed": seed, "entities": gold}, f, indent=1, ensure_ascii=False)
    sources = "\n".join(f"- {s} ({counts[s]} records): {SOURCE_DESC[s]}." for s in SOURCES)
    prompt = PROMPT.format(total=len(rows), n=n, sources=sources, wp=W_PAIR, wb=W_B3, wa=W_ATTR)
    return {"id": f"entities-n{n}-s{seed}", "prompt": prompt, "world_dir": world_dir, "n_units": n,
            "gold_path": gold_path}


class World:
    def __init__(self, world_dir):
        self.recs, self.pos = [], {}
        with open(os.path.join(world_dir, "records.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                self.pos[r["id"]] = len(self.recs)
                self.recs.append(r)
        self.by_src = {s: [i for i, r in enumerate(self.recs) if r["id"].startswith(s + "-")] for s in SOURCES}
        self.src_set = {s: set(v) for s, v in self.by_src.items()}
        self.index = index.Index(os.path.join(world_dir, "index.pkl"))
        self.scratch = Scratch(os.path.join(world_dir, "scratch.jsonl"))


class Scratch:
    # Shared key-value store backed by an append-only log; flock makes appends atomic across processes and the
    # tail is re-read before every operation so all processes see the same state.
    def __init__(self, path):
        self.path, self.lock, self.off, self.data, self.seq = path, threading.Lock(), 0, {}, 0

    def _sync(self):
        if not os.path.exists(self.path):
            return
        with open(self.path, "rb") as f:
            f.seek(self.off)
            chunk = f.read()
        end = chunk.rfind(b"\n") + 1
        for line in chunk[:end].splitlines():
            e = json.loads(line)
            self.data.setdefault(e["key"], []).append(e)
            self.seq += 1
        self.off += end

    def write(self, key, value, agent):
        line = json.dumps({"key": key, "value": value, "agent": agent, "t": round(time.time(), 3)}) + "\n"
        with self.lock:
            with open(self.path, "ab") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.write(line.encode())
                f.flush()
                fcntl.flock(f, fcntl.LOCK_UN)
            self._sync()
            return len(self.data[key])

    def get(self, key):
        with self.lock:
            self._sync()
            return list(self.data.get(key, []))

    def items(self):
        with self.lock:
            self._sync()
            return sorted((k, v[-1]) for k, v in self.data.items())


_WORLDS, _LOCK = {}, threading.Lock()


def load(world_dir):
    key = os.path.abspath(world_dir)
    if key not in _WORLDS:
        with _LOCK:
            if key not in _WORLDS:
                _WORLDS[key] = World(key)
    return _WORLDS[key]


def _int(v, default, lo, hi):
    try:
        return max(lo, min(hi, int(v)))
    except (TypeError, ValueError):
        return default


def summary(text, width=200):
    s = " / ".join(x for x in text.split("\n") if x.strip())
    return s[:width] + ("..." if len(s) > width else "")


def cap(s):
    return s if len(s) <= MAX_OUT else s[:MAX_OUT - 60] + "\n[output truncated at 8000 chars]"


def tools(world_dir):
    def list_records(args):
        w = load(world_dir)
        src = str(args.get("source", "")).strip().lower()
        if src not in w.by_src:
            return f"Error: source must be one of {SOURCES}."
        ids = w.by_src[src]
        off, lim = _int(args.get("offset"), 0, 0, len(ids)), _int(args.get("limit"), 20, 1, 30)
        lines = [f"{w.recs[i]['id']} | {summary(w.recs[i]['text'])}" for i in ids[off:off + lim]]
        more = f"[{src}: records {off}-{off + len(lines) - 1} of {len(ids)}" + \
               (f"; next offset={off + len(lines)}]" if off + len(lines) < len(ids) else "; end]")
        return cap("\n".join(lines + [more]))

    def read_record(args):
        w = load(world_dir)
        ids = re.findall(r"[a-z]{3}-\d{5}", str(args.get("ids", args.get("id", ""))).lower())
        if not ids:
            return "Error: give one or more record ids, e.g. 'reg-00012, crm-00401' (max 10)."
        out = []
        for rid in ids[:10]:
            i = w.pos.get(rid)
            out.append(f"=== {rid} ===\n" + (w.recs[i]["text"] if i is not None else "Error: unknown record id."))
        return cap("\n\n".join(out))

    def search_records(args):
        w = load(world_dir)
        src = str(args.get("source") or "").strip().lower()
        if src and src not in w.by_src:
            return f"Error: source must be empty or one of {SOURCES}."
        hits = w.index.search(str(args.get("query", ""))[:500], _int(args.get("k"), 10, 1, 20),
                              w.src_set[src] if src else None)
        if not hits:
            return "No results."
        return cap("\n".join(f"{w.recs[d]['id']} | score={s:.2f} | {summary(w.recs[d]['text'], 300)}"
                             for d, s in hits))

    def scratch_write(args):
        key, value = str(args.get("key", "")).strip()[:200], str(args.get("value", ""))[:4000]
        if not key:
            return "Error: empty key."
        v = load(world_dir).scratch.write(key, value, str(args.get("_agent", "?")))
        return f"ok: {key!r} now at version {v}"

    def scratch_read(args):
        key = str(args.get("key", "")).strip()
        hist = load(world_dir).scratch.get(key)
        if not hist:
            return f"No entry for {key!r}."
        last = hist[-1]
        older = ", ".join(f"v{i + 1} by {e['agent']}" for i, e in enumerate(hist[:-1]))
        return cap(f"{key} (version {len(hist)}, written by agent {last['agent']}"
                   + (f"; earlier: {older}" if older else "") + f"):\n{last['value']}")

    def scratch_list(args):
        prefix, contains = str(args.get("prefix", "")), str(args.get("contains", "")).lower()
        items = [(k, e) for k, e in load(world_dir).scratch.items()
                 if k.startswith(prefix) and (not contains or contains in e["value"].lower() or contains in k.lower())]
        off = _int(args.get("offset"), 0, 0, len(items))
        lines = [f"{k} [by {e['agent']}]: {summary(e['value'], 160)}" for k, e in items[off:off + 40]]
        tail = f"[{len(items)} matching keys; showing {off}-{off + len(lines) - 1}" + \
               (f"; next offset={off + len(lines)}]" if off + len(lines) < len(items) else "]")
        return cap("\n".join(lines + [tail])) if items else "No matching keys."

    obj = lambda props, req: {"type": "object", "properties": props, "required": req}
    s, i = {"type": "string"}, {"type": "integer"}
    return [
        {"name": "list_records", "description": "Page through one source (reg, crm, sig, ltr, evt) in id order. "
                                                "Returns 'id | one-line summary' for up to `limit` (<=30) records.",
         "parameters": obj({"source": s, "offset": i, "limit": i}, ["source"]), "fn": list_records},
        {"name": "read_record", "description": "Full text of up to 10 records; `ids` is a comma or space separated "
                                               "list such as 'reg-00012, crm-00401'.",
         "parameters": obj({"ids": s}, ["ids"]), "fn": read_record},
        {"name": "search_records", "description": "Offline BM25 search over all records (or one `source`). Phone "
                                                  "numbers match in any format and misspelled names still match. "
                                                  "Returns up to k (<=20) 'id | score | summary' lines.",
         "parameters": obj({"query": s, "source": s, "k": i}, ["query"]), "fn": search_records},
        {"name": "scratch_write", "description": "Write `value` (<=4000 chars) under `key` in the shared store seen "
                                                 "by every agent in this run. Overwrites; history is kept.",
         "parameters": obj({"key": s, "value": s}, ["key", "value"]), "fn": scratch_write},
        {"name": "scratch_read", "description": "Read the latest value of a shared-store key and who wrote it.",
         "parameters": obj({"key": s}, ["key"]), "fn": scratch_read},
        {"name": "scratch_list", "description": "List shared-store keys starting with `prefix` whose key or latest "
                                                "value contains `contains` (e.g. a record id), 40 per page.",
         "parameters": obj({"prefix": s, "contains": s, "offset": i}, []), "fn": scratch_list},
    ]


ID_RE = re.compile(r"\b(?:reg|crm|sig|ltr|evt)-\d{5}\b")
KV_RE = re.compile(r"^\s*\**([a-z_]+)\**\s*[=:]\s*(.*?)\s*$", re.I)


def parse_answer(answer):
    clusters = []
    for line in (answer or "").splitlines():
        parts = line.replace("`", "").split("|")
        ids = ID_RE.findall(parts[0].lower())
        # cluster lines start with their record ids; prose that mentions an id must not claim it
        if not ids or len(re.sub(r"[^a-z]", "", ID_RE.sub("", parts[0].lower()))) > 12:
            continue
        prof = {}
        for p in parts[1:]:
            m = KV_RE.match(p)
            if m:
                prof[m.group(1).lower()] = m.group(2)
        clusters.append((ids, prof))
    return clusters


def norm_value(key, v):
    v = str(v).strip().lower()
    if key == "phone":
        d = re.sub(r"\D", "", v)
        return d[-10:]
    if key == "website":
        return re.sub(r"^(https?://)?(www\.)?", "", v).rstrip("/")
    if key in ("name", "employer"):
        v = " ".join(re.sub(r"[^a-z0-9 ]", " ", v.replace("é", "e")).split())
        if key == "employer":  # "Varn Analytics" and "Varn Analytics Inc." name the same employer
            v = re.sub(r" (inc|llc|ltd|corp|gmbh|s a)$", "", v)
        return v
    return re.sub(r"\s+", " ", v)


def score(task, answer):
    with open(task["gold_path"]) as f:
        gold = json.load(f)["entities"]
    owner = {rid: gi for gi, g in enumerate(gold) for rid in g["records"]}
    pred, seen = [], set()
    for ids, prof in parse_answer(answer):
        ids = [x for x in dict.fromkeys(ids) if x in owner and x not in seen]
        seen.update(ids)
        if ids:
            pred.append((ids, prof))
    if not pred:  # all-singleton B-cubed would give a non-answer ~0.08
        return {"score": 0.0, "pairwise_f1": 0.0, "bcubed_f1": 0.0, "attr_acc": 0.0, "n_matched": 0, "n_correct": 0,
                "n_pred_clusters": 0}
    pred += [([rid], {}) for rid in owner if rid not in seen]
    pairs = lambda k: k * (k - 1) // 2
    tp = b3p = b3r = 0.0
    best = {}  # gold entity -> predicted cluster holding a majority of it, if that cluster is mostly it
    for ci, (ids, _) in enumerate(pred):
        c = Counter(owner[x] for x in ids)
        for gi, k in c.items():
            tp += pairs(k)
            g = len(gold[gi]["records"])
            b3p += k * k / len(ids)
            b3r += k * k / g
            if 2 * k > g and 2 * k > len(ids):
                best[gi] = ci
    pp, gp = sum(pairs(len(ids)) for ids, _ in pred), sum(pairs(len(g["records"])) for g in gold)
    prec, rec = (tp / pp if pp else 0.0), (tp / gp if gp else 0.0)
    f1 = lambda p, r: 2 * p * r / (p + r) if p + r else 0.0
    n_rec = len(owner)
    b3p, b3r = b3p / n_rec, b3r / n_rec
    n_attr = n_ok = n_exact = 0
    for gi, g in enumerate(gold):
        prof = pred[best[gi]][1] if gi in best else {}
        ok_all = gi in best and set(pred[best[gi]][0]) == set(g["records"])
        for k, v in g["profile"].items():
            ok = k in prof and norm_value(k, prof[k]) == norm_value(k, v)
            n_attr += 1
            n_ok += ok
            ok_all &= ok
        n_exact += ok_all
    attr = n_ok / n_attr if n_attr else 0.0
    pf1, bf1 = f1(prec, rec), f1(b3p, b3r)
    return {"score": W_PAIR * pf1 + W_B3 * bf1 + W_ATTR * attr, "pairwise_f1": pf1, "pairwise_precision": prec,
            "pairwise_recall": rec, "bcubed_f1": bf1, "bcubed_precision": b3p, "bcubed_recall": b3r,
            "attr_acc": attr, "n_entities": len(gold), "n_matched": len(best), "n_correct": n_exact,
            "n_records": n_rec, "n_pred_clusters": len(pred)}
