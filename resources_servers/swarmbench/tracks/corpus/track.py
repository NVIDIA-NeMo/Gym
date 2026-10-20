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
import datetime as dt
import functools
import json
import os
import random
import re

from .vendor import swarm_gen as sg

FAMS = ["refund", "billing", "expense", "incident", "meeting"]
PREFIX = {"refund": "T", "billing": "BIL", "expense": "EXP", "incident": "INC", "meeting": "TH"}
TYPE_NAME = {"refund": "support ticket", "billing": "billing case", "expense": "expense report",
             "incident": "incident report", "meeting": "email thread"}
ID_RE = re.compile(r"\b(?:T|BIL|EXP|INC|TH)-\d{5}\b")
MAX_OUT = 8000
EXCL_RATE, DECOY_RATE = 0.08, 0.05

# Exclusion notes remove {old} from every tally; decoy notes mention {old} but change nothing.
EXCLUDE = [
    "Records note: this record supersedes {old}, which covered the same matter. Count only this record; {old} is void.",
    "Records note: {old} was opened for the same matter and has been merged into this record. Treat {old} as closed and leave it out.",
    "Correction from Records: {old} was filed in error as a duplicate entry and must be excluded from all tallies.",
    "Records note: {old} is replaced by this record. Disregard {old} entirely, whatever it says.",
    "Audit note: after the data migration the case in {old} was re-recorded here, so {old} no longer counts on its own.",
    "Records note: please void {old}; it was a test entry created by mistake during the migration.",
]
DECOY = [
    "Records note: see also {old} for background. That record stands on its own and still counts.",
    "Records note: {old} involved the same customer but a different matter, so both records count separately.",
    "Records note: a request to void {old} was raised and then rejected after review, so {old} still counts.",
    "Records note: {old} was first flagged as a duplicate of this record, but the flag was withdrawn; {old} remains valid.",
    "Records note: this record is not a duplicate of {old}, despite the similar subject line. Both stand.",
]

PROMPT = """You are given a corpus of {n} business records (about {tok:,} tokens in total, far too much to read in one context). \
Records come in five types: support tickets (IDs T-xxxxx), billing cases (BIL-xxxxx), expense reports (EXP-xxxxx), \
incident reports (INC-xxxxx) and email threads (TH-xxxxx). Use the tools list_docs(offset, limit), read_doc(doc_id) and \
grep(pattern, max_hits) to access them.

A record QUALIFIES if it answers yes to the question for its type:
- refund (support tickets): {q_refund}
- billing (billing cases): {q_billing}
- expense (expense reports): {q_expense}
- incident (incident reports): {q_incident}
- meeting (email threads): {q_meeting}

Outcomes are often stated indirectly, change during the record, or are only mentioned hypothetically, so keyword search \
will not give the right answer: each record has to be read carefully.

Cross-record rule: some records start with a "Records note" saying that an EARLIER record (named by ID) is superseded, \
merged, replaced, voided or filed in error. Such an earlier record is EXCLUDED: it never qualifies, whatever its own content \
says. Notes that only mention another record, or say a void/duplicate flag was rejected or withdrawn, change nothing. The \
record carrying a note is judged on its own content as usual.

Each record header has an "Amount at issue" line. Use exactly that number for totals and ranking (ignore amounts in the body).

Final answer: after the line FINAL ANSWER, give exactly one row per type in this pipe-separated format (header row optional):
FINAL ANSWER
type | count | total | top3 | ids
refund | <number of qualifying tickets> | <sum of their amounts, 2 decimals> | <IDs of the 3 qualifying with the largest amounts> | <all qualifying IDs, comma-separated>
billing | ...
expense | ...
incident | ...
meeting | ...
Use - for an empty list and 0.00 for an empty total. Example row: refund | 2 | 1530.25 | T-10001, T-20002 | T-10001, T-20002"""


def _rng(n, seed):
    return random.Random(f"swarmbench-corpus:{seed}:{n}")


def _build(n, seed):
    rng = _rng(n, seed)
    weights = [rng.uniform(0.6, 1.4) for _ in FAMS]
    pos_rate = {f: rng.uniform(0.15, 0.35) for f in FAMS}
    fams = FAMS[: min(n, 5)] + rng.choices(FAMS, weights, k=max(0, n - 5))
    rng.shuffle(fams)
    ids, used = [], set()
    for f in fams:
        while True:
            i = f"{PREFIX[f]}-{rng.randrange(10000, 100000)}"
            if i not in used:
                used.add(i)
                ids.append(i)
                break
    cents = rng.sample(range(5000, 2000000), n)  # distinct amounts so top-3 has no ties
    labels = [rng.random() < pos_rate[f] for f in fams]

    # Exclusion targets lean positive so ignoring the notes costs precision.
    order = list(range(n))
    rng.shuffle(order)
    n_ex = max(1, round(EXCL_RATE * n))
    pos, neg = [k for k in order if labels[k]], [k for k in order if not labels[k]]
    targets = [(pos if pos and (rng.random() < 0.6 or not neg) else neg).pop() for _ in range(n_ex)]
    rest = [k for k in order if k not in set(targets)]
    referrers = rest[:n_ex]
    n_dec = max(1, round(DECOY_RATE * n))
    decoy_refs = rest[n_ex:n_ex + n_dec]
    decoy_tgts = rest[n_ex + n_dec:n_ex + 2 * n_dec]

    notes = {}
    base_date = dt.datetime(2026, rng.randint(1, 8), 1)
    # Filed after the body's events, which start up to 60 days past base_date.
    filed = [base_date.date() + dt.timedelta(days=70 + rng.randrange(90)) for _ in range(n)]
    for ref, tgt in zip(referrers, targets):
        notes[ref] = ("exclude", tgt, sg.pick(rng, EXCLUDE))
    for ref, tgt in zip(decoy_refs, decoy_tgts):
        notes[ref] = ("decoy", tgt, sg.pick(rng, DECOY))
    for ref, (_, tgt, _) in notes.items():
        filed[ref] = filed[tgt] + dt.timedelta(days=rng.randint(1, 30))

    docs = []
    for k in range(n):
        d = sg.FAMILIES[fams[k]]["gen"](rng, ids[k], labels[k], base_date)
        head = [f"Record: {ids[k]}", f"Type: {TYPE_NAME[fams[k]]}", f"Filed: {filed[k].isoformat()}",
                f"Amount at issue: ${cents[k] / 100:,.2f}"]
        if k in notes:
            head.append(notes[k][2].format(old=ids[notes[k][1]]))
        docs.append(dict(id=ids[k], family=fams[k], amount=cents[k], label=labels[k], trace=d.trace,
                         text="\n".join(head) + "\n---\n" + d.text))
    excluded = {ids[t] for t in targets}
    return docs, excluded, [(ids[r], ids[t]) for r, t in zip(referrers, targets)]


def aggregate(rows):
    # rows: (id, family, amount_cents) of qualifying records -> per-family table.
    table = {}
    for f in FAMS:
        mine = sorted([r for r in rows if r[1] == f], key=lambda r: -r[2])
        table[f] = dict(count=len(mine), total=sum(r[2] for r in mine), top3=[r[0] for r in mine[:3]],
                        ids=sorted(r[0] for r in mine))
    return table


def generate(n, seed, out_dir):
    docs, excluded, links = _build(n, seed)
    world = os.path.join(out_dir, f"corpus-n{n}-s{seed}")
    os.makedirs(world, exist_ok=True)
    with open(os.path.join(world, "corpus.jsonl"), "w") as f:
        for d in docs:
            f.write(json.dumps(dict(id=d["id"], family=d["family"], text=d["text"])) + "\n")
    qual = [(d["id"], d["family"], d["amount"]) for d in docs if d["label"] and d["id"] not in excluded]
    tok = sum(len(d["text"]) for d in docs) // 4
    gold = dict(table=aggregate(qual), excluded=sorted(excluded), links=links,
                labels={d["id"]: d["label"] for d in docs}, traces={d["id"]: d["trace"] for d in docs},
                amounts={d["id"]: d["amount"] for d in docs}, corpus_tokens_est=tok)
    gold_path = os.path.join(world, "gold.json")
    with open(gold_path, "w") as f:
        json.dump(gold, f)
    qs = {f"q_{f}": sg.FAMILIES[f]["question"] for f in FAMS}
    return dict(id=f"corpus-n{n}-s{seed}", prompt=PROMPT.format(n=n, tok=tok, **qs), world_dir=world, n_units=n,
                gold_path=gold_path, corpus_tokens_est=tok)


@functools.lru_cache(maxsize=16)
def _load(world_dir):
    with open(os.path.join(world_dir, "corpus.jsonl")) as f:
        docs = [json.loads(line) for line in f]
    lines = [(d["id"], line) for d in docs for line in d["text"].split("\n")]
    return docs, {d["id"]: d for d in docs}, lines


def _cap(s):
    return s if len(s) <= MAX_OUT else s[:MAX_OUT] + "\n[truncated]"


def tools(world_dir):
    def list_docs(args):
        docs = _load(world_dir)[0]
        off, lim = max(0, int(args.get("offset", 0))), max(1, min(200, int(args.get("limit", 100))))
        rows = [f"{d['id']}\t{d['family']}\t~{len(d['text']) // 4} tok" for d in docs[off:off + lim]]
        return _cap(f"{len(docs)} records total; showing {off}..{off + len(rows) - 1}\n" + "\n".join(rows))

    def read_doc(args):
        d = _load(world_dir)[1].get(str(args.get("doc_id", "")).strip().upper())
        return _cap(d["text"]) if d else f"no record with id {args.get('doc_id')!r}"

    def grep(args):
        pat = str(args.get("pattern", ""))[:200]
        try:
            rx = re.compile(pat, re.I)
        except re.error:
            rx = re.compile(re.escape(pat), re.I)
        lim = max(1, min(200, int(args.get("max_hits", 50))))
        hits = []
        for i, line in _load(world_dir)[2]:
            if rx.search(line):
                hits.append(f"{i}: {line[:300]}")
                if len(hits) >= lim:
                    return _cap("\n".join(hits) + f"\n[stopped at max_hits={lim}]")
        return _cap("\n".join(hits) or "no matches")

    return [
        dict(name="list_docs", description="List record IDs with their type and approximate token length.",
             parameters={"type": "object", "properties": {"offset": {"type": "integer", "default": 0},
                                                          "limit": {"type": "integer", "default": 100, "maximum": 200}}},
             fn=list_docs),
        dict(name="read_doc", description="Return the full text of one record by ID (e.g. T-12345).",
             parameters={"type": "object", "properties": {"doc_id": {"type": "string"}}, "required": ["doc_id"]},
             fn=read_doc),
        dict(name="grep", description="Case-insensitive regex search over all records; returns 'ID: line' hits.",
             parameters={"type": "object", "properties": {"pattern": {"type": "string"},
                                                          "max_hits": {"type": "integer", "default": 50, "maximum": 200}},
                         "required": ["pattern"]},
             fn=grep),
    ]


def _num(s):
    m = re.search(r"-?\d[\d,]*(?:\.\d+)?", s.replace(" ", ""))
    return float(m.group().replace(",", "")) if m else None


_ROW_RE = re.compile(r"^[\s|*`_]*(" + "|".join(FAMS) + r")s?[\s*`_]*\|", re.I | re.M)


def parse_answer(answer):
    # Last row per family after the last FINAL ANSWER marker; markdown and $ signs are tolerated.
    text = answer or ""
    # the last marker that has a family row after it ("that is my final answer" in closing prose does not count)
    for m in reversed(list(re.finditer("FINAL ANSWER", text.upper()))):
        if _ROW_RE.search(text[m.start():]):
            text = text[m.start():]
            break
    out = {}
    for line in text.splitlines():
        cells = [c.strip(" *`_\t") for c in line.strip().strip("|").split("|")]
        if len(cells) < 5:
            continue
        name = cells[0].lower().rstrip("s")
        if name not in FAMS:
            continue
        count, total = _num(cells[1]), _num(cells[2].replace("$", ""))
        out[name] = dict(count=None if count is None else int(round(count)),
                         total=None if total is None else round(total * 100),
                         top3=ID_RE.findall(cells[3]), ids=sorted(set(ID_RE.findall(cells[4]))))
    return out


def _f1(pred, gold):
    pred, gold = set(pred), set(gold)
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    return (2 * p * r / (p + r) if tp else 0.0), p, r


def _credit(pred, gold):
    # Full credit when exact, linear down to zero at 10% relative error (at least 1 unit).
    if pred is None:
        return 0.0
    return max(0.0, 1 - abs(pred - gold) / max(1.0, 0.1 * abs(gold)))


def score(task, answer):
    with open(task["gold_path"]) as f:
        gold = json.load(f)["table"]
    pred = parse_answer(answer)
    all_p = [i for f in FAMS for i in pred.get(f, {}).get("ids", [])]
    all_g = [i for f in FAMS for i in gold[f]["ids"]]
    id_f1, id_p, id_r = _f1(all_p, all_g)
    n_cells = n_correct = 0
    agg, top = [], []
    for f in FAMS:
        g, p = gold[f], pred.get(f, {})
        cnt, tot = _credit(p.get("count"), g["count"]), _credit(p.get("total"), g["total"])
        agg.append((cnt + tot) / 2)
        top.append(_f1(p.get("top3", []), g["top3"])[0])
        cells = [p.get("count") == g["count"], p.get("total") is not None and abs(p["total"] - g["total"]) <= 1,
                 set(p.get("top3", [])) == set(g["top3"]), set(p.get("ids", [])) == set(g["ids"])]
        n_cells += len(cells)
        n_correct += sum(cells)
    agg_credit, top3_f1 = sum(agg) / len(FAMS), sum(top) / len(FAMS)
    return dict(score=0.5 * id_f1 + 0.25 * agg_credit + 0.25 * top3_f1, id_f1=id_f1, precision=id_p, recall=id_r,
                agg_credit=agg_credit, top3_f1=top3_f1, n_correct=n_correct, n_cells=n_cells,
                cell_accuracy=n_correct / n_cells, exact=n_correct == n_cells, parsed_families=sorted(pred))


def oracle_answer(gold_table):
    rows = ["FINAL ANSWER", "type | count | total | top3 | ids"]
    for f in FAMS:
        g = gold_table[f]
        rows.append(f"{f} | {g['count']} | {g['total'] / 100:.2f} | {', '.join(g['top3']) or '-'} | {', '.join(g['ids']) or '-'}")
    return "\n".join(rows)
