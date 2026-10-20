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
import json
import os
import random
import re
import threading

from . import bm25
from .claims import LABELS, sample_claims
from .world import build_world, render_pages

CHUNK = 3000
MAX_OUT = 8000

PROMPT = """You are verifying {n} factual claims against an offline encyclopedia of {pages} pages \
(people, companies, cities, countries, events). The encyclopedia is the only source of truth. Use only its tools:
- search(query, k): BM25 keyword search, returns page ids, titles and snippets.
- open_page(page_id, offset): returns {chunk} characters of a page starting at offset.
- find_in_page(page_id, text): case-insensitive search for text inside one page.

Rules:
- Names in claims are exact page titles; people carry an occupation in parentheses, and companies may be named by \
a former name. Many entities have near-duplicate names (same person name with another occupation, sibling \
companies, other editions of an event, "Port X" next to "X"), so check you are on the right page.
- Facts can live on a different page than the subject (a company's headquarters may only be stated on the city's \
page) and some claims need 2-3 hops (founder -> birthplace -> country).
- SUPPORTED: the encyclopedia confirms the claim. REFUTED: the encyclopedia states a different value. \
NOT_ENOUGH_INFO: a fact the claim depends on is explicitly "not recorded".
- For REFUTED, also give the correct value: a city name, a year, or a country name, whichever the claim's asserted \
value is (for "born in the city that hosted the E" claims, give the city where the founder was actually born).

Answer format: one line per claim, exactly:
C0001 | SUPPORTED
C0002 | REFUTED | <correct value>
C0003 | NOT_ENOUGH_INFO
Every claim id must appear once. If a claim id appears more than once, the last line counts.

Claims:
{table}
"""


# v2 also puts the claim table in the world (world/claims.jsonl, ids and claim text only) behind a list_claims tool:
# harnesses that compact their context dropped the prompt's claim table and could not continue (seen at n=1024).
LIST_LINE = "- list_claims(offset, limit): the claim table again, in pages (useful if your context was compacted).\n"


def generate(n, seed, out_dir, version=1):
    rng = random.Random(seed)
    w = build_world(n, rng.randrange(2**31))
    pages, city_events = render_pages(w, random.Random(rng.randrange(2**31)))
    claims = sample_claims(w, city_events, n, seed)
    world_dir = os.path.join(out_dir, "world")
    os.makedirs(world_dir, exist_ok=True)
    with open(os.path.join(world_dir, "pages.jsonl"), "w") as f:
        for p in pages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    bm25.build((p["title"] + " " + p["title"] + " " + p["text"] for p in pages), os.path.join(world_dir, "index.pkl"))
    gold_path = os.path.join(out_dir, "gold.json")
    with open(gold_path, "w") as f:
        json.dump({"n": n, "seed": seed, "claims": claims}, f, indent=1)
    table = "\n".join(f"{c['id']} | {c['claim']}" for c in claims)
    prompt = PROMPT.format(n=n, pages=len(pages), chunk=CHUNK, table=table)
    task = {"id": f"search-n{n}-s{seed}", "prompt": prompt, "world_dir": world_dir, "n_units": n,
            "gold_path": gold_path}
    if version == 1:
        return task
    with open(os.path.join(world_dir, "claims.jsonl"), "w") as f:
        for c in claims:
            f.write(json.dumps({"id": c["id"], "claim": c["claim"]}, ensure_ascii=False) + "\n")
    marker = "\n\nRules:"
    return task | {"id": f"search-v{version}-n{n}-s{seed}", "version": version,
                   "prompt": prompt.replace(marker, "\n" + LIST_LINE.rstrip("\n") + marker, 1)}


class World:
    def __init__(self, world_dir):
        self.pages, self.by_title = [], {}
        with open(os.path.join(world_dir, "pages.jsonl")) as f:
            for line in f:
                p = json.loads(line)
                self.by_title[p["title"].lower()] = len(self.pages)
                self.pages.append(p)
        self.index = bm25.Index(os.path.join(world_dir, "index.pkl"))

    def page(self, ref):
        ref = str(ref).strip()
        m = re.fullmatch(r"pg(\d{6})", ref)
        if m and int(m.group(1)) < len(self.pages):
            return self.pages[int(m.group(1))]
        i = self.by_title.get(ref.lower())
        return None if i is None else self.pages[i]


_WORLDS, _LOCK = {}, threading.Lock()


def load(world_dir):
    # One read-only copy per process; the lock only guards the first load.
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


def tools(world_dir):
    def search(args):
        w = load(world_dir)
        k = _int(args.get("k"), 10, 1, 20)
        hits = w.index.search(str(args.get("query", ""))[:500], k)
        if not hits:
            return "No results."
        lines = []
        for d, s in hits:
            p = w.pages[d]
            body = p["text"].split("\n\n", 1)[-1].replace("\n", " ")
            lines.append(f"{p['id']} | {p['title']} | score={s:.2f} | {body[:200]}")
        return "\n".join(lines)[:MAX_OUT]

    def open_page(args):
        w = load(world_dir)
        p = w.page(args.get("page_id", ""))
        if p is None:
            return "Error: unknown page_id. Use an id from search results (e.g. pg000123) or an exact title."
        text = p["text"]
        off = _int(args.get("offset"), 0, 0, len(text))
        end = min(len(text), off + CHUNK)
        more = f"\n[more: call open_page with offset={end}]" if end < len(text) else "\n[end of page]"
        return f"[{p['id']} | {p['title']} | chars {off}-{end} of {len(text)}]\n{text[off:end]}{more}"

    def find_in_page(args):
        w = load(world_dir)
        p = w.page(args.get("page_id", ""))
        if p is None:
            return "Error: unknown page_id."
        needle = str(args.get("text", "")).strip().lower()
        if not needle:
            return "Error: empty text."
        text, low, out, i = p["text"], p["text"].lower(), [], 0
        while len(out) < 10:
            i = low.find(needle, i)
            if i < 0:
                break
            out.append(f"@{i}: ...{text[max(0, i - 200):i + len(needle) + 200]}...".replace("\n", " "))
            i += len(needle)
        if not out:
            return f"No match for {needle!r} in {p['id']} ({p['title']})."
        return f"[{p['id']} | {p['title']}] {len(out)} match(es)\n" + "\n".join(out)[:MAX_OUT]

    def list_claims(args):
        with open(os.path.join(world_dir, "claims.jsonl")) as f:
            rows = [json.loads(line) for line in f]
        off, lim = _int(args.get("offset"), 0, 0, len(rows)), _int(args.get("limit"), 100, 1, 200)
        body = "\n".join(f"{c['id']} | {c['claim']}" for c in rows[off:off + lim])
        return f"{len(rows)} claims; showing {off}-{min(off + lim, len(rows)) - 1}\n{body}"[:MAX_OUT]

    obj = lambda props, req: {"type": "object", "properties": props, "required": req}
    extra = [] if not os.path.exists(os.path.join(world_dir, "claims.jsonl")) else [
        {"name": "list_claims", "description": "The claim table: 'id | claim', up to 200 per call from offset.",
         "parameters": obj({"offset": {"type": "integer", "default": 0}, "limit": {"type": "integer", "default": 100}},
                           []),
         "fn": list_claims}]
    return extra + [
        {"name": "search", "description": "BM25 keyword search over the encyclopedia. Returns up to k (<=20) hits "
                                          "as 'page_id | title | score | snippet'.",
         "parameters": obj({"query": {"type": "string"}, "k": {"type": "integer", "default": 10}}, ["query"]),
         "fn": search},
        {"name": "open_page", "description": f"Read {CHUNK} characters of a page (by page_id or exact title) "
                                             "starting at offset.",
         "parameters": obj({"page_id": {"type": "string"}, "offset": {"type": "integer", "default": 0}},
                           ["page_id"]),
         "fn": open_page},
        {"name": "find_in_page", "description": "Case-insensitive substring search inside one page; returns up to "
                                                "10 matches with surrounding context.",
         "parameters": obj({"page_id": {"type": "string"}, "text": {"type": "string"}}, ["page_id", "text"]),
         "fn": find_in_page},
    ]


LABEL_RE = re.compile(r"^[\s|*`>-]*(C\d{4})[\s*`]*[|:,\t]?\s*[*`]*(SUPPORTED|REFUTED|NOT[_ ]ENOUGH[_ ]INFO|NEI)\b"
                      r"[*`]*\s*[|:,\t-]?\s*(.*?)[\s|*`]*$", re.I)


def parse_answer(answer):
    out = {}
    for line in (answer or "").splitlines():
        m = LABEL_RE.match(line)
        if not m:
            continue
        label = m.group(2).upper().replace(" ", "_")
        label = "NOT_ENOUGH_INFO" if label == "NEI" else label
        out[m.group(1).upper()] = (label, (m.group(3) or "").strip())
    return out


def norm(v):
    v = re.sub(r"[^a-z0-9 ]", " ", str(v).lower())
    v = re.sub(r"\s+", " ", v).strip()
    return v[4:] if v.startswith("the ") else v


def score(task, answer):
    with open(task["gold_path"]) as f:
        gold = json.load(f)["claims"]
    pred = parse_answer(answer)
    n = len(gold)
    n_correct = n_label = n_ref = n_val = 0
    tp, fp, fn = ({lab: 0 for lab in LABELS} for _ in range(3))
    by_hop = {}
    for c in gold:
        label, value = pred.get(c["id"], ("MISSING", ""))
        ok_label = label == c["label"]
        ok = ok_label
        if c["label"] == "REFUTED":
            n_ref += 1
            val_ok = ok_label and norm(value) in {norm(a) for a in c["accepted"]}
            n_val += val_ok
            ok = val_ok
        n_label += ok_label
        n_correct += ok
        h = by_hop.setdefault(c["hops"], [0, 0])
        h[0] += ok
        h[1] += 1
        if ok_label:
            tp[label] += 1
        else:
            fn[c["label"]] += 1
            if label in fp:
                fp[label] += 1
    f1s = [2 * tp[lab] / (2 * tp[lab] + fp[lab] + fn[lab]) if tp[lab] + fp[lab] + fn[lab] else 1.0 for lab in LABELS]
    answered = sum(1 for c in gold if c["id"] in pred)
    return {"score": n_correct / n if n else 0.0, "n": n, "n_correct": n_correct, "n_answered": answered,
            "label_acc": n_label / n if n else 0.0, "value_acc": n_val / n_ref if n_ref else 1.0,
            "macro_f1": sum(f1s) / 3, "precision": n_correct / answered if answered else 0.0,
            "recall": n_correct / n if n else 0.0,
            "acc_by_hops": {str(h): v[0] / v[1] for h, v in sorted(by_hop.items())}}
