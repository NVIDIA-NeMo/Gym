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
"""A09 pool = units whose plain-language K3 clause a blind K3 pass mapped back to the source paragraph out of 8
candidates. usage: python3 -I finalize_reg.py TYPE_DIR"""
import json
import re
import sys
from pathlib import Path

td = Path(sys.argv[1])
b = td / "data" / "build"
units = {json.loads(l)["cite"]: json.loads(l) for l in open(b / "units_candidates.jsonl")}
clauses = {}
for l in open(b / "clause2_k3_out.jsonl"):
    r = json.loads(l)
    m = re.findall(r"\{.*\}", r["content"], re.S)
    try:
        clauses[r["id"]] = json.loads(m[-1])["clause"].strip()
    except (ValueError, KeyError, IndexError):
        pass
st = {"units": len(units), "clauses": len(clauses), "verified": 0, "kept": 0}
keep = []
for l in open(b / "verify_k3_out.jsonl"):
    r = json.loads(l)
    st["verified"] += 1
    m = re.findall(r"\{[^{}]*\}", r["content"])
    try:
        pick = json.loads(m[-1])["cite"]
    except (ValueError, KeyError, IndexError):
        continue
    if re.sub(r"[\s\[\]]+", "", pick) == re.sub(r"\s+", "", r["id"]) and r["id"] in clauses:
        keep.append({**units[r["id"]], "clause": clauses[r["id"]]})
# adversarial filter against the lexical shortcut: drop clauses whose source paragraph is among the first TOPK
# distinct first-level paragraphs that BM25 (clause text as the query, the agent's own search tool) returns
# (top-1 only left a top-2 baseline at 0.2-0.3)
TOPK = 3
sys.path.insert(0, str(td.parent))
import loader  # noqa: E402
M = loader.load_type(td.name)
search = {t["name"]: t["fn"] for t in M.tools("/tmp")}["search"]
st["bm25_top1_hits_dropped"] = 0
kept = []
for u in keep:
    hits = [M.norm_cite(h.split("\t")[0]) for h in search({"query": u["clause"], "limit": 15}).splitlines() if "\t" in h]
    distinct = []
    for h in hits:
        if h and h not in distinct:
            distinct.append(h)
    if M.norm_cite(u["cite"]) in distinct[:TOPK]:
        st["bm25_top1_hits_dropped"] += 1
        continue
    kept.append(u)
keep = sorted(kept, key=lambda u: u["cite"])
# v3: drop units with a near-twin paragraph (tf-idf cosine >= 0.8 to another candidate paragraph, e.g. 1016.7(d)/(f),
# 1022.136(f)/1022.137(d), 1022.31(a)/1022.32(a)): a plain-language clause cannot tell the twins apart (audit of v2)
import math
from collections import Counter
W = re.compile(r"[a-z]{3,}")
allu = list(units.values())
docs = [Counter(W.findall(u["text"].lower())) for u in allu]
df = Counter(w for d in docs for w in d)


def vec(d):
    v = {w: c * math.log(len(docs) / df[w]) for w, c in d.items()}
    n = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {w: x / n for w, x in v.items()}


V = {u["cite"]: vec(d) for u, d in zip(allu, docs)}
twin = {}
for c in [u["cite"] for u in keep]:
    twin[c] = max(sum(V[c].get(w, 0) * x for w, x in V[o].items()) for o in V if o != c)
st["twin_dropped"] = sum(twin[u["cite"]] >= 0.8 for u in keep)
keep = [u for u in keep if twin[u["cite"]] < 0.8]
st["kept"] = len(keep)
st["by_part"] = {}
for u in keep:
    st["by_part"][u["part"]] = st["by_part"].get(u["part"], 0) + 1
(td / "data" / "pool.jsonl").write_text("".join(json.dumps(u) + "\n" for u in keep))
(b / "filter_stats.json").write_text(json.dumps(st, indent=1))
print(st)
