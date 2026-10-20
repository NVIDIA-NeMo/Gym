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
"""A06 data: arXiv cs.CL abstracts (arXiv metadata is CC0) -> data/refs.jsonl, nearest neighbours by tf-idf, and K3
request files. stage 'refs': parse OAI XML; 'gen': K3 writes one faithful and one distorted citing sentence per ref;
'verify': blind K3 judgement of each sentence against an abstract. usage: python3 -I cite_prep.py TYPE_DIR STAGE [XML_DIR]"""
import glob
import html
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

td, stage = Path(sys.argv[1]), sys.argv[2]
b = td / "data" / "build"
WORD = re.compile(r"[a-z][a-z0-9-]{2,}")
if stage == "refs":
    seen, refs = set(), []
    for f in sorted(glob.glob(sys.argv[3] + "/*.xml")):
        s = open(f, encoding="utf-8").read()
        for rec in re.findall(r"<arXiv .*?</arXiv>", s, re.S):
            g = lambda t: (re.search(rf"<{t}>(.*?)</{t}>", rec, re.S) or [None, ""])[1]
            aid, cats = g("id"), g("categories").split()
            if not cats or cats[0] != "cs.CL" or aid in seen or not aid.startswith("24"):
                continue
            title = re.sub(r"\s+", " ", html.unescape(g("title"))).strip()
            ab = re.sub(r"\s+", " ", html.unescape(g("abstract"))).strip()
            if not 600 <= len(ab) <= 2200:
                continue
            seen.add(aid)
            refs.append({"arxiv": aid, "title": title, "abstract": ab, "license": g("license") or "arXiv default"})
    refs.sort(key=lambda r: r["arxiv"])
    docs = [Counter(WORD.findall((r["title"] + " " + r["abstract"]).lower())) for r in refs]
    df = Counter(w for d in docs for w in d)
    vec = []
    for d in docs:
        v = {w: c * math.log(len(docs) / df[w]) for w, c in d.items()}
        nrm = math.sqrt(sum(x * x for x in v.values()))
        vec.append({w: x / nrm for w, x in v.items()})
    for i, r in enumerate(refs):
        sims = sorted(((sum(vec[i].get(w, 0) * x for w, x in vec[j].items()), j) for j in range(len(refs)) if j != i),
                      reverse=True)[:8]
        r["neighbours"] = [refs[j]["arxiv"] for _, j in sims]
    (td / "data" / "refs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in refs))
    print("refs", len(refs))
elif stage == "gen":
    refs = [json.loads(l) for l in open(td / "data" / "refs.jsonl")]
    FAITH = ("You are writing the related-work section of a survey on natural language processing. Write ONE sentence "
             "(20-45 words) that cites the paper below as [CITE] and states one specific, checkable finding, method "
             "detail or result from its abstract (for example a number, a dataset, a comparison or a named method), "
             "paraphrased in your own words and fully accurate. Do not use the paper's title. Reply with one JSON "
             "object {\"sentence\": \"...\"} and nothing else.")
    DIST = ("You are building a citation-checking exercise. Write ONE sentence (20-45 words) in the style of a survey's "
            "related-work section that cites the paper below as [CITE] and attributes to it a specific claim that its "
            "abstract CONTRADICTS: take one concrete detail from the abstract (a number, the direction of a result, the "
            "dataset or language, the task, or who or what was compared) and change it so the sentence is clearly "
            "false for this paper, while staying plausible for the topic. Do not use the paper's title. Reply with one "
            "JSON object {\"sentence\": \"...\", \"changed\": \"<what you changed, briefly>\"} and nothing else.")
    DIST2 = ("You are building a citation-checking exercise. Write ONE sentence (20-45 words) in the style of a survey's "
             "related-work section that cites the paper below as [CITE] and attributes to it a claim that its abstract "
             "CONTRADICTS. Build the false claim only from words, names and numbers that appear in the abstract, so it "
             "looks lexically just like a faithful summary: for example swap which method or setting did better, "
             "attach a reported number to the wrong metric, model or dataset, swap two named datasets, languages or "
             "baselines, reverse a reported effect, or turn a stated limitation into a strength. The sentence must be "
             "clearly false for this paper to a careful reader of the abstract. Do not use the paper's title. Reply "
             "with one JSON object {\"sentence\": \"...\", \"changed\": \"<what you changed, briefly>\"} and nothing else.")
    with open(b / "dist2_req.jsonl", "w") as f:
        for r in refs[:700]:
            f.write(json.dumps({"id": f"{r['arxiv']}|distorted", "max_tokens": 8000, "messages": [{"role": "user",
                    "content": f"{DIST2}\n\nTITLE: {r['title']}\nABSTRACT: {r['abstract']}"}]}) + "\n")
    with open(b / "gen_req.jsonl", "w") as f:
        for r in refs:
            for kind, ins in (("faithful", FAITH), ("distorted", DIST)):
                f.write(json.dumps({"id": f"{r['arxiv']}|{kind}", "max_tokens": 8000, "messages": [{"role": "user",
                        "content": f"{ins}\n\nTITLE: {r['title']}\nABSTRACT: {r['abstract']}"}]}) + "\n")
    print("gen requests", 2 * len(refs))
elif stage == "verify":
    refs = {json.loads(l)["arxiv"]: json.loads(l) for l in open(td / "data" / "refs.jsonl")}
    sents = {}
    # faithful sentences from gen_k3_out (v1 prompt); distorted ones from dist2_k3_out (lexically matched v2 prompt;
    # v1 distortions introduced words absent from the abstract, which a word-overlap heuristic caught)
    rows = [json.loads(l) for l in open(b / "gen_k3_out.jsonl")]
    rows = [r for r in rows if r["id"].endswith("|faithful")] + [json.loads(l) for l in open(b / "dist2_k3_out.jsonl")]
    for r in rows:
        m = re.findall(r"\{.*\}", r["content"], re.S)
        try:
            s = json.loads(m[-1])["sentence"].strip()
        except (ValueError, KeyError, IndexError):
            continue
        if "[CITE]" in s and 60 <= len(s) <= 400:
            sents[r["id"]] = s
    JUDGE = ("Does the abstract below support the sentence's statement about the cited paper [CITE]? Answer "
             "\"supported\" only if the abstract states it (paraphrase is fine); answer \"unsupported\" if the abstract "
             "contradicts it or does not state it. Reply with one JSON object {\"verdict\": \"supported\" or "
             "\"unsupported\"} and nothing else.")
    first300 = set(refs)  # (v1 verified neighbours for the first 300 papers only; v2: all)
    with open(b / "verify_req.jsonl", "w") as f:
        for sid, s in sents.items():
            aid, kind = sid.split("|")
            pairs = [(aid, "self")]
            if kind == "faithful" and aid in first300:  # judged against the nearest neighbour (wrong-source target)
                pairs.append((refs[aid]["neighbours"][0], "nn0"))
            for cited, tag in pairs:
                c = refs[cited]
                f.write(json.dumps({"id": f"{sid}|{tag}", "max_tokens": 6000, "messages": [{"role": "user", "content":
                        f"{JUDGE}\n\nSENTENCE: {s}\n\nCITED PAPER TITLE: {c['title']}\nABSTRACT: {c['abstract']}"}]})
                        + "\n")
    (b / "sentences.json").write_text(json.dumps(sents, indent=0))
    print("verify requests written for", len(sents), "sentences")
