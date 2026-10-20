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
"""A09 K3 requests. stage 'clause': K3 writes one company-policy clause per regulation paragraph (task text, K3-only).
stage 'verify': blind K3 mapping of each clause to one of 8 candidate paragraphs (target + 7 nearest by word overlap).
usage: python3 -I reg_prep.py TYPE_DIR clause|verify"""
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

td, stage = Path(sys.argv[1]), sys.argv[2]
b = td / "data" / "build"
units = [json.loads(l) for l in open(b / "units_candidates.jsonl")]
CLAUSE = ("You are drafting an internal compliance policy manual for a US consumer-finance company. Write ONE policy "
          "clause (2-4 sentences, at most 90 words) that implements the regulatory paragraph below, in the company's "
          "own voice (\"the Company\", \"we\", or named internal teams). Keep every specific detail that distinguishes "
          "this paragraph from similar ones (deadlines and day counts, dollar amounts, thresholds, who must act, "
          "triggering conditions, exceptions), because auditors must be able to trace the clause back to exactly this "
          "paragraph. Do not mention any regulation name, act, part, section or paragraph number, and do not copy more "
          "than six consecutive words from the paragraph. Reply with one JSON object {\"clause\": \"...\"} and nothing else.")
WORD = re.compile(r"[a-z]{3,}")
CLAUSE2 = ("You are drafting the operating procedures of a US consumer-finance company for front-line staff who have "
           "never read the regulations. Write ONE procedure clause (2-4 sentences, at most 90 words) that makes the "
           "company do exactly what the regulatory paragraph below requires, but describe it as concrete everyday "
           "work: who on our side does what, for which customers or accounts, when, and what they send or record. "
           "Use plain business vocabulary instead of the regulation's terms: do not reuse its defined terms, its "
           "distinctive nouns and verbs, or its phrasing (for example write customer, home loan, card, mistake, "
           "letter, tell, keep, look into, rather than consumer, mortgage loan, access device, error, notice, notify, "
           "retain, investigate). Keep every number exactly (days, dollar amounts, percentages, counts), because "
           "auditors must be able to trace the clause back to exactly this paragraph and not to similar ones. Do not "
           "mention any regulation, act, agency, part, section or paragraph. Reply with one JSON object "
           "{\"clause\": \"...\"} and nothing else.")
if stage == "clause2":
    with open(b / "clause2_req.jsonl", "w") as f:
        for u in units:
            f.write(json.dumps({"id": u["cite"], "max_tokens": 8000, "messages": [{"role": "user", "content":
                f"{CLAUSE2}\n\nSECTION HEADING: {u['heading']}\n\nPARAGRAPH:\n{u['text']}"}]}) + "\n")
elif stage == "clause":
    with open(b / "clause_req.jsonl", "w") as f:
        for u in units:
            f.write(json.dumps({"id": u["cite"], "max_tokens": 8000, "messages": [{"role": "user", "content":
                f"{CLAUSE}\n\nSECTION HEADING: {u['heading']}\n\nPARAGRAPH:\n{u['text']}"}]}) + "\n")
else:
    clauses = {}
    for l in open(b / "clause2_k3_out.jsonl"):
        r = json.loads(l)
        m = re.findall(r"\{.*\}", r["content"], re.S)
        try:
            clauses[r["id"]] = json.loads(m[-1])["clause"].strip()
        except (ValueError, KeyError, IndexError):
            pass
    docs = [Counter(WORD.findall(u["text"].lower())) for u in units]
    df = Counter(w for d in docs for w in d)
    with open(b / "verify_req.jsonl", "w") as f:
        for i, u in enumerate(units):
            c = clauses.get(u["cite"])
            if not c:
                continue
            # hard negatives: sibling paragraphs of the same section, then the paragraphs most similar to the SOURCE
            # paragraph (plain-language clauses share few words with any paragraph)
            q = set(WORD.findall(u["text"].lower()))
            sib = [j for j, v in enumerate(units) if v["section"] == u["section"] and j != i][:4]
            sc = sorted(range(len(units)), key=lambda j: -sum(math.log(len(units) / df[w]) for w in q if docs[j][w]))
            cand = (sib + [j for j in sc if j != i and j not in sib])[:7] + [i]
            random.Random(u["cite"]).shuffle(cand)
            body = "\n\n".join(f"[{units[j]['cite']}] {units[j]['heading']}\n{units[j]['text'][:2500]}" for j in cand)
            f.write(json.dumps({"id": u["cite"], "max_tokens": 8000, "messages": [{"role": "user", "content":
                "Which ONE of the regulatory paragraphs below does this company policy clause implement? Reply with one "
                "JSON object {\"cite\": \"<the bracketed citation>\"} and nothing else.\n\nCLAUSE: " + c +
                "\n\nPARAGRAPHS:\n" + body}]}) + "\n")
print(stage, "ok")
