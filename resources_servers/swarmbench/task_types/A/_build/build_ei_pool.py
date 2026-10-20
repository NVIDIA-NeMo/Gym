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
"""Build the A01 candidate pool from Evidence Inference 2.0 (MIT): articles with >=3 agreed, valid, unflagged prompts
and >=2 distinct labels. usage: python3 -I build_ei_pool.py EI_DIR OUT.jsonl"""
import collections
import csv
import json
import re
import sys

D, OUT = sys.argv[1], sys.argv[2]
csv.field_size_limit(10**9)
FLAGGED = set(map(str, [911, 912, 1262, 1261, 3044, 3248, 3111, 3620, 4308, 4490, 4491, 4324, 4325, 4492, 4824, 5000, 5001,
    5002, 5046, 5047, 4948, 5639, 5710, 5752, 5775, 5782, 5841, 5843, 5861, 5862, 5863, 5964, 5965, 5966, 5975, 4807, 5776,
    5777, 5778, 5779, 5780, 5781, 6034, 6065, 6066, 6666, 6667, 6668, 6669, 7040, 7042, 7944, 8590, 8605, 8606, 8639, 8640,
    8745, 8747, 8749, 8877, 8878, 8593, 8631, 8635, 8884, 8886, 8773, 10032, 10035, 8876, 8875, 8885, 8917, 8921, 8118,
    10885, 10886, 10887, 10888, 10889, 10890, 7811, 7812, 7813, 7814, 7815, 8197, 8198, 8199, 8200, 8201, 9429, 9430,
    9431, 8536, 9432, 3514, 346, 5037, 4715, 8767, 9295, 9297, 8870, 9862]))
LAB = {"significantly increased": "increased", "significantly decreased": "decreased",
       "no significant difference": "no_difference"}
prompts = {r["PromptID"]: r for r in csv.DictReader(open(f"{D}/prompts_merged.csv", encoding="utf-8"))}
ann = collections.defaultdict(list)
for r in csv.DictReader(open(f"{D}/annotations_merged.csv", encoding="utf-8")):
    ann[r["PromptID"]].append(r)


def norm(s):
    return re.sub(r"\W+", " ", s.lower()).strip()


by_art = collections.defaultdict(list)
for pid, rs in ann.items():
    if pid in FLAGGED or pid not in prompts:
        continue
    users = {}
    for r in rs:
        users.setdefault(r["UserID"], r)
    labels = {r["Label"] for r in users.values()}
    if len(users) < 2 or len(labels) != 1 or not all(r["Valid Label"] == "True" for r in users.values()):
        continue
    lab = labels.pop()
    if lab not in LAB:
        continue
    p = prompts[pid]
    ico = tuple(norm(p[k]) for k in ("Intervention", "Comparator", "Outcome"))
    if not all(ico) or any(len(x) > 300 for x in ico):
        continue
    by_art[p["PMCID"]].append({"pid": int(pid), "intervention": p["Intervention"].strip(),
                               "comparator": p["Comparator"].strip(), "outcome": p["Outcome"].strip(),
                               "label": LAB[lab], "in_abstract": any(r["In Abstract"] == "True" for r in rs),
                               "_key": ico})
n = 0
with open(OUT, "w") as f:
    for pmcid in sorted(by_art, key=int):
        seen, ps = set(), []
        for p in sorted(by_art[pmcid], key=lambda x: x["pid"]):
            if p["_key"] in seen or norm(p["outcome"]) in {norm(q["outcome"]) + "|" + q["label"] for q in ps}:
                continue
            seen.add(p.pop("_key"))
            ps.append(p)
        if len(ps) < 3 or len({p["label"] for p in ps}) < 2:
            continue
        try:
            text = open(f"{D}/txt_files/PMC{pmcid}.txt", encoding="utf-8").read()
        except OSError:
            continue
        if not 4000 <= len(text) <= 120000:
            continue
        title = text.split("\n", 1)[0].removeprefix("TITLE:").strip()
        f.write(json.dumps({"pmcid": f"PMC{pmcid}", "title": title, "text": text, "prompts": ps}) + "\n")
        n += 1
print("articles", n)
