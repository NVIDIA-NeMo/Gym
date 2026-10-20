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
"""A06 pool from the K3 sentences and blind K3 verdicts: faithful sentence kept if judged supported by its own
abstract; distorted kept if judged unsupported; wrong_cite = nearest neighbour if the faithful sentence is judged
unsupported by that neighbour's abstract. usage: python3 -I finalize_cite.py TYPE_DIR"""
import json
import re
import sys
from pathlib import Path

td = Path(sys.argv[1])
b = td / "data" / "build"
refs = {json.loads(l)["arxiv"]: json.loads(l) for l in open(td / "data" / "refs.jsonl")}
sents = json.loads((b / "sentences.json").read_text())
ver = {}
for l in open(b / "verify_k3_out.jsonl"):
    r = json.loads(l)
    m = re.findall(r"\{[^{}]*\}", r["content"])
    try:
        ver[r["id"]] = json.loads(m[-1])["verdict"].strip().lower()
    except (ValueError, KeyError, IndexError):
        pass
st = {"faithful": 0, "faithful_ok": 0, "distorted": 0, "distorted_ok": 0, "nn_ok": 0}
pool = []
for a in sorted(refs):
    f, d = sents.get(f"{a}|faithful"), sents.get(f"{a}|distorted")
    st["faithful"] += bool(f)
    st["distorted"] += bool(d)
    fok = f and ver.get(f"{a}|faithful|self") == "supported"
    dok = d and ver.get(f"{a}|distorted|self") == "unsupported"
    nok = fok and ver.get(f"{a}|faithful|nn0") == "unsupported"
    st["faithful_ok"] += bool(fok)
    st["distorted_ok"] += bool(dok)
    st["nn_ok"] += bool(nok)
    if fok:
        pool.append({"arxiv": a, "faithful": f, "distorted": d if dok else None,
                     "wrong_cite": refs[a]["neighbours"][0] if nok else None})
(td / "data" / "pool.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pool))
st["pool"] = len(pool)
(b / "filter_stats.json").write_text(json.dumps(st, indent=1))
print(st)
