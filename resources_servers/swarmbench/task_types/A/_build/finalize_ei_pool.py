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
"""A01 pool = candidates filtered by the K3 full-text labels: keep a prompt only if K3 agrees with the annotators'
label; keep an article only if >=3 prompts with >=2 distinct labels remain. Also writes data/build/filter_stats.json
and a disagreement sample for the hand audit. usage: python3 -I finalize_ei_pool.py TYPE_DIR [--dev]"""
import json
import random
import re
import sys
from pathlib import Path

td = Path(sys.argv[1])
b = td / "data" / "build"
cands = [json.loads(x) for x in open(b / "candidates.jsonl")]
if "--dev" in sys.argv:
    (td / "data" / "pool.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cands[:300]))
    sys.exit("dev pool written (unfiltered, never ships)")
k3 = {}
for x in open(b / "ei_k3_out.jsonl"):
    r = json.loads(x)
    m = re.findall(r"\{[^{}]*\}", r["content"])
    try:
        k3[r["id"]] = {k.lstrip("Q"): v for k, v in json.loads(m[-1]).items()} if m else {}
    except ValueError:
        k3[r["id"]] = {}
st = {"articles_labeled": 0, "prompts_labeled": 0, "prompts_agree": 0, "articles_kept": 0, "prompts_kept": 0}
dis, out = [], []
for c in cands:
    if c["pmcid"] not in k3:
        continue
    st["articles_labeled"] += 1
    keep = []
    for p in c["prompts"]:
        st["prompts_labeled"] += 1
        if k3[c["pmcid"]].get(str(p["pid"])) == p["label"]:
            st["prompts_agree"] += 1
            keep.append(p)
        else:
            dis.append({"pmcid": c["pmcid"], **p, "k3": k3[c["pmcid"]].get(str(p["pid"]))})
    if len(keep) >= 3 and len({p["label"] for p in keep}) >= 2:
        out.append({**c, "prompts": keep})
        st["articles_kept"] += 1
        st["prompts_kept"] += len(keep)
(td / "data" / "pool.jsonl").write_text("".join(json.dumps(c) + "\n" for c in out))
random.Random(0).shuffle(dis)
(b / "filter_stats.json").write_text(json.dumps({**st, "disagreements": len(dis), "sample": dis[:40]}, indent=1))
print(json.dumps(st))
