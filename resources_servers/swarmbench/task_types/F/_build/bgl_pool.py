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
"""Build the 52_loghub window pool from LogHub BGL (CC-BY-4.0): non-overlapping blocks of 50 consecutive lines.
Keeps blocks whose labels agree line-for-line with bgl_rules; samples mixed anomalous blocks and decoy-rich normal
blocks. usage: python3 bgl_pool.py BGL.log TYPE_DIR OUT.jsonl.gz"""
import collections, gzip, json, random, sys
sys.path.insert(0, sys.argv[2])
from bgl_rules import categorize, split

W = 50
anom, normal = [], []
block = []
def flush(i):
    labs = [l for l, _ in block]
    raws = [r for _, r in block]
    cats = [categorize(r) for r in raws]
    if any((c or "-") != l for c, l in zip(cats, labs)):
        return
    alert = [c for c in cats if c]
    decoys = sum(1 for r, c in zip(raws, cats) if not c and split(r)[1] not in ("INFO",))
    rec = {"start": i, "lines": raws, "cats": sorted(set(alert)), "n_alert": len(alert), "n_decoy": decoys}
    (anom if alert else normal).append(rec)
for i, line in enumerate(open(sys.argv[1], errors="replace")):
    lab, rest = line.rstrip("\n").split(" ", 1)
    block.append((lab, rest))
    if len(block) == W:
        flush(i - W + 1); block = []
rng = random.Random(52)
# anomalous: favour blocks that are mostly normal lines, and keep every rare category
mixed = [r for r in anom if r["n_alert"] <= W // 2]
by_cat = collections.defaultdict(list)
for r in anom:
    for c in r["cats"]:
        by_cat[c].append(r)
pick = {r["start"]: r for r in rng.sample(mixed, min(1500, len(mixed)))}
for c, rs in by_cat.items():
    for r in rng.sample(rs, min(60, len(rs))):
        pick[r["start"]] = r
full = [r for r in anom if r["n_alert"] > W // 2]
for r in rng.sample(full, min(300, len(full))):
    pick[r["start"]] = r
rich = [r for r in normal if r["n_decoy"] >= 5]
plain = [r for r in normal if r["n_decoy"] < 5]
npick = rng.sample(rich, min(3500, len(rich))) + rng.sample(plain, 1000)
pool = sorted(list(pick.values()) + npick, key=lambda r: r["start"])
with gzip.open(sys.argv[3], "wt") as f:
    for r in pool:
        f.write(json.dumps(r) + "\n")
cc = collections.Counter(c for r in pick.values() for c in r["cats"])
print("blocks anom", len(anom), "normal", len(normal), "pool", len(pool), "anom picked", len(pick), "rich", len(rich))
print(cc.most_common())
