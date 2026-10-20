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
"""Check 4 (calibration part): re-grade every calibration answer with the current code and compare to the recorded
score. usage: python3 _build/regrade.py ROWS.jsonl[,..] ANSWERS_DIR"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import loader  # noqa: E402

cache, n, diff, missing = {}, 0, [], 0
for f in sys.argv[1].split(","):
    for line in open(f):
        r = json.loads(line)
        a = Path(sys.argv[2]) / f"{r['rid']}.answer.txt"
        if not a.exists():
            missing += 1
            continue
        k = (r["track"], r["n"], r["seed"])
        if k not in cache:
            mod = loader.load_type(r["track"])
            cache[k] = (mod, mod.generate(r["n"], r["seed"], tempfile.mkdtemp()))
        mod, task = cache[k]
        s = mod.score(task, a.read_text())["score"]
        rec = r["score"].get("score") if isinstance(r["score"], dict) else None
        n += 1
        if rec is None or abs(rec - s) > 1e-9:
            diff.append((r["rid"], rec, s))
print(f"regraded {n}, differ {len(diff)}, missing answers {missing}")
for d in diff:
    print("  ", d)
