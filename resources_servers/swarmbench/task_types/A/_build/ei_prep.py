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
"""A01: K3 filter requests (full article in context, no tools). usage: python3 -I ei_prep.py TYPE_DIR"""
import json
import sys
from pathlib import Path

b = Path(sys.argv[1]) / "data" / "build"
INSTR = ("Below is the full text of a randomized controlled trial report, followed by questions. For each question, "
         "decide what the article reports for the OUTCOME in the INTERVENTION group relative to the COMPARATOR group: "
         "'increased' (the outcome measure is statistically significantly higher with the intervention), 'decreased' "
         "(significantly lower), or 'no_difference' (no statistically significant difference reported). Answer with one "
         "JSON object mapping each question id to one of these three strings, and nothing else.")
with open(b / "ei_req.jsonl", "w") as f:
    for l in open(b / "candidates.jsonl"):
        rec = json.loads(l)
        qs = "\n".join(f"Q{p['pid']}: INTERVENTION: {p['intervention']} | COMPARATOR: {p['comparator']} | "
                       f"OUTCOME: {p['outcome']}" for p in rec["prompts"])
        f.write(json.dumps({"id": rec["pmcid"], "max_tokens": 24000, "messages": [
            {"role": "user", "content": f"{INSTR}\n\nARTICLE:\n{rec['text'][:110000]}\n\nQUESTIONS:\n{qs}"}]}) + "\n")
