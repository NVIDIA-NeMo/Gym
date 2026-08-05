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
import re


def _norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", re.sub(r"\s+", " ", (s or "").lower())).strip()


def _toks(s):
    return set(_norm(s).split())


def parse_answer(generation: str):
    ans, pages = "", []
    for m in re.finditer(r"\{[^{}]*\}", generation, re.S):
        try:
            d = json.loads(m.group(0))
        except Exception:
            continue
        if isinstance(d, dict) and "answer" in d:
            ans = str(d.get("answer") or "")
            cp = d.get("cited_pages") or d.get("citations") or []
            pages = [str(p) for p in cp] if isinstance(cp, list) else [str(cp)]
    if not ans:
        ans = generation.strip()
    return ans, pages


def parse_choice(generation: str):
    """The chosen letter: JSON answer field, then "answer: C", then a final letter."""
    for m in re.finditer(r"\{[^{}]*\}", generation, re.S):
        try:
            d = json.loads(m.group(0))
        except Exception:
            continue
        if isinstance(d, dict) and "answer" in d:
            t = str(d.get("answer") or "").strip().upper()
            mm = re.search(r"[A-D]", t)
            if mm:
                return mm.group(0)
    m = re.search(r"answer\s*(?:is|:)?\s*\(?([A-D])\)?", generation, re.I)
    if m:
        return m.group(1).upper()
    m = re.findall(r"\b([A-D])\b", generation.strip()[-40:])
    return m[-1].upper() if m else ""


def mcqa_match(gold_letter: str, generation: str) -> float:
    gl = (gold_letter or "").strip().upper()[:1]
    return 1.0 if gl and parse_choice(generation) == gl else 0.0


def citation_match(gold_page: str, cited_pages) -> float:
    gp = (gold_page or "").strip().lower()
    if not gp:
        return 0.0
    return 1.0 if any(gp in (c or "").lower() or (c or "").lower() in gp for c in (cited_pages or [])) else 0.0
