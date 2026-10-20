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
"""A09: parse eCFR full-text XML (public domain, US Government work) into sections and paragraphs.
usage: python3 -I ecfr_parse.py ECFR_DIR OUT_DIR -> OUT_DIR/regs.json (sections with paragraph lines) and
OUT_DIR/build/units_candidates.jsonl (level-1 paragraphs that can carry a policy clause)."""
import html
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
ROMAN = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii", "xiii", "xiv", "xv", "xvi", "xvii",
         "xviii", "xix", "xx"}
DES = re.compile(r"^\(([a-z]{1,4}|\d{1,2}|[A-Z])\)")
NAMES = {"1002": "Regulation B (Equal Credit Opportunity Act)", "1005": "Regulation E (Electronic Fund Transfers)",
         "1016": "Regulation P (Privacy of Consumer Financial Information)", "1022": "Regulation V (Fair Credit Reporting)",
         "1024": "Regulation X (Real Estate Settlement Procedures Act)", "1030": "Regulation DD (Truth in Savings)",
         "312": "Children's Online Privacy Protection Rule", "314": "Standards for Safeguarding Customer Information",
         "681": "Identity Theft Rules", "1020": "Rules for Banks (Bank Secrecy Act)"}


def text_of(el):
    s = ET.tostring(el, encoding="unicode", method="text")
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


def level(tok, prev1):
    if tok.isdigit():
        return 2
    if tok.isupper():
        return 4
    if tok in ROMAN and not (tok == "i" and prev1 == "h") and not (tok == "v" and prev1 == "u") and \
            not (tok == "x" and prev1 == "w"):
        return 3
    return 1


sections, units = [], []
for f in sorted(src.glob("t*-*.xml")):
    title, part = re.match(r"t(\d+)-(\d+)", f.stem).groups()
    root = ET.parse(f).getroot()
    for sec in root.iter("DIV8"):
        if sec.get("TYPE") != "SECTION":
            continue
        num = sec.get("N")
        head = text_of(sec.find("HEAD")) if sec.find("HEAD") is not None else num
        paras, cur1, path = [], None, []
        for p in sec.iter("P"):
            t = text_of(p)
            if not t:
                continue
            toks = []
            rest = t
            while True:
                m = DES.match(rest)
                if not m:
                    break
                toks.append(m.group(1))
                rest = rest[m.end():].lstrip()
            if toks:
                for tok in toks:
                    lv = level(tok, cur1)
                    if lv == 1:
                        cur1 = tok
                    path = path[:lv - 1] + [tok]
            des = "".join(f"({x})" for x in path)
            paras.append({"des": des, "text": t})
        sections.append({"cite": f"{title} CFR {num}", "section": num, "part": part, "title": title,
                         "heading": head, "paras": paras})
        if re.search(r"definition|authority|purpose|scope|reserved|effective date|severab|staff commentary|"
                     r"model|appendix|exemptions? for|preemption|relation to state|record retention", head, re.I):
            continue
        by1 = {}
        for p in paras:
            if p["des"]:
                by1.setdefault(p["des"].split(")")[0] + ")", []).append(p["text"])
        for d1, ts in by1.items():
            body = " ".join(ts)
            if 250 <= len(body) <= 3500 and "[Reserved]" not in body:
                units.append({"cite": f"{title} CFR {num}{d1}", "section": num, "part": part, "heading": head,
                              "text": body})
(out / "data").mkdir(parents=True, exist_ok=True)
(out / "data" / "regs.json").write_text(json.dumps({"parts": NAMES, "sections": sections}))
with open(out / "data" / "build" / "units_candidates.jsonl", "w") as fh:
    for u in units:
        fh.write(json.dumps(u) + "\n")
import collections
print(len(sections), "sections", len(units), "units", collections.Counter(u["part"] for u in units))
