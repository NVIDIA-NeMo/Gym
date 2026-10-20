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
"""Build-time only: sanitize the downloaded pages (data/raw) into data/pages.json (run with python3 -I from 37_a11y).

Keeps the real element structure and accessibility-relevant attributes; drops scripts, styles, head metadata, classes
and Parsoid bookkeeping; truncates each page to a character budget (later siblings are dropped once it is reached)."""
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import a11ydom as D  # noqa: E402

DROP = {"script", "style", "link", "meta", "noscript", "svg", "math", "iframe", "object", "embed", "video", "audio",
        "source", "template", "head", "base", "canvas", "track"}
UNWRAP = {"picture", "html", "body", "span", "abbr", "bdi", "small", "b", "i", "em", "strong", "code", "sup", "sub",
          "cite", "kbd", "var", "samp", "q", "s", "u", "mark", "time", "wbr", "font", "big", "data"}
KEEP_ATTR = {"id", "href", "alt", "title", "src", "role", "lang", "for", "type", "value", "name", "placeholder",
             "hidden", "scope", "colspan", "rowspan"}
BUDGET = 5000


def clean_attrs(n):
    out = {}
    for k, v in n.attrs.items():
        k = k.lower()
        if k.startswith("aria-") or k in KEEP_ATTR:
            if k == "id" and re.fullmatch(r"mw[\w-]{1,8}", v):
                continue
            if k == "src":
                v = v.split("?")[0].rstrip("/").split("/")[-1][:80]
            if k == "href":
                v = v.split("?")[0][:120]
            out[k] = v
        elif k == "style" and D._style_hidden(v):
            out["style"] = "display: none"
    n.attrs = out


def sanitize(n, budget):
    """Returns the cleaned children list of n; budget is a one-element list of remaining text chars."""
    kids = []
    for c in n.children:
        if budget[0] <= 0:
            break
        if isinstance(c, str):
            t = re.sub(r"\s+", " ", c)
            if t.strip() or kids:
                kids.append(t)
                budget[0] -= len(t.strip())
            continue
        if c.tag in DROP or c.tag.startswith("mw:"):
            continue
        clean_attrs(c)
        c.children = sanitize(c, budget)
        if c.tag in UNWRAP and not c.attrs:
            kids.extend(c.children)
            for g in c.children:
                if isinstance(g, D.Node):
                    g.parent = n
            continue
        if c.tag in ("span", "div", "section", "p") and not c.children and not c.attrs:
            continue
        c.parent = n
        kids.append(c)
        budget[0] -= 20 if c.tag in D.REF_TAGS else 0
    return kids


def main():
    raw = HERE / "data" / "raw"
    pages, files = [], {}
    for src, sub in (("wikipedia", "wiki"), ("python-docs", "pydocs")):
        for f in sorted((raw / sub).glob("*.html")):
            b = f.read_bytes()
            files[str(f.relative_to(HERE))] = hashlib.sha256(b).hexdigest()
            s = b.decode("utf-8", "replace")
            root = D.parse(s)
            body = next((x for x in root.iter() if x.tag == "body"), root)
            if src == "wikipedia":
                title = re.search(r"<title>(.*?)</title>", s).group(1)
                rev = re.search(r'revision/(\d+)"', s).group(1)
                url = f"https://en.wikipedia.org/w/index.php?title={f.stem}&oldid={rev}"
            else:
                title = re.sub(r"\s+", " ", re.search(r"<title>(.*?)</title>", s, re.S).group(1)).strip()
                title = title.replace("&#8212;", "-").replace("—", "-")
                url = "https://docs.python.org/3.13/" + f.stem.replace("_", "/", 1) + ".html"
            new = D.Node("div")
            new.children = sanitize(body, [BUDGET])
            for c in new.children:
                if isinstance(c, D.Node):
                    c.parent = new
            html = "".join(D.serialize(c) if isinstance(c, D.Node) else c for c in new.children)
            pages.append({"key": f"{src}:{f.stem}", "source": src, "title": title, "url": url, "body": html})
    out = HERE / "data" / "pages.json"
    out.write_text(json.dumps(pages, indent=0, ensure_ascii=False))
    print(len(pages), "pages", sum(len(p["body"]) for p in pages) // len(pages), "chars avg")
    print(json.dumps({"raw_sha256": files, "pages_sha256": hashlib.sha256(out.read_bytes()).hexdigest()})[:300])
    (HERE / "data" / "raw_hashes.json").write_text(json.dumps(files, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
