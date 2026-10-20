"""Type 37 solutions: tool-only oracle (reads every page through the tools and runs the reference checker),
plausible-wrong baselines (a regex auditor, answers shifted by one page), careless readers that miss one exception
kind each, and single units (check 5)."""
import json
import re

from .. import common
from . import a11ydom as D
from . import type as track

T = track.T


def read_pages(task):
    fns = {t["name"]: t["fn"] for t in track.tools(task["world_dir"])}
    ids, off = [], 0
    while True:
        out = fns["list_pages"]({"offset": off, "limit": 100})
        ids += re.findall(r"^(P\d{3}) \|", out, re.M)
        m = re.search(r"\(next offset (\d+)\)$", out)
        if not m:
            break
        off = int(m.group(1))
    pages = {}
    for pid in ids:
        lines, start = [], 1
        while True:
            out = fns["read_page"]({"page_id": pid, "start_line": start})
            body, tail = out.rsplit("\n[", 1)
            lines += [re.sub(r"^\d+: ", "", ln, count=1) for ln in body.split("\n")]
            m = re.search(r"next start_line (\d+)\]$", tail)
            if not m:
                break
            start = int(m.group(1))
        pages[pid] = "\n".join(lines) + "\n"
    return pages


def _answer(pages, opts=None):
    out = {}
    for pid, html in pages.items():
        g = D.check(D.parse(html), opts)
        out[pid] = {f: g[f] for f in D.FIELDS}
    return json.dumps(out)


def oracle(task):
    return _answer(read_pages(task))


# ---------- regex auditor ----------
TAG = re.compile(r"<(img|input|select|textarea)\b([^>]*)>", re.I)
ATTR = lambda name: re.compile(name + r'="([^"]*)"', re.I)  # noqa: E731
REFX = ATTR("data-ref")


def regex_page(html):
    out = {f: [] for f in D.FIELDS}
    fors = set(re.findall(r'<label[^>]*for="([^"]*)"', html))
    for m in TAG.finditer(html):
        tag, attrs = m.group(1).lower(), m.group(2)
        ref = REFX.search(attrs).group(1)
        if tag == "img":
            if "alt=" not in attrs:
                out["img_alt"].append(ref)
            continue
        typ = (ATTR("type").search(attrs) or [None, "text"])[1].lower()
        if typ in ("hidden", "submit", "reset"):
            continue
        i = ATTR("id").search(attrs)
        if not (re.search(r"aria-label(ledby)?=|title=", attrs) or (i and i.group(1) in fors)):
            out["control_name"].append(ref)
    for m in re.finditer(r"<(a|button)\b([^>]*)>(.*?)</\1>", html, re.S | re.I):
        attrs, inner = m.group(2), m.group(3)
        if m.group(1) == "a" and "href=" not in attrs:
            continue
        txt = " ".join(re.sub(r"<[^>]+>", " ", inner).split())
        if not re.search(r"aria-label(ledby)?=|title=", attrs) and (not txt or D.is_generic(txt)):
            out["link_name"].append(REFX.search(attrs).group(1))
    prev = 0
    for m in re.finditer(r"<h([1-6])\b([^>]*)>", html):
        lv = int(m.group(1))
        if lv > prev + 1:
            out["heading_skip"].append(REFX.search(m.group(2)).group(1))
        prev = lv
    ids = re.findall(r'\sid="([^"]*)"', html)
    out["duplicate_id"] = sorted({x for x in ids if ids.count(x) > 1})
    lm = re.search(r'<html[^>]*lang="([^"]*)"', html)
    out["lang_ok"] = bool(lm and D.LANG_RX.fullmatch(lm.group(1)))
    for m in re.finditer(r"<table\b([^>]*)>(.*?)</table>", html, re.S):
        if "<th" not in m.group(2):
            out["table_header"].append(REFX.search(m.group(1)).group(1))
    return out


def wrong(task):
    pages = read_pages(task)
    ids = list(pages)
    shifted = {ids[(j + 1) % len(ids)]: v for j, v in enumerate(json.loads(_answer(pages)).values())}
    return {
        "regex": json.dumps({pid: regex_page(h) for pid, h in pages.items()}),
        "shifted_pages": json.dumps(shifted),
        "ignore_hidden": _answer(pages, {"no_hidden": True}),
        "placeholder_is_label": _answer(pages, {"placeholder": True}),
        "no_wrapping_labels": _answer(pages, {"no_wrap": True}),
        "labelledby_unchecked": _answer(pages, {"lb_any": True}),
        "filename_alt_ok": _answer(pages, {"file_alt_ok": True}),
        "link_content_first": _answer(pages, {"content_first": True}),
    }


def units(task, k, rng):
    g = common.gold_of(task)["pages"]
    pids = rng.sample(sorted(g), min(k, len(g)))
    out = []
    for pid in pids:
        gp = g[pid]
        ex = {pid: {"img_alt": ["r7"], "control_name": [], "link_name": ["r12"], "heading_skip": [],
                    "duplicate_id": [], "lang_ok": True, "table_header": []}}
        prompt = "\n\n".join([common.CANARY, common.fill(T["unit_prompt"], page=pid, example=json.dumps(ex)),
                              track.rules_text()])

        def check(text, pid=pid, gp=gp):
            p = track.parse_answer(text)
            if pid not in p:
                return None
            return all(track.page_fields_ok(p[pid], gp).values())
        out.append({"id": f"{task['id']}:{pid}", "kind": gp["source"].split(":")[0], "prompt": prompt,
                    "gold": {f: gp[f] for f in D.FIELDS}, "check": check})
    return out
