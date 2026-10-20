"""Type 37 a11y: accessibility audit of N static web pages (real Wikipedia / Python-docs snapshots + seeded edits).

Unit = one page: the exact offenders of seven rules (images, form controls, links/buttons, heading skips, duplicate
ids, html lang, data-table headers), located by data-ref. Gold = the deterministic checker in a11ydom.py run on the
page exactly as the agent reads it. Task prose is K3's (text/k3.json); page text is real or from K3 phrase banks."""
import json
import random
import re
import threading
from html import escape
from pathlib import Path

from .. import common
from . import a11ydom as D

TYPE_ID = "37_a11y"
TIERS = {"easy": (4, 1800), "medium": (16, 1200), "hard": (56, 1200)}  # hard: 56 x ~60 s (OpenCode) ~ 2.8x budget
HERE = Path(__file__).parent
T = common.load_text(HERE)
GENERIC = [g.lower().strip() for g in T["generic"]]
D.GENERIC[:] = GENERIC
POOL = json.loads((HERE / "data" / "pages.json").read_text())
HIDERS = ['hidden', 'aria-hidden="true"', 'style="display: none"', 'style="visibility:hidden"',
          'style="display:none !important"']
LANG_OK = ["en", "en", "en", "en", "en-GB", "en-US", "EN", "fr"]
LANG_BAD = [None, "", "english", "en_US", "en-", "e"]
FILES = ["IMG_2041.JPG", "banner-final.png", "photo 3.jpeg", "hero_v2.webp", "DSC00172.jpg", "icon.svg", "image.gif"]
GLYPHS = ["✕", "☰", "→", "★", "✉", "⌂"]


def _e(s):
    return escape(str(s))


class _Seed:
    """Builds the injected blocks of one page; every builder records its intended effect (for the tests)."""

    def __init__(self, rng, pidx):
        self.rng, self.k, self.pidx, self.kinds = rng, 0, pidx, []
        self.header, self.aside, self.footer, self.form, self.late = [], [], [], [], []

    def uid(self, pre):
        self.k += 1
        return f"{pre}-{self.pidx}{self.k}"

    def pick(self, key):
        return _e(self.rng.choice(T[key]))

    def hider(self):
        return self.rng.choice(HIDERS)

    def generic_text(self):
        g = self.rng.choice(GENERIC)
        g = self.rng.choice([g, g.capitalize(), g.upper(), g.title()])
        return _e(g + self.rng.choice(["", "", " »", ".", "…", " ›", ":"]))


def _img_kind(s, kind):
    src = f"img-{s.rng.randint(100, 999)}.png"
    if kind == "img_decor":
        return f'<img src="divider.png" alt="">'
    if kind == "img_role":
        return f'<img src="{src}" role="{s.rng.choice(["presentation", "none"])}">'
    if kind == "img_hidden":
        return f'<div {s.hider()}><img src="{src}"></div>' if s.rng.random() < 0.6 else f'<img src="{src}" {s.hider()}>'
    if kind == "img_ariafalse":
        return f'<div aria-hidden="false"><img src="{src}"></div>'
    if kind in ("img_lb_ok", "img_lb_bad"):
        cid = s.uid("cap")
        ref = cid if kind == "img_lb_ok" else cid + "x"
        return (f'<figure><img src="{src}" aria-labelledby="{ref}">'
                f'<figcaption id="{cid}">{s.pick("captions")}</figcaption></figure>')
    if kind == "img_file":
        return f'<img src="{src}" alt="{s.rng.choice(FILES)}">'
    if kind == "img_title":
        return f'<img src="{src}" title="{s.pick("alts")}">'
    if kind == "img_noalt":
        return f'<img src="{src}">'
    if kind == "img_alt_ok":
        return f'<img src="{src}" alt="{s.pick("alts")}">'
    raise KeyError(kind)


def _ctl_kind(s, kind):
    """Returns (html in the form, html placed earlier on the page or '')."""
    lab = s.pick("labels")
    typ = s.rng.choice(["text", "email", "search", "tel", "number", "checkbox", "url"])
    if kind == "ctl_for_far":
        i = s.uid("fld")
        return f'<input type="{typ}" id="{i}" name="{i}">', f'<label for="{i}">{lab}</label>'
    if kind == "ctl_wrap":
        return f'<label>{lab} <input type="{typ}" name="w{s.k}"></label>', ""
    if kind == "ctl_placeholder":
        return f'<input type="{typ}" name="p{s.k}" placeholder="{s.pick("placeholders")}">', ""
    if kind in ("ctl_lb_ok", "ctl_lb_bad"):
        i = s.uid("hint")
        ref = i if kind == "ctl_lb_ok" else s.uid("hint")
        return f'<p id="{i}">{lab}</p><input type="{typ}" aria-labelledby="{ref}">', ""
    if kind == "ctl_hidden_input":
        return f'<input type="hidden" name="token" value="{s.rng.randint(10**7, 10**8)}">', ""
    if kind == "ctl_in_hidden":
        return f'<div {s.hider()}><input type="{typ}" name="h{s.k}"></div>', ""
    if kind == "ctl_submit":
        return '<input type="submit">' if s.rng.random() < 0.5 else '<input type="reset">', ""
    if kind == "ctl_btn_novalue":
        return f'<input type="button" name="b{s.k}">', ""
    if kind == "ctl_btn_value":
        return f'<input type="button" value="{s.pick("buttons")}">', ""
    if kind == "ctl_select_nolabel":
        opts = "".join(f"<option>{s.pick('cells')}</option>" for _ in range(3))
        return f"<select name=\"s{s.k}\">{opts}</select>", ""
    if kind == "ctl_title":
        return f'<textarea name="t{s.k}" title="{lab}"></textarea>', ""
    if kind == "ctl_for_mismatch":
        i = s.uid("fld")
        return f'<label for="{i}">{lab}</label><input type="{typ}" id="{s.uid("fld")}">', ""
    if kind == "ctl_empty_label":
        i = s.uid("fld")
        return f'<label for="{i}"> </label><input type="{typ}" id="{i}">', ""
    if kind == "ctl_label_for_wrap":  # wrapping label that has a for= pointing elsewhere: wrapping does not apply
        i = s.uid("fld")
        return f'<label for="{i}">{lab} <input type="{typ}" name="x{s.k}"></label>', ""
    raise KeyError(kind)


def _link_kind(s, kind):
    href = f"/{s.rng.choice(['docs', 'news', 'about', 'help', 'events', 'shop'])}/{s.rng.randint(10, 999)}"
    if kind == "lk_generic":
        return f'<a href="{href}">{s.generic_text()}</a>'
    if kind == "lk_generic_aria":
        return f'<a href="{href}" aria-label="{s.pick("links")}">{s.generic_text()}</a>'
    if kind == "lk_aria_generic":
        return f'<a href="{href}" aria-label="{s.generic_text()}">{s.pick("links")}</a>'
    if kind == "lk_img_alt":
        return f'<a href="{href}"><img src="thumb-{s.k}.png" alt="{s.pick("alts")}"></a>'
    if kind == "lk_img_empty":
        return f'<a href="{href}"><img src="thumb-{s.k}.png" alt=""></a>'
    if kind == "lk_nohref":
        return f'<a id="{s.uid("anchor")}"></a>'
    if kind == "lk_hidden":
        return f'<a href="{href}" {s.hider()}></a>'
    if kind == "lk_title":
        return f'<a href="{href}" title="{s.pick("links")}"></a>'
    if kind == "lk_partial_generic":
        return f'<a href="{href}">{_e(s.rng.choice(GENERIC).capitalize())} {s.pick("links").lower()}</a>'
    if kind == "btn_icon":
        return f'<button type="button"><span aria-hidden="true">{s.rng.choice(GLYPHS)}</span></button>'
    if kind == "btn_icon_label":
        return (f'<button type="button" aria-label="{s.pick("icon_names")}">'
                f'<span aria-hidden="true">{s.rng.choice(GLYPHS)}</span></button>')
    if kind == "btn_text":
        return f'<button type="button">{s.pick("buttons")}</button>'
    if kind == "lk_empty":
        return f'<a href="{href}"> </a>'
    if kind == "lk_lb":
        i = s.uid("lbl")
        return f'<span id="{i}">{s.pick("links")}</span> <a href="{href}" aria-labelledby="{i}">{s.generic_text()}</a>'
    raise KeyError(kind)


IMG_KINDS = ["img_decor", "img_role", "img_hidden", "img_ariafalse", "img_lb_ok", "img_lb_bad", "img_file",
             "img_title", "img_noalt", "img_alt_ok"]
CTL_KINDS = ["ctl_for_far", "ctl_wrap", "ctl_placeholder", "ctl_lb_ok", "ctl_lb_bad", "ctl_hidden_input",
             "ctl_in_hidden", "ctl_submit", "ctl_btn_novalue", "ctl_btn_value", "ctl_select_nolabel", "ctl_title",
             "ctl_for_mismatch", "ctl_empty_label", "ctl_label_for_wrap"]
LINK_KINDS = ["lk_generic", "lk_generic_aria", "lk_aria_generic", "lk_img_alt", "lk_img_empty", "lk_nohref",
              "lk_hidden", "lk_title", "lk_partial_generic", "btn_icon", "btn_icon_label", "btn_text", "lk_empty",
              "lk_lb"]
REAL_KINDS = ["real_strip_alt", "real_hide_img", "real_heading_up", "real_th_td", "dup_real", "dup_hidden",
              "dup_case", "h_hidden_bridge", "h_hidden_skip", "tbl_layout", "tbl_nohead", "tbl_nested", "tbl_hidden"]


def _visible_nodes(root, tag_pred):
    return [n for n in root.iter() if tag_pred(n) and not D.hidden(n)]


def _frag(html, parent):
    r = D.parse(html)
    kids = r.children
    for c in kids:
        if isinstance(c, D.Node):
            c.parent = parent
    return kids


def _prev_heading_level(root, node):
    lv = 0
    for n in root.iter():
        if n is node:
            return lv
        if n.tag in ("h1", "h2", "h3", "h4", "h5", "h6") and not D.hidden(n):
            lv = int(n.tag[1])
    return lv


def build_page(i, seed, perm):
    rng = random.Random(f"a11y:{seed}:{i}")
    base = POOL[perm[i]]
    s = _Seed(rng, i + 1)
    body = D.Node("body")
    main = D.Node("main")
    body.add(main)
    if base["source"] == "wikipedia" and rng.random() < 0.85:
        main.add(D.Node("h1")).add(base["title"])
    for c in _frag(base["body"], main):
        main.add(c)
    real_ids = sorted({n.get("id") for n in main.iter() if n.get("id")})
    # choose trap kinds: every page mixes several kinds that a pattern reader gets wrong
    kinds = (rng.sample(IMG_KINDS, rng.randint(2, 3)) + rng.sample(CTL_KINDS, rng.randint(3, 5))
             + rng.sample(LINK_KINDS, rng.randint(3, 5)) + rng.sample(REAL_KINDS, rng.randint(3, 5)))
    s.kinds = kinds
    head_links = [f'<a href="/{x}">{s.pick("links")}</a>' for x in range(rng.randint(2, 4))]
    header = f'<header><a href="/"><img src="logo.png" alt="{s.pick("alts")}"></a><nav>{" ".join(head_links)}</nav>'
    aside, footer, form, early = [], [], [], []
    for k in kinds:
        if k in IMG_KINDS:
            (aside if rng.random() < 0.6 else footer).append(_img_kind(s, k))
        elif k in CTL_KINDS:
            h, e = _ctl_kind(s, k)
            form.append(f"<div>{h}</div>")
            if e:
                early.append(e)
        elif k in LINK_KINDS:
            (aside if rng.random() < 0.5 else footer).append(_link_kind(s, k))
    form_html = (f'<form><h2 id="{s.uid("formhead")}">{s.pick("headings")}</h2>{"".join(form)}'
                 f'<button type="submit">{s.pick("buttons")}</button></form>') if form else ""
    header += "".join(early) + "</header>"
    for c in _frag(header, body):
        body.add(c, at=0)
    aside_node = D.Node("aside")
    aside_node.add(D.Node("h2")).add(_e(rng.choice(T["headings"])))
    for c in _frag("".join(aside), aside_node):
        aside_node.add(c)
    blocks = [c for c in main.children if isinstance(c, D.Node)]
    at = main.children.index(blocks[len(blocks) // 2]) if blocks else len(main.children)
    aside_node.parent = main
    main.add(aside_node, at=at)
    foot = D.Node("footer")
    foot.parent = body
    body.add(foot)
    for c in _frag("".join(footer) + form_html, foot):
        foot.add(c)
    # edits of real content and structural traps
    for k in kinds:
        if k not in REAL_KINDS:
            continue
        if k == "real_strip_alt":
            c = _visible_nodes(main, lambda n: n.tag == "img" and n.get("alt"))
            if c:
                del rng.choice(c).attrs["alt"]
        elif k == "real_hide_img":
            c = _visible_nodes(main, lambda n: n.tag == "img" and "alt" not in n.attrs)
            if c:
                tgt = rng.choice(c)
                tgt = tgt.parent if tgt.parent.tag == "a" else tgt
                h = rng.choice(HIDERS)
                if h == "hidden":
                    tgt.attrs["hidden"] = ""
                else:
                    a, v = h.split("=", 1)
                    tgt.attrs[a] = v.strip('"')
        elif k == "real_heading_up":
            c = _visible_nodes(main, lambda n: n.tag in ("h2", "h3"))
            if c:
                h = rng.choice(c)
                h.tag = f"h{int(h.tag[1]) + 2}"
        elif k == "real_th_td":
            c = _visible_nodes(main, lambda n: n.tag == "table" and D._has_own_th(n))
            if c:
                for n in rng.choice(c).iter():
                    if n.tag == "th":
                        n.tag = "td"
        elif k in ("dup_real", "dup_hidden"):
            val = rng.choice(real_ids) if real_ids and k == "dup_real" else s.uid("note")
            if k == "dup_hidden":
                foot.add(D.Node("span", {"id": val}))
            h = f'<div {s.hider()}><p id="{_e(val)}">{s.pick("notices")}</p></div>' if k == "dup_hidden" else \
                f'<p id="{_e(val)}">{s.pick("notices")}</p>'
            for c in _frag(h, aside_node):
                aside_node.add(c)
        elif k == "dup_case":
            v = s.uid("Top")
            for c in _frag(f'<span id="{v}"></span><span id="{v.lower()}"></span>', foot):
                foot.add(c)
        elif k in ("h_hidden_bridge", "h_hidden_skip"):
            p = _prev_heading_level(body, aside_node)
            p = min(p, 3)
            if k == "h_hidden_bridge":
                h = (f'<div {s.hider()}><h{p + 1}>{s.pick("headings")}</h{p + 1}></div>'
                     f'<h{p + 2}>{s.pick("headings")}</h{p + 2}>')
            else:
                h = f'<h{p + 3} {s.hider()}>{s.pick("headings")}</h{p + 3}>'
            ch = _frag(h, aside_node)
            for j, c in enumerate(ch):
                aside_node.add(c, at=j)
        elif k.startswith("tbl_"):
            row = lambda tag: "<tr>" + "".join(f"<{tag}>{s.pick('cells' if tag == 'td' else 'col_heads')}</{tag}>"  # noqa
                                               for _ in range(3)) + "</tr>"
            if k == "tbl_layout":
                h = f'<table role="{rng.choice(["presentation", "none"])}">{row("td")}</table>'
            elif k == "tbl_nohead":
                h = f"<table>{row('td')}{row('td')}</table>"
            elif k == "tbl_nested":
                h = f"<table><tr><td><table>{row('th')}{row('td')}</table></td></tr></table>"
            else:
                h = f"<table {s.hider()}>{row('td')}</table>"
            for c in _frag(h, foot):
                foot.add(c)
    lang = rng.choice(LANG_OK) if rng.random() < 0.65 else rng.choice(LANG_BAD)
    la = "" if lang is None else f' lang="{_e(lang)}"'
    html = D.Node("html", {} if lang is None else {"lang": lang})
    head = html.add(D.Node("head"))
    head.add(D.Node("title")).add(base["title"])
    body.parent = html
    html.add(body)
    D.assign_refs(html)
    text = "<!DOCTYPE html>\n" + D.serialize(html)
    assert la == "" or 'lang="' in text.split("\n")[1]
    gold = D.check(D.parse(text))
    return text, gold, kinds, base


def generate(n, seed, out_dir):
    assert n <= len(POOL)
    perm = list(range(len(POOL)))
    random.Random(f"a11y-perm:{seed}").shuffle(perm)
    pages, gold = [], {}
    for i in range(n):
        text, g, kinds, base = build_page(i, seed, perm)
        pid = f"P{i + 1:03d}"
        pages.append({"id": pid, "title": base["title"], "html": text})
        gold[pid] = {**g, "kinds": kinds, "source": base["key"]}
    ex = {pages[0]["id"]: {"img_alt": ["r7"], "control_name": [], "link_name": ["r12", "r40"], "heading_skip": [],
                           "duplicate_id": ["intro"], "lang_ok": True, "table_header": []}}
    prompt = "\n\n".join([common.fill(T["intro"], n=n, first=pages[0]["id"], last=pages[-1]["id"]),
                          rules_text(), common.fill(T["answer_line"], example=json.dumps(ex))])
    return common.finish(TYPE_ID, n, seed, out_dir, prompt, {"pages": gold}, TIERS,
                         files={"pages.json": pages})


def rules_text():
    gen = "; ".join(f'"{g}"' for g in GENERIC)
    return "\n\n".join([T["rule_scope"], T["rule_hidden"], T["rule_img"], T["rule_control"],
                        common.fill(T["rule_link"], generic=gen), T["rule_heading"], T["rule_dup"], T["rule_lang"],
                        T["rule_table"]])


# ---------- tools ----------
_cache, _lock = {}, threading.Lock()


def _world(world_dir):
    with _lock:
        if world_dir not in _cache:
            pages = json.loads((Path(world_dir) / "pages.json").read_text())
            _cache[world_dir] = ({p["id"]: p["html"].split("\n")[:-1] for p in pages},
                                 [(p["id"], p["title"]) for p in pages])
        return _cache[world_dir]


def _page_lines(world_dir, a):
    by, _ = _world(world_dir)
    pid = str(a.get("page_id") or "").strip().upper()
    if pid not in by:
        raise KeyError(f"unknown page {pid!r}; use list_pages")
    return pid, by[pid]


def tools(world_dir):
    def list_pages(a):
        by, order = _world(world_dir)
        off, lim = common.page_args(a, 100, 100)
        rows = [f"{p} | {t} | {len(by[p])} lines" for p, t in order[off:off + lim]]
        more = f"\n(next offset {off + lim})" if off + lim < len(order) else ""
        return "page | title | lines\n" + "\n".join(rows) + more

    def read_page(a):
        pid, lines = _page_lines(world_dir, a)
        start = int(a.get("start_line") or 1)
        if start < 1:
            raise ValueError("start_line must be >= 1")
        out, size, j = [], 0, start - 1
        while j < len(lines):
            row = f"{j + 1}: {lines[j]}"
            if out and size + len(row) > 7400:
                break
            out.append(row[:7400])
            size += len(row) + 1
            j += 1
        tail = f"\n[lines {start}-{j} of {len(lines)}; next start_line {j + 1}]" if j < len(lines) else \
            f"\n[end of page {pid}, {len(lines)} lines]"
        return "\n".join(out) + tail

    def search_page(a):
        pid, lines = _page_lines(world_dir, a)
        q = str(a.get("query") or "")
        if not q.strip():
            raise ValueError("query is required")
        hits = [f"{j + 1}: {ln if len(ln) <= 300 else ln[:300] + ' ...'}" for j, ln in enumerate(lines)
                if q.lower() in ln.lower()]
        return f"{len(hits)} matching lines" + (" (first 40 shown)" if len(hits) > 40 else "") + "\n" + \
            "\n".join(hits[:40])

    Dd = T["tools"]
    S, I = {"type": "string"}, {"type": "integer"}
    return [common.tool("list_pages", Dd["list_pages"], {"offset": I, "limit": I}, [], list_pages),
            common.tool("read_page", Dd["read_page"], {"page_id": S, "start_line": I}, ["page_id"], read_page),
            common.tool("search_page", Dd["search_page"], {"page_id": S, "query": S}, ["page_id", "query"],
                        search_page)]


# ---------- scoring ----------
REF = re.compile(r"^\s*(?:data-ref\s*=\s*)?[\"']?r?(\d+)[\"']?\s*$", re.I)


def _refs(v):
    if not isinstance(v, list):
        return None
    out = set()
    for x in v:
        m = REF.match(str(x))
        if not m:
            return None
        out.add(f"r{int(m.group(1))}")
    return out


def _bool(v):
    if isinstance(v, bool):
        return v
    if str(v).strip().lower() in ("true", "yes", "valid", "ok", "pass"):
        return True
    if str(v).strip().lower() in ("false", "no", "invalid", "fail", "missing"):
        return False
    return None


def parse_answer(answer):
    obj = common.last_json(answer, lambda o: isinstance(o, dict) and any(re.fullmatch(r"P\d{3}", str(k).strip().upper())
                                                                         for k in o))
    out = {}
    for k, v in (obj or {}).items():
        k = str(k).strip().upper()
        if not re.fullmatch(r"P\d{3}", k) or not isinstance(v, dict):
            continue
        p = {}
        for f in D.FIELDS:
            if f == "lang_ok":
                p[f] = _bool(v.get(f))
            elif f == "duplicate_id":
                d = v.get(f)
                p[f] = {str(x).strip().strip("#") for x in d} if isinstance(d, list) else None
            else:
                p[f] = _refs(v.get(f))
        out[k] = p
    return out


def page_fields_ok(pa, g):
    ok = {}
    for f in D.FIELDS:
        if f == "lang_ok":
            ok[f] = pa.get(f) is g[f]
        else:
            ok[f] = pa.get(f) is not None and pa[f] == set(g[f])
    return ok


def score(task, answer):
    g = common.gold_of(task)["pages"]
    p = parse_answer(answer)
    n_ok, part = 0, 0.0
    for pid, gp in g.items():
        if pid not in p:
            continue
        ok = page_fields_ok(p[pid], gp)
        n_ok += all(ok.values())
        part += sum(ok.values()) / len(ok)
    n = len(g)
    return {"score": n_ok / n, "score_exact": float(n_ok == n), "score_partial": part / n, "n_correct": n_ok,
            "n_units": n}
