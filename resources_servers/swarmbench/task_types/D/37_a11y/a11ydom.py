"""Tiny HTML tree (stdlib html.parser), serializer and the exact accessibility checker of type 37 (code only).

The checker is the definition of gold; the K3 rulebook in text/k3.json states the same rules in prose."""
import re
from html import escape
from html.parser import HTMLParser

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
BLOCK = {"html", "head", "body", "title", "header", "footer", "main", "nav", "aside", "section", "article", "div", "p",
         "ul", "ol", "li", "dl", "dt", "dd", "table", "thead", "tbody", "tfoot", "tr", "td", "th", "caption", "figure",
         "figcaption", "form", "fieldset", "legend", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "blockquote", "hr",
         "label", "select", "textarea", "option", "img", "input", "button", "br"}
REF_TAGS = {"img", "input", "select", "textarea", "button", "a", "h1", "h2", "h3", "h4", "h5", "h6", "table"}
GENERIC = ["click here", "here", "read more", "more", "learn more", "link", "this link", "details", "more info"]
FIELDS = ["img_alt", "control_name", "link_name", "heading_skip", "duplicate_id", "lang_ok", "table_header"]
LANG_RX = re.compile(r"[A-Za-z]{2,3}(-[A-Za-z0-9]{1,8})*")
FILE_RX = re.compile(r"[\w\-. ]+\.(jpe?g|png|gif|svg|webp)", re.I)
STRIP = " .,:;!?…»›>→"


class Node:
    __slots__ = ("tag", "attrs", "children", "parent")

    def __init__(self, tag, attrs=None, parent=None):
        self.tag, self.attrs, self.children, self.parent = tag, dict(attrs or {}), [], parent

    def add(self, child, at=None):
        if isinstance(child, Node):
            child.parent = self
        if at is None:
            self.children.append(child)
        else:
            self.children.insert(at, child)
        return child

    def iter(self):
        yield self
        for c in self.children:
            if isinstance(c, Node):
                yield from c.iter()

    def get(self, k, d=None):
        return self.attrs.get(k, d)


class _P(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node("#root")
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        n = self.stack[-1].add(Node(tag, [(k, v if v is not None else "") for k, v in attrs]))
        if tag not in VOID:
            self.stack.append(n)

    def handle_startendtag(self, tag, attrs):
        self.stack[-1].add(Node(tag, [(k, v if v is not None else "") for k, v in attrs]))

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]
                return

    def handle_data(self, data):
        self.stack[-1].add(data)


def parse(html):
    p = _P()
    p.feed(html)
    p.close()
    return p.root


def serialize(node):
    """HTML text with a line break around every block element (so pages can be read by line)."""
    out = []

    def walk(n):
        if isinstance(n, str):
            out.append(escape(n, quote=False))
            return
        if n.tag == "#root":
            for c in n.children:
                walk(c)
            return
        attrs = "".join(f' {k}="{escape(v)}"' if v != "" or k not in ("hidden",) else f" {k}"
                        for k, v in n.attrs.items())
        blk = n.tag in BLOCK
        if blk:
            out.append("\n")
        out.append(f"<{n.tag}{attrs}>")
        if n.tag in VOID:
            if blk:
                out.append("\n")
            return
        for c in n.children:
            walk(c)
        out.append(f"</{n.tag}>")
        if blk:
            out.append("\n")
    walk(node)
    text = "".join(out)
    lines = [ln.rstrip() for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln.strip()) + "\n"


# ---------- checker ----------
def ws(s):
    return " ".join(str(s).split())


def _style_hidden(style):
    for decl in (style or "").split(";"):
        if ":" in decl:
            k, v = decl.split(":", 1)
            k, v = k.strip().lower(), v.strip().lower().replace("!important", "").strip()
            if (k, v) in (("display", "none"), ("visibility", "hidden")):
                return True
    return False


def self_hidden(n):
    return ("hidden" in n.attrs or str(n.get("aria-hidden", "")).strip().lower() == "true"
            or _style_hidden(n.get("style")))


def hidden(n):
    while isinstance(n, Node) and n.tag != "#root":
        if self_hidden(n):
            return True
        n = n.parent
    return False


def text_of(n, skip_controls=False, visible_only=True, with_alt=True):
    """Text content: visible descendant text (hidden subtrees skipped) plus alt of visible descendant images."""
    parts = []

    def walk(x):
        if isinstance(x, str):
            parts.append(x)
            return
        if visible_only and self_hidden(x):
            return
        if skip_controls and x.tag in ("select", "textarea", "button", "option", "input"):
            return
        if x.tag == "img" and with_alt:
            parts.append(" " + x.get("alt", "") + " ")
        for c in x.children:
            walk(c)
    for c in n.children:
        walk(c)
    return ws("".join(parts))


class Page:
    def __init__(self, root):
        self.root = root
        self.ids = {}
        for n in root.iter():
            if n.tag != "#root" and "id" in n.attrs:
                self.ids.setdefault(n.attrs["id"], []).append(n)

    def labelledby(self, n):
        """Text of the aria-labelledby targets (hidden targets count, as in the accessible-name rules)."""
        refs = str(n.get("aria-labelledby", "")).split()
        got = [text_of(self.ids[r][0], visible_only=False) for r in refs if r in self.ids]
        return ws(" ".join(got))

    def label_text(self, ctl, no_wrap=False):
        texts = []
        cid = ctl.get("id")
        if cid:
            if self.ids.get(cid, [None])[0] is ctl:
                for lab in self.root.iter():
                    if lab.tag == "label" and lab.get("for") == cid:
                        texts.append(text_of(lab, skip_controls=True))
        p = ctl.parent
        while not no_wrap and isinstance(p, Node) and p.tag != "#root":
            if p.tag == "label" and "for" not in p.attrs:
                texts.append(text_of(p, skip_controls=True))
                break
            p = p.parent
        return ws(" ".join(texts))


def is_generic(name):
    return name.lower().strip(STRIP).strip() in GENERIC


def check(root, opts=None):
    """Exact per-rule offenders of one page (refs are data-ref values). opts switch on the careless-reader variants
    used only by the plausible-wrong baselines (solutions.py); gold always uses opts=None."""
    o = opts or {}
    pg = Page(root)
    lb = (lambda n: str(n.get("aria-labelledby", "")).strip()) if o.get("lb_any") else pg.labelledby
    hid = (lambda n: False) if o.get("no_hidden") else hidden
    out = {f: [] for f in FIELDS}
    prev = 0
    html = next((n for n in root.iter() if n.tag == "html"), None)
    out["lang_ok"] = bool(html is not None and LANG_RX.fullmatch(html.get("lang", "") or ""))
    out["duplicate_id"] = sorted(k for k, v in pg.ids.items() if len(v) > 1)
    for n in root.iter():
        t, ref = n.tag, n.get("data-ref")
        if t not in REF_TAGS or hid(n):
            continue
        if t == "img":
            alt = n.get("alt")
            named = (alt is not None or str(n.get("role", "")).strip().lower() in ("presentation", "none")
                     or ws(n.get("aria-label", "")) or lb(n) or ws(n.get("title", "")))
            if not named or (alt is not None and FILE_RX.fullmatch(alt.strip()) and not o.get("file_alt_ok")):
                out["img_alt"].append(ref)
        elif t in ("input", "select", "textarea"):
            typ = str(n.get("type", "text")).strip().lower() if t == "input" else ""
            if typ in ("hidden", "submit", "reset"):
                continue
            named = (ws(n.get("aria-label", "")) or lb(n) or pg.label_text(n, o.get("no_wrap")) or ws(n.get("title", ""))
                     or (o.get("placeholder") and ws(n.get("placeholder", "")))
                     or (typ == "button" and ws(n.get("value", ""))) or (typ == "image" and ws(n.get("alt", ""))))
            if not named:
                out["control_name"].append(ref)
        elif t in ("a", "button"):
            if t == "a" and "href" not in n.attrs:
                continue
            name = ws(n.get("aria-label", "")) or lb(n) or text_of(n) or ws(n.get("title", ""))
            if o.get("content_first"):
                name = text_of(n) or ws(n.get("aria-label", "")) or lb(n) or ws(n.get("title", ""))
            if not name or is_generic(name):
                out["link_name"].append(ref)
        elif t in ("h1", "h2", "h3", "h4", "h5", "h6"):
            lv = int(t[1])
            if lv > prev + 1:
                out["heading_skip"].append(ref)
            prev = lv
        elif t == "table":
            if str(n.get("role", "")).strip().lower() in ("presentation", "none"):
                continue
            if not _has_own_th(n):
                out["table_header"].append(ref)
    for f in FIELDS:
        if f not in ("lang_ok", "duplicate_id"):
            out[f] = sorted(out[f], key=lambda r: int(r[1:]))
    return out


def _has_own_th(table):
    def walk(x):
        for c in x.children:
            if isinstance(c, Node):
                if c.tag == "th":
                    return True
                if c.tag != "table" and walk(c):
                    return True
        return False
    return walk(table)


def assign_refs(root):
    k = 0
    for n in root.iter():
        if n.tag in REF_TAGS:
            k += 1
            n.attrs["data-ref"] = f"r{k}"
    return k
