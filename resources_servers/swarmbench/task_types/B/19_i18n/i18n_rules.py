"""Type 19 rules: which calls are translation calls, the inverse (oracle -> hard-coded strings), the static signature
of a function, and splicing one function of the agent's file into the oracle file."""
import ast
import re
import string

KINDS = {"_": "gettext", "gettext": "gettext", "_gettext": "gettext", "ngettext": "ngettext", "_ngettext": "ngettext",
         "P_": "pgettext", "_pgettext": "pgettext", "pgettext": "pgettext", "NS_": "ngettext_noop",
         "_ngettext_noop": "ngettext_noop"}


def offsets(src):
    starts, pos = [0], 0
    for line in src.split(b"\n"):
        pos += len(line) + 1
        starts.append(pos)
    return starts


def span(starts, node):
    return starts[node.lineno - 1] + node.col_offset, starts[node.end_lineno - 1] + node.end_col_offset


def kind_of(call):
    return KINDS.get(call.func.id) if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) else None


def literal_args(call):
    if call.keywords or not call.args or not all(isinstance(a, ast.Constant) and isinstance(a.value, str) for a in call.args):
        return None
    return [a.value for a in call.args]


def qualnames(tree):
    """{id(node): qualname of the innermost enclosing function in co_qualname style, or None at module/class level},
    plus {qualname: function node}."""
    owner, funcs = {}, {}

    def visit(node, prefix, fn_qual):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                q = prefix + child.name
                funcs[q] = child
                visit(child, q + ".<locals>.", q)
            elif isinstance(child, ast.ClassDef):
                visit(child, prefix + child.name + ".", None)
            else:
                owner[id(child)] = fn_qual
                visit(child, prefix, fn_qual)

    visit(tree, "", None)
    return owner, funcs


def signature(fn):
    """Sorted list of [kind, args...] for translation calls with literal arguments directly in fn (not nested defs)."""
    out = []

    def visit(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            k = kind_of(child)
            if k:
                args = literal_args(child)
                if k == "ngettext" and len(child.args) == 3 and not child.keywords and all(
                        isinstance(a, ast.Constant) and isinstance(a.value, str) for a in child.args[:2]):
                    args = [child.args[0].value, child.args[1].value]  # the count is not part of the msgid
                if args is not None:
                    out.append([k] + args)
            visit(child)

    visit(fn)
    return sorted(out)


def has_nested_sites(fn):
    for n in ast.walk(fn):
        if n is not fn and isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and signature(n):
            return True
    return False


def esc(text):
    out = text.replace("\\", "\\\\").replace('"', '\\"').replace("{", "{{").replace("}", "}}")
    return re.sub(r"[\x00-\x1f]", lambda m: {"\n": "\\n", "\t": "\\t", "\r": "\\r"}.get(m[0], "\\x%02x" % ord(m[0])), out)


def to_fstring(template, kw):
    """f-string source equivalent to template.format(**kw) (kw: name -> (expr source, is simple name)), or None."""
    parts, used = [], set()
    try:
        parsed = list(string.Formatter().parse(template))
    except ValueError:
        return None
    for lit, field, spec, conv in parsed:
        parts.append(esc(lit))
        if field is None:
            continue
        m = re.match(r"[A-Za-z_]\w*", field)
        if not m or (spec and "{" in spec):
            return None
        base, rest = m.group(0), field[m.end():]
        if base not in kw:
            return None
        used.add(base)
        expr, simple = kw[base]
        if "\n" in expr:
            return None
        if rest and not simple:
            expr = f"({expr})"
        parts.append("{" + expr + rest + (f"!{conv}" if conv else "") + (f":{spec}" if spec else "") + "}")
    if used != set(kw):
        return None
    return 'f"' + "".join(parts) + '"'


def inverse_edits(src_bytes):
    """[(start, end, replacement, qualname)] that turn translation calls into hard-coded strings. Only sites in
    functions (not module or class level) whose function has no nested definitions with sites are rewritten."""
    tree = ast.parse(src_bytes)
    starts = offsets(src_bytes)
    owner, funcs = qualnames(tree)
    text = src_bytes.decode()
    skip_fn = {q for q, fn in funcs.items() if has_nested_sites(fn)}
    edits, covered = [], []

    def seg(node):
        s, e = span(starts, node)
        return src_bytes[s:e].decode()

    def inside(s, e):
        return any(a <= s and e <= b for a, b in covered)

    nodes = sorted((n for n in ast.walk(tree) if isinstance(n, ast.Call)), key=lambda n: (span(starts, n)[0], -span(starts, n)[1]))
    for n in nodes:
        q = owner.get(id(n))
        if q is None or q in skip_fn:
            continue
        s, e = span(starts, n)
        if inside(s, e):
            continue
        # X(...).format(**kw)
        fmt_kw = None
        inner = n
        if isinstance(n.func, ast.Attribute) and n.func.attr == "format" and isinstance(n.func.value, ast.Call):
            inner = n.func.value
            if n.args or any(k.arg is None for k in n.keywords):
                continue
            fmt_kw = {k.arg: (seg(k.value), isinstance(k.value, ast.Name)) for k in n.keywords}
        k = kind_of(inner)
        if not k:
            continue
        args = literal_args(inner)
        if args is None:
            continue
        if k == "gettext" and len(args) == 1:
            rep = seg(inner.args[0]) if fmt_kw is None else to_fstring(args[0], fmt_kw)
        elif k == "pgettext" and len(args) == 2 and fmt_kw is None:
            rep = seg(inner.args[1])
        elif k == "ngettext_noop" and len(args) == 2 and fmt_kw is None:
            rep = f"({seg(inner.args[0])}, {seg(inner.args[1])})"
        else:
            continue
        if rep is None:
            continue
        edits.append((s, e, rep, q))
        covered.append((s, e))
    # ngettext(singular, plural, count) [.format(**kw)]
    for n in nodes:
        q = owner.get(id(n))
        if q is None or q in skip_fn:
            continue
        s, e = span(starts, n)
        if inside(s, e):
            continue
        fmt_kw, inner = None, n
        if isinstance(n.func, ast.Attribute) and n.func.attr == "format" and isinstance(n.func.value, ast.Call):
            inner = n.func.value
            if n.args or any(k.arg is None for k in n.keywords):
                continue
            fmt_kw = {k.arg: (seg(k.value), isinstance(k.value, ast.Name)) for k in n.keywords}
        if kind_of(inner) != "ngettext" or inner.keywords or len(inner.args) != 3:
            continue
        sg, pl, cnt = inner.args
        if not all(isinstance(a, ast.Constant) and isinstance(a.value, str) for a in (sg, pl)):
            continue
        if "\n" in seg(cnt):
            continue
        if fmt_kw is None:
            a, b = seg(sg), seg(pl)
            if "\n" in a or "\n" in b:
                a, b = repr(sg.value), repr(pl.value)
        else:
            a, b = to_fstring(sg.value, fmt_kw), to_fstring(pl.value, fmt_kw)
            if a is None or b is None:
                continue
        edits.append((s, e, f"({a} if {seg(cnt)} == 1 else {b})", q))
        covered.append((s, e))
    return sorted(edits)


def apply(src_bytes, edits):
    out = src_bytes
    for s, e, rep, _ in sorted(edits, reverse=True):
        out = out[:s] + rep.encode() + out[e:]
    return out.decode()


def splice(oracle_text, agent_text, qualname):
    """The oracle file with the function `qualname` replaced by the agent's version (decorators included), or None
    if the agent's file has no such function or does not parse."""
    try:
        at = ast.parse(agent_text)
    except SyntaxError:
        return None
    _, afuncs = qualnames(at)
    _, ofuncs = qualnames(ast.parse(oracle_text))
    if qualname not in afuncs or qualname not in ofuncs:
        return None
    a, o = afuncs[qualname], ofuncs[qualname]
    alines, olines = agent_text.split("\n"), oracle_text.split("\n")
    a0 = min([a.lineno] + [d.lineno for d in a.decorator_list]) - 1
    o0 = min([o.lineno] + [d.lineno for d in o.decorator_list]) - 1
    ind_a = len(alines[a0]) - len(alines[a0].lstrip())
    ind_o = len(olines[o0]) - len(olines[o0].lstrip())
    if ind_a != ind_o:
        return None
    return "\n".join(olines[:o0] + alines[a0:a.end_lineno] + olines[o.end_lineno:])
