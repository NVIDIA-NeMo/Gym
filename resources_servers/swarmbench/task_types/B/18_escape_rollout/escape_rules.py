"""Type 18 rules: escaping-call sites, their removal, the static count, and function splicing."""
import ast

NAMES = {"escape", "escape_text", "escape_url", "safe_entity", "html_escape", "_escape", "code_escape",
         "_escape_attrib", "_escape_cdata"}


def offsets(src):
    starts, pos = [0], 0
    for line in src.split(b"\n"):
        pos += len(line) + 1
        starts.append(pos)
    return starts


def qualnames(tree):
    """({id(node): innermost function qualname or None}, {qualname: function node}) in co_qualname style."""
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


def call_name(n):
    f = n.func
    return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None


def sites(src_bytes):
    """[(start, end, replacement, line, qualname)] for escaping calls inside functions: replacement = the call's
    first argument (parenthesized), i.e. the unescaped text."""
    tree = ast.parse(src_bytes)
    owner, _ = qualnames(tree)
    starts = offsets(src_bytes)
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and call_name(n) in NAMES and n.args and owner.get(id(n)):
            a = n.args[0]
            s = starts[n.lineno - 1] + n.col_offset
            e = starts[n.end_lineno - 1] + n.end_col_offset
            arg = src_bytes[starts[a.lineno - 1] + a.col_offset:starts[a.end_lineno - 1] + a.end_col_offset].decode()
            out.append((s, e, f"({arg})", n.lineno, owner[id(n)]))
    # nested escaping calls: keep only the outermost
    out.sort(key=lambda x: (x[0], -x[1]))
    kept, end = [], -1
    for x in out:
        if x[0] >= end:
            kept.append(x)
            end = x[1]
    return kept


def apply(src_bytes, edits):
    out = src_bytes
    for s, e, rep, *_ in sorted(edits, reverse=True):
        out = out[:s] + rep.encode() + out[e:]
    return out.decode()


def escape_count(fn):
    return sum(1 for n in ast.walk(fn) if isinstance(n, ast.Call) and call_name(n) in NAMES)


def splice(oracle_text, agent_text, qualname):
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
    if len(alines[a0]) - len(alines[a0].lstrip()) != len(olines[o0]) - len(olines[o0].lstrip()):
        return None
    return "\n".join(olines[:o0] + alines[a0:a.end_lineno] + olines[o.end_lineno:])
