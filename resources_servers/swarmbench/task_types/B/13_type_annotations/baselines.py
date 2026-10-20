"""Baselines for type 13. Each takes a generated task and rewrites the unit files in its world."""
import ast
import json
from pathlib import Path

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("b13_type", Path(__file__).with_name("type.py"))
T = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(T)
eng = T.eng


def units(task):
    return json.loads((Path(task["gold_path"]) / "meta.json").read_text())["units"]


def oracle(task):
    eng.oracle_fill(task)


def unannotated(tree):
    """Function nodes with no annotation at all (the stripped ones)."""
    out = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.returns is None and \
                all(a.annotation is None for a in n.args.args + n.args.kwonlyargs + n.args.posonlyargs):
            out.append(n)
    return out


def annotate(src, param, ret):
    """Text-insert annotations into every unannotated def. param(arg, default) / ret(fn) return a string or None."""
    tree = ast.parse(src)
    lines = src.split("\n")
    inserts = []  # (line, col, text)
    for fn in unannotated(tree):
        a = fn.args
        pos = a.posonlyargs + a.args
        defaults = [None] * (len(pos) - len(a.defaults)) + list(a.defaults)
        pairs = list(zip(pos, defaults)) + list(zip(a.kwonlyargs, a.kw_defaults))
        for arg, d in pairs + [(x, None) for x in (a.vararg, a.kwarg) if x]:
            if arg.arg in ("self", "cls"):
                continue
            t = param(arg, d)
            if t:
                inserts.append((arg.end_lineno, arg.end_col_offset, f": {t}"))
        r = ret(fn)
        if r:
            # insert before the ':' that ends the signature (the line before the first body statement)
            body0 = fn.body[0]
            ln = body0.lineno - 1
            while ln >= fn.lineno:
                line = lines[ln - 1]
                idx = line.rstrip().rfind(":")
                if idx >= 0 and line.rstrip().endswith(":"):
                    inserts.append((ln, idx, f" -> {r}"))
                    break
                ln -= 1
    for ln, col, text in sorted(inserts, reverse=True):
        line = lines[ln - 1]
        b = line.encode()
        lines[ln - 1] = (b[:col] + text.encode() + b[col:]).decode()
    return "\n".join(lines)


def returns_value(fn):
    return any(isinstance(n, ast.Return) and n.value is not None for n in ast.walk(fn)) or \
        any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(fn))


def any_all(task):
    """Everything typed as Any (blocked by the Any cap)."""
    w = Path(task["world_dir"])
    for u in units(task):
        p = w / u["id"]
        src = annotate(p.read_text(), lambda a, d: "Any", lambda fn: "Any")
        p.write_text("from typing import Any\n" + src)


def heuristic(task):
    """No Any: literal defaults give their type, -> None when nothing is returned, object otherwise."""
    w = Path(task["world_dir"])
    names = {bool: "bool", int: "int", str: "str", float: "float", bytes: "bytes"}

    def param(a, d):
        if isinstance(d, ast.Constant):
            return names.get(type(d.value), "object") if d.value is not None else "object | None"
        return "object"

    for u in units(task):
        p = w / u["id"]
        p.write_text(annotate(p.read_text(), param, lambda fn: "object" if returns_value(fn) else "None"))


def partial(task):
    """Oracle annotations back for half of the stripped functions only."""
    cat = {u["id"]: u for u in T.catalog()["units"]}
    for u in units(task):
        g = cat[u["id"]]["groups"]
        keep_stripped = u["functions"][: (len(u["functions"]) + 1) // 2]
        base = (Path(task["gold_path"]) / "base" / u["id"]).read_bytes()
        (Path(task["world_dir"]) / u["id"]).write_text(T.apply(base, [e for k in keep_stripped for e in g[k]]))


def stub(task):
    """Oracle annotations, but the stripped functions' bodies replaced by raise NotImplementedError."""
    eng.oracle_fill(task)
    w = Path(task["world_dir"])
    for u in units(task):
        p = w / u["id"]
        src = p.read_text()
        tree = ast.parse(src)
        lines = src.split("\n")
        targets = []
        for n in ast.walk(tree):
            if isinstance(n, ast.ClassDef):
                for m in n.body:
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        targets.append((f"{n.name}.{m.name}", m))
            if isinstance(n, ast.Module):
                for m in n.body:
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        targets.append((m.name, m))
        for name, fn in sorted(targets, key=lambda x: -x[1].lineno):
            if name not in u["functions"] or any(isinstance(d, ast.Name) and d.id == "overload" or
                                                 isinstance(d, ast.Attribute) and d.attr == "overload"
                                                 for d in fn.decorator_list):
                continue
            first, last = fn.body[0], fn.body[-1]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and len(fn.body) > 1:
                first = fn.body[1]
            ind = " " * first.col_offset
            lines[first.lineno - 1:last.end_lineno] = [ind + "raise NotImplementedError"]
        p.write_text("\n".join(lines))


BASELINES = {"oracle": oracle, "any": any_all, "heuristic": heuristic, "partial": partial, "stub": stub}
