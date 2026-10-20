"""Baselines for type 18. Each takes a generated task and rewrites the unit files in its world."""
import ast
import json
import re
from pathlib import Path

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("b18_type", Path(__file__).with_name("type.py"))
T = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(T)
eng, R = T.eng, T.R


def units(task):
    return json.loads((Path(task["gold_path"]) / "meta.json").read_text())["units"]


def oracle(task):
    eng.oracle_fill(task)


def generic_helper(task):
    """Right places, wrong helper: every removed escape comes back as html.escape(...) (quote=True)."""
    cat = {u["id"]: u for u in T.catalog()["units"]}
    w, g = Path(task["world_dir"]), Path(task["gold_path"]) / "base"
    by_file = {}
    for u in units(task):
        by_file.setdefault(u["files"][0], []).extend(cat[u["id"]]["edits"])
    for rel, eds in by_file.items():
        src = (g / rel).read_bytes()
        new = [(s, e, "__import__('html').escape(str(" + rep[1:-1] + "))") for s, e, rep, *_ in eds]
        (w / rel).write_text(R.apply(src, new))


def escape_all(task):
    """Escape every non-literal value that is concatenated or formatted into a string in the unit functions."""
    w = Path(task["world_dir"])
    for u in units(task):
        p = w / u["files"][0]
        src = p.read_text()
        tree = ast.parse(src)
        _, funcs = R.qualnames(tree)
        fn = funcs[u["qualname"]]
        starts = R.offsets(src.encode())
        spans = []
        for n in ast.walk(fn):
            if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
                for side in (n.left, n.right):
                    if isinstance(side, (ast.Name, ast.Attribute, ast.Subscript, ast.Call)) and not (
                            isinstance(side, ast.Call) and R.call_name(side) in R.NAMES):
                        spans.append(side)
            if isinstance(n, ast.FormattedValue) and not isinstance(n.value, ast.Constant):
                spans.append(n.value)
        b = src.encode()
        edits, last = [], None
        for side in sorted(spans, key=lambda x: (x.lineno, x.col_offset), reverse=True):
            s = starts[side.lineno - 1] + side.col_offset
            e = starts[side.end_lineno - 1] + side.end_col_offset
            if last is not None and e > last:
                continue
            edits.append((s, e, "__import__('html').escape(str(" + b[s:e].decode() + "))"))
            last = s
        p.write_text(R.apply(b, edits))


def escape_return(task):
    """Escape late: wrap every returned value of the unit functions in html.escape (double-escapes markup)."""
    w = Path(task["world_dir"])
    for u in units(task):
        p = w / u["files"][0]
        src = p.read_text()
        _, funcs = R.qualnames(ast.parse(src))
        b = src.encode()
        starts = R.offsets(b)
        edits = []
        for n in ast.walk(funcs[u["qualname"]]):
            if isinstance(n, ast.Return) and n.value is not None:
                v = n.value
                s0, e0 = starts[v.lineno - 1] + v.col_offset, starts[v.end_lineno - 1] + v.end_col_offset
                edits.append((s0, e0, "__import__('html').escape(str(" + b[s0:e0].decode() + "))"))
        p.write_text(R.apply(b, edits))


BASELINES = {"oracle": oracle, "generic_helper": generic_helper, "escape_all": escape_all,
             "escape_return": escape_return}
