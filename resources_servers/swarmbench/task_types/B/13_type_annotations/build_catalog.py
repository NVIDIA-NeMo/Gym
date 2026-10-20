"""Build-time: catalog for type 13 (restore type annotations so mypy --strict passes again).

Oracle = real, strictly typed modules of click 8.1.7, itsdangerous 2.2.0, jinja2 3.1.4 and markupsafe 2.1.5.
"Before" = the module with every function-signature annotation and every annotation inside function bodies and at
module level removed (class-level annotations stay: dataclass/NamedTuple fields depend on them at run time).
A module is a unit if stripping removes >= 3 annotations, the real test suite still passes with the stripped module
(no run-time effect), and mypy reports new errors.

Usage: python build_catalog.py BUILD_DIR
"""
import ast
import collections
import io
import json
import os
import re
import shutil
import sys
import tokenize
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
B = HERE.parent
sys.path.insert(0, str(B))
import swb_b_engine as eng  # noqa: E402

REPOS = B / "repos"
# repo -> (tarball, source prefix in tarball, package, extra sys.path entries for its tests and mypy)
SPECS = {
    "click": ("click-8.1.7.tar.gz", "src/", "click", []),
    "itsdangerous": ("itsdangerous-2.2.0.tar.gz", "src/", "itsdangerous", []),
    "jinja": ("jinja-3.1.4.tar.gz", "src/", "jinja2", ["repos/markupsafe/src"]),
    "markupsafe": ("markupsafe-2.1.5.tar.gz", "src/", "markupsafe", []),
}
MYPY_ARGS = ["--strict", "--no-warn-unused-ignores", "--python-version", "3.12", "--no-error-summary",
             "--show-error-codes", "--hide-error-context", "--no-pretty"]
MIN_GROUPS = 3


def meta():
    paths = {r: [f"repos/{r}/src"] + extra for r, (_, _, _, extra) in SPECS.items()}
    return {"paths": paths, "cwd": {r: f"repos/{r}" for r in SPECS}}


def extract_tree(dest):
    for repo, (tar, pre, pkg, _) in SPECS.items():
        eng.extract(REPOS / tar, dest / "repos" / repo, [pre + pkg + "/", "LICENSE.rst", "LICENSE.txt"])
        for p in (dest / "repos" / repo / "src" / pkg).rglob("*.c"):
            p.unlink()  # C speedups are never built; the pure-Python fallbacks are used


MYPY_TEST = '''import collections
import json
import os
import os
import re
import tempfile

import pytest
from mypy import api

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(open(os.path.join(HERE, "mypy.json")))
LINE = re.compile(r"^(?P<file>[^:]+):\\d+(?::\\d+)?: error: (?P<msg>.*?)\\s+\\[(?P<code>[\\w-]+)\\]$")


def errors():
    os.environ["MYPYPATH"] = os.pathsep.join(os.path.join(HERE, "..", "..", "..", p) for p in DATA["mypy_path"])
    with tempfile.TemporaryDirectory() as cache:
        out, err, rc = api.run(DATA["args"] + ["--cache-dir", cache, "-p", DATA["package"]])
    found = []
    for line in out.splitlines():
        m = LINE.match(line.strip())
        if m:
            found.append(((os.path.normpath(m["file"]).split("src" + os.sep)[-1], m["code"], m["msg"]), line.strip()))
    return found


@pytest.mark.parametrize("unit", DATA["units"])
def test_mypy_strict(unit):
    found = errors()
    new = collections.Counter(k for k, _ in found) - collections.Counter(map(tuple, DATA["baseline"]))
    lines = []
    for key, line in found:
        if new[key] > 0:
            new[key] -= 1
            lines.append(line)
    assert not lines, f"mypy --strict reports {len(lines)} new errors:\\n" + "\\n".join(lines[:60])
'''


def mypy_errors(root, repo):
    """Runs the generated mypy test machinery directly (oracle baseline)."""
    code = f"""
import json, os, re, sys, tempfile
sys.path.insert(0, {str(B / '_vendor')!r})
from mypy import api
os.environ["MYPYPATH"] = {os.pathsep.join(str(root / p) for p in meta()['paths'][repo])!r}
os.chdir({str(root / 'repos' / repo)!r})
with tempfile.TemporaryDirectory() as c:
    out, err, rc = api.run({MYPY_ARGS!r} + ["--cache-dir", c, "-p", {SPECS[repo][2]!r}])
print(out)
"""
    import subprocess
    r = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", code], capture_output=True, text=True, timeout=900)
    found = []
    for line in r.stdout.splitlines():
        m = re.match(r"^(?P<file>[^:]+):\d+(?::\d+)?: error: (?P<msg>.*?)\s+\[(?P<code>[\w-]+)\]$", line.strip())
        if m:
            found.append([os.path.normpath(m["file"]).split("src" + os.sep)[-1], m["code"], m["msg"]])
    return found


# ---------- stripping ----------

def offsets(src):
    starts, pos = [0], 0
    for line in src.split(b"\n"):
        pos += len(line) + 1
        starts.append(pos)
    return starts


def pos(starts, line, col):
    return starts[line - 1] + col


def arrow_start(src, starts, fn):
    """Byte offset of the '->' that precedes the return annotation of fn."""
    end = pos(starts, fn.returns.lineno, fn.returns.col_offset)
    i = src.rfind(b"->", 0, end)
    while i > 0 and src[i - 1:i] in (b" ", b"\t"):
        i -= 1
    return i


def strip_edits(src):
    """Groups {name: [(start, end, replacement)]} that remove annotations. One group per top-level function or
    method (nested functions and annotated assignments in its body included). Class bodies keep their
    annotations, module-level annotated assignments are kept too (stripping them is often not a mypy error)."""
    tree = ast.parse(src)
    starts = offsets(src)
    groups = {}

    def fn_edits(fn, out):
        a = fn.args
        for arg in a.posonlyargs + a.args + a.kwonlyargs + ([a.vararg] if a.vararg else []) + ([a.kwarg] if a.kwarg else []):
            if arg.annotation is not None:
                s0 = pos(starts, arg.lineno, arg.col_offset) + len(arg.arg.encode())
                out.append((s0, pos(starts, arg.annotation.end_lineno, arg.annotation.end_col_offset), b""))
        if fn.returns is not None:
            out.append((arrow_start(src, starts, fn), pos(starts, fn.returns.end_lineno, fn.returns.end_col_offset), b""))
        body(fn, out)

    def body(node, out):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn_edits(child, out)
            elif isinstance(child, ast.ClassDef):
                continue
            elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                s0 = pos(starts, child.target.end_lineno, child.target.end_col_offset)
                if child.value is not None:
                    out.append((s0, pos(starts, child.value.lineno, child.value.col_offset), b" = "))
                else:
                    out.append((pos(starts, child.lineno, child.col_offset),
                                pos(starts, child.end_lineno, child.end_col_offset), b"pass"))
            else:
                body(child, out)

    def top(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out = []
                fn_edits(child, out)
                # overloads and redefinitions share a name: keep them in one group
                if out:
                    groups.setdefault(prefix + child.name, []).extend(out)
            elif isinstance(child, ast.ClassDef):
                top(child, prefix + child.name + ".")
            elif isinstance(child, (ast.If, ast.Try)):
                top(child, prefix)

    top(tree, "")
    return {k: sorted(v) for k, v in groups.items()}


def apply(src, edits):
    out = src
    for s, e, r in sorted(edits, reverse=True):
        out = out[:s] + r + out[e:]
    return out


def counts(text):
    code = "\n".join(line.split("#")[0] if "type:" not in line else line for line in text.splitlines())
    return {"Any": len(re.findall(r"\bAny\b", code)), "ignore": len(re.findall(r"#\s*type:\s*ignore", text)),
            "cast": len(re.findall(r"\bcast\(", code))}


def sandbox(build, name, overlay):
    d = build / "sb" / name
    if d.exists():
        shutil.rmtree(d)
    eng.copytree(build / "base", d)
    eng.copytree(build / "tests", d)
    for rel, text in overlay.items():
        (d / rel).write_bytes(text)
    return d


def main(build):
    build = Path(build).resolve()
    base, tests = build / "base", build / "tests"
    if not base.exists():
        extract_tree(base)
        for repo, (tar, pre, pkg, _) in SPECS.items():
            eng.extract(REPOS / tar, tests / "repos" / repo, ["tests/"])
    m = meta()
    baseline, passing = {}, {}
    root = sandbox(build, "oracle", {})
    for repo in SPECS:
        baseline[repo] = mypy_errors(root, repo)
        runs = [eng.run_pytest(str(root), ["tests"], m["paths"][repo], cwd=m["cwd"][repo], timeout=900) for _ in range(2)]
        passing[repo] = sorted(t for t, o in runs[0]["outcomes"].items() if o == "passed" and runs[1]["outcomes"].get(t) == "passed")
        print(repo, len(baseline[repo]), "oracle mypy errors;", len(passing[repo]), "passing tests")
    cands = []
    for repo, (_, _, pkg, _) in SPECS.items():
        for p in sorted((base / "repos" / repo / "src" / pkg).rglob("*.py")):
            rel = str(p.relative_to(base))
            src = p.read_bytes()
            groups = strip_edits(src)
            if len(groups) >= MIN_GROUPS:
                cands.append((repo, rel, groups))
    print(len(cands), "candidates")

    def check(c):
        repo, rel, groups = c
        before = apply((base / rel).read_bytes(), [e for g in groups.values() for e in g])
        compile(before, rel, "exec")
        root = sandbox(build, "iso-" + rel.replace("/", "_"), {rel: before})
        res = eng.run_pytest(str(root), ["tests"], m["paths"][repo], cwd=m["cwd"][repo], timeout=900)
        broken = [t for t in passing[repo] if res["outcomes"].get(t) != "passed"]
        errs = mypy_errors(root, repo)
        shutil.rmtree(root)
        new = collections.Counter(map(tuple, errs)) - collections.Counter(map(tuple, baseline[repo]))
        return repo, rel, groups, broken, sum(new.values())

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(check, cands))
    units = []
    for repo, rel, groups, broken, new in results:
        ok = not broken and new > 0
        n_ann = sum(len(g) for g in groups.values())
        print(f"{rel}: {len(groups)} functions, {n_ann} annotations stripped, {new} new mypy errors, "
              f"{len(broken)} tests broken", "" if ok else "DROP")
        if ok:
            oracle_text = (base / rel).read_text()
            units.append({"id": rel, "repo": repo, "counts": counts(oracle_text), "lines": oracle_text.count("\n"),
                          "groups": {k: [[s, e, r.decode()] for s, e, r in v] for k, v in groups.items()}})
    catalog = {"units": units, "passing": passing, "baseline": baseline, "meta": m, "mypy_args": MYPY_ARGS,
               "mypy_test": MYPY_TEST, "packages": {r: s[2] for r, s in SPECS.items()}}
    (HERE / "catalog.json").write_text(json.dumps(catalog, indent=1))
    print(len(units), "units")


if __name__ == "__main__":
    main(sys.argv[1])
