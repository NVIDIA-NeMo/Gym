"""Build-time: catalog of units for type 11 (Markdown 3.6 deprecated-API removal, real extension code).

Oracle = the real code of Markdown 3.6's bundled extensions and PyMdown Extensions 10.8.1 (both use the new API).
"Before" = the same modules rewritten to the API that Markdown 3.4 removed (Registry.add with locations,
reg[name] = item, del reg[name], extendMarkdown(md, md_globals), util.etree, util.isBlockLevel, Processor.markdown).
Locations are derived from the registry order recorded while the real test suites run, and every generated legacy
module is checked by running the real tests against Markdown 3.6 with the 3.3.7 legacy API re-added (recorder.py
compat mode): outcomes and registry orders must match the oracle's.

Usage: python build_catalog.py BUILD_DIR   (writes catalog.json next to this file)
"""
import ast
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
B = HERE.parent
sys.path.insert(0, str(B))
import swb_b_engine as eng  # noqa: E402

REPOS = B / "repos"
PY = sys.executable
REG_ATTRS = {"preprocessors", "blockprocessors", "inlinePatterns", "treeprocessors", "postprocessors", "patterns"}
META = {"paths": {"markdown": ["repos/markdown"], "pymdownx": ["repos/pymdownx", "repos/markdown"]},
        "cwd": {"markdown": "repos/markdown", "pymdownx": "repos/pymdownx"}}
ENTRY_POINTS = None  # filled from pyproject


def extract_all(build):
    base, tests = build / "base", build / "tests"
    if base.exists():
        return base, tests
    md = REPOS / "markdown-3.6.tar.gz"
    px = REPOS / "pymdownx-10.8.1.tar.gz"
    eng.extract(md, base / "repos/markdown", ["markdown/", "docs/changelog.md", "LICENSE.md"])
    eng.extract(px, base / "repos/pymdownx", ["pymdownx/", "LICENSE.md"])
    eng.extract(md, tests / "repos/markdown", ["tests/"])
    eng.extract(px, tests / "repos/pymdownx", ["tests/"])
    eng.sort_pymdownx_walk(tests / "repos/pymdownx")
    # entry points (short extension names such as 'toc') come from the installed distribution's metadata
    import tomllib
    import tarfile
    with tarfile.open(md) as tf:
        name = next(m for m in tf.getnames() if m.endswith("/pyproject.toml") and m.count("/") == 1)
        pp = tomllib.loads(tf.extractfile(name).read().decode())
    eps = pp["project"]["entry-points"]
    dist = tests / "repos/markdown/Markdown-3.6.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: Markdown\nVersion: 3.6\n")
    lines = []
    for group, items in eps.items():
        lines.append(f"[{group}]")
        lines += [f"{k} = {v}" for k, v in items.items()]
        lines.append("")
    (dist / "entry_points.txt").write_text("\n".join(lines))
    return base, tests


def sandbox(build, name, overlay):
    """build/sb/<name>: base + tests + overlay {rel: text}."""
    d = build / "sb" / name
    if d.exists():
        shutil.rmtree(d)
    eng.copytree(build / "base", d)
    eng.copytree(build / "tests", d)
    for rel, text in overlay.items():
        (d / rel).write_text(text)
    return d


def all_ids(repo):
    return ["tests"]


def record(root, mode, unit_files, repo):
    out = root / f"rec-{repo}-{mode}.json"
    job = {"ids": ["tests"], "paths": [str(root / p) for p in META["paths"][repo]],
           "unit_files": [str(root / f) for f in unit_files]}
    subprocess.run([PY, "-I", "-S", "-B", str(HERE / "recorder.py"), str(out), str(root / META["cwd"][repo]), mode],
                   input=json.dumps(job).encode(), check=True, timeout=900)
    return json.loads(out.read_text())


def candidates(base):
    out = []
    for p in sorted((base / "repos/markdown/markdown/extensions").glob("*.py")):
        if p.name != "__init__.py":
            out.append(str(p.relative_to(base)))
    for p in sorted((base / "repos/pymdownx/pymdownx").rglob("*.py")):
        if p.name not in ("__init__.py", "__meta__.py") or p.parent.name == "blocks":
            out.append(str(p.relative_to(base)))
    return out


def module_name(rel):
    parts = rel.split("/")[2:]
    mod = ".".join(parts)[:-3]
    return mod[:-9] if mod.endswith(".__init__") else mod


def processor_classes(base, rel):
    """Names of classes in the module that derive from markdown's Processor or Pattern (they had .markdown)."""
    code = f"""
import importlib, inspect, json, sys
sys.path[:0] = {[str(base / 'repos/pymdownx'), str(base / 'repos/markdown'), str(B / '_vendor')]!r}
import markdown.util as u, markdown.inlinepatterns as ip
m = importlib.import_module({module_name(rel)!r})
print(json.dumps(sorted(n for n, c in vars(m).items() if inspect.isclass(c) and c.__module__ == m.__name__
                  and issubclass(c, (u.Processor, ip.Pattern)))))
"""
    r = subprocess.run([PY, "-I", "-S", "-B", "-c", code], capture_output=True, text=True, timeout=120)
    return set(json.loads(r.stdout)) if r.returncode == 0 else set()


# ---------- legacy rewrite ----------

def byte_offsets(src_bytes):
    starts, pos = [0], 0
    for line in src_bytes.split(b"\n"):
        pos += len(line) + 1
        starts.append(pos)
    return starts


def seg(src_bytes, starts, node):
    return src_bytes[starts[node.lineno - 1] + node.col_offset: starts[node.end_lineno - 1] + node.end_col_offset].decode()


def span(starts, node):
    return starts[node.lineno - 1] + node.col_offset, starts[node.end_lineno - 1] + node.end_col_offset


def call_args(call, names):
    """Positional/keyword arguments of a call mapped to `names` (None if the call shape is unusual)."""
    if any(isinstance(a, ast.Starred) for a in call.args) or any(k.arg is None for k in call.keywords):
        return None
    out = dict(zip(names, call.args))
    for k in call.keywords:
        if k.arg not in names or k.arg in out:
            return None
        out[k.arg] = k.value
    return out


def locations(op):
    """Legacy locations that put the item where register(item, name, priority) puts it, given the order before."""
    before = [tuple(x) for x in op["before"]]
    names = [n for n, _ in before]
    kind = "add"
    if op["name"] in names:
        prio = dict(before)[op["name"]]
        if prio == op["priority"]:
            return "set", set()
        kind = "readd"
        before = [x for x in before if x[0] != op["name"]]
    k = sum(1 for _, p in before if p >= op["priority"])
    locs = set()
    if k < len(before):
        locs.add("<" + before[k][0])
    if k > 0:
        locs.add(">" + before[k - 1][0])
    if k == 0 and not before:
        locs.add("_begin")
    if k == len(before) and not before:
        locs.add("_end")
    return kind, locs


def legacy_edits(base, rel, ops):
    """Returns a list of edits {start, end, options: [text, ...], kind} (byte offsets) for one module."""
    src_bytes = (base / rel).read_bytes()
    starts = byte_offsets(src_bytes)
    tree = ast.parse(src_bytes)
    procs = processor_classes(base, rel)
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    by_line = {}
    for op in ops:
        by_line.setdefault(op["line"], []).append(op)
    edits = []

    def stmt_of(node):
        p = parents.get(node)
        return p if isinstance(p, ast.Expr) else None

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("register", "deregister"):
            lines = range(node.lineno, node.end_lineno + 1)
            site_ops = [op for ln in lines for op in by_line.get(ln, []) if op["op"] == node.func.attr]
            if not site_ops:
                continue
            recv = seg(src_bytes, starts, node.func.value)
            st = stmt_of(node)
            if node.func.attr == "register":
                a = call_args(node, ["item", "name", "priority"])
                if not a or len(a) != 3:
                    continue
                kinds, locs = set(), None
                for op in site_ops:
                    k, l = locations(op)
                    kinds.add(k)
                    locs = l if locs is None else locs & l
                if len(kinds) != 1:
                    continue
                kind = kinds.pop()
                item, name = seg(src_bytes, starts, a["item"]), seg(src_bytes, starts, a["name"])
                if kind == "set":
                    if st is None:
                        continue
                    s, e = span(starts, st)
                    edits.append({"start": s, "end": e, "kind": "reg-set", "options": [f"{recv}[{name}] = {item}"]})
                    continue
                if not locs:
                    continue
                opts = [f"{recv}.add({name}, {item}, {loc!r})" for loc in sorted(locs)]
                if kind == "readd":
                    if st is None:
                        continue
                    indent = " " * st.col_offset
                    opts = [f"del {recv}[{name}]\n{indent}{o}" for o in opts]
                    s, e = span(starts, st)
                else:
                    s, e = span(starts, node)
                edits.append({"start": s, "end": e, "kind": "reg-add", "options": opts})
            else:
                a = call_args(node, ["name", "strict"])
                if not a or st is None:
                    continue
                name = seg(src_bytes, starts, a["name"])
                strict = a.get("strict")
                s, e = span(starts, st)
                if strict is None or (isinstance(strict, ast.Constant) and strict.value is True):
                    text = f"del {recv}[{name}]"
                elif isinstance(strict, ast.Constant) and strict.value is False:
                    indent = " " * st.col_offset
                    text = f"if {name} in {recv}:\n{indent}    del {recv}[{name}]"
                else:
                    continue
                edits.append({"start": s, "end": e, "kind": "reg-del", "options": [text]})
        elif isinstance(node, ast.FunctionDef) and node.name == "extendMarkdown":
            args = node.args
            if len(args.args) == 2 and not args.vararg and not args.kwonlyargs and not args.kwarg and not args.defaults:
                md_arg = args.args[1]
                s, e = span(starts, md_arg)
                text = seg(src_bytes, starts, md_arg)
                edits.append({"start": s, "end": e, "kind": "md-globals", "options": [text + ", md_globals"]})
                for sub in ast.walk(node):
                    if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                            and sub.func.attr == "extendMarkdown" and len(sub.args) == 1 and not sub.keywords):
                        s2, e2 = span(starts, sub.args[0])
                        edits.append({"start": s2, "end": e2, "kind": "md-globals-call",
                                      "options": [seg(src_bytes, starts, sub.args[0]) + ", md_globals"]})
        elif isinstance(node, ast.Import) and len(node.names) == 1 and node.names[0].name == "xml.etree.ElementTree":
            alias = node.names[0].asname
            if not alias:
                continue
            src_mod = "..util" if rel.startswith("repos/markdown/") else "markdown.util"
            s, e = span(starts, node)
            text = f"from {src_mod} import etree" + ("" if alias == "etree" else f" as {alias}")
            edits.append({"start": s, "end": e, "kind": "etree", "options": [text]})
        elif isinstance(node, ast.ClassDef) and node.name in procs:
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Attribute) and sub.attr == "md" and isinstance(sub.ctx, ast.Load)
                        and isinstance(sub.value, ast.Name) and sub.value.id == "self"):
                    s, e = span(starts, sub)
                    edits.append({"start": s, "end": e, "kind": "self-markdown", "options": ["self.markdown"]})
                if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr == "is_block_level"
                        and len(sub.args) == 1 and has_util_import(tree)):
                    s, e = span(starts, sub)
                    edits.append({"start": s, "end": e, "kind": "isblocklevel",
                                  "options": [f"util.isBlockLevel({seg(src_bytes, starts, sub.args[0])})"]})
    # an is_block_level call contains a self.md attribute: keep the outer edit only
    edits.sort(key=lambda x: (x["start"], -x["end"]))
    out, last_end = [], -1
    for ed in edits:
        if ed["start"] >= last_end:
            out.append(ed)
            last_end = ed["end"]
    return out


def has_util_import(tree):
    for n in tree.body:
        if isinstance(n, ast.ImportFrom) and any(a.name == "util" and not a.asname for a in n.names):
            return True
    return False


def apply_edits(src_bytes, edits, choice):
    """choice(i, edit) -> option index. Edits must not overlap."""
    out = src_bytes
    for i, ed in sorted(enumerate(edits), key=lambda x: -x[1]["start"]):
        text = ed["options"][choice(i, ed)].encode()
        out = out[:ed["start"]] + text + out[ed["end"]:]
    return out.decode()


# ---------- main ----------

ORDER_TEST = """import json
import os

import markdown
import pytest

DATA = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "orders.json")))
REGS = ["preprocessors", "parser.blockprocessors", "inlinePatterns", "treeprocessors", "postprocessors"]


def snapshot(md):
    out = {}
    for r in REGS:
        obj = md
        for part in r.split("."):
            obj = getattr(obj, part)
        out[r] = [p.name for p in sorted(obj._priority, key=lambda p: p.priority, reverse=True)]
    return out


@pytest.mark.parametrize("unit", sorted(DATA))
def test_registry_order(unit):
    md = markdown.Markdown(extensions=[DATA[unit]["ext"]])
    assert snapshot(md) == DATA[unit]["orders"]
"""


def order_id(rel):
    return f"swb_hidden/test_registry_order.py::test_registry_order[{module_name(rel)}]"


def repo_of(rel):
    return rel.split("/")[1]


def standalone_orders(build, units):
    """Registry orders after loading each unit's extension alone (oracle code)."""
    data = {}
    for repo in ("markdown", "pymdownx"):
        code = f"""
import importlib, json, sys
sys.path[:0] = {[str(build / 'base' / p) for p in META['paths'][repo]] + [str(build / 'tests/repos/markdown'), str(B / '_vendor')]!r}
import markdown
REGS = ["preprocessors", "parser.blockprocessors", "inlinePatterns", "treeprocessors", "postprocessors"]
out = {{}}
for mod in {[module_name(r) for r in units if repo_of(r) == repo]!r}:
    try:
        md = markdown.Markdown(extensions=[mod])
    except Exception as e:
        continue
    snap = {{}}
    for r in REGS:
        obj = md
        for part in r.split("."):
            obj = getattr(obj, part)
        snap[r] = [p.name for p in sorted(obj._priority, key=lambda p: p.priority, reverse=True)]
    out[mod] = {{"ext": mod, "orders": snap}}
print(json.dumps(out))
"""
        r = subprocess.run([PY, "-I", "-S", "-B", "-c", code], capture_output=True, text=True, timeout=300, check=True)
        data[repo] = json.loads(r.stdout)
    for repo, d in data.items():
        tdir = build / "tests/repos" / repo / "swb_hidden"
        tdir.mkdir(parents=True, exist_ok=True)
        (tdir / "orders.json").write_text(json.dumps(d, indent=1, sort_keys=True))
        (tdir / "test_registry_order.py").write_text(ORDER_TEST)
    return {r: module_name(r) in data[repo_of(r)] for r in units}


def moved(src, ed, prio):
    """Oracle source with the priority of the register call at this edit replaced by `prio`."""
    seg_bytes = src[ed["start"]:ed["end"]]
    node = ast.parse(seg_bytes.decode(), mode="eval").body
    p0, p1 = span(byte_offsets(seg_bytes), call_args(node, ["item", "name", "priority"])["priority"])
    return (src[:ed["start"]] + seg_bytes[:p0] + prio.encode() + seg_bytes[p1:] + src[ed["end"]:]).decode()


def main(build):
    build = Path(build).resolve()
    base, tests = extract_all(build)
    cands = candidates(base)
    runs = []
    for i in range(2):
        root = sandbox(build, f"oracle{i}", {})
        runs.append({repo: record(root, "record", cands, repo) for repo in ("markdown", "pymdownx")})
    oracle = runs[0]
    passing = {repo: sorted(t for t, o in oracle[repo]["outcomes"].items()
                            if o == "passed" and runs[1][repo]["outcomes"].get(t) == "passed")
               for repo in oracle}
    print({r: len(v) for r, v in passing.items()}, "oracle-passing tests")
    ops_by_file = {}
    for repo in oracle:
        for op in oracle[repo]["ops"]:
            rel = str(Path(op["file"]).relative_to(build / "sb/oracle0"))
            ops_by_file.setdefault(rel, []).append(op)
    units = {}
    for rel in cands:
        eds = legacy_edits(base, rel, ops_by_file.get(rel, []))
        if any(e["kind"] == "reg-add" for e in eds):
            units[rel] = eds
    print(len(units), "candidate units")
    has_order = standalone_orders(build, units)

    def check(rel):
        eds = units[rel]
        log = []
        # 1. hidden tests: real tests that fail when only this unit is legacy, plus the registry-order test
        text = apply_edits((base / rel).read_bytes(), eds, lambda i, e: 0)
        root = sandbox(build, "iso-" + rel.replace("/", "_"), {rel: text})
        hidden = {}
        for repo in ("markdown", "pymdownx"):
            res = eng.run_pytest(str(root), ["tests"], META["paths"][repo], cwd=META["cwd"][repo], timeout=900)
            hidden[repo] = sorted(t for t in passing[repo] if res["outcomes"].get(t) != "passed")
        shutil.rmtree(root)
        # 2. on those tests the legacy module must be a correct program for the legacy API (3.3.7 API re-added):
        # same outcomes and the same registry orders as the oracle
        for v in (0, 1):
            text = apply_edits((base / rel).read_bytes(), eds, lambda i, e: 0 if v == 0 else len(e["options"]) - 1)
            root = sandbox(build, f"c{v}-" + rel.replace("/", "_"), {rel: text})
            bad = []
            for repo in ("markdown", "pymdownx"):
                if not hidden[repo]:
                    continue
                rec = record(root, "compat", [], repo)
                bad += [t for t in hidden[repo] if rec["outcomes"].get(t) != "passed"
                        or rec["finals"].get(t) != oracle[repo]["finals"].get(t)]
            shutil.rmtree(root)
            if bad and v == 0:
                return rel, None, None, [f"compat v0 mismatch: {bad[:3]}"]
            if bad:
                eds = [dict(e, options=e["options"][:1]) for e in eds]
                log.append(f"compat v1 mismatch ({len(bad)} tests): first option only")
        if has_order[rel]:
            hidden[repo_of(rel)].append(order_id(rel))
        # 3. every registry site must be checked: moving its item to the other end must fail a hidden test
        keep = []
        for ed in eds:
            if ed["kind"] != "reg-add":
                keep.append(ed)
                continue
            src = (base / rel).read_bytes()
            caught = False
            for prio in ("1e9", "-1e9"):
                mutated = moved(src, ed, prio)
                root = sandbox(build, "mut-" + rel.replace("/", "_"), {rel: mutated})
                ok_all = True
                for repo, ids in hidden.items():
                    if not ids:
                        continue
                    res = eng.run_pytest(str(root), ids, META["paths"][repo], cwd=META["cwd"][repo], timeout=900)
                    if any(res.get("outcomes", {}).get(t) != "passed" for t in ids):
                        ok_all = False
                shutil.rmtree(root)
                if not ok_all:
                    caught = True
                    break
            if caught:
                keep.append(ed)
            else:
                log.append(f"site at byte {ed['start']} not discriminated: kept in new form")
        if not any(e["kind"] == "reg-add" for e in keep):
            return rel, None, None, log + ["no discriminated registry site"]
        return rel, keep, hidden, log

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(check, sorted(units)))
    orders = {repo: json.loads((build / "tests/repos" / repo / "swb_hidden/orders.json").read_text())
              for repo in ("markdown", "pymdownx")}
    catalog = {"units": [], "passing": passing, "meta": META, "orders": orders, "order_test": ORDER_TEST}
    for rel, eds, hidden, log in results:
        print(rel, "DROP" if eds is None else "ok", "; ".join(log))
        if eds is None:
            continue
        n_tests = sum(len(v) for v in hidden.values())
        catalog["units"].append({"id": rel, "module": module_name(rel), "edits": eds, "tests": hidden,
                                 "n_tests": n_tests, "kinds": sorted({e["kind"] for e in eds}),
                                 "n_sites": len(eds)})
    print(len(catalog["units"]), "units in catalog")
    (HERE / "catalog.json").write_text(json.dumps(catalog, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
