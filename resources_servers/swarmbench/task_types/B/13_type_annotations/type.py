"""Type 13: restore type annotations so mypy --strict passes again (real strictly typed packages).

Real code: click 8.1.7, itsdangerous 2.2.0, jinja2 3.1.4, markupsafe 2.1.5 (pinned tarballs in ../repos). Each unit
is one module in which the annotations of up to MAX_FUNCS seeded functions/methods were stripped
(build_catalog.py). Hidden tests per unit: mypy --strict on the module's package (no errors beyond the oracle's;
all other files are the oracle) plus the package's real test suite; a static check caps Any / type: ignore / cast
at the oracle module's counts and forbids inline mypy configuration.
"""
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
import importlib.util as _ilu  # noqa: E402
import sys as _sys  # noqa: E402


def _load(name, path):
    """Load a sibling module by path under a name that includes its content hash, so a long-lived process (the
    MCP tool server) never keeps using a stale copy after the file changes."""
    import hashlib
    key = f"{name}_{hashlib.sha1(Path(path).read_bytes()).hexdigest()[:10]}"
    if key not in _sys.modules:
        spec = _ilu.spec_from_file_location(key, path)
        mod = _ilu.module_from_spec(spec)
        _sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return _sys.modules[key]


eng = _load("swb_b_engine", HERE.parent / "swb_b_engine.py")
REPOS = HERE.parent / "repos"
SPECS = {"click": ("click-8.1.7.tar.gz", "click"), "itsdangerous": ("itsdangerous-2.2.0.tar.gz", "itsdangerous"),
         "jinja": ("jinja-3.1.4.tar.gz", "jinja2"), "markupsafe": ("markupsafe-2.1.5.tar.gz", "markupsafe")}
TIERS = {"easy": (6, 1800), "medium": (16, 1200), "hard": (42, 1200)}
MAX_FUNCS = 12
VISIBLE_CAP = 16
_catalog = None


def catalog():
    global _catalog
    if _catalog is None:
        _catalog = json.loads((HERE / "catalog.json").read_text())
    return _catalog


def tier_of(n):
    return next((t for t, (k, _) in TIERS.items() if k == n), "custom")


def extract_tree(dest):
    for repo, (tar, pkg) in SPECS.items():
        eng.extract(REPOS / tar, dest / "repos" / repo, [f"src/{pkg}/", "LICENSE.rst", "LICENSE.txt"])
        for p in (dest / "repos" / repo / "src" / pkg).rglob("*.c"):
            p.unlink()


def unit_key(rel):
    return rel.split("/src/")[-1][:-3].replace("/", ".")


def apply(src, edits):
    out = src
    for s, e, r in sorted(edits, reverse=True):
        out = out[:s] + r.encode() + out[e:]
    return out.decode()


def generate(n, seed, out_dir):
    cat = catalog()
    if not 1 <= n <= len(cat["units"]):
        raise ValueError(f"n must be 1..{len(cat['units'])}")
    rng = random.Random(f"b13:{n}:{seed}")
    world, gold = eng.fresh(out_dir)
    chosen = sorted(rng.sample(cat["units"], n), key=lambda u: u["id"])
    extract_tree(world)
    extract_tree(gold / "base")
    tests = gold / "tests"
    for repo, (tar, pkg) in SPECS.items():
        eng.extract(REPOS / tar, tests / "repos" / repo, ["tests/"])
        d = tests / "repos" / repo / "swb_hidden"
        d.mkdir(parents=True)
        mine = [unit_key(u["id"]) for u in chosen if u["repo"] == repo]
        (d / "mypy.json").write_text(json.dumps({
            "args": cat["mypy_args"], "package": pkg, "baseline": cat["baseline"][repo], "units": mine,
            "mypy_path": cat["meta"]["paths"][repo]}, indent=1))
        (d / "test_mypy_strict.py").write_text(cat["mypy_test"])
    units = []
    for u in chosen:
        names = sorted(u["groups"])
        picked = sorted(rng.sample(names, min(MAX_FUNCS, len(names))))
        edits = [e for k in picked for e in u["groups"][k]]
        (world / u["id"]).write_text(apply((world / u["id"]).read_bytes(), edits))
        mypy_id = f"swb_hidden/test_mypy_strict.py::test_mypy_strict[{unit_key(u['id'])}]"
        real = cat["passing"][u["repo"]]
        vis = sorted(random.Random(f"b13-vis:{u['id']}:{seed}").sample(real, min(VISIBLE_CAP, len(real))))
        units.append({"id": u["id"], "files": [u["id"]], "functions": picked, "counts": u["counts"],
                      "tests": {u["repo"]: [mypy_id] + real}, "visible": {u["repo"]: [mypy_id] + vis}})
    used = {u["repo"] for u in chosen}
    controls = {repo: ids for repo, ids in cat["passing"].items() if repo not in used}
    meta = dict(cat["meta"], units=units, controls=controls, type="13_type_annotations", seed=seed, n=n)
    (gold / "meta.json").write_text(json.dumps(meta, indent=1))
    eng.write_common(world, [u["id"] for u in units])
    tier = tier_of(n)
    prompt = eng.CANARY + "\n\n" + (HERE / "prompt.txt").read_text().strip().replace("{n}", str(n))
    return {"id": f"b13-n{n}-s{seed}", "prompt": prompt, "world_dir": str(world), "n_units": n,
            "gold_path": str(gold), "tier": tier, "budget_s": TIERS.get(tier, (0, 1200))[1], "canary": eng.CANARY}


def counts(text):
    code = "\n".join(line.split("#")[0] if "type:" not in line else line for line in text.splitlines())
    return {"Any": len(re.findall(r"\bAny\b", code)), "ignore": len(re.findall(r"#\s*type:\s*ignore", text)),
            "cast": len(re.findall(r"\bcast\(", code))}


def static_check(unit, texts):
    for rel, text in texts.items():
        if text is None:
            return f"{rel} is missing"
        if re.search(r"#\s*mypy\s*:", text):
            return f"{rel}: inline '# mypy:' configuration is not allowed"
        c = counts(text)
        for k, limit in unit["counts"].items():
            if c[k] > limit:
                return f"{rel}: {c[k]} uses of {k} (the original module has {limit}); not allowed"
    return None


def tools(world_dir):
    return eng.make_tools(world_dir, static=static_check, unit_hint="a module path from UNITS.txt")


def score(task, answer=None):
    return eng.score_world(task, static=static_check)


def parse_answer(text):
    return True  # the repository is graded, not the reply
