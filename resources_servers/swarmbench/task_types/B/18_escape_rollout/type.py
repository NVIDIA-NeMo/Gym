"""Type 18: security fix rollout. In N functions of real HTML-producing code (mistune, Python-Markdown, PyMdown
Extensions, jinja2) the escaping of text that ends up in HTML was removed; the agents restore it with the right
helper at the right places (no double escaping of already-rendered HTML).

Oracle = the real function. Each unit is scored by splicing only that function from the agent's file into the
oracle file. Hidden checks: the package's real test suite (only sites whose removal a real test catches are used)
and a static rule: the function makes at least as many escaping calls as the original.
"""
import ast
import json
import random
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
R = _load("swb_b18_rules", HERE / "escape_rules.py")
TIERS = {"easy": (5, 1800), "medium": (12, 1200), "hard": (27, 600)}
VISIBLE_CAP = 16
_catalog = None
_bc = None


def catalog():
    global _catalog
    if _catalog is None:
        _catalog = json.loads((HERE / "catalog.json").read_text())
    return _catalog


def builder():
    global _bc
    if _bc is None:
        _bc = _load("swb_b18_build", HERE / "build_catalog.py")
    return _bc


def tier_of(n):
    return next((t for t, (k, _) in TIERS.items() if k == n), "custom")


def generate(n, seed, out_dir):
    cat = catalog()
    if not 1 <= n <= len(cat["units"]):
        raise ValueError(f"n must be 1..{len(cat['units'])}")
    rng = random.Random(f"b18:{n}:{seed}")
    world, gold = eng.fresh(out_dir)
    chosen = sorted(rng.sample(cat["units"], n), key=lambda u: u["id"])
    bc = builder()
    bc.extract_tree(world)
    bc.extract_tree(gold / "base")
    bc.tests_overlay(gold / "tests")
    bc.write_probes(gold / "tests", cat["probes"])
    by_file = {}
    for u in chosen:
        by_file.setdefault(u["file"], []).extend(u["edits"])
    for rel, eds in by_file.items():
        (world / rel).write_text(R.apply((world / rel).read_bytes(), eds))
    units = []
    for u in chosen:
        passing = cat["passing"][u["repo"]] + bc.probe_ids(cat["probes"], u["repo"])
        pick = set(random.Random(f"b18-vis:{u['id']}:{seed}").sample(passing, min(VISIBLE_CAP, len(passing))))
        pick |= set(u["catching"][:8])
        units.append({"id": u["id"], "files": [u["file"]], "qualname": u["qualname"], "n_escape": u["n_escape"],
                      "tests": {u["repo"]: list(passing)}, "visible": {u["repo"]: [t for t in passing if t in pick]}})
    used = {u["repo"] for u in chosen}
    controls = {repo: ids for repo, ids in cat["passing"].items() if repo not in used}
    meta = dict(cat["meta"], units=units, controls=controls, type="18_escape_rollout", seed=seed, n=n)
    (gold / "meta.json").write_text(json.dumps(meta, indent=1))
    eng.write_common(world, [u["id"] for u in units])
    tier = tier_of(n)
    prompt = eng.CANARY + "\n\n" + (HERE / "prompt.txt").read_text().strip().replace("{n}", str(n))
    return {"id": f"b18-n{n}-s{seed}", "prompt": prompt, "world_dir": str(world), "n_units": n,
            "gold_path": str(gold), "tier": tier, "budget_s": TIERS.get(tier, (0, 1200))[1], "canary": eng.CANARY}


def overlay(gold, world, unit):
    rel = unit["files"][0]
    oracle = (Path(gold) / "base" / rel).read_text()
    try:
        agent = (Path(world) / rel).read_text()
    except OSError:
        agent = ""
    spliced = R.splice(oracle, agent, unit["qualname"])
    return {rel: ("text", spliced if spliced is not None else "raise SyntaxError('unit function missing')\n")}


def static_check(unit, texts):
    for rel, text in texts.items():
        if text is None:
            return f"{rel} is missing"
        try:
            tree = ast.parse(text)
        except SyntaxError as e:
            return f"{rel}: syntax error: {e}"
        _, funcs = R.qualnames(tree)
        fn = funcs.get(unit["qualname"])
        if fn is None:
            return f"{rel}: function {unit['qualname']} not found"
        if R.escape_count(fn) < unit["n_escape"]:
            return f"{unit['qualname']}: text still reaches the HTML unescaped somewhere"
    return None


def tools(world_dir):
    return eng.make_tools(world_dir, static=static_check, overlay=overlay,
                          unit_hint="a unit from UNITS.txt (path::function)")


def score(task, answer=None):
    return eng.score_world(task, static=static_check, overlay=overlay)


def parse_answer(text):
    return True  # the repository is graded, not the reply
