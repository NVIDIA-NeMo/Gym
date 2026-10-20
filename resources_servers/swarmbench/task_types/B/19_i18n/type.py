"""Type 19: internationalization pass. Hard-coded user-facing strings in N functions of real click and humanize
code must go back through the translation catalog (gettext/ngettext/pgettext with the catalog's exact msgids).

Real code: click 8.1.7 and humanize 4.9.0 (pinned tarballs in ../repos); the oracle is the real code, which already
translates. The world has humanize's real .po catalogs and a messages.pot for click extracted from the real code.
Each unit (one function) is scored by splicing only that function from the agent's file into the oracle file.
Hidden checks: static signature (translation calls with their literal msgids), per-test runtime trace of the
function's translation calls, and humanize's real locale tests (compiled .mo).
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
R = _load("swb_b19_rules", HERE / "i18n_rules.py")
BC = None
REPOS = HERE.parent / "repos"
SPECS = {"click": ("click-8.1.7.tar.gz", "src/click", "tests"),
         "humanize": ("humanize-4.9.0.tar.gz", "src/humanize", "tests")}
TIERS = {"easy": (6, 1800), "medium": (14, 1200), "hard": (36, 300)}
VISIBLE_CAP = 16
_catalog = None


def catalog():
    global _catalog
    if _catalog is None:
        _catalog = json.loads((HERE / "catalog.json").read_text())
    return _catalog


def tier_of(n):
    return next((t for t, (k, _) in TIERS.items() if k == n), "custom")


def builder():
    global BC
    if BC is None:
        BC = _load("swb_b19_build", HERE / "build_catalog.py")
    return BC


def generate(n, seed, out_dir):
    cat = catalog()
    if not 1 <= n <= len(cat["units"]):
        raise ValueError(f"n must be 1..{len(cat['units'])}")
    rng = random.Random(f"b19:{n}:{seed}")
    world, gold = eng.fresh(out_dir)
    chosen = sorted(rng.sample(cat["units"], n), key=lambda u: u["id"])
    bc = builder()
    bc.extract_tree(world)
    bc.extract_tree(gold / "base")
    bc.tests_overlay(gold / "tests", gold / "base")
    for repo, exp in cat["expected"].items():
        (gold / "tests/swb" / f"expected_{repo}.json").write_text(json.dumps(exp))
    (gold / "tests/swb/swb_i18n_plugin.py").write_text(cat["plugin"])
    (world / "repos/click/messages.pot").write_text(cat["click_pot"])
    by_file = {}
    for u in chosen:
        by_file.setdefault(u["file"], []).extend(u["edits"])
    for rel, eds in by_file.items():
        (world / rel).write_text(R.apply((world / rel).read_bytes(), [(s, e, r, None) for s, e, r in eds]))
    units = []
    for u in chosen:
        passing = cat["passing"][u["repo"]]
        vr = random.Random(f"b19-vis:{u['id']}:{seed}")
        # ids stay in the suite's collection order: some humanize i18n tests change module state
        pick = set(vr.sample(u["event_tests"], min(12, len(u["event_tests"]))) or vr.sample(passing, 8))
        units.append({"id": u["id"], "files": [u["file"]], "qualname": u["qualname"], "signature": u["signature"],
                      "tests": {u["repo"]: list(passing)}, "visible": {u["repo"]: [t for t in passing if t in pick]}})
    used = {u["repo"] for u in chosen}
    controls = {repo: ids for repo, ids in cat["passing"].items() if repo not in used}
    meta = dict(cat["meta"], units=units, controls=controls, type="19_i18n", seed=seed, n=n)
    (gold / "meta.json").write_text(json.dumps(meta, indent=1))
    eng.write_common(world, [u["id"] for u in units])
    tier = tier_of(n)
    prompt = eng.CANARY + "\n\n" + (HERE / "prompt.txt").read_text().strip().replace("{n}", str(n))
    return {"id": f"b19-n{n}-s{seed}", "prompt": prompt, "world_dir": str(world), "n_units": n,
            "gold_path": str(gold), "tier": tier, "budget_s": TIERS.get(tier, (0, 1200))[1], "canary": eng.CANARY}


def overlay(gold, world, unit):
    """Only the unit's function comes from the agent's file; the rest of the file is the oracle."""
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
        got = R.signature(fn)
        if got != unit["signature"]:
            want = [tuple(x) for x in unit["signature"]]
            have = [tuple(x) for x in got]
            missing = [x for x in want if x not in have or want.count(x) > have.count(x)]
            extra = [x for x in have if x not in want or have.count(x) > want.count(x)]
            return (f"{unit['qualname']}: translation calls with literal msgids differ from the expected ones; "
                    f"missing {sorted(set(missing))[:6]}, unexpected {sorted(set(extra))[:6]}")
    return None


def tools(world_dir):
    return eng.make_tools(world_dir, static=static_check, overlay=overlay,
                          unit_hint="a unit from UNITS.txt (path::function)")


def score(task, answer=None):
    return eng.score_world(task, static=static_check, overlay=overlay)


def parse_answer(text):
    return True  # the repository is graded, not the reply
