"""Type 11: API migration after a dependency bump (Markdown 3.6 removed the 3.0-deprecated extension API).

Real code: Python-Markdown 3.6 bundled extensions and PyMdown Extensions 10.8.1 (pinned tarballs in ../repos).
Each unit is one extension module rewritten (build_catalog.py) to the removed legacy API; the oracle is the real
module. Hidden tests: the real test suites' tests that fail while the module is legacy, plus a registry-order test.
"""
import ast
import json
import random
import tarfile
import tomllib
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
MD_TAR, PX_TAR = REPOS / "markdown-3.6.tar.gz", REPOS / "pymdownx-10.8.1.tar.gz"
TIERS = {"easy": (5, 1800), "medium": (11, 1200), "hard": (22, 600)}
REG_ATTRS = {"preprocessors", "blockprocessors", "inlinePatterns", "treeprocessors", "postprocessors", "patterns"}
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
    eng.extract(MD_TAR, dest / "repos/markdown", ["markdown/", "docs/changelog.md", "LICENSE.md"])
    eng.extract(PX_TAR, dest / "repos/pymdownx", ["pymdownx/", "LICENSE.md"])


def extract_tests(dest, cat):
    eng.extract(MD_TAR, dest / "repos/markdown", ["tests/"])
    eng.extract(PX_TAR, dest / "repos/pymdownx", ["tests/"])
    eng.sort_pymdownx_walk(dest / "repos/pymdownx")
    with tarfile.open(MD_TAR) as tf:
        name = next(m for m in tf.getnames() if m.endswith("/pyproject.toml") and m.count("/") == 1)
        eps = tomllib.loads(tf.extractfile(name).read().decode())["project"]["entry-points"]
    dist = dest / "repos/markdown/Markdown-3.6.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: Markdown\nVersion: 3.6\n")
    lines = []
    for group, items in eps.items():
        lines += [f"[{group}]"] + [f"{k} = {v}" for k, v in items.items()] + [""]
    (dist / "entry_points.txt").write_text("\n".join(lines))
    for repo, orders in cat["orders"].items():
        d = dest / "repos" / repo / "swb_hidden"
        d.mkdir(parents=True)
        (d / "orders.json").write_text(json.dumps(orders, indent=1, sort_keys=True))
        (d / "test_registry_order.py").write_text(cat["order_test"])


def apply_edits(src_bytes, edits, picks):
    out = src_bytes
    for ed, k in sorted(zip(edits, picks), key=lambda x: -x[0]["start"]):
        out = out[:ed["start"]] + ed["options"][k].encode() + out[ed["end"]:]
    return out.decode()


def generate(n, seed, out_dir):
    cat = catalog()
    if not 1 <= n <= len(cat["units"]):
        raise ValueError(f"n must be 1..{len(cat['units'])}")
    rng = random.Random(f"b11:{n}:{seed}")
    world, gold = eng.fresh(out_dir)
    chosen = sorted(rng.sample(cat["units"], n), key=lambda u: u["id"])
    extract_tree(world)
    extract_tree(gold / "base")
    extract_tests(gold / "tests", cat)
    units = []
    for u in chosen:
        picks = [rng.randrange(len(e["options"])) for e in u["edits"]]
        (world / u["id"]).write_text(apply_edits((world / u["id"]).read_bytes(), u["edits"], picks))
        units.append({"id": u["id"], "files": [u["id"]], "tests": u["tests"], "visible": visible(u, seed)})
    hidden_files = {(repo, t.split("::")[0]) for u in chosen for repo, ids in u["tests"].items() for t in ids}
    controls = {repo: [t for t in ids if (repo, t.split("::")[0]) not in hidden_files]
                for repo, ids in cat["passing"].items()}
    meta = dict(cat["meta"], units=units, controls=controls, type="11_api_migration", seed=seed, n=n)
    (gold / "meta.json").write_text(json.dumps(meta, indent=1))
    eng.write_common(world, [u["id"] for u in units])
    tier = tier_of(n)
    prompt = eng.CANARY + "\n\n" + (HERE / "prompt.txt").read_text().strip().replace("{n}", str(n))
    return {"id": f"b11-n{n}-s{seed}", "prompt": prompt, "world_dir": str(world), "n_units": n,
            "gold_path": str(gold), "tier": tier, "budget_s": TIERS.get(tier, (0, 1200))[1], "canary": eng.CANARY}


def visible(u, seed):
    """The registry-order test plus up to VISIBLE_CAP real tests (deterministic sample)."""
    out = {}
    rng = random.Random(f"b11-vis:{u['id']}:{seed}")
    for repo, ids in u["tests"].items():
        order = [t for t in ids if t.startswith("swb_hidden/")]
        real = [t for t in ids if not t.startswith("swb_hidden/")]
        # small units show every hidden real test (the multi-extension cases expose priority ties); large ones
        # a deterministic sample of VISIBLE_CAP
        k = len(real) if len(real) <= 24 else VISIBLE_CAP
        out[repo] = order + sorted(rng.sample(real, k))
    return out


def legacy_left(text):
    """Static check: no use of the API removed in Markdown 3.4 may remain in a migrated module."""
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        return f"syntax error: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "md_globals" or isinstance(node, ast.arg) and node.arg == "md_globals":
            return "md_globals is still used"
        if isinstance(node, ast.Attribute):
            if node.attr in ("isBlockLevel", "etree") and isinstance(node.value, ast.Name) and node.value.id == "util":
                return f"util.{node.attr} is still used"
            if node.attr == "markdown" and isinstance(node.value, ast.Name) and node.value.id == "self":
                return "self.markdown is still used"
            if node.attr == "add" and isinstance(node.value, ast.Attribute) and node.value.attr in REG_ATTRS:
                return f"{node.value.attr}.add(...) is still used"
        if isinstance(node, ast.ImportFrom) and node.module and node.module.endswith("util") and \
                any(a.name in ("etree", "isBlockLevel") for a in node.names):
            return "import of a removed util name"
        if isinstance(node, (ast.Assign, ast.Delete)):
            for t in getattr(node, "targets", []):
                if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Attribute) and t.value.attr in REG_ATTRS:
                    return f"{t.value.attr}[...] assignment or del is still used"
    return None


def static_check(unit, texts):
    for rel, text in texts.items():
        if text is None:
            return f"{rel} is missing"
        bad = legacy_left(text)
        if bad:
            return f"{rel}: {bad}"
    return None


def tools(world_dir):
    return eng.make_tools(world_dir, static=static_check, unit_hint="a module path from UNITS.txt")


def score(task, answer=None):
    return eng.score_world(task, static=static_check)


def parse_answer(text):
    return True  # the repository is graded, not the reply
