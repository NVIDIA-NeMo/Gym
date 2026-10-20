"""Build-time: catalog for type 18 (roll out the fix for one vulnerable pattern: unescaped text in generated HTML).

Real code: mistune 3.0.2, Python-Markdown 3.6, PyMdown Extensions 10.8.1 and jinja2 3.1.4 (+markupsafe). A site is a
call to one of the projects' escaping helpers (escape, escape_text, escape_url, safe_entity, html_escape, _escape,
code_escape, _escape_attrib, _escape_cdata); "before" replaces the call by its first argument, so the text reaches
the HTML unescaped. Only sites whose removal makes a real test fail are used (each tested alone, hermetically).
Units = functions with at least one such site; all of a function's caught sites are removed together, and the
function is scored by splicing it into the oracle file.

Usage: python build_catalog.py BUILD_DIR
"""
import ast
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
B = HERE.parent
sys.path[:0] = [str(B), str(HERE)]
import escape_rules as R  # noqa: E402
import probes as P  # noqa: E402
import swb_b_engine as eng  # noqa: E402

SPECS = {  # repo -> tarball, package dir, tests dir, sys.path entries (sandbox-relative)
    "mistune": ("mistune-v3.0.2.tar.gz", "src/mistune", "tests", ["repos/mistune/src"]),
    "markdown": ("markdown-3.6.tar.gz", "markdown", "tests", ["repos/markdown"]),
    "pymdownx": ("pymdownx-10.8.1.tar.gz", "pymdownx", "tests", ["repos/pymdownx", "repos/markdown"]),
    "jinja": ("jinja-3.1.4.tar.gz", "src/jinja2", "tests", ["repos/jinja/src", "repos/markupsafe/src"]),
}


def extract_tree(dest):
    for repo, (tar, pkg, tests, _) in SPECS.items():
        eng.extract(B / "repos" / tar, dest / "repos" / repo, [pkg + "/", "LICENSE", "LICENSE.md", "LICENSE.txt"])
    eng.extract(B / "repos/markupsafe-2.1.5.tar.gz", dest / "repos/markupsafe", ["src/markupsafe/", "LICENSE.rst"])
    for p in (dest / "repos/markupsafe").rglob("*.c"):
        p.unlink()


def tests_overlay(dest):
    import tarfile
    import tomllib
    for repo, (tar, pkg, tests, _) in SPECS.items():
        eng.extract(B / "repos" / tar, dest / "repos" / repo, [tests + "/"])
    eng.sort_pymdownx_walk(dest / "repos/pymdownx")
    with tarfile.open(B / "repos/markdown-3.6.tar.gz") as tf:
        name = next(m for m in tf.getnames() if m.endswith("/pyproject.toml") and m.count("/") == 1)
        eps = tomllib.loads(tf.extractfile(name).read().decode())["project"]["entry-points"]
    dist = dest / "repos/markdown/Markdown-3.6.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: Markdown\nVersion: 3.6\n")
    lines = []
    for group, items in eps.items():
        lines += [f"[{group}]"] + [f"{k} = {v}" for k, v in items.items()] + [""]
    (dist / "entry_points.txt").write_text("\n".join(lines))


def write_probes(dest, data):
    """data: {repo: {"kind", "cases", "expected"?}} -> swb_hidden/test_escape_probes.py + probes.json per repo."""
    for repo, d in data.items():
        h = dest / "repos" / repo / "swb_hidden"
        h.mkdir(parents=True, exist_ok=True)
        (h / "test_escape_probes.py").write_text(P.MODULE)
        (h / "probes.json").write_text(json.dumps(d, indent=1, ensure_ascii=False))


def probe_ids(data, repo):
    return [f"swb_hidden/test_escape_probes.py::test_escape_probe[{i}]" for i in range(len(data[repo]["cases"]))]


def meta():
    return {"paths": {r: s[3] for r, s in SPECS.items()}, "cwd": {r: f"repos/{r}" for r in SPECS}}


def main(build):
    build = Path(build).resolve()
    if build.exists():
        shutil.rmtree(build)
    base, tests = build / "base", build / "tests"
    extract_tree(base)
    tests_overlay(tests)
    m = meta()

    def sandbox(name, overlay):
        d = build / "sb" / name
        if d.exists():
            shutil.rmtree(d)
        eng.copytree(base, d)
        eng.copytree(tests, d)
        for rel, text in overlay.items():
            (d / rel).write_text(text)
        return d

    def run(root, repo):
        return eng.run_pytest(str(root), [SPECS[repo][2]], m["paths"][repo], cwd=m["cwd"][repo], timeout=1800)["outcomes"]

    # probes: record the oracle's outputs (twice, must agree)
    probes = {repo: {"kind": repo, "cases": P.CASES[repo]} for repo in SPECS}
    rec = []
    for i in range(2):
        write_probes(tests, probes)
        root = sandbox(f"probe{i}", {})
        for repo in SPECS:
            import subprocess
            code = (f"import sys; sys.path[:0] = {[str(root / p) for p in m['paths'][repo]] + [str(B / '_vendor')]!r}; "
                    f"import runpy; runpy.run_path({str(root / 'repos' / repo / 'swb_hidden/test_escape_probes.py')!r}, run_name='__main__')")
            subprocess.run([sys.executable, "-I", "-S", "-B", "-c", code], cwd=root / "repos" / repo, check=True, timeout=600)
        rec.append({repo: json.loads((root / "repos" / repo / "swb_hidden/probes.json").read_text()) for repo in SPECS})
    assert rec[0] == rec[1], "probe outputs are not deterministic"
    probes = rec[0]
    write_probes(tests, probes)
    passing = {}
    for repo in SPECS:
        outs = [run(sandbox(f"oracle{i}", {}), repo) for i in range(2)]
        passing[repo] = [t for t, o in outs[0].items() if o == "passed" and outs[1].get(t) == "passed"]
        print(repo, len(passing[repo]), "passing")
    sites = []
    for repo, (tar, pkg, _, _) in SPECS.items():
        for p in sorted((base / "repos" / repo / pkg).rglob("*.py")):
            rel = str(p.relative_to(base))
            for s in R.sites(p.read_bytes()):
                sites.append((repo, rel, s))
    print(len(sites), "escaping call sites in functions")

    def probe(job):
        repo, rel, s = job
        text = R.apply((base / rel).read_bytes(), [s])
        root = sandbox(f"p-{rel.replace('/', '_')}-{s[0]}", {rel: text})
        out = run(root, repo)
        pr = eng.run_pytest(str(root), probe_ids(probes, repo), m["paths"][repo], cwd=m["cwd"][repo])["outcomes"]
        shutil.rmtree(root)
        return job, [t for t in passing[repo] if out.get(t) != "passed"] + [t for t in probe_ids(probes, repo)
                                                                            if pr.get(t) != "passed"]

    with ThreadPoolExecutor(8) as pool:
        probed = list(pool.map(probe, sites))
    by_fn = {}
    for (repo, rel, s), failed in probed:
        print(f"{rel}:{s[3]} {s[2]} in {s[4]}: {len(failed)} tests catch its removal")
        if failed:
            by_fn.setdefault((repo, rel, s[4]), []).append((s, failed))
    units = []
    for (repo, rel, q), ss in sorted(by_fn.items()):
        _, funcs = R.qualnames(ast.parse((base / rel).read_bytes()))
        catching = [t for t in passing[repo] + probe_ids(probes, repo) if any(t in f for _, f in ss)]
        units.append({"id": f"{rel}::{q}", "repo": repo, "file": rel, "qualname": q,
                      "edits": [list(s) for s, _ in ss], "n_escape": R.escape_count(funcs[q]), "catching": catching})
    catalog = {"units": units, "passing": passing, "meta": m, "probes": probes}
    (HERE / "catalog.json").write_text(json.dumps(catalog, indent=1))
    print(len(units), "units")


if __name__ == "__main__":
    main(sys.argv[1])
