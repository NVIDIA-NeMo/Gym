"""Build-time: catalog for type 19 (move hard-coded user-facing strings back into the translation catalog).

Oracle = real click 8.1.7 and humanize 4.9.0 (both translate user-facing strings with gettext). Units = functions
whose translation calls are turned back into hard-coded strings by i18n_rules.inverse_edits (f-strings for
_("...").format(...), conditionals for ngettext, the bare message for pgettext). Hidden checks per unit: the static
signature (multiset of translation calls with their literal arguments) of the function, and a runtime trace: a
plugin records every translation call (calling module, function, kind, msgids) per test; the trace must equal the
oracle's. humanize's real translations (.po compiled to .mo) make its locale tests a third check.

Usage: python build_catalog.py BUILD_DIR
"""
import ast
import json
import os
import shutil
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
B = HERE.parent
sys.path[:0] = [str(B), str(HERE)]
import i18n_rules as R  # noqa: E402
import swb_b_engine as eng  # noqa: E402

SPECS = {"click": ("click-8.1.7.tar.gz", "src/click", "tests"),
         "humanize": ("humanize-4.9.0.tar.gz", "src/humanize", "tests")}
PLUGIN = r'''# Loaded with -p swb_i18n_plugin: records every translation call (calling module and function, kind, msgids)
# per test and, unless SWB_I18N_RECORD is set, compares the sequence with the oracle's.
import gettext
import json
import os
import sys
import threading

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.basename(os.getcwd())
_path = os.path.join(HERE, f"expected_{REPO}.json")
EXPECTED = json.load(open(_path)) if os.path.exists(_path) else {}
RECORD = os.environ.get("SWB_I18N_RECORD")
_events, _recorded, _busy = [], {}, threading.local()
MARK = os.sep + "src" + os.sep


def _caller():
    f = sys._getframe(2)
    while f is not None:
        fn = f.f_code.co_filename
        if MARK in fn and (os.sep + "click" + os.sep in fn or os.sep + "humanize" + os.sep in fn) \
                and not fn.endswith("i18n.py"):
            mod = fn.split(MARK)[-1][:-3].replace(os.sep, ".")
            return mod, f.f_code.co_qualname
        f = f.f_back
    return None, None


def _wrap(fn, kind, nmsg):
    def wrapper(*args, **kwargs):
        if getattr(_busy, "on", False):
            return fn(*args, **kwargs)
        _busy.on = True
        try:
            msgs = [a for a in args if isinstance(a, str)][:nmsg]
            mod, qual = _caller()
            if mod:
                _events.append([mod, qual, kind] + msgs)
            return fn(*args, **kwargs)
        finally:
            _busy.on = False
    return wrapper


for _name, _kind, _n in (("gettext", "gettext", 1), ("ngettext", "ngettext", 2), ("pgettext", "pgettext", 2)):
    setattr(gettext, _name, _wrap(getattr(gettext, _name), _kind, _n))
    for _cls in (gettext.NullTranslations, gettext.GNUTranslations):
        setattr(_cls, _name, _wrap(getattr(_cls, _name), _kind, _n))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    _events.clear()
    res = yield
    got = list(_events)
    if RECORD:
        _recorded[item.nodeid] = got
    elif item.nodeid in EXPECTED and got != EXPECTED[item.nodeid]:
        exp = EXPECTED[item.nodeid]
        diff = next((i for i, (a, b) in enumerate(zip(exp, got)) if a != b), min(len(exp), len(got)))
        raise AssertionError(f"translation calls differ at call {diff}: expected {exp[diff:diff + 2]}, "
                             f"got {got[diff:diff + 2]} ({len(exp)} expected, {len(got)} made)")
    return res


def pytest_sessionfinish(session):
    if RECORD:
        with open(RECORD, "w") as f:
            json.dump(_recorded, f)
'''


# ---------- msgfmt ----------

def parse_po(text):
    """[(msgctxt, msgid, msgid_plural, [msgstr...], fuzzy)] from a .po file."""
    entries, cur, field, fuzzy = [], None, None, False

    def unq(s):
        return ast.literal_eval(s.strip())

    def flush():
        nonlocal cur
        if cur is not None and "msgid" in cur:
            entries.append((cur.get("msgctxt"), cur["msgid"], cur.get("msgid_plural"),
                            [cur[k] for k in sorted(k for k in cur if k.startswith("msgstr"))], cur.get("fuzzy", False)))
        cur = None

    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("#,") and "fuzzy" in line:
            flush()
            cur, fuzzy = {"fuzzy": True}, True
            continue
        if not line or line.startswith("#"):
            if not line:
                flush()
            continue
        for key in ("msgctxt", "msgid_plural", "msgid", "msgstr"):
            if line.startswith(key):
                rest = line[len(key):]
                k = key
                if key == "msgstr" and rest.startswith("["):
                    k = "msgstr" + rest[1:rest.index("]")].zfill(3)
                    rest = rest[rest.index("]") + 1:]
                elif key == "msgstr":
                    k = "msgstr000"
                if key in ("msgctxt", "msgid") and cur is not None and "msgstr000" in cur:
                    flush()
                if cur is None:
                    cur = {}
                cur[k] = unq(rest)
                field = k
                break
        else:
            if line.startswith('"') and cur is not None and field:
                cur[field] += unq(line)
    flush()
    return entries


def msgfmt(po_text):
    """Compile .po text to .mo bytes (GNU format), skipping fuzzy and untranslated entries (header kept)."""
    cat = {}
    for ctxt, mid, plural, strs, fuzzy in parse_po(po_text):
        if fuzzy and mid:
            continue
        if mid and not any(strs):
            continue
        key = mid if plural is None else mid + "\0" + plural
        if ctxt is not None:
            key = ctxt + "\x04" + key
        cat[key.encode()] = "\0".join(strs).encode()
    keys = sorted(cat)
    ids = strs = b""
    offs = []
    for k in keys:
        offs.append((len(ids), len(k), len(strs), len(cat[k])))
        ids += k + b"\0"
        strs += cat[k] + b"\0"
    n = len(keys)
    keystart = 7 * 4 + 16 * n
    valuestart = keystart + len(ids)
    koffsets, voffsets = [], []
    for o1, l1, o2, l2 in offs:
        koffsets += [l1, o1 + keystart]
        voffsets += [l2, o2 + valuestart]
    out = struct.pack("Iiiiiii", 0x950412DE, 0, n, 7 * 4, 7 * 4 + n * 8, 0, 0)
    out += struct.pack(f"{len(koffsets)}i", *koffsets) + struct.pack(f"{len(voffsets)}i", *voffsets)
    return out + ids + strs


# ---------- trees ----------

def extract_tree(dest):
    for repo, (tar, pkg, tests) in SPECS.items():
        eng.extract(B / "repos" / tar, dest / "repos" / repo, [pkg + "/", "LICENSE.rst", "LICENCE"])


def tests_overlay(dest, base):
    for repo, (tar, pkg, tests) in SPECS.items():
        eng.extract(B / "repos" / tar, dest / "repos" / repo, [tests + "/"])
    for po in sorted((base / "repos/humanize/src/humanize/locale").rglob("*.po")):
        mo = dest / po.relative_to(base).with_suffix(".mo")
        mo.parent.mkdir(parents=True, exist_ok=True)
        mo.write_bytes(msgfmt(po.read_text()))
    d = dest / "swb_meta/humanize-4.9.0.dist-info"
    d.mkdir(parents=True)
    (d / "METADATA").write_text("Metadata-Version: 2.1\nName: humanize\nVersion: 4.9.0\n")
    (dest / "swb").mkdir(parents=True, exist_ok=True)
    (dest / "swb/swb_i18n_plugin.py").write_text(PLUGIN)


def pot(base):
    """messages.pot for click: every translation call with literal arguments in the oracle code."""
    seen = {}
    for p in sorted((base / "repos/click/src/click").glob("*.py")):
        rel = str(p.relative_to(base / "repos/click"))
        for n in ast.walk(ast.parse(p.read_text())):
            k = R.kind_of(n)
            args = R.literal_args(n) if k else None
            if k == "ngettext" and n.args[:2] and all(isinstance(a, ast.Constant) for a in n.args[:2]):
                args = [n.args[0].value, n.args[1].value]
            if k in ("gettext", "ngettext") and args:
                seen.setdefault(tuple(args), set()).add(rel)

    def q(s):
        return json.dumps(s, ensure_ascii=False)

    out = ['msgid ""', 'msgstr ""', q("Content-Type: text/plain; charset=UTF-8\n"), ""]
    for args, refs in seen.items():
        out.append("#: " + " ".join(sorted(refs)))
        if "{" in args[0] or (len(args) > 1 and "{" in args[1]):
            out.append("#, python-brace-format")
        out.append("msgid " + q(args[0]))
        if len(args) > 1:
            out += ["msgid_plural " + q(args[1]), 'msgstr[0] ""', 'msgstr[1] ""']
        else:
            out.append('msgstr ""')
        out.append("")
    return "\n".join(out)


def meta():
    return {"paths": {"click": ["repos/click/src", "swb"], "humanize": ["repos/humanize/src", "swb_meta", "swb"]},
            "cwd": {"click": "repos/click", "humanize": "repos/humanize"}, "pytest_args": ["-p", "swb_i18n_plugin"]}


def main(build):
    build = Path(build).resolve()
    if build.exists():
        shutil.rmtree(build)
    base, tests = build / "base", build / "tests"
    extract_tree(base)
    tests_overlay(tests, base)
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

    def run(root, repo, record=None):
        out = root / f"out-{repo}.json"
        job = {"ids": [SPECS[repo][2]], "paths": [str(root / p) for p in m["paths"][repo]], "args": m["pytest_args"]}
        env = dict(os.environ, **({"SWB_I18N_RECORD": str(record)} if record else {}))
        subprocess.run([sys.executable, "-I", "-S", "-B", str(B / "swb_b_pytest_runner.py"), str(out), str(root / m["cwd"][repo])],
                       input=json.dumps(job).encode(), env=env, timeout=1800, check=True)
        return json.loads(out.read_text())["outcomes"]

    passing, expected = {}, {}
    for repo in SPECS:
        outs, recs = [], []
        for i in range(2):
            root = sandbox(f"oracle{i}", {})
            outs.append(run(root, repo, root / "rec.json"))
            recs.append(json.loads((root / "rec.json").read_text()))
        passing[repo] = [t for t, o in outs[0].items() if o == "passed" and outs[1].get(t) == "passed"]  # run order
        expected[repo] = {t: recs[0][t] for t in passing[repo] if t in recs[0] and recs[0][t] == recs[1].get(t)}
        print(repo, len(passing[repo]), "passing,", sum(1 for v in expected[repo].values() if v), "with translation calls")
        (tests / "swb" / f"expected_{repo}.json").write_text(json.dumps(expected[repo]))

    cands = []
    for repo, (tar, pkg, _) in SPECS.items():
        for p in sorted((base / "repos" / repo / pkg).rglob("*.py")):
            rel = str(p.relative_to(base))
            src = p.read_bytes()
            by_q = {}
            for e in R.inverse_edits(src):
                by_q.setdefault(e[3], []).append(e)
            _, funcs = R.qualnames(ast.parse(src))
            for q, eds in sorted(by_q.items()):
                cands.append({"repo": repo, "file": rel, "qualname": q, "edits": eds,
                              "signature": R.signature(funcs[q]), "module": rel.split("/src/")[-1][:-3].replace("/", ".")})

    def check(c):
        text = R.apply((base / c["file"]).read_bytes(), c["edits"])
        root = sandbox("iso-" + c["file"].replace("/", "_") + "-" + c["qualname"].replace("<", "").replace(">", ""),
                       {c["file"]: text})
        rec = root / "rec.json"
        plain = run(root, c["repo"], rec)  # record mode: no trace check, so failures are behavior changes
        traces = json.loads(rec.read_text())
        checked = run(root, c["repo"])
        shutil.rmtree(root)
        changed = [t for t in passing[c["repo"]] if plain.get(t) != "passed"]
        hidden = [t for t in passing[c["repo"]] if checked.get(t) != "passed"]
        mine = lambda evs: [e for e in evs if e[0] == c["module"] and e[1] == c["qualname"]]  # noqa: E731
        traced = [t for t in passing[c["repo"]] if mine(expected[c["repo"]].get(t, []))]
        return c, changed, hidden, traced, traces

    with ThreadPoolExecutor(8) as pool:
        res = list(pool.map(check, cands))
    units = []
    for c, changed, hidden, traced, _ in res:
        locale_only = all("i18n" in t for t in changed)
        ok = locale_only
        print(f"{c['file']}::{c['qualname']}: {len(c['edits'])} sites, {len(hidden)} hidden tests fail, "
              f"{len(traced)} traced, behavior changes outside locale tests: {0 if locale_only else len(changed)}",
              "" if ok else "DROP")
        if ok:
            units.append({"id": f"{c['file']}::{c['qualname']}", "repo": c["repo"], "file": c["file"],
                          "qualname": c["qualname"], "module": c["module"], "signature": c["signature"],
                          "edits": [[s, e, r] for s, e, r, _ in c["edits"]], "event_tests": hidden})
    catalog = {"units": units, "passing": passing, "expected": expected, "meta": m, "plugin": PLUGIN,
               "click_pot": pot(base)}
    (HERE / "catalog.json").write_text(json.dumps(catalog, indent=1))
    print(len(units), "units")


if __name__ == "__main__":
    main(sys.argv[1])
