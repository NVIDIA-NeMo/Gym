# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import ast
import fcntl
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import worldgen

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "repo_template" / "acme"
RUNNER = HERE / "runner.py"
FAMILIES = ["money", "store", "clock", "events", "config"]
MAX_OUT = 8000
MODULE_RE = re.compile(r"^m\d{4}$")

PROMPT = """You are leading a repo-wide code migration in a shared Python repository (pure stdlib).

The internal API `acme.legacy` is being deleted and replaced by `acme.api`. {n} modules under `acme/mods/` still use
the legacy API; they are listed in `UNITS.txt` (one module name per line). Each must be rewritten to use `acme.api`
while keeping exactly the same observable behavior. A few other modules are already migrated; do not break them.

The new API changes behavior in these ways (MIGRATION.md and the "Replacing legacy" docstrings in acme/api/*.py give
exact recipes):
- money: convert(amount, dst, src="USD") -> exchange(src, dst, cents): src first; int cents = round(amount * 100);
  returns Result (value in cents, /100 for the old float); errors give ok=False instead of raising ConversionError.
- store: fetch -> lookup returns a Record (.value, .version; not a tuple) or None instead of KeyError/default;
  put(k, v) -> save(k, v).version.
- clock: span(start, end) seconds -> elapsed_ms(end_ms, start_ms): end first, milliseconds, negative instead of
  ValueError; shift(ts, minutes, hours) seconds -> offset(ts_ms, delta_ms) milliseconds.
- events: emit(topic, payload) -> list becomes publish(topic, payload, on_reply) with replies delivered in REVERSE
  order; returns the handler count.
- config: get(section, key, default, cast) -> setting("section.key", default): raises MissingSetting when absent
  without a default; legacy cast=bool is bool(raw) (so "false" is True), not kind="bool".
- Cross-module: public load_* loaders must return the Record, and their callers in other modules must be updated.

Rules checked by the tests: every legacy call site becomes an acme.api call (no hand-inlined logic), no function is
removed or renamed, and no compatibility wrappers (no new top-level names). Call sites use aliases, keyword
arguments, partial application, wrapper helpers, dispatch tables and error-handling paths, so each module (about
100-200 lines) must be read and migrated by hand. run_visible_tests(module) covers every function that has to change.

Tools (paths are relative to the repo root): list_files, read_file, search, write_file, apply_edit,
run_visible_tests(module). All agents edit the same working copy, so only edit the modules you were assigned.

Scoring reads the final state of the repository, not your answer: score = fraction of the {n} modules whose hidden
tests pass (minus already-migrated modules you broke). When the work is finished, reply with "done" and a short
summary."""


# ---------- generate ----------

def generate(n, seed, out_dir):
    rng = random.Random(f"code:{n}:{seed}")
    out = Path(out_dir)
    world, gold = out / "world", out / "gold"
    for d in (world, gold):
        if d.exists() and any(d.iterdir()):
            raise FileExistsError(f"{d} already exists; generate into a fresh directory")
    w = worldgen.make_world_data(rng)

    n_ctrl = max(2, n // 16)
    names = [f"m{i:04d}" for i in range(n + n_ctrl)]
    shuffled = names[:]
    rng.shuffle(shuffled)
    controls, units = sorted(shuffled[:n_ctrl]), sorted(shuffled[n_ctrl:])
    n_link = max(1, n // 16)
    linked = rng.sample(units, 2 * n_link)
    providers = {m: f"{rng.choice(worldgen.NOUNS)}_{m[1:]}" for m in linked[:n_link]}
    dependents = {m: rng.choice(sorted(providers)) for m in linked[n_link:]}

    modules = []
    for name in names:
        fams = rng.sample(FAMILIES, rng.choice([2, 2, 3]))
        k = rng.randint(7, 11)
        dep_ref = None
        if name in dependents:
            p = dependents[name]
            loader = f"load_{providers[p]}"
            if rng.random() < 0.5:
                dep_ref = ([f"from acme.mods.{p} import {loader} as _{loader}"], f"_{loader}")
            else:
                dep_ref = ([f"from acme.mods import {p}"], f"{p}.{loader}")
        modules.append(worldgen.build_module(rng, w, name, fams, k, control=name in controls,
                                             provider=providers.get(name), dep_ref=dep_ref))

    shutil.copytree(TEMPLATE, world / "acme")
    (world / "acme" / "api" / "_data.py").write_text(worldgen.data_module_source(w))
    for m in modules:
        (world / "acme" / "mods" / f"{m.name}.py").write_text(m.old)
    (world / "MIGRATION.md").write_text(worldgen.MIGRATION_MD)
    (world / "UNITS.txt").write_text("\n".join(units) + "\n")

    base = gold / "acme_base" / "acme"
    shutil.copytree(world / "acme", base, ignore=shutil.ignore_patterns("legacy", "m[0-9]*.py"))
    (gold / "mods").mkdir()
    for m in modules:
        (gold / "mods" / f"{m.name}.py").write_text(m.gold)

    expected = expected_results(gold, {m.name: m.cases for m in modules})
    hidden = {}
    for m in modules:
        rows = expected[m.name]
        if "cases" not in rows or any("raises" in r for r in rows["cases"]):
            raise RuntimeError(f"gold module {m.name} failed: {rows}")
        hidden[m.name] = [{"fn": fn, "args": args, "expect": r["ok"], "api": r["calls"] > 0}
                          for (fn, args), r in zip(m.cases, rows["cases"])]
    (gold / "hidden_cases.json").write_text(json.dumps(hidden, sort_keys=True))
    names = {m.name: sorted(top_names(m.old)) for m in modules}
    for m in modules:
        assert top_names(m.gold) <= set(names[m.name]), m.name
    meta = {"units": units, "controls": controls, "providers": sorted(providers), "dependents": dependents,
            "top_names": names}
    (gold / "meta.json").write_text(json.dumps(meta, sort_keys=True, indent=1))

    # Visible tests: every function that must change, on inputs different from the hidden ones.
    vis_expected = expected_results(gold, {m.name: m.visible for m in modules})
    visible = {}
    for m in modules:
        rows = vis_expected[m.name].get("cases", [])
        if len(rows) != len(m.visible) or any("raises" in r for r in rows):
            raise RuntimeError(f"gold module {m.name} failed its visible cases")
        visible[m.name] = {"cases": [{"fn": fn, "args": args, "expect": r["ok"], "api": r["calls"] > 0}
                                     for (fn, args), r in zip(m.visible, rows)],
                           "top_names": names[m.name]}
    (world / "tests").mkdir()
    (world / "tests" / "visible_cases.json").write_text(json.dumps(visible, sort_keys=True))

    return {"id": f"code-n{n}-s{seed}", "prompt": PROMPT.format(n=n), "world_dir": str(world),
            "n_units": n, "gold_path": str(gold)}


def build_sandbox(gold, mods_dir, dst, modules):
    # Only the task's own modules are copied: new helper files are not part of the migrated repo.
    shutil.copytree(Path(gold) / "acme_base" / "acme", Path(dst) / "acme")
    for name in modules:
        p = Path(mods_dir) / f"{name}.py"
        if p.is_file() and not p.is_symlink():
            shutil.copyfile(p, Path(dst) / "acme" / "mods" / p.name)


def top_names(src):
    """Names bound at module top level (outside def/class bodies), excluding imports."""
    out = set()

    def visit(stmts):
        for st in stmts:
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out.add(st.name)
            elif isinstance(st, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.For, ast.With)):
                targets = st.targets if isinstance(st, ast.Assign) else [getattr(st, "target", None)]
                targets += [i.optional_vars for i in getattr(st, "items", []) if i.optional_vars]
                for t in targets:
                    if t is not None:
                        out.update(n.id for n in ast.walk(t) if isinstance(n, ast.Name))
            for field in ("body", "orelse", "finalbody", "handlers"):
                if not isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    visit(getattr(st, field, []) or [])

    visit(ast.parse(src).body)
    return out


def shim_violation(path, allowed):
    try:
        new = top_names(Path(path).read_text()) - set(allowed)
    except (OSError, SyntaxError, ValueError):
        return None
    if new:
        return "new top-level names are not allowed (no compatibility wrappers): " + ", ".join(sorted(new))[:200]
    return None


def run_cases(root, jobs, timeout, block_legacy=False):
    fd, out_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.unlink(out_path)
    cmd = [sys.executable, "-I", "-B", str(RUNNER), str(root), out_path] + (["--block-legacy"] if block_legacy else [])
    try:
        proc = subprocess.run(cmd, input=json.dumps(jobs).encode(), stdout=subprocess.DEVNULL,
                              stderr=subprocess.PIPE, timeout=timeout)
        if not os.path.exists(out_path):
            return {"error": f"crashed (exit {proc.returncode}): {proc.stderr.decode(errors='replace')[-300:]}"}
        with open(out_path) as f:
            return json.load(f)
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {timeout}s"}
    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)


def expected_results(gold, jobs):
    with tempfile.TemporaryDirectory() as tmp:
        build_sandbox(gold, Path(gold) / "mods", tmp, list(jobs))
        res = run_cases(tmp, jobs, timeout=600)
    if "error" in res:
        raise RuntimeError(f"gold run failed: {res['error']}")
    return res


# ---------- comparison ----------

def same(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return a == b


def check_module(cases, result):
    """Returns (failure reason or None, number of passing cases)."""
    if "error" in result:
        return result["error"], 0
    if "import_error" in result:
        return "import failed: " + result["import_error"], 0
    first, passed = None, 0
    for case, got in zip(cases, result["cases"]):
        label = f"{case['fn']}({json.dumps(case['args'])[1:-1][:120]})"
        if "raises" in got:
            reason = f"{label} raised {got['raises']}: {got['msg'][:150]}"
        elif not same(case["expect"], got["ok"]):
            reason = f"{label}: expected {json.dumps(case['expect'])[:200]}, got {json.dumps(got['ok'])[:200]}"
        elif case.get("api") and got["calls"] == 0:
            reason = f"{label}: correct value but acme.api was never called"
        else:
            passed += 1
            continue
        first = first or reason
    return first, passed


# ---------- score ----------

def score(task, answer):
    gold = Path(task["gold_path"])
    meta = json.loads((gold / "meta.json").read_text())
    hidden = json.loads((gold / "hidden_cases.json").read_text())
    timeout = task.get("timeout_s", 30)
    mods_dir = Path(task["world_dir"]) / "acme" / "mods"
    modules = meta["units"] + meta["controls"]
    with tempfile.TemporaryDirectory() as tmp:
        build_sandbox(gold, mods_dir, tmp, modules)

        def one(name):
            bad = shim_violation(Path(tmp) / "acme" / "mods" / f"{name}.py", meta["top_names"][name])
            if bad:
                return name, (bad, 0)
            jobs = {name: [[c["fn"], c["args"]] for c in hidden[name]]}
            res = run_cases(tmp, jobs, timeout)
            return name, check_module(hidden[name], res if "error" in res else res[name])

        with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 4)) as pool:
            outcome = dict(pool.map(one, modules))
    failures = {m: r for m, (r, _) in outcome.items()}
    unit_cases = sum(len(hidden[m]) for m in meta["units"])

    units_ok = [m for m in meta["units"] if failures[m] is None]
    controls_broken = [m for m in meta["controls"] if failures[m] is not None]
    legacy_left = [m for m in meta["units"] if uses_legacy(mods_dir / f"{m}.py")]
    n = len(meta["units"])
    return {
        "score": max(0, len(units_ok) - len(controls_broken)) / n,
        "n_correct": len(units_ok),
        "n_units": n,
        "unit_pass_rate": len(units_ok) / n,
        "controls_broken": len(controls_broken),
        "n_controls": len(meta["controls"]),
        "legacy_imports_left": len(legacy_left),
        "case_pass_rate": sum(outcome[m][1] for m in meta["units"]) / unit_cases,
        "failures": dict([(m, r) for m, r in sorted(failures.items()) if r][:50]),
        "answer_chars": len(answer or ""),
    }


def uses_legacy(path):
    try:
        src = path.read_text()
    except OSError:
        return False
    return bool(re.search(r"^\s*(from\s+acme(\.legacy|\s+import\s+legacy)|import\s+acme\.legacy)", src, re.M))


# ---------- tools ----------

def tools(world_dir):
    root = os.path.realpath(world_dir)
    lock = threading.Lock()
    test_slots = threading.BoundedSemaphore(max(2, (os.cpu_count() or 4) * 2))

    def resolve(path):
        p = os.path.realpath(os.path.join(root, str(path).lstrip("/")))
        if p != root and not p.startswith(root + os.sep):
            raise ValueError("path is outside the repository")
        return p

    def rel(p):
        return os.path.relpath(p, root)

    def cap(text):
        return text if len(text) <= MAX_OUT else text[:MAX_OUT] + "\n...[output truncated]"

    def all_files():
        out = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith((".", "__pycache__")))
            out += [rel(os.path.join(dirpath, f)) for f in sorted(filenames) if not f.startswith(".")]
        return out

    class edit_lock:
        # Thread lock for in-process callers plus flock for multi-process harnesses.
        def __enter__(self):
            lock.acquire()
            self.f = open(os.path.join(root, ".edit.lock"), "a")
            try:
                fcntl.flock(self.f, fcntl.LOCK_EX)
            except OSError:
                pass

        def __exit__(self, *exc):
            self.f.close()
            lock.release()

    def atomic_write(p, content):
        tmp = os.path.join(os.path.dirname(p), f".{os.path.basename(p)}.{uuid.uuid4().hex}.tmp")
        with open(tmp, "w") as f:
            f.write(content)
        os.replace(tmp, p)

    def list_files(args):
        prefix = str(args.get("prefix", "") or "")
        offset = int(args.get("offset", 0) or 0)
        files = [f for f in all_files() if f.startswith(prefix)]
        page = files[offset:offset + 400]
        text = "\n".join(page)
        if offset + len(page) < len(files):
            text += f"\n... {len(files) - offset - len(page)} more (call again with offset={offset + len(page)})"
        return cap(text or "(no files)")

    def read_file(args):
        p = resolve(args["path"])
        with open(p) as f:
            lines = f.read().splitlines()
        start = max(1, int(args.get("start", 1) or 1))
        end = min(len(lines), int(args.get("end") or len(lines)))
        out, size = [], 0
        for i in range(start, end + 1):
            line = f"{i:4d}  {lines[i - 1]}"
            if size + len(line) > MAX_OUT:
                out.append(f"...[truncated; call again with start={i}]")
                break
            out.append(line)
            size += len(line) + 1
        return "\n".join(out) if out else f"(empty range; file has {len(lines)} lines)"

    def search(args):
        text = str(args["text"])
        prefix = str(args.get("prefix", "") or "")
        hits = []
        for f in all_files():
            if not f.startswith(prefix) or not f.endswith((".py", ".md", ".txt")):
                continue
            with open(os.path.join(root, f), errors="replace") as fh:
                for i, line in enumerate(fh, 1):
                    if text in line:
                        hits.append(f"{f}:{i}: {line.rstrip()[:160]}")
            if len(hits) >= 200:
                hits.append("...[more matches omitted]")
                break
        return cap("\n".join(hits) or "(no matches)")

    def write_file(args):
        p = resolve(args["path"])
        content = str(args["content"])
        if len(content) > 200_000:
            return "error: content too large (200k char limit)"
        with edit_lock():
            os.makedirs(os.path.dirname(p), exist_ok=True)
            atomic_write(p, content)
        return f"wrote {rel(p)} ({content.count(chr(10)) + 1} lines)"

    def apply_edit(args):
        p = resolve(args["path"])
        old, new = str(args["old"]), str(args["new"])
        if not old:
            return "error: old must be non-empty"
        with edit_lock():
            with open(p) as f:
                src = f.read()
            count = src.count(old)
            if count != 1:
                return f"error: old text found {count} times in {rel(p)}; it must match exactly once (include more context)"
            atomic_write(p, src.replace(old, new, 1))
        return f"edited {rel(p)}"

    def run_visible_tests(args):
        module = str(args["module"]).removesuffix(".py").split("/")[-1]
        if not MODULE_RE.match(module):
            return "error: module must look like m0042"
        with open(os.path.join(root, "tests", "visible_cases.json")) as f:
            spec = json.load(f).get(module)
        if spec is None:
            return f"error: no visible tests for {module}"
        path = os.path.join(root, "acme", "mods", f"{module}.py")
        bad = shim_violation(path, spec["top_names"])
        if bad:
            return f"{module}: FAIL: {bad}"
        try:  # the hidden tests also cover helpers that need no change, so a deleted name always fails
            gone = set(spec["top_names"]) - top_names(Path(path).read_text())
        except (OSError, SyntaxError, ValueError):
            gone = set()
        if gone:
            return f"{module}: FAIL: top-level names were removed (keep every function and constant): " + \
                ", ".join(sorted(gone))[:200]
        cases = spec["cases"]
        with test_slots:
            res = run_cases(root, {module: [[c["fn"], c["args"]] for c in cases]}, timeout=30, block_legacy=True)
        if "error" in res:
            return f"{module}: FAILED to run: {res['error']}"
        res = res[module]
        if "import_error" in res:
            return f"{module}: import failed (acme.legacy is removed during tests): {res['import_error']}"
        lines, passed = [], 0
        for c, got in zip(cases, res["cases"]):
            label = f"{c['fn']}({json.dumps(c['args'])[1:-1][:150]})"
            if "raises" in got:
                lines.append(f"FAIL {label}: raised {got['raises']}: {got['msg'][:200]}")
            elif not same(c["expect"], got["ok"]):
                lines.append(f"FAIL {label}: expected {json.dumps(c['expect'])[:300]}, got {json.dumps(got['ok'])[:300]}")
            elif c.get("api") and got["calls"] == 0:
                lines.append(f"FAIL {label}: correct value, but this path must call acme.api (do not inline the logic)")
            else:
                passed += 1
                lines.append(f"ok   {label}")
        return cap(f"{module}: {passed}/{len(cases)} visible cases passed\n" + "\n".join(lines))

    def schema(props, required):
        return {"type": "object", "properties": props, "required": required}

    s, i = {"type": "string"}, {"type": "integer"}
    return [
        {"name": "list_files", "description": "List repository files (400 per page), optionally under a path prefix.",
         "parameters": schema({"prefix": s, "offset": i}, []), "fn": guard(list_files)},
        {"name": "read_file", "description": "Read a file with line numbers; optional 1-based inclusive start/end lines.",
         "parameters": schema({"path": s, "start": i, "end": i}, ["path"]), "fn": guard(read_file)},
        {"name": "search", "description": "Find lines containing a literal string in .py/.md/.txt files (optional path prefix).",
         "parameters": schema({"text": s, "prefix": s}, ["text"]), "fn": guard(search)},
        {"name": "write_file", "description": "Atomically replace a whole file with new content.",
         "parameters": schema({"path": s, "content": s}, ["path", "content"]), "fn": guard(write_file)},
        {"name": "apply_edit", "description": "Replace one exact occurrence of `old` with `new` in a file (atomic).",
         "parameters": schema({"path": s, "old": s, "new": s}, ["path", "old", "new"]), "fn": guard(apply_edit)},
        {"name": "run_visible_tests", "description": "Run the visible (partial) tests of one module, e.g. m0042, with acme.legacy removed.",
         "parameters": schema({"module": s}, ["module"]), "fn": guard(run_visible_tests)},
    ]


def guard(fn):
    def wrapped(args):
        try:
            return fn(args or {})
        except (KeyError, TypeError) as e:
            return f"error: bad arguments ({e})"
        except (OSError, ValueError, UnicodeDecodeError) as e:
            return f"error: {e}"
    return wrapped
