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
"""Group workspace builder, vendored so the judge service's code archive is self-contained.

SOURCE: turn_judge/tools/group_workspace.py (research/ppo/turn_judge, not a git repo), sha256
5479830a07307a14410ea7e4f62d47dd05bb6d6b815c4a58c1d147cb93c56b97 (mtime 2026-11-13 23:29), the version that built the
judge development workspaces (iterations/judge_ws, 031-train-r2/ws, ...). Copied verbatim from HARNESS through
build; changes:
- build(..., answer_key=None): the service builds each task's TASK.md + answer_key/ once into a node-local cache and
  passes write_answer_key's return value (None = original behaviour);
- TB_TASK_ROOT may be reassigned by the caller (service --tb-task-root);
- dropped: eval_source / train_source / main (collection-root readers; the service builds from its request).
Keep in sync with the source: tests/test_workspace_fidelity.py compares service-built workspaces with offline ones.

Layout of a workspace (what the judge sessions read):
  TASK.md, answer_key/, R<k>/{trajectory.txt, final.patch, tests.json, test_output.txt, failing_assertions.txt},
  PREPASS.md, group.json (per R<k>: label locator file + line, reward, calls, flags).
"""

import ast
import json
import os
import re
import shutil
from pathlib import Path

HARNESS = os.environ.get("TURN_JUDGE_GYM_ROOT", str(Path(__file__).resolve().parents[2]))
# Training Terminal-Bench tasks: mounted as /terminal-bench-tasks in training (launchers/nt4/nt4_common.sh TB_TASK_ROOT)
TB_TASK_ROOT = os.environ.get("TURN_JUDGE_TB_TASK_ROOT", "/terminal-bench-tasks")
# Task-folder files copied into answer_key/ for Terminal-Bench (what exists varies by suite): designer notes (README.md
# eval; SPEC.md atcb-2609/3500; qc_check.md atcb-1 = instruction<->test alignment), metadata, solution, tests, env.
TB_DOCS = ("README.md", "SPEC.md", "qc_check.md", "task.toml")
TB_DIRS = ("solution", "tests", "environment")
# Data-row fields that are not designer information (big, duplicated elsewhere, or infrastructure).
SKIP_FIELDS = {
    "responses_create_params",
    "patch",
    "test_patch",
    "golden_patch",
    "pr_patch",
    "f2p_patch",
    "agent_ref",
    "install_config",
    "docker_image",
    "image_name",
    "image_ref",
    "image_url",
    "nydus_ref",
    "config_json",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
    "fail_to_pass",
    "pass_to_pass",
}


# ------------------------------------------------------------------------------------------------ text helpers


def as_list(v):
    """Test-id fields are JSON or python-repr strings, sometimes a list holding one list-string."""
    if isinstance(v, str):
        for parse in (json.loads, ast.literal_eval):
            try:
                return as_list(parse(v))
            except Exception:
                pass
        return [v]
    if isinstance(v, list) and len(v) == 1 and isinstance(v[0], str) and v[0].strip().startswith("["):
        return as_list(v[0])
    return v or []


def text_of(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return ""


def item_text(it):
    t = it.get("type")
    if t == "reasoning":
        return "\n".join(s.get("text", "") for s in it.get("summary") or []) or text_of(it.get("content"))
    if t == "function_call":
        return f"{it.get('name')}({it.get('arguments')})"
    if t == "function_call_output":
        o = it.get("output")
        return o if isinstance(o, str) else json.dumps(o)
    if t == "message":
        return f"[{it.get('role')}] " + text_of(it.get("content"))
    return json.dumps(it)


def call_ordinals(out):
    """Model-call ordinal per item: every model call starts with a reasoning item and owns the items up to the next one
    (parallel tool calls and their outputs interleave inside one call). Matches the trainer's turn count on 99.67% of
    twin rollouts (LOG 010, user ruling LOG 014)."""
    ords, c = [], -1
    for it in out:
        if c < 0 or it.get("type") == "reasoning":
            c += 1
        ords.append(c)
    return ords


def patch_files(patch):
    return sorted(set(re.findall(r"^diff --git a/(\S+) b/", patch or "", re.M)))


def added_lines(patch):
    """[(file, line text)] of '+' lines in a unified diff."""
    out, f = [], None
    for ln in (patch or "").splitlines():
        m = re.match(r"^\+\+\+ b/(\S+)", ln)
        if m:
            f = m.group(1)
        elif ln.startswith("+") and not ln.startswith("+++"):
            out.append((f, ln[1:]))
    return out


WRAP = 1500  # the agents' Read tool cuts lines far longer than this; wrap (never cut) so every character stays visible


def wrap_long(text):
    """Hard-wrap lines longer than WRAP into WRAP-sized pieces; continuation lines start with '↪ '. Nothing is dropped."""
    out = []
    for ln in (text or "").split("\n"):
        while len(ln) > WRAP:
            out.append(ln[:WRAP])
            ln = "↪ " + ln[WRAP:]
        out.append(ln)
    return "\n".join(out)


def write_text(path, text):
    """Write a workspace text file with long lines wrapped (see wrap_long)."""
    with open(path, "w") as f:
        f.write(wrap_long(text))


# A line that starts another test's report (ends the current failure block).
NEXT_REPORT = re.compile(
    r"^(---- \S+ stdout ----|\*?\s*\[(ok|err|exception)\]:|--- (PASS|FAIL): |_{3,} .+ _{3,}$|test \S+ \.\.\. |"
    r"=+ .+ =+$|failures:$|\[(INFO|ERROR)\] Tests run:)"
)


def block_from(lines, start):
    """Lines from `start` up to (not including) the next line that starts another report; no length cap."""
    end = next((j for j in range(start + 1, len(lines)) if NEXT_REPORT.match(lines[j].strip())), len(lines))
    return "\n".join(lines[start:end])


def prompt_text(row):
    """The user message the agent received (responses_create_params.input)."""
    msgs = ((row or {}).get("responses_create_params") or {}).get("input") or []
    return "\n\n".join(text_of(m.get("content")) for m in msgs if isinstance(m, dict) and m.get("role") == "user")


# ------------------------------------------------------------------------------------------------ test results


def failing_blocks(log, failing):
    """{test: excerpt} of the failure report of each failing test: pytest FAILURES blocks when present, else the 40
    lines from the cargo `---- name stdout ----` block, the Tcl `[err]: name` line, or the first line naming the test
    next to a failure marker (Go / JUnit / JS)."""
    out, log = {}, log or ""
    m = re.search(r"^=+ FAILURES =+$", log, re.M)
    if m:
        end = re.search(r"^=+ (short test summary info|warnings summary|PASSES|ERRORS) =+$", log[m.end() :], re.M)
        sec = log[m.end() : m.end() + end.start()] if end else log[m.end() :]
        parts = re.split(r"(?m)^_{3,} (.+?) _{3,}$", sec)
        for name, body in zip(parts[1::2], parts[2::2]):
            name = name.strip()
            for t in failing:
                short = t.split("::")[-1]
                if short == name or short == name.split(".")[-1] or name.endswith(short):
                    out.setdefault(t, body.strip())
    jest = jest_results(log)
    vit = vitest_results(log, failing)
    for t in failing:
        if t not in out and jest.get(t.split("::")[-1], ("", ""))[1]:
            out[t] = jest[t.split("::")[-1]][1]
        if t not in out and vit.get(t, ("", ""))[1]:
            out[t] = vit[t][1]
        if t not in out and (ce := collection_error(log, t)):
            out[t] = "(the test never ran: collection / import error)\n" + ce
    lines = log.splitlines()
    for t in failing:
        if t in out:
            continue
        names = [t.replace("#", "."), t, t.split("::")[-1], t.split("#")[-1]]
        pats = [r"^---- " + re.escape(t) + r" stdout ----", r"\[(err|exception)\]: " + re.escape(t)]
        hit = next((i for p in pats for i, ln in enumerate(lines) if re.search(p, ln)), None)
        if hit is None:
            hit = next(
                (
                    i
                    for nm in names
                    for i, ln in enumerate(lines)
                    if nm and nm in ln and re.search(r"FAIL|✕|×|<<< ERROR|\[err\]|panic", ln)
                ),
                None,
            )
        if hit is not None:
            out[t] = block_from(lines, hit)
    return out


def jest_results(log):
    """{fullName: (status, failure message)} from Jest JSON reports embedded in a log (swemer_oml)."""
    out = {}
    for m in re.finditer(
        r'"failureMessages":\[(.*?)\],"fullName":"((?:[^"\\]|\\.)*)"[^{}]*?"status":"(\w+)"', log or ""
    ):
        msg = m.group(1).encode().decode("unicode_escape", errors="replace") if m.group(1) else ""
        out[m.group(2)] = ("PASSED" if m.group(3) == "passed" else "FAILED", msg)
    return out


def vitest_results(log, tests):
    """{test id: (status, failure message)} from a Vitest console report (swemer_oml frontends): per-test lines
    `✓|×|↓ Suite > sub > name [12ms]` (failure detail on the following `→` lines), a whole file that failed to load
    (`FAIL  <file> [ <file> ]` + its error block: every test of the file FAILED with that block), and a passing file
    printed only as `✓ <file> (N tests)` (every test of the file PASSED). Test ids look like
    `<abs path>::Suite sub name`."""
    out, lines = {}, re.sub(r"\x1b\[[0-9;]*m", "", log or "").splitlines()  # Vitest 4 colours its output
    per = {}
    for i, ln in enumerate(lines):
        m = re.match(r"^\s*(✓|×|↓)\s+(.+?)(?:\s+\d+\s*ms)?\s*$", ln)
        if not m or " > " not in m.group(2):
            continue
        name = m.group(2)
        first = name.split(" > ")[0]
        if re.search(r"\.(test|spec)\.[cm]?[jt]sx?$", first):  # Vitest 4 prefixes the file: `file > Suite > name`
            name = name[len(first) + 3 :]
        msg = []
        for nxt in lines[i + 1 :]:
            if re.match(r"^\s*→", nxt):
                msg.append(nxt.strip())
            else:
                break
        per[name.replace(" > ", " ")] = ({"✓": "PASSED", "×": "FAILED", "↓": "SKIPPED"}[m.group(1)], "\n".join(msg))
    for i, ln in enumerate(lines):  # detailed report: ` FAIL  <file> > Suite > sub > name` up to the next ⎯ rule
        m = re.match(r"^\s*FAIL\s+(\S+) > (.+?)\s*$", ln)
        if m and m.group(2).replace(" > ", " ") in per:
            end = next((j for j in range(i + 1, len(lines)) if lines[j].startswith("⎯")), len(lines))
            per[m.group(2).replace(" > ", " ")] = ("FAILED", "\n".join(lines[i:end]).strip())
    for t in tests:
        name = t.split("::")[-1]
        if name in per:
            out[t] = per[name]
    for i, ln in enumerate(lines):
        m = re.match(r"^\s*FAIL\s+(\S+)\s+\[\s*\S+\s*\]\s*$", ln)
        if m:
            end = next((j for j in range(i + 1, len(lines)) if lines[j].startswith("⎯")), len(lines))
            block = "\n".join(lines[i:end]).strip()
            for t in tests:
                if t not in out and t.split("::")[0].endswith(m.group(1)):
                    out[t] = ("FAILED", "(the test file failed to load; no test ran)\n" + block)
        m = re.match(r"^\s*✓\s+(\S+)\s+\((\d+) tests?\)", ln)
        if m:
            for t in tests:
                if t not in out and t.split("::")[0].endswith(m.group(1)):
                    out[t] = ("PASSED", "")
    return out


def collection_error(log, test):
    """Excerpt of a pytest collection / import error for the module of `test` (tests that never ran)."""
    mod = test.split("::")[0]
    path = mod if mod.endswith(".py") else mod.replace(".", "/") + ".py"
    lines = (log or "").splitlines()
    hit = next((i for i, ln in enumerate(lines) if path in ln and re.search(r"ImportError|ERROR|error", ln)), None)
    if hit is None and re.search(r"error(s)? during collection", log or ""):
        hit = next((i for i, ln in enumerate(lines) if re.match(r"^_+ ERROR collecting", ln)), None)
    return block_from(lines, hit) if hit is not None else None


def status_from_log(log, tests):
    """Per-test status parsed from a raw test log, for records without test_results: Tcl `[ok]|[err]: name`, cargo
    `test name ... ok|FAILED`, Go `--- PASS|FAIL: name`, JUnit/Maven `Class.method ... <<< FAILURE!|ERROR!` (a -Dtest run
    without one counts as PASSED), pytest `PASSED|FAILED path::name`, Jest/Mocha marks. Tests not found are unknown."""
    st, log = {}, log or ""
    jest = jest_results(log)
    vit = vitest_results(log, tests)
    for t in tests:
        n = re.escape(t)
        if t.split("::")[-1] in jest:
            st[t] = jest[t.split("::")[-1]][0]
        elif t in vit:
            st[t] = vit[t][0]
        elif re.search(r"\[(err|exception)\]: " + n, log):
            st[t] = "FAILED"
        elif re.search(r"\[ok\]: " + n, log):
            st[t] = "PASSED"
        elif m := re.search(r"^test " + n + r" \.\.\. (\w+)", log, re.M):
            st[t] = {"ok": "PASSED", "FAILED": "FAILED", "ignored": "SKIPPED"}.get(m.group(1), m.group(1))
        elif m := re.search(r"--- (PASS|FAIL): " + n + r"\b", log):
            st[t] = "PASSED" if m.group(1) == "PASS" else "FAILED"
        elif re.search(re.escape(t.replace("#", ".")) + r"\b.*<<< (FAILURE|ERROR)!", log):
            st[t] = "FAILED"
        elif "#" in t and re.search(r"-Dtest=" + n + r"\b", log):
            st[t] = "PASSED"
        elif m := re.search(r"^(PASSED|FAILED|ERROR) \S*" + n + r"\b", log, re.M):
            st[t] = "PASSED" if m.group(1) == "PASSED" else "FAILED"
        elif re.search(r"(✕|✗|×)\s+" + n, log):
            st[t] = "FAILED"
        elif re.search(r"(✓|✔)\s+" + n, log):
            st[t] = "PASSED"
    return st


def test_status(rec, tests, tb):
    if tb:  # pytest -rA short summary (`PASSED file::name`) or -v lines (`file::name PASSED`) in the verifier log
        log = rec.get("test_output") or ""
        st = {}
        for m in re.finditer(r"^\S*?::(\w+) (PASSED|FAILED|ERROR|SKIPPED)\b", log, re.M):
            st[m.group(1)] = m.group(2)
        for m in re.finditer(r"^(PASSED|FAILED|ERROR) \S*?::(\w+)", log, re.M):
            st[m.group(2)] = m.group(1)
        if not st and re.search(r"^=*\s*\d+ passed\b", log, re.M) and not re.search(r"\d+ (failed|errors?)\b", log):
            st = {t: "PASSED" for t in tests}  # quiet run (`1 passed in 0.36s`): no per-test lines, nothing failed
        return st
    tr = rec.get("test_results")
    st = {}
    if isinstance(tr, dict) and isinstance(tr.get("tests"), list):  # eval SWE-bench Pro: {"tests": [{name, status}]}
        st = {t.get("name"): t.get("status") for t in tr["tests"]}
    elif isinstance(tr, dict):  # training SWE servers: {"FAIL_TO_PASS": {"success": [...], "failure": [...]}, ...}
        for part in ("FAIL_TO_PASS", "PASS_TO_PASS"):
            v = tr.get(part)
            if isinstance(v, dict):
                st.update({t: "PASSED" for t in v.get("success") or []})
                st.update({t: "FAILED" for t in v.get("failure") or []})
    # swemer_oml lists the test ids without status; Multilingual has no test_results: parse the log
    return st or status_from_log(rec.get("test_output") or "", tests)


# ------------------------------------------------------------------------------------------------ answer keys


def tb_task_dir(task_folder):
    """Host path of a Terminal-Bench task folder: eval rows hold a Gym-relative path, training rows /terminal-bench-tasks/…"""
    if task_folder.startswith("/terminal-bench-tasks/"):
        return os.path.join(TB_TASK_ROOT, task_folder[len("/terminal-bench-tasks/") :])
    return os.path.join(HARNESS, task_folder)


def copy_tree(src, dst):
    """Copy every file (no size limit). A text file with lines longer than WRAP also gets a '<name>.wrapped.txt' copy
    (same content, long lines wrapped) so it can be read in full with the Read tool."""
    for root, _, names in os.walk(src):
        for n in names:
            f = os.path.join(root, n)
            d = os.path.join(dst, os.path.relpath(f, src))
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.copy(f, d)
            try:
                text = open(f, encoding="utf-8").read()
            except (UnicodeDecodeError, OSError):
                continue
            if any(len(ln) > WRAP for ln in text.split("\n")):
                write_text(d + ".wrapped.txt", text)


def write_answer_key(out_dir, task, key):
    """Write TASK.md + answer_key/; return (tb, golden, hidden, f2p, p2p)."""
    ak = f"{out_dir}/answer_key"
    os.makedirs(ak, exist_ok=True)
    tb = bool(key.get("task_folder"))
    prompt = prompt_text(key)
    if tb:
        folder = tb_task_dir(key["task_folder"])
        read = lambda rel: (
            open(os.path.join(folder, rel), errors="replace").read()
            if os.path.isfile(os.path.join(folder, rel))
            else ""
        )
        golden = read("solution/solve.sh") or key.get("golden_patch") or ""
        tests_dir = os.path.join(folder, "tests")
        # every file under tests/, including subfolders such as tests/fixtures/ (the whole folder is also copied as is)
        test_files = (
            sorted(os.path.relpath(os.path.join(dp, f), folder) for dp, _, fs in os.walk(tests_dir) for f in fs)
            if os.path.isdir(tests_dir)
            else []
        )
        text_ext = (
            ".py",
            ".sh",
            ".js",
            ".ts",
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".txt",
            ".md",
            ".csv",
            ".sql",
            ".cfg",
        )
        test_files.sort(key=lambda n: (n.count("/") > 1, n))  # tests/*.py, tests/test.sh first, then subfolders
        hidden = (
            "".join(
                f"===== {n} =====\n{read(n)}\n"
                if n.endswith(text_ext)
                else f"===== {n} =====\n(non-text file; the full copy is answer_key/{n})\n"
                for n in test_files
            )
            if test_files
            else (key.get("test_patch") or "")
        )
        f2p = re.findall(r"^def (test_\w+)", read("tests/test_outputs.py") or key.get("test_patch") or "", re.M)
        p2p = []
        write_text(
            f"{out_dir}/TASK.md", f"# {task}\n\n## Prompt the agent received\n{prompt or read('instruction.md')}\n"
        )
        write_text(f"{ak}/golden_solution.sh", golden)
        write_text(f"{ak}/hidden_tests.txt", hidden)
        for doc in TB_DOCS:
            if os.path.exists(os.path.join(folder, doc)):
                shutil.copy(os.path.join(folder, doc), f"{ak}/{doc}")
        for d in TB_DIRS:
            if os.path.isdir(os.path.join(folder, d)):
                copy_tree(os.path.join(folder, d), f"{ak}/{d}")
        if not os.path.isdir(folder):
            open(f"{ak}/MISSING_TASK_FOLDER.txt", "w").write(f"{folder} not found; golden/tests from the data row\n")
    else:
        golden = key.get("patch") or key.get("golden_patch") or ""
        hidden = key.get("test_patch") or ""
        f2p = as_list(key.get("fail_to_pass") or key.get("FAIL_TO_PASS"))
        p2p = as_list(key.get("pass_to_pass") or key.get("PASS_TO_PASS"))
        if not prompt:  # eval rows: reconstruct from the benchmark fields
            prompt = (
                f"{key.get('problem_statement', '')}\n\n## Requirements\n{key.get('requirements', '') or '(none)'}"
                f"\n\n## Interface\n{key.get('interface', '') or '(none)'}"
            )
        write_text(f"{out_dir}/TASK.md", f"# {task}\n\n## Prompt the agent received\n{prompt}\n")
        write_text(f"{ak}/golden.patch", golden)
        write_text(f"{ak}/hidden_tests.patch", hidden)
    fields = {k: v for k, v in key.items() if k not in SKIP_FIELDS and v not in (None, "", [], {})}
    json.dump(fields, open(f"{ak}/task_fields.json", "w"), indent=1, ensure_ascii=False, default=str)
    json.dump({"fail_to_pass": f2p, "pass_to_pass": p2p}, open(f"{ak}/tests.json", "w"), indent=1)
    return tb, golden, hidden, f2p, p2p


# ------------------------------------------------------------------------------------------------ build


def build(out_dir, task, key, rollouts, answer_key=None):
    """answer_key (service addition): the (tb, golden, hidden, f2p, p2p) that write_answer_key returned for this task
    when TASK.md and answer_key/ were already copied into out_dir from the service's node-local cache; None = write
    them here (the original behaviour)."""
    os.makedirs(out_dir, exist_ok=True)
    tb, golden, hidden, f2p, p2p = answer_key if answer_key is not None else write_answer_key(out_dir, task, key)
    gold_files, test_files = patch_files(golden), patch_files(hidden)
    pre = [f"# Pre-pass for {task}\n", "Diagnostic only (LABELING.md rule 4): confirm everything on the full turns.\n"]
    if tb:
        pre.append(
            "Terminal-Bench: no patch exists; the grader runs the hidden tests (answer_key/hidden_tests.txt) "
            "against the final container state. answer_key/ holds the task folder: solution/ (the reference, one "
            "valid way), tests/, environment/ (the starting container), and the designer's notes (README.md / "
            "SPEC.md / qc_check.md: what the tests verify, instruction-test alignment, solution approach) where "
            "the suite has them.\n"
        )
    pre.append(
        f"Golden patch files: {gold_files}\nHidden test files: {test_files}\n"
        f"fail_to_pass ({len(f2p)}): {f2p}\npass_to_pass: {len(p2p)} tests\n"
    )
    rew = lambda r: float(r.get("resolved") if r.get("resolved") is not None else (r.get("reward") or 0))
    n_pass = sum(rew(r) > 0.5 for _, r in rollouts)
    pre.append(f"Group: {len(rollouts)} rollouts, {n_pass} pass, {len(rollouts) - n_pass} fail.\n")
    group = {"task": task, "terminal_bench": tb, "rollouts": []}
    for k, (loc, r) in enumerate(rollouts):
        rd = f"{out_dir}/R{k}"
        os.makedirs(rd, exist_ok=True)
        out = (r.get("response") or {}).get("output") or []
        ords = call_ordinals(out)
        write_text(
            f"{rd}/trajectory.txt",
            "".join(
                f"===== #{i} [call {c}] {it.get('type')} =====\n{item_text(it)}\n\n"
                for i, (it, c) in enumerate(zip(out, ords))
            ),
        )
        patch = r.get("model_patch") or ""
        write_text(f"{rd}/final.patch", patch)
        tests = test_status(r, f2p + p2p, tb)
        json.dump(tests, open(f"{rd}/tests.json", "w"), indent=1)
        log = r.get("test_output") or ""
        write_text(f"{rd}/test_output.txt", log)
        reward = rew(r)
        killed = "signal: killed" in (r.get("opencode_run_stderr") or "")
        group["rollouts"].append(
            {
                "R": k,
                **loc,
                "reward": reward,
                "items": len(out),
                "calls": ords[-1] + 1 if ords else 0,
                "cap_killed": killed,
                "context_limited": bool(r.get("opencode_context_limited")),
                "mask_sample": bool(r.get("mask_sample")),
            }
        )
        failing = [t for t in f2p + p2p if tests.get(t) not in ("PASSED", None)]
        unknown = [t for t in f2p + p2p if t not in tests]
        flags = "; 3 h CAP KILL" if killed else ""
        flags += "; CONTEXT LIMITED" if r.get("opencode_context_limited") else ""
        flags += "; MASKED (outcome not measured: infrastructure problem)" if r.get("mask_sample") else ""
        if not tb and len(log) == 100_000:  # first collection's Gym SWE servers kept only the log's last 100k chars
            flags += (
                "; GRADER LOG TRUNCATED UPSTREAM (only its last 100,000 characters were kept; earlier failure "
                "reports may be missing: rely on tests.json for which tests failed and on the hidden test source)"
            )
        pre.append(
            f"\n## R{k} ({'PASS' if reward > 0.5 else 'FAIL'}): {len(out)} items, "
            f"{group['rollouts'][-1]['calls']} model calls{flags}\n"
        )
        tag = lambda t: (
            "hidden test (Terminal-Bench: every test is hidden)"
            if tb
            else "pass_to_pass: unchanged by the hidden patch (a real regression, rule 15)"
            if t in p2p
            else "fail_to_pass: added/modified by the hidden patch"
        )
        pre.append("- failing tests:" + (" none" if not failing else "") + "\n")
        pre += [f"  - {t} [{tag(t)}]\n" for t in failing]
        if unknown:
            pre.append(
                f"  - status unknown (not in the report / log): {len(unknown)}"
                + (
                    " <- the outcome cannot be traced to a test; read test_output.txt"
                    if reward <= 0.5 and not failing
                    else ""
                )
                + "\n"
            )
        blocks = failing_blocks(log, failing)
        if blocks:
            write_text(
                f"{rd}/failing_assertions.txt", "".join(f"===== {t} =====\n{body}\n\n" for t, body in blocks.items())
            )
            pre.append(
                f"- failure reports for {len(blocks)}/{len(failing)} failing tests, in full: R{k}/failing_assertions.txt "
                f"(read them there; the whole grader log is R{k}/test_output.txt)\n"
            )
        comp = [
            i
            for i, it in enumerate(out)
            if it.get("type") == "message" and it.get("role") != "assistant" and not text_of(it.get("content")).strip()
        ]
        pre.append(f"- compaction boundaries: {comp or 'none'}\n")
        if tb:
            continue
        pf = patch_files(patch)
        pre.append(
            f"- files in final patch: {pf}; golden files not touched: {[g for g in gold_files if g not in pf]}\n"
        )
        pre.append(
            "- golden files, first mention (call args = the agent asked for it; output = it appeared in an "
            "observation):\n"
        )
        first_edit = next(
            (
                i
                for i, it in enumerate(out)
                if it.get("type") == "function_call"
                and re.search(r"\b(edit|write|apply_patch)\b", str(it.get("name")))
            ),
            None,
        )
        pre.append(f"  (first edit/write call: #{first_edit})\n")
        for g in gold_files + test_files:
            base = os.path.basename(g)
            in_args = next(
                (
                    i
                    for i, it in enumerate(out)
                    if it.get("type") == "function_call" and base in str(it.get("arguments"))
                ),
                None,
            )
            in_out = next(
                (i for i, it in enumerate(out) if it.get("type") == "function_call_output" and base in item_text(it)),
                None,
            )
            pre.append(f"  - {g}: first in call args #{in_args}, first in an output #{in_out}\n")
        pre.append(
            "- edit replay (final-patch '+' lines -> items whose tool-call arguments contain the line; first and "
            "last such item):\n"
        )
        rows = []
        for f_, ln in added_lines(patch):
            s = ln.strip()
            if len(s) < 8:
                continue
            esc = json.dumps(s)[1:-1]
            hits = [
                i
                for i, it in enumerate(out)
                if it.get("type") == "function_call"
                and (s in str(it.get("arguments")) or esc in str(it.get("arguments")))
            ]
            rows.append((f_, s, hits))
        for f_, line_text, hits in rows:
            pre.append(
                f"  - {f_}: `{line_text}` -> "
                + (
                    f"#{hits[0]}" + (f" … #{hits[-1]}" if len(hits) > 1 else "")
                    if hits
                    else "not found in any call (written by a script/command?)"
                )
                + "\n"
            )
    write_text(f"{out_dir}/PREPASS.md", "".join(pre))
    json.dump(group, open(f"{out_dir}/group.json", "w"), indent=1)
    return group
