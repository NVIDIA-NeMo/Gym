#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""DeNovoSWE in-container evaluator.

Runs ``pytest`` per-file on the ``passed_ptp`` test IDs and computes a pass
rate, then writes a JSON report + a 0/1 reward file. Designed to be bind-
mounted into the eval Apptainer container by :class:`DeNovoSWEDatasetProcessor`
and invoked AFTER:
  1. The repo has been checked out at ``parent_commit``,
  2. (Optional, agent path only) the agent's patch has been applied,
  3. All pre-existing test files have been removed,
  4. ``test_patch`` has been applied,
  5. The package has been reinstalled with ``pip install -e .``.

Reads (paths are bind-mounted by the harness):
  /root/denovoswe_meta.json   {"workdir": "...", "passed_ptp": [...], "failed_ptp": [...]}

Writes:
  /trajectories_mount/eval_results/report.json
  /trajectories_mount/eval_results/reward.txt          (\"1\" iff every ``passed_ptp`` test passes; else \"0\")

The Python stdlib is intentionally the only dependency — eval images vary
widely in what's installed beyond pytest/pip.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


META_PATH = Path("/root/denovoswe_meta.json")
REPORT_DIR = Path("/trajectories_mount/eval_results")
PER_FILE_TIMEOUT = int(os.environ.get("DENOVOSWE_PER_FILE_TIMEOUT", "600"))


def _load_meta() -> Dict[str, Any]:
    return json.loads(META_PATH.read_text())


def _group_by_file(test_ids: List[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = defaultdict(list)
    for tid in test_ids:
        f = tid.split("::", 1)[0]
        groups[f].append(tid)
    return dict(groups)


# Match pytest's final summary line. Tolerates both the verbose form
# ("===== 5 passed, 1 warning in 0.16s =====") and the quiet -q form
# ("19 passed in 0.16s") — the latter omits the ``=`` brackets entirely.
_SUMMARY_LINE = re.compile(
    r"^\s*=*\s*((?:\d+\s+(?:passed|failed|error|errors|skipped|xfailed|xpassed|deselected|warning|warnings)(?:\s*,\s*)?)+)\s+in\s+[\d.]+s\s*=*\s*$",
    re.MULTILINE,
)
_NUM_KIND = re.compile(r"(\d+)\s+(passed|failed|error|errors|skipped|xfailed|xpassed|deselected)")


def _parse_pytest_counts(output: str) -> Dict[str, int]:
    """Best-effort parse of ``passed`` / ``failed`` / ``errors`` from pytest's
    final summary line. Returns zeros if no summary was found (e.g. crash).
    """
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    matches = list(_SUMMARY_LINE.finditer(output))
    if not matches:
        return counts
    summary = matches[-1].group(1)
    for m in _NUM_KIND.finditer(summary):
        n = int(m.group(1))
        kind = m.group(2)
        if kind == "errors":
            kind = "error"
        if kind == "error":
            counts["errors"] = counts.get("errors", 0) + n
        elif kind in ("passed", "failed", "skipped"):
            counts[kind] = counts.get(kind, 0) + n
    return counts


def _collect_test_ids(workdir: str, filepath: str, timeout: int = 120) -> Optional[List[str]]:
    """Pre-flight collection: ``pytest --collect-only -q <filepath>``.

    Returns the list of collected test ids on success, or None if collection
    itself fails (import error in the file, syntax error, etc.). Mirrors
    AweAgent's ``collect_test_ids`` mitigation: a single stale id in
    ``passed_ptp`` (parametrize label drift, hypothesis seed change, etc.)
    aborts the ENTIRE batch with exit 4 / "no tests ran" — so we collect
    first, intersect with passed_ptp, then run only the resolvable ids.
    """
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        "--no-header",
        "-p",
        "no:cacheprovider",
        "-o",
        "addopts=",
        filepath,
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout,
            errors="replace",
        )
    except (subprocess.TimeoutExpired, Exception):
        return None
    if proc.returncode not in (0, 5):
        # Exit 5 = "no tests collected" (treated as no-op, not failure).
        # Non-zero / non-5 means collection itself crashed (import error etc.) —
        # signal that by returning None so the caller scores the whole file as
        # errors (matching what pytest would do when run with the raw ids).
        return None
    # Permissive id extraction: pytest's ``-q --collect-only`` typically emits
    # ``<path>::<rest>`` one per line, but format varies by version / plugin
    # set. Accept ANY line that contains ``::`` and isn't a summary/warning;
    # the caller intersects with ``passed_ptp`` anyway, so false positives are
    # harmlessly filtered out — false NEGATIVES (valid ids we drop) would
    # cause v1→v2 regressions, so err permissive.
    ids: List[str] = []
    for raw_line in (proc.stdout or "").splitlines():
        line = raw_line.strip()
        if not line or "::" not in line:
            continue
        low = line.lower()
        if low.startswith(("warning", "error", "deprecation")):
            continue
        if "collected" in low and " in " in low and low.rstrip().endswith("s"):
            continue
        ids.append(line)
    return ids


def _run_pytest(workdir: str, test_ids: List[str], timeout: int) -> Tuple[Dict[str, int], str, int]:
    """Run ``python -m pytest <ids>`` in ``workdir``; return (counts, output, exit)."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--tb=short",
        "-q",
        "--no-header",
        "-p",
        "no:cacheprovider",
        "-o",
        "addopts=",  # neutralize any project-level coverage / xdist opts
        *test_ids,
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout,
            errors="replace",
        )
        out = (proc.stdout or "") + ("\n--- stderr ---\n" + proc.stderr if proc.stderr else "")
        return _parse_pytest_counts(out), out, proc.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + (e.stderr or "") + f"\n[TIMEOUT after {timeout}s]"
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        return {"passed": 0, "failed": 0, "errors": len(test_ids), "skipped": 0}, out, -1
    except Exception as e:
        return {"passed": 0, "failed": 0, "errors": len(test_ids), "skipped": 0}, f"[crash] {e!r}", -2


def _head_tail(text: str, head: int = 4000, tail: int = 4000) -> str:
    if len(text) <= head + tail + 64:
        return text
    return text[:head] + f"\n... [trimmed {len(text) - head - tail} bytes] ...\n" + text[-tail:]


def main() -> int:
    start = time.monotonic()
    meta = _load_meta()
    workdir = str(meta.get("workdir") or "")
    passed_ptp: List[str] = list(meta.get("passed_ptp") or [])
    total_expected = len(passed_ptp)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not workdir or not Path(workdir).is_dir():
        report = {
            "_test_completed": True,
            "error": f"workdir missing: {workdir!r}",
            "pass_rate": 0.0,
            "passed": 0,
            "failed": 0,
            "errors": total_expected,
            "total_expected": total_expected,
            "reward": "0",
            "duration": time.monotonic() - start,
        }
        (REPORT_DIR / "report.json").write_text(json.dumps(report, indent=2))
        (REPORT_DIR / "reward.txt").write_text("0")
        print("DeNovoSWE eval: workdir missing → reward=0", flush=True)
        return 0

    if not passed_ptp:
        report = {
            "_test_completed": True,
            "error": "no passed_ptp test ids",
            "pass_rate": 0.0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "total_expected": 0,
            "reward": "0",
            "duration": time.monotonic() - start,
        }
        (REPORT_DIR / "report.json").write_text(json.dumps(report, indent=2))
        (REPORT_DIR / "reward.txt").write_text("0")
        print("DeNovoSWE eval: empty passed_ptp → reward=0", flush=True)
        return 0

    groups = _group_by_file(passed_ptp)
    per_file: Dict[str, Any] = {}
    total_passed = total_failed = total_errors = 0
    for fpath, ids in groups.items():
        # Pre-flight collection: enumerate the ids pytest can ACTUALLY resolve
        # right now, intersect with passed_ptp, run only the intersection.
        # Mirrors AweAgent's mitigation against parametrize-label drift /
        # hypothesis-seed shifts that would otherwise abort the entire batch
        # (one stale id → exit 4 / "no tests ran" → all-or-nothing zero).
        collected = _collect_test_ids(workdir, fpath, timeout=120)
        if collected is None:
            # Collection itself failed (import error in the test file, syntax
            # error, missing dep). Fall through with the original id list so
            # the ensuing pytest run surfaces the diagnostic in its output;
            # every id in the file counts as an error.
            runnable = list(ids)
            missing: List[str] = []
            collected_n = 0
        else:
            collected_set = set(collected)
            runnable = [t for t in ids if t in collected_set]
            missing = [t for t in ids if t not in collected_set]
            collected_n = len(collected)
            # Safety net: if collect-only parsing yielded ZERO intersection
            # despite returning ids (parser failure / format drift), fall
            # back to running the original id list — same behaviour as if
            # collect-only had failed outright. Prevents a parsing bug from
            # silently scoring every test as missing.
            if not runnable and collected_n > 0 and ids:
                runnable = list(ids)
                missing = []

        if runnable:
            counts, output, exit_code = _run_pytest(workdir, runnable, PER_FILE_TIMEOUT)
        else:
            counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
            output = (
                f"[no runnable test ids for {fpath}; "
                f"expected={len(ids)} collected={collected_n} missing={len(missing)}]"
            )
            exit_code = 5  # treat as "nothing to run"

        # Cap "passed" at expected count to prevent parametrize-expansion
        # inflation (a single id can collect to N parametrizations).
        passed = min(counts.get("passed", 0), len(ids))
        failed = counts.get("failed", 0)
        errors = counts.get("errors", 0)
        # Account missing ids individually as errors (don't crash the batch).
        errors += len(missing)
        # If pytest non-zero AND no signal parsed, score remaining runnable
        # ids as errors (e.g. SIGSEGV mid-batch).
        if (
            exit_code not in (0, 5)
            and (failed + counts.get("errors", 0)) == 0
            and counts.get("passed", 0) == 0
            and runnable
        ):
            errors += len(runnable)
        total_passed += passed
        total_failed += failed
        total_errors += errors
        per_file[fpath] = {
            "expected": len(ids),
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "exit": exit_code,
            "collected": collected_n,
            "missing_ids": len(missing),
            "collection_failed": collected is None,
            "output_tail": _head_tail(output, head=2000, tail=2000),
        }

    # Account for any expected-but-not-counted ids (collection skew etc.).
    accounted = total_passed + total_failed + total_errors
    if accounted < total_expected:
        total_errors += total_expected - accounted

    all_passed = (total_passed == total_expected) and total_failed == 0 and total_errors == 0
    pass_rate = (total_passed / total_expected) if total_expected else 0.0
    reward = "1" if all_passed else "0"

    report = {
        "_test_completed": True,
        "passed": total_passed,
        "failed": total_failed,
        "errors": total_errors,
        "total_expected": total_expected,
        "pass_rate": pass_rate,
        "accepted": all_passed,
        "reward": reward,
        "expected_coverage_percent": meta.get("expected_coverage_percent"),
        "per_file": per_file,
        "duration": time.monotonic() - start,
    }
    (REPORT_DIR / "report.json").write_text(json.dumps(report, indent=2))
    (REPORT_DIR / "reward.txt").write_text(reward)
    print(
        f"DeNovoSWE eval: passed={total_passed}/{total_expected} "
        f"failed={total_failed} errors={total_errors} pass_rate={pass_rate:.3f} reward={reward}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
