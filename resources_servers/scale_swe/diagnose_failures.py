#!/usr/bin/env python3
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
"""Separate golden-patch failures caused by the sandbox from failures of the row itself.

This exists because `evaluation_completed` does not answer the question. A build that is
OOM-killed, or a test suite that runs out of disk, still produces a parsed log and a verdict --
the tests simply failed. That grades identically to a genuinely broken golden patch, so raising
the memory limit would silently change the "supported" set.

The signatures below are read out of the captured test output, so a failure is attributed to
resources only on direct evidence, never inferred from the fact that it failed.

    python resources_servers/scale_swe/diagnose_failures.py results/.../pass1_job_X.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# Ordered: the first match wins, so the most specific cause is reported.
RESOURCE_SIGNATURES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "disk_full",
        ("no space left on device", "enospc", "disk quota exceeded", "write error: no space"),
    ),
    (
        "oom",
        (
            "out of memory",
            "outofmemoryerror",
            "cannot allocate memory",
            "oom-kill",
            "oom killed",
            "killed process",
            "memoryerror",
            "signal: killed",
            "runtime: out of memory",
            "fatal error: out of memory",
            "std::bad_alloc",
            "exit status 137",
            "javaheapspace",
            "java heap space",
            "gc overhead limit exceeded",
        ),
    ),
    (
        "process_killed",
        # A bare SIGKILL with no allocator message. Usually the cgroup OOM killer, but the
        # evidence is weaker, so it is reported separately rather than folded into `oom`.
        ("killed", "sigkill", "terminated by signal 9"),
    ),
    (
        "too_many_files",
        ("too many open files", "emfile"),
    ),
    (
        "network",
        # An egress fault -- pip/git failing to reach a registry -- rather than evidence the
        # golden patch is wrong. Confirm against the actual output before trusting a raised
        # limit or a retry policy change: SWE-rebench's version of this bucket turned out to be
        # one specific unreachable host, not a sandbox-wide restriction (see that server's
        # diagnose_failures.py history for the false starts before the real cause was found).
        (
            "could not resolve host",
            "connection refused",
            "network is unreachable",
            "temporary failure in name resolution",
            "etimedout",
            "econnreset",
            "proxyconnect",
        ),
    ),
)


def classify_output(text: str) -> str | None:
    """Return the resource cause evidenced in the output, or None."""
    lowered = text.lower()
    for label, needles in RESOURCE_SIGNATURES:
        if any(needle in lowered for needle in needles):
            return label
    return None


def _rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rollouts_jsonl", type=Path)
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=1200.0,
        help="The configured evaluation timeout; runs within --near of it are treated as timing out.",
    )
    parser.add_argument("--near", type=float, default=0.9, help="Fraction of the timeout that counts as 'near'.")
    parser.add_argument("--show", type=int, default=4, help="Example instance ids to print per cause.")
    args = parser.parse_args()

    if not args.rollouts_jsonl.exists():
        print(f"no such file: {args.rollouts_jsonl}", file=sys.stderr)
        return 2

    total = 0
    resolved = 0
    failures: list[dict[str, Any]] = []
    durations: list[float] = []

    for row in _rows(args.rollouts_jsonl):
        total += 1
        duration = float(row.get("patch_verification_time_taken") or 0.0)
        durations.append(duration)
        if row.get("resolved"):
            resolved += 1
            continue
        failures.append(row)

    if total == 0:
        print("no rows", file=sys.stderr)
        return 2

    causes: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    by_language: dict[str, Counter[str]] = defaultdict(Counter)

    for row in failures:
        instance_id = str(row.get("instance_id") or "?")
        language = str(row.get("language") or "unknown")
        duration = float(row.get("patch_verification_time_taken") or 0.0)
        output = str(row.get("test_output") or "")
        error = str(row.get("error") or "")

        if not row.get("evaluation_completed", False):
            cause = "no_verdict"
        elif duration >= args.timeout_s * args.near:
            # Ran to (or near) the ceiling. With CPU requests well below limits this is the
            # shape CPU starvation takes: the work is not wrong, it just never finished.
            cause = "timeout"
        elif (row.get("test_results") or {}).get("tests_observed", 0) == 0 and (row.get("test_results") is not None):
            # pytest saw zero tests, so nothing ran: pre_commands or the patch/f2p-script
            # apply died before the suite started, or the named test files never exist in
            # this checkout. A different thing from "the tests failed" -- the golden patch is
            # not implicated either way, so this is not scored as evidence against the row.
            cause = classify_output(output + "\n" + error) or "no_tests_ran"
        else:
            cause = classify_output(output + "\n" + error) or "row_failure"

        causes[cause] += 1
        by_language[language][cause] += 1
        if len(examples[cause]) < args.show:
            examples[cause].append(f"{instance_id} ({language}, {duration:.0f}s)")

    # Causes that are about the environment rather than the row. `no_tests_ran` is included:
    # a suite that never started says nothing about whether the golden patch is correct.
    resource_causes = {
        "oom",
        "disk_full",
        "process_killed",
        "too_many_files",
        "network",
        "timeout",
        "no_verdict",
        "no_tests_ran",
    }
    resource_total = sum(count for cause, count in causes.items() if cause in resource_causes)

    print(f"rows: {total}   resolved: {resolved} ({100 * resolved / total:.1f}%)   failures: {len(failures)}")
    if durations:
        ordered = sorted(durations)
        p50 = ordered[len(ordered) // 2]
        p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
        print(
            f"verification seconds: p50={p50:.0f}  p95={p95:.0f}  max={ordered[-1]:.0f}  (timeout {args.timeout_s:.0f})"
        )
    print()
    print("failure attribution:")
    for cause, count in causes.most_common():
        marker = "ENV" if cause in resource_causes else "row"
        share = 100 * count / max(len(failures), 1)
        print(f"  {cause:<16} {count:>5} ({share:5.1f}% of failures)  [{marker}]")
        for example in examples[cause]:
            print(f"      e.g. {example}")

    print()
    if not failures:
        print("VERDICT: no failures to attribute")
    elif resource_total == 0:
        print("VERDICT: no failure shows resource evidence — these look like genuine row failures.")
        print("         Raising sandbox limits would not change the supported set.")
    else:
        share = 100 * resource_total / len(failures)
        print(
            f"VERDICT: {resource_total}/{len(failures)} failures ({share:.1f}%) show resource or environment evidence."
        )
        print("         Raise the relevant limit and re-measure before treating these rows as broken.")

    if by_language:
        print("\n  per-language failure causes:")
        for language, counts in sorted(by_language.items(), key=lambda kv: -sum(kv[1].values())):
            summary = " ".join(f"{cause}={count}" for cause, count in counts.most_common())
            print(f"    {language:<10} {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
