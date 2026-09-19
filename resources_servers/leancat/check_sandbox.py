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

"""Pre-flight check that a Lean sandbox can state every LeanCat problem.

Compiles the 100 reference statements unmodified. Each still contains its ``sorry``, so each
must compile with a "declaration uses 'sorry'" warning and no errors. A hard error means the
sandbox's Mathlib is not v4.19.0. No model or GPU needed.

Usage:
    python check_sandbox.py --host <node> --port 6000
    python check_sandbox.py --host <node> --port 6000 --limit 5
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

from resources_servers.math_formal_lean.sandbox_client import Lean4SandboxClient
from resources_servers.math_formal_lean.toolchain import TOOLCHAIN_PROBE, parse_lean_version


EXPECTED_LEAN_VERSION = "4.19.0"

DATA_DIR = Path(__file__).absolute().parent / "data"
REPO_ROOT = Path(__file__).absolute().parents[2]
# Prefer the full 100-problem set; fall back to the committed 5-row example for a smoke test.
DATASET_CANDIDATES = (
    REPO_ROOT / "benchmarks/leancat/data/leancat_benchmark.jsonl",
    DATA_DIR / "example.jsonl",
)


def load_statements(limit: int | None) -> list[tuple[str, str, str]]:
    for path in DATASET_CANDIDATES:
        if path.exists():
            break
    else:
        raise SystemExit("No dataset found. Run benchmarks/leancat/prepare.py first.")
    if path != DATASET_CANDIDATES[0]:
        print(f"WARNING: {DATASET_CANDIDATES[0]} is missing; checking only the {path.name} rows.")

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    # Rows are flat (see prepare.py); the nested form is tolerated too.
    out = [
        (
            (r.get("verifier_metadata") or r)["problem_id"],
            (r.get("verifier_metadata") or r)["level"],
            (r.get("verifier_metadata") or r)["formal_statement"],
        )
        for r in rows
    ]
    return out[:limit] if limit else out


def build_client(args: argparse.Namespace):
    return Lean4SandboxClient(host=args.host, port=args.port, max_output_characters=4000, timeout_buffer=30.0)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--limit", type=int, default=None, help="Check only the first N problems.")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    client = build_client(args)
    if not await client.health_check():
        print(f"FAIL: Lean is not reachable via {type(client).__name__}", file=sys.stderr)
        return 2

    # Report the toolchain first so per-problem failures below can be read in context.
    probe = await client.execute_lean4(code=TOOLCHAIN_PROBE, timeout=args.timeout)
    found = parse_lean_version(probe)
    if found is None:
        print("FAIL: `import Mathlib` did not compile -- this sandbox cannot state any LeanCat problem.")
        print(f"       stderr: {probe.get('stderr', '')[:400]}")
        return 2
    if found != EXPECTED_LEAN_VERSION:
        print(f"WARNING: sandbox is Lean/Mathlib {found}, expected {EXPECTED_LEAN_VERSION}.")
        print("         Statements written against v4.19.0 may fail to compile below.\n")
    else:
        print(f"Sandbox toolchain: Lean/Mathlib {found} (expected {EXPECTED_LEAN_VERSION})\n")

    problems = load_statements(args.limit)
    print(f"Compiling {len(problems)} reference statements via {type(client).__name__}\n")

    semaphore = asyncio.Semaphore(args.concurrency)
    failures: list[tuple[str, str, str]] = []

    async def check(problem_id: str, level: str, statement: str) -> None:
        async with semaphore:
            result = await client.execute_lean4(code=statement, timeout=args.timeout)
        combined = f"{result.get('stdout', '')}\n{result.get('stderr', '')}"
        status = result.get("process_status", "unknown")
        return_code = result.get("return_code")

        if status != "completed":
            failures.append((problem_id, level, f"sandbox status {status!r}"))
            print(f"  {problem_id} [{level:<6}] FAIL  {status}")
        elif return_code not in (None, 0):
            first = next((ln for ln in combined.splitlines() if "error:" in ln.lower()), f"exit {return_code}")
            failures.append((problem_id, level, first.strip()))
            print(f"  {problem_id} [{level:<6}] FAIL  {first.strip()[:100]}")
        elif "error:" in combined.lower():
            first = next((ln for ln in combined.splitlines() if "error:" in ln.lower()), "")
            failures.append((problem_id, level, first.strip()))
            print(f"  {problem_id} [{level:<6}] FAIL  {first.strip()[:100]}")
        elif re.search(r"\bsorry\b", combined, re.I):
            print(f"  {problem_id} [{level:<6}] ok    (sorry warning, as expected)")
        else:
            # No error and no sorry warning: the placeholder did not survive, so this check
            # did not test what it should have.
            failures.append((problem_id, level, "compiled with no sorry warning -- unexpected"))
            print(f"  {problem_id} [{level:<6}] ODD   no sorry warning")

    await asyncio.gather(*(check(p, lv, s) for p, lv, s in problems))

    print(f"\n{len(problems) - len(failures)}/{len(problems)} reference statements compiled as expected.")
    if failures:
        print(f"\n{len(failures)} FAILED — this sandbox is not usable for LeanCat.")
        print(
            "Almost always this means its Mathlib is not v4.19.0. See README.md, 'Get a sandbox on Mathlib v4.19.0'.\n"
        )
        for problem_id, level, reason in failures[:10]:
            print(f"  {problem_id} [{level}]: {reason[:140]}")
        return 1

    print("Sandbox looks correct: Mathlib can state every LeanCat problem.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
