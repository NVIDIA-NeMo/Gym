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

"""Check a Lean sandbox is the right one for LeanCat, before spending anything on inference.

Compiles the 100 reference statements **unmodified**. Each still contains its ``sorry``, so
each must compile with a "declaration uses 'sorry'" warning and no errors. A hard error means
the sandbox cannot even state the problem, which in practice means its Mathlib is not v4.19.0.

This matters because the failure is silent otherwise. The stock NeMo-Skills sandbox pins
Mathlib v4.12.0; LeanCat's ``CategoryTheory`` statements fail on it with ordinary-looking
"unknown identifier" errors. A full eval against that sandbox produces a plausible near-zero
score that looks like a model result and is not one.

No model and no GPU needed.

Usage:
    python check_sandbox.py                      # all 100
    python check_sandbox.py --limit 5            # quick smoke test
    python check_sandbox.py --host h --port 6000
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

from resources_servers.leancat.sandbox_client import Lean4SandboxClient


DATA_DIR = Path(__file__).absolute().parent / "data"


def load_statements(limit: int | None) -> list[tuple[str, str, str]]:
    for name in ("train.jsonl", "example.jsonl"):
        path = DATA_DIR / name
        if path.exists():
            break
    else:
        raise SystemExit("No dataset found. Run prepare_leancat.py first.")

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    out = [
        (
            r["verifier_metadata"]["problem_id"],
            r["verifier_metadata"]["level"],
            r["verifier_metadata"]["formal_statement"],
        )
        for r in rows
    ]
    return out[:limit] if limit else out


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--limit", type=int, default=None, help="Check only the first N problems.")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    client = Lean4SandboxClient(host=args.host, port=args.port, max_output_characters=4000)
    if not await client.health_check():
        print(f"FAIL: no healthy sandbox at {args.host}:{args.port}", file=sys.stderr)
        return 2

    problems = load_statements(args.limit)
    print(f"Compiling {len(problems)} reference statements against {args.host}:{args.port}\n")

    semaphore = asyncio.Semaphore(args.concurrency)
    failures: list[tuple[str, str, str]] = []

    async def check(problem_id: str, level: str, statement: str) -> None:
        async with semaphore:
            result = await client.execute_lean4(code=statement, timeout=args.timeout)
        combined = f"{result.get('stdout', '')}\n{result.get('stderr', '')}"
        status = result.get("process_status", "unknown")

        if status != "completed":
            failures.append((problem_id, level, f"sandbox status {status!r}"))
            print(f"  {problem_id} [{level:<6}] FAIL  {status}")
        elif "error:" in combined.lower():
            first = next((ln for ln in combined.splitlines() if "error:" in ln.lower()), "")
            failures.append((problem_id, level, first.strip()))
            print(f"  {problem_id} [{level:<6}] FAIL  {first.strip()[:100]}")
        elif re.search(r"\bsorry\b", combined, re.I):
            print(f"  {problem_id} [{level:<6}] ok    (sorry warning, as expected)")
        else:
            # No error and no sorry warning: the file did not carry its placeholder through,
            # which means this check is not testing what it thinks it is.
            failures.append((problem_id, level, "compiled with no sorry warning -- unexpected"))
            print(f"  {problem_id} [{level:<6}] ODD   no sorry warning")

    await asyncio.gather(*(check(p, lv, s) for p, lv, s in problems))

    print(f"\n{len(problems) - len(failures)}/{len(problems)} reference statements compiled as expected.")
    if failures:
        print(f"\n{len(failures)} FAILED — this sandbox is not usable for LeanCat.")
        print("Almost always this means its Mathlib is not v4.19.0. See build_sandbox.sh.\n")
        for problem_id, level, reason in failures[:10]:
            print(f"  {problem_id} [{level}]: {reason[:140]}")
        return 1

    print("Sandbox looks correct: Mathlib can state every LeanCat problem.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
