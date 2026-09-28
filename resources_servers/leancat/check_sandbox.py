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
    python check_sandbox.py --snapshot-id <id>
    python check_sandbox.py --image <registry/image:tag> --limit 5
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.config import resolve_provider_config
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


def provider_config(args: argparse.Namespace) -> dict:
    """Inline provider config, so this script needs no Gym global config to run.

    Defaults to OpenSandbox against `OPENSANDBOX_DOMAIN`/`OPENSANDBOX_API_KEY`. The cells use
    self-signed certificates, hence `tls_verify: false` -- the same default the shipped
    provider config carries.
    """
    if args.provider == "enroot":
        # bypass_entrypoint passes `--rc /dev/null`, which enroot resolves inside the
        # container namespace before /dev is mounted -- it fails with "No such file or
        # directory". The Lean image sets CMD, not ENTRYPOINT, so there is nothing to bypass.
        return {"enroot": {"create": {"bypass_entrypoint": False}}}
    if args.provider != "opensandbox":
        return {args.provider: {}}
    return {
        "opensandbox": {
            "connection": {
                "domain": args.domain,
                "api_key": os.environ.get("OPENSANDBOX_API_KEY"),
                "tls_verify": False,
            }
        }
    }


async def start_sandbox(args: argparse.Namespace) -> AsyncSandbox:
    """Start one sandbox from a snapshot or image, exactly as the server does."""
    sandbox = AsyncSandbox(resolve_provider_config(provider_config(args)))
    await sandbox.start(
        SandboxSpec(
            image=args.image,
            ttl_s=args.ttl,
            ready_timeout_s=args.ready_timeout,
            workdir=args.project_dir,
            resources=SandboxResources.from_mapping({"cpu": args.cpu, "memory_mib": args.memory_mib}),
            provider_options={"snapshot_id": args.snapshot_id} if args.snapshot_id else {},
            metadata={"benchmark": "leancat", "purpose": "check-sandbox"},
        )
    )
    return sandbox


async def run_lean(sandbox: AsyncSandbox, code: str, project_dir: str, timeout: float) -> dict:
    """Compile one file, the same way app.py does: heredoc in, `lake env lean`, temp file out."""
    import uuid

    path = f"check_{uuid.uuid4().hex}.lean"
    delimiter = f"LEANCAT_EOF_{uuid.uuid4().hex}"
    command = (
        f"cat > {path} <<'{delimiter}'\n{code}\n{delimiter}\n"
        f"lake env lean {path}; status=$?; rm -f {path}; exit $status"
    )
    result = await sandbox.exec(command, cwd=project_dir, timeout_s=timeout + 30)
    return {
        "stdout": result.stdout or "",
        "stderr": result.stderr or "",
        "return_code": result.return_code,
        "error_type": result.error_type,
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="opensandbox", help="Sandbox provider name.")
    parser.add_argument(
        "--domain",
        default=os.environ.get("OPENSANDBOX_DOMAIN"),
        help="OpenSandbox endpoint; defaults to $OPENSANDBOX_DOMAIN.",
    )
    parser.add_argument("--snapshot-id", default=os.environ.get("LEANCAT_SANDBOX_SNAPSHOT_ID"))
    parser.add_argument("--image", default=None, help="Image URI, when there is no snapshot.")
    parser.add_argument("--project-dir", default="/lean4/my_project", help="Lake project to compile in.")
    parser.add_argument("--ttl", type=float, default=7200)
    parser.add_argument("--ready-timeout", type=float, default=1200)
    parser.add_argument("--cpu", type=float, default=4)
    parser.add_argument("--memory-mib", type=int, default=16384)
    parser.add_argument("--limit", type=int, default=None, help="Check only the first N problems.")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    if not args.snapshot_id and not args.image:
        print("FAIL: pass --snapshot-id or --image; there is no default Lean environment.", file=sys.stderr)
        return 2

    try:
        sandbox = await start_sandbox(args)
    except Exception as exc:  # noqa: BLE001 - the reason matters more than the type here
        print(f"FAIL: could not start a sandbox: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    # Report the toolchain first so per-problem failures below can be read in context.
    probe = await run_lean(sandbox, TOOLCHAIN_PROBE, args.project_dir, args.timeout)
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
    print(f"Compiling {len(problems)} reference statements in the sandbox\n")

    semaphore = asyncio.Semaphore(args.concurrency)
    failures: list[tuple[str, str, str]] = []

    async def check(problem_id: str, level: str, statement: str) -> None:
        async with semaphore:
            result = await run_lean(sandbox, statement, args.project_dir, args.timeout)
        combined = f"{result.get('stdout', '')}\n{result.get('stderr', '')}"
        error_type = result.get("error_type")
        return_code = result.get("return_code")

        if error_type:
            failures.append((problem_id, level, f"sandbox error {error_type!r}"))
            print(f"  {problem_id} [{level:<6}] FAIL  {error_type}")
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
