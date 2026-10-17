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

from resources_servers.lean_proof.lean_sandbox import DEFAULT_LEAN_PROJECT_DIR, LeanSandbox
from resources_servers.lean_proof.toolchain import TOOLCHAIN_PROBE, parse_lean_version


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
    """Inline provider config, so this script needs no Gym global config to run."""
    if args.provider == "enroot":
        # bypass_entrypoint passes `--rc /dev/null`, which enroot resolves inside the
        # container namespace before /dev is mounted. The image sets CMD, not ENTRYPOINT.
        return {"enroot": {"create": {"bypass_entrypoint": False}}}
    if args.provider != "opensandbox":
        return {args.provider: {}}
    return {
        "opensandbox": {
            "connection": {
                "domain": args.domain,
                "api_key": os.environ.get("OPENSANDBOX_API_KEY"),
                # The cells use self-signed certificates, as the shipped provider config does.
                "tls_verify": False,
                "use_server_proxy": True,
                "request_timeout_s": 300,
            },
            # Creating a sandbox includes pulling the image and waiting for readiness. A
            # multi-GB Lean image takes minutes on a cell that has not cached it, so these
            # mirror the shipped provider config rather than the SDK's short defaults.
            "create": {
                "request_timeout_s": 1200,
                "timeout_s": 1500,
                "connect_attempt_timeout_s": 1500,
                "retries": 10,
                "retry_delay_s": 5.0,
                "retry_max_delay_s": 90.0,
            },
        }
    }


def image_auth(args: argparse.Namespace) -> dict | None:
    """Registry credentials for a private image, which the cell needs to pull it."""
    if not args.image_username or not args.image_password:
        return None
    return {"username": args.image_username, "password": args.image_password}


async def compile_in(lean: LeanSandbox, code: str, timeout: float) -> dict:
    """Compile one file through the server's own path and flatten the result."""
    result = await lean.compile(code, timeout_s=timeout)
    return {
        "stdout": result.stdout or "",
        "stderr": result.stderr or "",
        "return_code": result.return_code,
        "error_type": result.error_type,
    }


def build_sandbox(args: argparse.Namespace) -> LeanSandbox:
    """The same LeanSandbox the server uses, so the gate exercises the real compile path."""
    return LeanSandbox(
        sandbox_provider=provider_config(args),
        sandbox_config={
            "image": args.image,
            "ttl_s": args.ttl,
            "ready_timeout_s": args.ready_timeout,
            "resources": {"cpu": args.cpu, "memory_mib": args.memory_mib},
            "provider_options": {
                **({"snapshot_id": args.snapshot_id} if args.snapshot_id else {}),
                **({"image_auth": auth} if (auth := image_auth(args)) else {}),
            },
            "metadata": {"benchmark": "leancat", "purpose": "check-sandbox"},
        },
        project_dir=args.project_dir,
        server_name="leancat-check-sandbox",
    )


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
    parser.add_argument(
        "--image-username",
        default=os.environ.get("LEAN_IMAGE_REGISTRY_USERNAME"),
        help="Registry username, for an image in a private registry.",
    )
    parser.add_argument(
        "--image-password",
        default=os.environ.get("LEAN_IMAGE_REGISTRY_PASSWORD"),
        help="Registry password or token; prefer $LEAN_IMAGE_REGISTRY_PASSWORD over the flag.",
    )
    parser.add_argument(
        "--project-dir",
        default=DEFAULT_LEAN_PROJECT_DIR,
        help="Lake project to compile in; the shipped image builds Mathlib there.",
    )
    parser.add_argument("--ttl", type=float, default=7200)
    parser.add_argument("--ready-timeout", type=float, default=1200)
    parser.add_argument("--cpu", type=float, default=4)
    parser.add_argument("--memory-mib", type=int, default=16384)
    parser.add_argument("--limit", type=int, default=None, help="Check only the first N problems.")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    # Gym's global aiohttp client parses the CLI through Hydra the first time it is used,
    # which happens inside the sandbox provider's first request. It would reject this
    # script's own flags. They are consumed by now, so take them out of its way.
    sys.argv = sys.argv[:1]

    if not args.snapshot_id and not args.image:
        print("FAIL: pass --snapshot-id or --image; there is no default Lean environment.", file=sys.stderr)
        return 2

    lean = build_sandbox(args)
    try:
        await lean.start()
    except Exception as exc:  # noqa: BLE001 - the reason matters more than the type here
        print(f"FAIL: could not start a sandbox: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    # Report the toolchain first so per-problem failures below can be read in context.
    probe = await compile_in(lean, TOOLCHAIN_PROBE, args.timeout)
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
            result = await compile_in(lean, statement, args.timeout)
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
        print("Almost always this means its Mathlib is not v4.19.0. See the 'Requirements' section of README.md.\n")
        for problem_id, level, reason in failures[:10]:
            print(f"  {problem_id} [{level}]: {reason[:140]}")
        return 1

    print("Sandbox looks correct: Mathlib can state every LeanCat problem.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
