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

Three ways to reach Lean, matching the three ways the server can be configured:

    --lean-prefix DIR   Run `lake env lean` directly, no sandbox at all. Use this first,
                        right after setup_lean.sh: it isolates "is Mathlib correct?" from
                        "is my sandbox wiring correct?", so a failure has one meaning.
    --enroot-image IMG  Go through nemo_gym.sandbox's enroot provider, exercising the same
                        path configs/leancat_enroot.yaml uses. Needs --lean-prefix too.
    --host/--port       Talk to a NeMo-Skills HTTP sandbox (the default backend).

Usage:
    python check_sandbox.py --lean-prefix /lustre/<...>/lean4-mathlib-v4.19.0
    python check_sandbox.py --lean-prefix /lustre/<...> --enroot-image base.sqsh
    python check_sandbox.py --host h --port 6000 --limit 5
"""

import argparse
import asyncio
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

from resources_servers.leancat.sandbox_client import GymSandboxLean4Client, Lean4SandboxClient


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


class LocalLeanClient:
    """Run `lake env lean` as a subprocess. Same contract as the sandbox clients.

    Deliberately no container: this answers "does this Mathlib build state every LeanCat
    problem?" on its own, so a failure here is never ambiguous between a bad Mathlib and a
    bad mount.
    """

    def __init__(self, lean_prefix: Path, max_output_characters: int = 4000):
        self.project_dir = lean_prefix / "my_project"
        self.elan_bin = lean_prefix / "elan" / "bin"
        self.max_output_characters = max_output_characters

    async def health_check(self, timeout: float = 5.0) -> bool:
        if not self.project_dir.is_dir():
            print(f"No Lean project at {self.project_dir} -- run setup_lean.sh first.", file=sys.stderr)
            return False
        if not (self.elan_bin / "lake").exists() and shutil.which("lake") is None:
            print(f"No `lake` in {self.elan_bin} or on PATH.", file=sys.stderr)
            return False
        return True

    async def execute_lean4(self, code: str, timeout: float = 300.0) -> Dict[str, Any]:
        import os

        with tempfile.NamedTemporaryFile("w", suffix=".lean", delete=False, encoding="utf-8") as handle:
            handle.write(code)
            path = handle.name
        env = {**os.environ, "PATH": f"{self.elan_bin}{os.pathsep}{os.environ.get('PATH', '')}"}
        try:
            proc = await asyncio.create_subprocess_exec(
                "lake",
                "env",
                "lean",
                path,
                cwd=str(self.project_dir),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                return {"process_status": "timeout", "stdout": "", "stderr": "", "return_code": None}
        except OSError as exc:
            return {"process_status": "error", "stdout": "", "stderr": str(exc), "return_code": None}
        finally:
            Path(path).unlink(missing_ok=True)

        return {
            "process_status": "completed",
            "stdout": stdout.decode("utf-8", "replace")[: self.max_output_characters],
            "stderr": stderr.decode("utf-8", "replace")[: self.max_output_characters],
            "return_code": proc.returncode,
        }


def build_client(args: argparse.Namespace):
    if args.enroot_image:
        if not args.lean_prefix:
            raise SystemExit("--enroot-image also needs --lean-prefix (Lean comes from the mount).")
        return GymSandboxLean4Client(
            provider={"enroot": {"exec": {"concurrency": args.concurrency, "default_timeout_s": args.timeout + 30}}},
            spec={
                "image": args.enroot_image,
                "provider_options": {"mounts": [f"{Path(args.lean_prefix).absolute()}:/lean4:none:ro,rbind"]},
                "env": {"PATH": "/lean4/elan/bin:/usr/local/bin:/usr/bin:/bin", "ELAN_HOME": "/lean4/elan"},
            },
            lean_project_dir="/lean4/my_project",
        )
    if args.lean_prefix:
        return LocalLeanClient(Path(args.lean_prefix).absolute())
    return Lean4SandboxClient(host=args.host, port=args.port, max_output_characters=4000)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--limit", type=int, default=None, help="Check only the first N problems.")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--lean-prefix", help="Directory produced by setup_lean.sh; runs lake directly.")
    parser.add_argument("--enroot-image", help="Base image to mount --lean-prefix into, via nemo_gym.sandbox.")
    args = parser.parse_args()

    client = build_client(args)
    if not await client.health_check():
        print(f"FAIL: Lean is not reachable via {type(client).__name__}", file=sys.stderr)
        return 2

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
