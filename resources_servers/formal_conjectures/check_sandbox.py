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

"""Compile the Formal Conjectures reference files and check each target is actually proved.

This is both the pre-flight gate and the sweep that defines the benchmark. It compiles the
reference version of a task -- the file with upstream's own proof still in place -- and asks
Lean `#print axioms <target>`. A task counts as verified only when the file compiles clean and
the target is free of `sorryAx`.

Two failure modes it separates, which is the whole point:

* the sandbox's Mathlib is not v4.33.1, so statements fail for reasons unrelated to the task;
* upstream's "proof" is not one -- it transitively depends on a `sorry` elsewhere in the file.
  Only `#print axioms` catches that, and it is why `data/verified_task_ids.txt` is smaller than the
  set `extract.py` can produce.

No model or GPU needed.

Usage:
    # gate: check the committed task list still holds in this sandbox
    python check_sandbox.py --image gym-lean:v4.33.1 --limit 25

    # regenerate the benchmark definition (slow: every candidate task is compiled)
    python check_sandbox.py --image gym-lean:v4.33.1 --all --write-verified
"""

import argparse
import asyncio
import os
import sys

from resources_servers.formal_conjectures.extract import extract_file
from resources_servers.formal_conjectures.prepare import VERIFIED_TASKS_FPATH, fetch_sources
from resources_servers.lean_proof.lean_sandbox import DEFAULT_LEAN_PROJECT_DIR, LeanSandbox
from resources_servers.lean_proof.status import STATUS_COMPLETED, determine_proof_status
from resources_servers.lean_proof.toolchain import TOOLCHAIN_PROBE, normalize_version, parse_lean_version


EXPECTED_LEAN_VERSION = "4.33.1"


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
            # A multi-GB Lean image takes minutes to pull on a cell that has not cached it.
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
            "metadata": {"benchmark": "formal_conjectures", "purpose": "check-sandbox"},
        },
        project_dir=args.project_dir,
        server_name="formal-conjectures-check-sandbox",
    )


async def compile_in(lean: LeanSandbox, code: str, timeout: float) -> dict:
    """Compile one file through the server's own path and flatten the result."""
    result = await lean.compile(code, timeout_s=timeout)
    return {
        "stdout": result.stdout or "",
        "stderr": result.stderr or "",
        "return_code": result.return_code,
        "error_type": result.error_type,
    }


def load_candidates(check_all: bool, limit: int | None) -> list:
    """Extract candidate tasks from the pinned upstream revision.

    Without ``--all`` this is narrowed to the ids already in ``data/verified_task_ids.txt``, which is
    the gate: it re-checks the committed benchmark rather than rediscovering it.
    """
    sources = fetch_sources()
    tasks = []
    for path, text in sorted(sources.items()):
        if not path.startswith("FormalConjectures/"):
            continue
        tasks.extend(extract_file(path, text, fc_only_names=set()))

    # Ids are namespace-qualified, so they are unique. Assert it rather than assume: a
    # duplicate would mean the sweep verifies one declaration and prepare.py ships another.
    seen = {}
    for task in tasks:
        if task.task_id in seen:
            raise ValueError(f"duplicate task_id {task.task_id!r}; extraction is ambiguous")
        seen[task.task_id] = task

    if not check_all:
        wanted = set(VERIFIED_TASKS_FPATH.read_text(encoding="utf-8").split("\n")) - {""}
        tasks = [t for t in tasks if t.task_id in wanted]
        missing = wanted - {t.task_id for t in tasks}
        if missing:
            print(f"WARNING: {len(missing)} verified task(s) are no longer extractable, e.g. {sorted(missing)[:3]}")

    tasks.sort(key=lambda t: t.task_id)
    return tasks[:limit] if limit else tasks


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="opensandbox", help="Sandbox provider name.")
    parser.add_argument(
        "--domain",
        default=os.environ.get("OPENSANDBOX_DOMAIN"),
        help="OpenSandbox endpoint; defaults to $OPENSANDBOX_DOMAIN.",
    )
    parser.add_argument("--snapshot-id", default=os.environ.get("FORMAL_CONJECTURES_SANDBOX_SNAPSHOT_ID"))
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
    parser.add_argument("--limit", type=int, default=None, help="Check only the first N tasks.")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Check every extractable candidate, not just the committed verified list.",
    )
    parser.add_argument(
        "--write-verified",
        action="store_true",
        help=f"Rewrite {VERIFIED_TASKS_FPATH.name} with the tasks that passed. Implies a full sweep.",
    )
    args = parser.parse_args()
    # Gym's global aiohttp client parses the CLI through Hydra the first time it is used,
    # which happens inside the sandbox provider's first request. It would reject this
    # script's own flags. They are consumed by now, so take them out of its way.
    sys.argv = sys.argv[:1]

    if not args.snapshot_id and not args.image:
        print("FAIL: pass --snapshot-id or --image; there is no default Lean environment.", file=sys.stderr)
        return 2
    if args.write_verified and not args.all:
        print("FAIL: --write-verified rewrites the benchmark definition, so it requires --all.", file=sys.stderr)
        return 2
    if args.write_verified and args.limit:
        print("FAIL: --write-verified with --limit would silently shrink the benchmark.", file=sys.stderr)
        return 2

    lean = build_sandbox(args)
    try:
        await lean.start()
    except Exception as exc:  # noqa: BLE001 - the reason matters more than the type here
        print(f"FAIL: could not start a sandbox: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    # Report the toolchain first so per-task failures below can be read in context.
    probe = await compile_in(lean, TOOLCHAIN_PROBE, args.timeout)
    found = parse_lean_version(probe)
    if found is None:
        print("FAIL: `import Mathlib` did not compile -- this sandbox cannot state any FC task.")
        print(f"       stderr: {probe.get('stderr', '')[:400]}")
        return 2
    if found != normalize_version(EXPECTED_LEAN_VERSION):
        print(f"WARNING: sandbox is Lean/Mathlib {found}, expected {EXPECTED_LEAN_VERSION}.")
        print("         Statements written against v4.33.1 may fail to compile below.\n")
    else:
        print(f"Sandbox toolchain: Lean/Mathlib {found} (expected {EXPECTED_LEAN_VERSION})\n")

    tasks = load_candidates(args.all, args.limit)
    print(f"Compiling {len(tasks)} reference files in the sandbox\n")

    semaphore = asyncio.Semaphore(args.concurrency)
    passed: list[str] = []
    failures: list[tuple[str, str]] = []

    async def check(task) -> None:
        # The reference file, with upstream's proof intact, plus the question that decides it.
        code = f"{task.reference_file}\n\n#print axioms {task.full_name}\n"
        async with semaphore:
            result = await compile_in(lean, code, args.timeout)

        # `sorry_is_error=False` for the same reason the server uses it: the file is allowed to
        # keep the open conjecture's hole. `#print axioms` is what rules on the target.
        status, reason = determine_proof_status(result, sorry_is_error=False)
        if status != STATUS_COMPLETED:
            combined = f"{result['stdout']}\n{result['stderr']}"
            first = next((ln for ln in combined.splitlines() if "error:" in ln.lower()), reason or status)
            failures.append((task.task_id, first.strip()[:140]))
            print(f"  FAIL  {task.task_id}  {first.strip()[:90]}")
            return

        # Imported here rather than at module scope to keep the sweep's rule and the server's
        # rule literally the same function.
        from resources_servers.formal_conjectures.app import target_is_proved

        proved = target_is_proved(result, task.full_name)
        if proved is None:
            failures.append((task.task_id, f"`#print axioms {task.full_name}` produced no output"))
            print(f"  FAIL  {task.task_id}  no axiom line; declaration missing or renamed")
        elif not proved:
            failures.append((task.task_id, "upstream's proof depends on sorryAx"))
            print(f"  FAIL  {task.task_id}  upstream's proof depends on sorryAx")
        else:
            passed.append(task.task_id)
            print(f"  ok    {task.task_id}")

    await asyncio.gather(*(check(t) for t in tasks))

    print(f"\n{len(passed)}/{len(tasks)} reference files compiled with their target proved.")

    if args.write_verified:
        VERIFIED_TASKS_FPATH.write_text("\n".join(sorted(passed)) + "\n", encoding="utf-8")
        print(f"Wrote {len(passed)} task ids to {VERIFIED_TASKS_FPATH}")
        return 0

    if failures:
        print(f"\n{len(failures)} FAILED — this sandbox is not usable for Formal Conjectures as committed.")
        print("Almost always this means its Mathlib is not v4.33.1. See 'Requirements' in README.md.\n")
        for task_id, reason in failures[:10]:
            print(f"  {task_id}: {reason}")
        return 1

    print("Sandbox looks correct: every committed task's reference proof checks out.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
