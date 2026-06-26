# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run the swe_env sandbox infra over SWE-bench Verified (gold-patch eval).

For each instance: provision the official SWE-bench docker image through the
swe_env sandbox provider and lifecycle (``acquire_sandbox``), apply the gold
patch, run the SWE-bench ``eval_script`` (the real per-repo test command), and
grade with the official ``swebench`` parser. A gold run should resolve nearly
all instances.

This is a driver/operational script (not a unit test). Requires extra deps:
    uv pip install swebench datasets    # + docker (provider=docker) or apptainer

Examples:
    # smoke (5 instances), docker provider, prune images to bound disk
    python responses_api_agents/swe_env/scripts/run_swebench_verified.py --limit 5

    # full 500, 4 in parallel, keep an incremental results file
    python responses_api_agents/swe_env/scripts/run_swebench_verified.py \\
        --concurrency 4 --output /tmp/swebench_gold_results.jsonl

    # apptainer provider (converts each image to .sif on the fly, then removes it)
    python responses_api_agents/swe_env/scripts/run_swebench_verified.py --provider apptainer --limit 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
# Use a writable HF cache (the default ~/.cache/huggingface may be root-polluted by
# docker containers that mount it). Override with HF_HOME in the environment.
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[3] / ".hf_cache"))

from nemo_gym.sandbox import SandboxSpec  # noqa: E402
from responses_api_agents.swe_env.lifecycle import acquire_sandbox  # noqa: E402


def _load_instances(limit, instance_ids):
    """Load SWE-bench Verified test instances.

    Args:
        limit: Maximum number of instances to return, or a falsy value for all.
        instance_ids: Optional iterable of instance ids to filter to.

    Returns:
        list[dict]: The selected dataset rows.
    """
    from datasets import load_dataset

    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    rows = list(ds)
    if instance_ids:
        wanted = set(instance_ids)
        rows = [r for r in rows if r["instance_id"] in wanted]
    if limit:
        rows = rows[:limit]
    return rows


def _docker(*args, timeout=None):
    """Run a ``docker`` subcommand, capturing its output.

    Args:
        *args: Arguments passed through to the ``docker`` CLI.
        timeout: Optional timeout in seconds for the subprocess.

    Returns:
        subprocess.CompletedProcess: The completed process with captured output.
    """
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


async def _eval_one(instance, *, keep_images, eval_timeout):
    """Evaluate one SWE-bench Verified instance with its gold patch.

    Pulls the official image, provisions a sandbox through the selected
    provider, applies the gold patch, runs the SWE-bench eval script, parses the
    logs, and computes whether the instance resolved. Docker (and any built
    ``.sif``) images are removed afterwards unless ``keep_images`` is set.

    Args:
        instance: The dataset row for the instance.
        keep_images: When True, do not remove images after the run.
        eval_timeout: Eval-script timeout in seconds.

    Returns:
        dict: Result with ``instance_id``, ``resolved``, ``status``, and
            ``error`` keys.
    """
    from swebench.harness.constants import FAIL_TO_PASS, PASS_TO_PASS, TestStatus
    from swebench.harness.grading import get_logs_eval
    from swebench.harness.test_spec.test_spec import make_test_spec

    iid = instance["instance_id"]
    # namespace="swebench" -> Docker Hub image key (swebench/sweb.eval.x86_64.<munged>:latest);
    # the default (None) yields a namespace-less local name that isn't pullable.
    spec = make_test_spec(instance, namespace="swebench")
    image = spec.instance_image_key
    try:
        _docker("pull", image, timeout=3600)
        provider = {"docker": {}}
        sbox_image = image
        provider_options = {}

        sandbox_spec = SandboxSpec(
            image=sbox_image,
            workdir="/testbed",
            ttl_s=eval_timeout + 600,
            ready_timeout_s=900,
            provider_options=provider_options,
        )
        async with acquire_sandbox(provider, sandbox_spec, instance_id=iid) as env:
            await env.write_text("/root/gold.patch", instance["patch"])
            await env.execute(
                "cd /testbed && (git apply -v /root/gold.patch || git apply -v --3way /root/gold.patch)",
                cwd="/testbed",
            )
            await env.write_text("/root/eval.sh", spec.eval_script)
            result = await env.execute("bash /root/eval.sh", timeout_s=eval_timeout, is_eval=True)

        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh:
            fh.write(result.get("output", ""))
            log_path = fh.name
        # swebench's per-repo log parser -> {test_id: status}; we compute resolution
        # ourselves from the instance's gold FAIL_TO_PASS / PASS_TO_PASS.
        status_map, found = get_logs_eval(spec, log_path)
        Path(log_path).unlink(missing_ok=True)
        f2p = instance.get(FAIL_TO_PASS) or []
        p2p = instance.get(PASS_TO_PASS) or []
        if isinstance(f2p, str):
            f2p = json.loads(f2p)
        if isinstance(p2p, str):
            p2p = json.loads(p2p)
        passed = {t for t, s in status_map.items() if s == TestStatus.PASSED.value}
        resolved = bool(found) and all(t in passed for t in f2p) and all(t in passed for t in p2p)
        status = "RESOLVED" if resolved else ("NO_LOG" if not found else "UNRESOLVED")
        return {"instance_id": iid, "resolved": resolved, "status": status, "error": None}
    except Exception as exc:  # noqa: BLE001
        return {"instance_id": iid, "resolved": False, "status": "ERROR", "error": repr(exc)}
    finally:
        if not keep_images:
            _docker("rmi", "-f", image)


async def _main_async(args):
    """Evaluate the selected instances concurrently and print a summary.

    Loads the instances, runs them with bounded concurrency, optionally streams
    each result to an output JSONL file, and prints per-instance progress plus a
    final resolved/error summary.

    Args:
        args: Parsed command-line arguments.
    """
    instances = _load_instances(args.limit, args.instances.split(",") if args.instances else None)
    print(f"Running {len(instances)} SWE-bench Verified instances (gold) via docker", flush=True)
    sem = asyncio.Semaphore(args.concurrency)
    out = open(args.output, "w") if args.output else None
    results = []

    async def _runner(inst):
        """Evaluate one instance under the concurrency semaphore and record it.

        Args:
            inst: The dataset row for the instance to evaluate.
        """
        async with sem:
            res = await _eval_one(
                inst,
                keep_images=args.keep_images,
                eval_timeout=args.eval_timeout,
            )
            results.append(res)
            mark = "PASS" if res["resolved"] else ("ERR " if res["error"] else "fail")
            print(f"  [{len(results):>3}/{len(instances)}] {mark} {res['instance_id']} ({res['status']})", flush=True)
            if out:
                out.write(json.dumps(res) + "\n")
                out.flush()

    await asyncio.gather(*[_runner(i) for i in instances])
    if out:
        out.close()
    resolved = sum(r["resolved"] for r in results)
    errors = sum(1 for r in results if r["error"])
    print(
        f"\n=== RESULT: resolved {resolved}/{len(results)} ({100 * resolved / max(1, len(results)):.1f}%); errors {errors} ==="
    )


def main():
    """Parse command-line arguments and run the gold-patch evaluation."""
    p = argparse.ArgumentParser(description="Gold-patch eval of SWE-bench Verified via swe_env providers")
    p.add_argument("--limit", type=int, default=None, help="only the first N instances")
    p.add_argument("--instances", type=str, default="", help="comma-separated instance_ids")
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--eval-timeout", type=int, default=1800)
    p.add_argument("--keep-images", action="store_true", help="do not docker rmi after each instance")
    p.add_argument("--output", type=str, default="", help="incremental results JSONL path")
    asyncio.run(_main_async(p.parse_args()))


if __name__ == "__main__":
    main()
