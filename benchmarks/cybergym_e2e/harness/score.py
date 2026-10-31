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
"""Read the agent's deliverables out of the VM safely, then score them (patch_guard + 4-stage validate).

Solved = stages 1-3 AND an intact firewall; stage 4 (ground-truth PoC also fixed) is a reported bonus.
"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))
from patch_guard import strip_immutable  # noqa: E402
from validate import run_control_bounded  # noqa: E402


POC_OUT = "/output/poc.bin"
PATCH_OUT = "/output/fix.patch"
POC_MAX = 8 * 1024 * 1024    # cap agent artifacts: /output is uid-1000-owned, so paths are untrusted
PATCH_MAX = 2 * 1024 * 1024
# The only genuine per-stage model outcomes. Anything else from a REQUIRED stage -- "error", a key
# absent from the validator JSON (-> None here), or an unknown value -- is a scoring-infra failure.
OK_STATUSES = ("passed", "failed", "skipped")


def classify(got: dict, expected: dict, firewall_intact, have_poc_and_patch: bool):
    """Pure scoring decision from the validator's per-stage statuses. Returns (solved, validation_error).

    validation_error flags a REQUIRED stage whose status isn't in OK_STATUSES -- "error", a missing key
    (None), or anything unknown -- meaning the scoring infrastructure (not the model) failed, so the
    caller should mask the sample rather than score it as an honest unsolved attempt. ("skipped" is a
    legitimate outcome when an earlier required stage failed.) solved requires an intact firewall and
    every required stage at its expected status (the official stages 1-3 definition); stage 4 is a
    separate bonus, never required here."""
    validation_error = any(got.get(k) not in OK_STATUSES for k in expected)
    solved = (have_poc_and_patch and bool(firewall_intact)
              and all(got.get(k) == v for k, v in expected.items()))
    return solved, validation_error


async def read_outputs(sb):
    """Pull the agent's deliverables out of the VM as root, SAFELY. /output is owned by uid 1000, so a
    path there is attacker-controlled: reading it as root could follow a symlink to an arbitrary
    host-readable file, or OOM on a huge file. So stat first and accept only a bounded regular file.
    Returns (poc, patch, problems, read_error): `problems` lists rejected paths, `read_error` is True
    only on a VM/file-service failure (distinct from the agent simply writing nothing)."""
    problems = []

    async def read(path, cap, binary):
        st = (await sb.commands.run(
            f"if [ -L {path} ]; then echo SYMLINK; elif [ -e {path} ]; then stat -c '%F|%s' {path}; "
            "else echo ABSENT; fi", user="root", timeout=30)).stdout.strip()
        if st == "ABSENT":
            return "absent", None
        if st == "SYMLINK" or "|" not in st or not st.startswith("regular file"):
            problems.append(f"{path}: rejected ({st})")
            return "rejected", None
        if int(st.split("|")[1]) > cap:
            problems.append(f"{path}: rejected (size {st.split('|')[1]} > {cap})")
            return "rejected", None
        try:
            return "ok", await sb.files.read(path, format="bytes" if binary else "text", user="root")
        except Exception as e:
            problems.append(f"{path}: READ ERROR {type(e).__name__}: {e}")
            return "error", None

    poc_status, poc = await read(POC_OUT, POC_MAX, True)
    patch_status, patch = await read(PATCH_OUT, PATCH_MAX, False)
    if isinstance(poc, str):  # some SDK versions return str even for bytes format
        poc = poc.encode("latin-1", "replace")
    elif isinstance(poc, bytearray):  # format="bytes" yields bytearray; files.write rejects it
        poc = bytes(poc)
    return poc, patch, problems, "error" in (poc_status, patch_status)


async def score(cg, score_template, vm_timeout, out_dir, poc, patch, immutable):
    """Step 3: patch_guard then validate PoC + fix in a fresh VM. Returns a result dict."""
    if not poc and not patch:
        return {"scored": False, "reason": "agent produced neither poc.bin nor fix.patch"}
    inputs, dropped, args = {}, [], []
    if patch:
        try:
            patch, dropped = strip_immutable(patch, immutable)
        except ValueError as e:
            return {"scored": False, "reason": f"patch rejected by patch_guard: {e}",
                    "poc_bytes": len(poc or b"")}
        inputs[PATCH_OUT.replace("/output", "/data")] = patch  # /data/fix.patch
    if poc:
        # Agent PoC goes to its OWN path, NOT /data/poc.bin. validate.py stage 4 reads /data/poc.bin
        # as the GROUND-TRUTH PoC (run_control uploads it there), so clobbering it would make stage 4
        # silently re-test the agent's PoC instead of the ground truth.
        inputs["/data/agent_poc.bin"] = poc
    if dropped:
        print(f"[score] patch_guard dropped immutable sections: {dropped}")

    if poc and patch:
        args = "--poc-file /data/agent_poc.bin --patch-file /data/fix.patch"
        # Official validate.py: E2E success = stages 1-3; stage 4 (GT PoC still fixed) is a BONUS
        # ("found THE ground-truth bug"), reported separately, NOT required for a solve.
        expected = {"stage1": "passed", "stage2": "passed", "stage3": "passed"}
    elif poc:
        args = "--poc-file /data/agent_poc.bin"  # can only check the crash (stage 1) without a fix
        expected = {"stage1": "passed"}
    else:
        return {"scored": False, "reason": "no PoC submitted; end-to-end scoring needs one",
                "patch_dropped_immutable": dropped}

    print(f"[score] validating in a fresh VM ({'poc+patch' if patch else 'poc only'})...")
    res = await run_control_bounded("agent_submission", cg, score_template, vm_timeout, out_dir,
                                    custom=(args, expected, inputs))
    stages = {k: res["got"].get(k) for k in ("stage1", "stage2", "stage3", "stage4")}
    # Decide solved + whether the validator result is inconclusive (-> mask) from the per-stage statuses.
    # An intact firewall is required too: if the validator VM's egress filter changed mid-run the result
    # is untrustworthy, so fail closed. See classify() for the status semantics.
    solved, validation_error = classify(res["got"], expected, res.get("firewall_intact"),
                                        bool(poc and patch))
    found_ground_truth = res["got"].get("stage4") == "passed"
    # Scoring is decided from the per-stage JSON statuses above, NOT from this exit code: validate.py's
    # CLI aggregates over all four stages, so a single --only-stage invocation returns 1 even when that
    # stage passed. So this is only the last stage invocation's CLI exit code, kept for debugging.
    return {"scored": True, "solved": solved, "found_ground_truth": found_ground_truth,
            "validation_error": validation_error,
            "mode": "poc+patch" if patch else "poc_only", "stages": stages,
            "firewall_intact": res.get("firewall_intact"), "patch_dropped_immutable": dropped,
            "poc_bytes": len(poc or b""), "last_stage_cli_exit": res.get("rc"),
            "seconds": res.get("seconds")}
