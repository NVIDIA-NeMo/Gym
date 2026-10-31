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
"""CyberGym validation on an E2B microVM: task config, templates, and the 4-stage scorer.

Each control runs in its OWN fresh microVM: egress deny-all -> firewall + lockdown -> egress check ->
upload inputs -> validate.py per stage -> firewall re-read (must be unchanged). validate.py runs as
root inside bwrap (empty net ns + seccomp), since compile.sh/test.sh execute the untrusted patch.
Stages reuse this one VM (restoring /src per stage); upstream uses a fresh container per stage (see the
README's Comparability).

Task assets come from the managed dir prepare.py populates (data/assets). Env: the E2B provider's
variables (E2B_API_URL, E2B_API_KEY, ...).
"""

import argparse
import asyncio
import base64
import hashlib
import json
import os
import shlex
import sys
import time
import tomllib
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))
import e2b_tls  # noqa: E402
from e2b import AsyncSandbox, AsyncTemplate  # noqa: E402
from isolation import (  # noqa: E402
    REMOTE_SECCOMP,
    SANDBOX_RUNTIME_DIR,
    SECCOMP_DENY,
    firewall_script,
    seccomp_bpf,
    sh,
    verify_isolation,
)
from patch_guard import strip_immutable  # noqa: E402


# Current task + its per-build-image template names. set_task() fills these in; a real run always
# calls it first (there is no default task).
PROJECT = TASK = TEMPLATE = BASE_TEMPLATE = None
# The managed asset tree populated by prepare.py (benchmarks/cybergym_e2e/data/assets): upstream
# code + configs + gated HF data. Derived from this file's location, so cwd doesn't matter.
DEFAULT_CYBERGYM = Path(__file__).resolve().parents[1] / "data" / "assets"

# The sandbox-tooling packages baked into both templates (agent base + scoring). Shared here so the
# recipe fingerprint tracks the real list, and template.py uses the same one -- no two-place drift.
APT_PACKAGES = ["sudo", "git", "iptables", "iproute2", "conntrack", "bubblewrap", "util-linux", "procps"]
# Bump when the in-code template recipe changes (the run_cmd steps in build_base/build_template). The
# deps script and APT_PACKAGES are hashed automatically, so edits to those need no bump.
RECIPE_REVISION = "1"


def recipe_tag(cg: Path) -> str:
    """Short fingerprint of the template build recipe -- the validator deps script + APT_PACKAGES (both
    hashed, so any edit invalidates cached templates on its own) + RECIPE_REVISION. Folded into template
    names so a recipe change forces a rebuild instead of silently reusing a stale cached image, and so
    the name recorded with a result pins the recipe that produced it."""
    deps = (cg / "scripts" / "install_validate_deps.sh").read_bytes()
    blob = deps + repr(APT_PACKAGES).encode() + RECIPE_REVISION.encode()
    return hashlib.sha256(blob).hexdigest()[:8]
DEADLINE_MARGIN_S = 120  # stop waiting this long before the microVM's own vm_timeout would reap it


def set_task(spec, cg=None):
    """Point every script at project/task_id (e.g. 'jq/arvo_64574'), reading the build image from the
    assets dir `cg` (defaults to DEFAULT_CYBERGYM). A task is required -- there is no default. Templates
    are per build-image, so all tasks sharing an image reuse one template."""
    global PROJECT, TASK, TEMPLATE, BASE_TEMPLATE
    cg = cg or DEFAULT_CYBERGYM
    if not spec or "/" not in spec:
        raise SystemExit(f"a task is required as 'project/arvo_id', got {spec!r}")
    PROJECT, TASK = spec.split("/", 1)
    img = tomllib.loads((cg / "projects" / PROJECT / "project.toml").read_text()
                        ).get("build_image", "")
    tag = img.split("@sha256:")[-1][:12] if "@sha256:" in img else PROJECT
    rtag = recipe_tag(cg)  # changes whenever the build recipe (deps/apt/recipe-rev) changes
    TEMPLATE = f"cybergym-score-{PROJECT}-{tag}-r{rtag}"
    BASE_TEMPLATE = f"cybergym-agent-{PROJECT}-{tag}-r{rtag}"
    return PROJECT, TASK


# (validate.py args, expected stage statuses) known-answer controls for maintainers.
CONTROLS = {
    "gt_poc_unpatched": ("--poc-file /data/poc.bin", {"stage1": "passed"}),
    "gt_patch": ("--poc-file /data/poc.bin --patch-file /data/patch.diff",
                 {"stage1": "passed", "stage2": "passed", "stage3": "passed", "stage4": "passed"}),
    "random_poc": ("--poc-file /data/random.bin", {"stage1": "failed"}),
}

# Like isolation.LOCKDOWN_SCRIPT, minus the setuid strip: validate.py runs everything via sudo.
VALIDATION_LOCKDOWN = """
sysctl -w user.max_user_namespaces=0
sysctl -w kernel.unprivileged_bpf_disabled=1
sysctl -w kernel.perf_event_paranoid=3
sysctl -w kernel.kptr_restrict=2
sysctl -w kernel.dmesg_restrict=1
sysctl -w vm.mmap_rnd_bits=28 || echo "WARN: could not set vm.mmap_rnd_bits=28 (ASan may misbehave)"
sysctl -w kernel.modules_disabled=1 2>/dev/null || true
echo "mmap_rnd_bits=$(cat /proc/sys/vm/mmap_rnd_bits)"
"""

# test.sh may run the project's suite with LeakSanitizer on (only run_poc.sh sets detect_leaks=0),
# and LSan ptrace()s its own threads at exit; with ptrace blocked that dies. So validation allows
# ptrace. The agent sandbox (run.py) keeps blocking it.
VALIDATION_SECCOMP_DENY = {k: v for k, v in SECCOMP_DENY.items() if k != "ptrace"}

# Run as root right after the firewall, before anything is uploaded. Prints REACHABLE for any target
# that answers: public IP, public DNS, cloud metadata, and the RFC1918 ranges. These are generic probe
# targets; add any addresses specific to YOUR deployment (its resolver / gateway) via $EGRESS_EXTRA
# (space-separated host/port), e.g. EGRESS_EXTRA="<your-resolver-ip>/53 <your-gateway-ip>/443".
EGRESS_CHECK = r"""
for t in 1.1.1.1/443 8.8.8.8/53 169.254.169.254/80 10.0.0.1/443 172.16.0.1/443 192.168.0.1/80 \
         ${EGRESS_EXTRA:-}; do
  if timeout 5 bash -c "exec 3<>/dev/tcp/$t" 2>/dev/null; then echo "REACHABLE tcp $t"; fi
done
if timeout 5 getent hosts example.com >/dev/null 2>&1; then echo "REACHABLE dns example.com"; fi
echo egress-check-done
"""

# Root, but no network, own pid/ipc/uts, seccomp; /etc/cybergym.env restores the image's ENV
# (CC, CFLAGS, SANITIZER flags, SRC/OUT/WORK) captured at template build time.
BWRAP_ROOT = (
    f"10<{REMOTE_SECCOMP} "
    "bwrap --unshare-net --unshare-pid --unshare-ipc --unshare-uts "
    "--die-with-parent --new-session --seccomp 10 "
    f"--bind / / --dev /dev --proc /proc --tmpfs {SANDBOX_RUNTIME_DIR} -- "
)


def task_config(cg):
    cfg = tomllib.loads((cg / "projects" / PROJECT / "project.toml").read_text())
    cfg.update(tomllib.loads((cg / "projects" / PROJECT / TASK / "config.toml").read_text()))
    return cfg


async def build_template(cg, cpu, mem):
    # Project-agnostic: just the task's build image + our sandbox tools + validator deps. prepare.sh is
    # NOT baked -- it runs per VM during the trusted-setup window (like CyberGym's own harness), so ANY
    # task works regardless of what its prepare.sh does. One template serves every task sharing an image.
    cfg = task_config(cg)
    if cfg.get("pre_patches") or cfg.get("pre_patch"):
        raise SystemExit("pre_patches not supported by this harness yet")
    deps = base64.b64encode((cg / "scripts" / "install_validate_deps.sh").read_bytes()).decode()
    t = (AsyncTemplate()
         .from_image(cfg["build_image"])
         .apt_install(APT_PACKAGES)
         .run_cmd("update-alternatives --set iptables /usr/sbin/iptables-nft || true; "
                  "update-alternatives --set ip6tables /usr/sbin/ip6tables-nft || true", user="root")
         .run_cmd("mkdir -p /config /data /output /scripts /src /out /work", user="root")
         .run_cmd(f"echo {deps} | base64 -d > /install_validate_deps.sh && "
                  "bash -eux /install_validate_deps.sh", user="root")
         .run_cmd("bash -c 'export -p > /etc/cybergym.env'", user="root"))
    t0 = time.time()
    await AsyncTemplate.build(t, name=TEMPLATE, cpu_count=cpu, memory_mb=mem)
    print(f"built template {TEMPLATE} in {time.time() - t0:.0f}s")


async def _run_stage(sb, name, stage, args, t0, deadline, out_dir):
    """Run one validate.py --only-stage in the VM, with a freshly cleaned /out and /work, detached and
    polled. Returns (status, rc, log). Wiping between stages stops stale build state from leaking from
    one stage into the next (upstream CyberGym gets this from a fresh container per stage); the engine
    lib lives in /usr/lib, not /out, so it survives the wipe. validate.py restores /src itself."""
    await sb.commands.run("rm -rf /out /work; mkdir -p /out /work; "
                          "rm -f /output/rc /output/stage.log /output/result.json",
                          user="root", timeout=120)
    inner = (". /etc/cybergym.env; exec /scripts/.venv/bin/python /scripts/validate.py "
             f"--src-dir /src --config-dir /config --data-dir /data "
             f"--json-output /output/result.json --only-stage {stage} {args}")
    job = f"{BWRAP_ROOT}bash -c {shlex.quote(inner)} > /output/stage.log 2>&1; echo $? > /output/rc"
    await sb.commands.run(f"nohup bash -c {shlex.quote(job)} >/dev/null 2>&1 &", user="root", timeout=30)
    while True:
        await asyncio.sleep(30)
        if time.time() > deadline:  # don't rely on the backend reaping the VM: give up ourselves
            try:
                slog = await sb.files.read("/output/stage.log", user="root")
                (out_dir / f"{name}.log").write_text(slog + f"\n[TIMEOUT in stage {stage}: VM killed]\n")
            except Exception:
                pass
            raise TimeoutError(f"validate.py stage {stage} still running after {time.time() - t0:.0f}s; VM killed")
        r = await sb.commands.run("cat /output/rc 2>/dev/null || echo RUNNING; "
                                  "tail -n1 /output/stage.log 2>/dev/null || true", user="root", timeout=30)
        lines = r.stdout.splitlines() or ["RUNNING"]
        if lines[0].strip() != "RUNNING":
            rc = int(lines[0])
            break
        print(f"[{name}] stage{stage} {time.time() - t0:5.0f}s  {(lines[1] if len(lines) > 1 else '')[:85]}")
    slog = await sb.files.read("/output/stage.log", user="root")
    try:
        res = json.loads(await sb.files.read("/output/result.json", user="root"))
        status = res.get(f"stage{stage}")
    except Exception:
        status = "error"
    return status, rc, slog


async def run_control(name, cg, template, vm_timeout, out_dir, custom=None):
    """custom=(args, expected, {vm_path: bytes|str}) validates arbitrary inputs (run.py's deliverables)."""
    args, expected = custom[:2] if custom else CONTROLS[name]
    data = cg / "data" / "projects" / PROJECT / TASK
    t0 = time.time()
    # Network is OPEN during trusted setup (upload + prepare.sh), exactly as CyberGym's own harness
    # runs it, so ANY task's prepare.sh works. Nothing untrusted runs in this window. The in-VM
    # firewall + lockdown + egress-check then go up BEFORE validate.py (which builds the untrusted
    # patch); validate.py also runs inside bwrap --unshare-net, so it has no network regardless.
    sb = await AsyncSandbox.create(template=template, timeout=vm_timeout,
                                   metadata={"purpose": f"cybergym-validate:{name}"})
    try:
        print(f"[{name}] microVM {sb.sandbox_id} up in {time.time() - t0:.1f}s")
        cfg = task_config(cg)
        repo = cfg["repo_to_patch"]
        task = cg / "projects" / PROJECT / TASK
        # JSON strings/arrays are valid TOML; same sanitized keys as upstream utils.setup_workspace.
        sanitized = (f"repo_to_patch = {json.dumps(repo)}\n"
                     f"immutable_files = {json.dumps(cfg.get('immutable_files', []))}\n")
        await sb.files.write("/scripts/validate.py", (cg / "scripts" / "validate.py").read_text(), user="root")
        await sb.files.write("/config/config.toml", sanitized, user="root")
        await sb.files.write("/src/src.tgz", (data / "src.tgz").read_bytes(), user="root",
                             request_timeout=600)
        for s in ("prepare.sh", "compile.sh", "run_poc.sh", "test.sh"):
            await sb.files.write(f"/src/{s}", (task / s).read_text(), user="root")
        print(f"[{name}] running prepare.sh (network open, trusted setup)...")
        r = await sb.commands.run(
            f"set -e; tar xf /src/src.tgz -C /src; rm /src/src.tgz; rm -rf /src/{repo}/.git; "
            ". /etc/cybergym.env; export SRC=/src OUT=/out WORK=/work; cd /src; "
            "if [ -f /src/prepare.sh ]; then bash -eux /src/prepare.sh > /output/prepare.log 2>&1; fi; "
            "cp -a /src /src_backup", user="root", timeout=1800)
        if r.exit_code != 0:
            tail = (await sb.commands.run("tail -40 /output/prepare.log 2>/dev/null", user="root", timeout=30)).stdout
            raise RuntimeError(f"setup/prepare failed: {r.stderr[-800:]}\n{tail}")
        # Now lock down: firewall + lockdown + verify, before any untrusted build runs.
        await sh(sb, name, firewall_script([]))
        await sh(sb, name, VALIDATION_LOCKDOWN)
        await verify_isolation(sb, name)  # fail closed: both iptables sets + required sysctls
        r = await sb.commands.run(EGRESS_CHECK, user="root", timeout=120,
                                  envs={"EGRESS_EXTRA": os.environ.get("EGRESS_EXTRA", "")})
        # Fail closed: a target answered OR the probe didn't run to completion (no marker) -- a probe
        # that errored out early must not be read as "all blocked".
        if "REACHABLE" in r.stdout or "egress-check-done" not in r.stdout:
            raise RuntimeError(f"egress not blocked after setup:\n{r.stdout}{r.stderr}")
        print(f"[{name}] egress check: all blocked (as root)")
        fw_before = (await sb.commands.run("iptables -S; ip6tables -S 2>/dev/null || true", user="root", timeout=30)).stdout
        await sb.files.write("/data/poc.bin", (data / "poc.bin").read_bytes(), user="root")
        if name == "gt_patch":
            raw = (cg / "projects" / PROJECT / TASK / "patch.diff").read_text()
            # Normally we drop immutable (tests/) hunks to mirror the agent policy. But the GT patch is
            # TRUSTED ground truth, and some tasks' GT patches are binary (or otherwise can't be stripped
            # safely) -- strip_immutable guards UNTRUSTED agent patches, not the benchmark's own answer.
            # When it can't strip, apply the GT verbatim rather than failing the known-answer check.
            try:
                gt, dropped = strip_immutable(raw, cfg.get("immutable_files", []))
                print(f"[{name}] patch: dropped immutable sections {dropped}")
            except ValueError as e:
                gt = raw
                print(f"[{name}] patch: applying GT verbatim (strip skipped: {e})")
            await sb.files.write("/data/patch.diff", gt, user="root")
        if name == "random_poc":
            await sb.files.write("/data/random.bin", os.urandom(4096), user="root")
        for path, blob in (custom[2] if custom else {}).items():
            await sb.files.write(path, bytes(blob) if isinstance(blob, bytearray) else blob, user="root")
        await sb.files.write(REMOTE_SECCOMP, seccomp_bpf(VALIDATION_SECCOMP_DENY), user="root")

        # Run each stage as its own validate.py --only-stage, wiping /out+/work between stages (see
        # _run_stage). Stage plan + skip logic follow validate.py/run_agent.py: stage 2 needs a PoC and
        # a passing stage 1; stage 3 (e2e) needs a passing stage 2; stage 4 needs a passing stage 3.
        has_poc, has_patch = "--poc-file" in args, "--patch-file" in args
        plan = ([1] if has_poc else []) + ([2] if has_poc and has_patch else []) \
            + ([3, 4] if has_patch else [])
        got, rc, log = {}, 0, ""
        deadline = t0 + vm_timeout - DEADLINE_MARGIN_S
        for stage in plan:
            if stage == 2 and got.get("stage1") != "passed":
                got["stage2"] = "skipped"
                continue
            if stage == 3 and has_poc and got.get("stage2") != "passed":
                got["stage3"] = "skipped"
                continue
            if stage == 4 and got.get("stage3") != "passed":
                got["stage4"] = "skipped"
                continue
            status, rc, slog = await _run_stage(sb, name, stage, args, t0, deadline, out_dir)
            got[f"stage{stage}"] = status
            log += f"\n===== STAGE {stage} (rc={rc}) =====\n{slog}\n"

        (out_dir / f"{name}.log").write_text(log)
        fw_after = (await sb.commands.run("iptables -S; ip6tables -S 2>/dev/null || true", user="root", timeout=30)).stdout
        fw_intact = fw_after == fw_before
        ok = fw_intact and all(got.get(k) == v for k, v in expected.items())
        print(f"[{name}] done in {time.time() - t0:.0f}s rc={rc} {'OK' if ok else 'MISMATCH'}"
              f"{'' if fw_intact else ' FIREWALL CHANGED'}")
        return {"rc": rc, "got": got, "expected": expected, "ok": ok, "firewall_intact": fw_intact,
                "seconds": round(time.time() - t0)}
    finally:
        try:
            await sb.kill()
            print(f"[{name}] microVM {sb.sandbox_id} killed")
        except Exception as e:  # already gone (backend reaped it) or API down; it expires at vm_timeout
            print(f"[{name}] WARN kill failed: {type(e).__name__}: {e}")


async def run_control_bounded(name, cg, template, vm_timeout, out_dir, custom=None):
    """Hard cap on the whole control (setup, uploads, polling). Cancellation still runs the kill."""
    return await asyncio.wait_for(run_control(name, cg, template, vm_timeout, out_dir, custom),
                                  vm_timeout - DEADLINE_MARGIN_S // 2)


async def main():
    e2b_tls.install()  # optional custom-CA/self-signed trust for the E2B backend (no-op by default)
    ap = argparse.ArgumentParser(description="CyberGym validator known-answer checks (maintainers).")
    ap.add_argument("--task", required=True, help="project/arvo_id")
    ap.add_argument("--build-template", action="store_true")
    ap.add_argument("--template", default=None)
    ap.add_argument("--cybergym-dir", type=Path, default=DEFAULT_CYBERGYM)
    ap.add_argument("--only", action="append", choices=list(CONTROLS), help="run only these controls")
    ap.add_argument("--cpu", type=int, default=4)
    ap.add_argument("--memory-mb", type=int, default=8192)
    ap.add_argument("--vm-timeout", type=int, default=3 * 3600, help="microVM lifetime, seconds")
    ap.add_argument("--out-dir", type=Path, default=Path("validate_check"))
    a = ap.parse_args()
    sys.argv = sys.argv[:1]
    set_task(a.task, a.cybergym_dir)
    a.template = a.template or TEMPLATE

    if a.build_template:
        await build_template(a.cybergym_dir, a.cpu, a.memory_mb)
        return 0

    a.out_dir.mkdir(parents=True, exist_ok=True)
    names = a.only or list(CONTROLS)
    if a.vm_timeout <= 2 * DEADLINE_MARGIN_S:
        raise SystemExit(f"--vm-timeout must be > {2 * DEADLINE_MARGIN_S}s")
    results = await asyncio.gather(*(run_control_bounded(n, a.cybergym_dir, a.template, a.vm_timeout, a.out_dir)
                                     for n in names), return_exceptions=True)
    report = {}
    print(f"\n=== validator known-answer checks: {PROJECT}/{TASK} ===")
    for name, res in zip(names, results):
        if isinstance(res, Exception):
            label = "TIMEOUT" if isinstance(res, TimeoutError) else "ERROR"
            report[name] = {"ok": False, "error": f"{label}: {type(res).__name__}: {res}"}
            print(f"  {label:<8} {name:<17} {report[name]['error'][:120]}")
            continue
        report[name] = res
        got = " ".join(f"{k}={res['got'].get(k)}" for k in ("stage1", "stage2", "stage3", "stage4"))
        want = " ".join(f"{k}={v}" for k, v in res["expected"].items())
        print(f"  {'PASS' if res['ok'] else 'MISMATCH':<8} {name:<17} got: {got}\n"
              f"  {'':<8} {'':<17} want: {want}")
    (a.out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"logs + report: {a.out_dir}/")
    return 0 if all(r.get("ok") for r in report.values()) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
