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
"""E2B template provisioning for a CyberGym agent run -- a SEPARATE preparation step.

The base template is project-agnostic: the task's OSS-Fuzz base-builder image + our sandbox tools
(iptables/bubblewrap/util-linux) + the upstream validator deps + an `agent` user with uid 1000.
prepare.sh is NOT baked -- it runs per VM during the trusted-setup window (network open), so any task
sharing a build image reuses one template. Guest-kernel lockdown is applied per-VM at run time, not
baked in (modules_disabled is one-way).

Build the base template once per build image before running tasks on it:

    python template.py --base --task jq/arvo_64574

Env: the E2B provider's own variables (e.g. E2B_API_URL, E2B_API_KEY). The CyberGym checkout is
read from the managed asset tree prepare.py populates (benchmarks/cybergym_e2e/data/assets).
"""

import argparse
import asyncio
import base64
import json
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))
import e2b_tls  # noqa: E402
import validate as vc  # noqa: E402  (task-configurable: vc.PROJECT/TASK/BASE_TEMPLATE)
from e2b import AsyncTemplate  # noqa: E402
from validate import DEFAULT_CYBERGYM, task_config  # noqa: E402


AGENT_UID = 1000
AGENT_DIRS = "/src /out /work /output"
# Never in the agent VM: they give away the answer.
FORBIDDEN = ["/data/poc.bin", "/src/poc.bin", "/src/crash.log", "/data/patch.diff"]


async def build_base(cg, cpu, mem):
    cfg = task_config(cg)
    # pre_patches would need to be applied to the source before the agent sees it; not wired yet.
    if cfg.get("pre_patches") or cfg.get("pre_patch"):
        raise SystemExit("pre_patches not supported by this harness yet")
    # The validator deps install script is read from the (public) repo and baked in base64 so it can be
    # passed inline to a run_cmd (the template builder has no file-upload step).
    deps = base64.b64encode((cg / "scripts" / "install_validate_deps.sh").read_bytes()).decode()
    t = (AsyncTemplate()
         # Start from the task's own OSS-Fuzz base-builder image (CC/CFLAGS/sanitizer env already set).
         .from_image(cfg["build_image"])
         # Our sandbox tooling: firewall (iptables), bubblewrap, setpriv (util-linux), process tools.
         # Shared with validate.py so the recipe fingerprint tracks the real list (see vc.recipe_tag).
         .apt_install(vc.APT_PACKAGES)
         # Use the nft iptables backend where available (matches modern kernels; harmless otherwise).
         .run_cmd("update-alternatives --set iptables /usr/sbin/iptables-nft || true; "
                  "update-alternatives --set ip6tables /usr/sbin/ip6tables-nft || true", user="root")
         # The unprivileged agent user the loop drops to (uid 1000).
         .run_cmd(f"id -u {AGENT_UID} >/dev/null 2>&1 || useradd -m -u {AGENT_UID} -s /bin/bash agent",
                  user="root")
         # Fixed mount points the task is staged into at run time.
         .run_cmd("mkdir -p /config /data /output /scripts /src /out /work", user="root")
         # Install the validator's Python venv + deps now, so scoring needs no network later.
         .run_cmd(f"echo {deps} | base64 -d > /install_validate_deps.sh && "
                  "bash -eux /install_validate_deps.sh && rm /install_validate_deps.sh", user="root")
         # Snapshot the image's build env (CC, CFLAGS, SANITIZER, SRC/OUT/WORK) so later shells can
         # restore it with `. /etc/cybergym.env` without re-deriving it.
         .run_cmd("bash -c 'export -p > /etc/cybergym.env'", user="root"))
    # prepare.sh is NOT baked -- it runs per VM at stage time, so this one template serves every task
    # that shares the build image.
    t0 = time.time()
    await AsyncTemplate.build(t, name=vc.BASE_TEMPLATE, cpu_count=cpu, memory_mb=mem)
    print(f"built base template {vc.BASE_TEMPLATE} in {time.time() - t0:.0f}s")


async def stage_task(sb, cg, name="stage"):
    """Upload + unpack the task and run prepare.sh. The caller keeps the network OPEN here (trusted
    setup) and raises the firewall afterwards. No ground truth is uploaded."""
    cfg = task_config(cg)
    repo = cfg["repo_to_patch"]
    data = cg / "data" / "projects" / vc.PROJECT / vc.TASK  # gated HF assets (src.tgz, ...)
    task = cg / "projects" / vc.PROJECT / vc.TASK            # repo per-task scripts
    # Only the two keys the validator needs; ground-truth fields are deliberately NOT written.
    # Same sanitized keys as upstream utils.setup_workspace (JSON strings/arrays are valid TOML).
    sanitized = (f"repo_to_patch = {json.dumps(repo)}\n"
                 f"immutable_files = {json.dumps(cfg.get('immutable_files', []))}\n")
    await sb.files.write("/scripts/validate.py", (cg / "scripts" / "validate.py").read_text(), user="root")
    await sb.files.write("/config/config.toml", sanitized, user="root")
    print(f"[{name}] uploading src.tgz...")
    await sb.files.write("/src/src.tgz", (data / "src.tgz").read_bytes(), user="root",
                         request_timeout=900)
    # Build/run scripts only -- NOT poc.bin/crash.log/patch.diff (those would hand the agent the answer).
    for s in ("prepare.sh", "compile.sh", "run_poc.sh", "test.sh"):
        await sb.files.write(f"/src/{s}", (task / s).read_text(), user="root")
    print(f"[{name}] running prepare.sh (network open, trusted setup)...")
    # Unpack, drop the repo's .git, run the task's prepare.sh, snapshot a pristine /src_backup (the
    # validator rebuilds from it), hand the work dirs to uid 1000, and fail if any answer file leaked in.
    r = await sb.commands.run(
        f"set -e; tar xf /src/src.tgz -C /src; rm /src/src.tgz; rm -rf /src/{repo}/.git; "
        ". /etc/cybergym.env; export SRC=/src OUT=/out WORK=/work; cd /src; "
        "if [ -f /src/prepare.sh ]; then bash -eux /src/prepare.sh > /output/prepare.log 2>&1; fi; "
        "rm -rf /src_backup; cp -a /src /src_backup; "
        f"chown -R {AGENT_UID}:{AGENT_UID} {AGENT_DIRS}; "
        f"for f in {' '.join(FORBIDDEN)}; do test ! -e $f || {{ echo LEAK $f; exit 1; }}; done; "
        "sync; du -sh /src /src_backup", user="root", timeout=1800)
    if r.exit_code != 0:
        tail = (await sb.commands.run("tail -40 /output/prepare.log 2>/dev/null", user="root", timeout=30)).stdout
        raise RuntimeError(f"staging/prepare failed: {r.stdout[-600:]} {r.stderr[-600:]}\n{tail}")
    print(f"[{name}] {r.stdout.strip()}")


async def main():
    e2b_tls.install()  # optional custom-CA/self-signed trust for the E2B backend (no-op by default)
    ap = argparse.ArgumentParser(description="Build the project-agnostic agent base template.")
    ap.add_argument("--base", action="store_true", required=True, help="build the base template")
    ap.add_argument("--task", required=True, help="project/arvo_id (selects the build image)")
    ap.add_argument("--cybergym-dir", type=Path, default=DEFAULT_CYBERGYM)
    ap.add_argument("--cpu", type=int, default=4)
    ap.add_argument("--memory-mb", type=int, default=8192)
    a = ap.parse_args()
    sys.argv = sys.argv[:1]
    vc.set_task(a.task, a.cybergym_dir)
    await build_base(a.cybergym_dir, a.cpu, a.memory_mb)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
