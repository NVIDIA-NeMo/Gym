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

"""Build a Lean + Mathlib sandbox environment and snapshot it.

**This is the recipe for the environment every Lean benchmark's numbers depend on.** Run it
once per Mathlib version; it prints a snapshot id that later runs boot from in seconds.
Nobody runs it again unless the snapshot is lost.

The version is an argument because the benchmarks disagree and the environments are not
interchangeable: minif2f, mobench, proofnet and putnam_bench run against this server at
Mathlib v4.12.0, while leancat's statements are written against v4.19.0. Compiled oleans do
not carry across versions, so each version needs its own snapshot. The cost of getting it
wrong is quiet: on v4.12.0, 36 of leancat's 100 reference statements fail to compile with
their ``sorry`` still intact, which caps any score at 64/100 and skews it by difficulty.

Why a snapshot rather than an image: no published image carries Mathlib v4.19.0 --
``leanprovercommunity/mathlib`` ships only ``latest``/``gitpod``/``debian``, and NeMo-Skills'
sandbox image is pinned to v4.12.0. Building inside a sandbox and snapshotting needs no
docker, no registry and no push.

The base is a slim Debian image, not the NeMo-Skills sandbox: that one is
``uwsgi-nginx-flask`` plus pypy and a large Python stack, all of it there to serve the HTTP
``/execute`` API -- roughly 15 GB pulled on every start and stored in every snapshot. Its
prebuilt Mathlib does not help either, since oleans do not carry across versions and the
cache is refetched regardless.

A snapshot is an opaque blob tied to one cell, so treat it as a cache and this script as the
source of truth: if the snapshot is lost or the cell retired, re-run this.

Usage:
    export OPENSANDBOX_DOMAIN=https://<cell-endpoint>
    export OPENSANDBOX_API_KEY=<key>
    python build_sandbox.py --lean-version v4.19.0   # leancat
    python build_sandbox.py --lean-version v4.12.0   # minif2f, mobench, proofnet, putnam_bench
    python build_sandbox.py --no-snapshot --keep     # build only, leave it up to inspect
"""

import argparse
import asyncio
import json
import os
import ssl
import sys
import urllib.request
from typing import Any, Dict, Optional

from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.config import resolve_provider_config


# Slim base: this script installs elan and Mathlib itself, so nothing unused is inherited.
BASE_IMAGE = "debian:bookworm-slim"
LEAN_PROJECT_DIR = "/lean4/my_project"

# leancat's pin (its upstream configs/evaluation_protocol.json); the other benchmarks on this
# server are v4.12.0, so this is always worth passing explicitly.
DEFAULT_LEAN_VERSION = "v4.19.0"


def build_script(lean_version: str, mathlib_version: str) -> str:
    """Install elan, create a lake project pinned to `lean_version`, and fetch Mathlib.

    `lake exe cache get` downloads prebuilt oleans, which is what keeps this minutes rather
    than the hours a from-source Mathlib build takes. `lake build` afterwards is a no-op when
    the cache is complete and a fallback when it is not.
    """
    return f"""set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends curl git ca-certificates build-essential
rm -rf /var/lib/apt/lists/*

curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
  | sh -s -- -y --default-toolchain leanprover/lean4:{lean_version}
export PATH="/root/.elan/bin:$PATH"

mkdir -p {LEAN_PROJECT_DIR}
cd {LEAN_PROJECT_DIR}
echo 'leanprover/lean4:{lean_version}' > lean-toolchain
cat > lakefile.lean <<'LAKEFILE'
import Lake
open Lake DSL

package «my_project» where

require mathlib from git "https://github.com/leanprover-community/mathlib4" @ "{mathlib_version}"
LAKEFILE

# elan only exports PATH through ~/.profile, which the non-login shells that `exec` spawns
# never read -- and the server runs `lake env lean` through exactly such a shell. Symlink the
# binaries somewhere already on PATH so the environment works without a login shell.
ln -sf /root/.elan/bin/lake /usr/local/bin/lake
ln -sf /root/.elan/bin/lean /usr/local/bin/lean
ln -sf /root/.elan/bin/elan /usr/local/bin/elan

lake update
lake exe cache get
lake build

# Prove the binaries resolve the way the server will call them: no PATH, no login shell.
env -i /usr/local/bin/lake --version
"""


# The check that decides whether the environment is usable at all.
PROBE = "import Mathlib\n#eval Lean.versionString"


def _api(path: str, method: str = "GET", body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Call the OpenSandbox control plane directly.

    Snapshot creation is not exposed through Gym's provider (it only consumes
    ``provider_options.snapshot_id``), so this reaches the REST API for that one step.
    The cells use self-signed certificates, hence the unverified context -- the same choice
    the shipped provider config makes with ``tls_verify: false``.
    """
    domain = os.environ["OPENSANDBOX_DOMAIN"].rstrip("/")
    request = urllib.request.Request(
        f"{domain}{path}",
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={
            "OPEN-SANDBOX-API-KEY": os.environ["OPENSANDBOX_API_KEY"],
            "Content-Type": "application/json",
        },
    )
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(request, context=context, timeout=300) as response:
        raw = response.read().decode()
    return json.loads(raw) if raw else {}


def provider_config(tls_verify: bool = False) -> Dict[str, Any]:
    return {
        "opensandbox": {
            "connection": {
                "domain": os.environ["OPENSANDBOX_DOMAIN"],
                "api_key": os.environ["OPENSANDBOX_API_KEY"],
                "tls_verify": tls_verify,
            }
        }
    }


async def run(sandbox: AsyncSandbox, command: str, timeout_s: float, label: str) -> str:
    print(f"==> {label}")
    result = await sandbox.exec(command, cwd=LEAN_PROJECT_DIR, timeout_s=timeout_s)
    if result.error_type or result.return_code != 0:
        print((result.stdout or "")[-4000:])
        print((result.stderr or "")[-4000:], file=sys.stderr)
        raise SystemExit(f"FAIL: {label} (exit {result.return_code}, error_type={result.error_type})")
    return f"{result.stdout or ''}\n{result.stderr or ''}"


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-image", default=BASE_IMAGE)
    parser.add_argument(
        "--lean-version",
        default=DEFAULT_LEAN_VERSION,
        help="Lean toolchain tag. Mathlib is pinned to the same tag unless --mathlib-version is given.",
    )
    parser.add_argument("--mathlib-version", default=None, help="Defaults to --lean-version.")
    parser.add_argument("--cpu", type=float, default=8, help="Build is CPU-bound; more is faster.")
    parser.add_argument("--memory-mib", type=int, default=32768)
    parser.add_argument("--disk-gib", type=int, default=60, help="Mathlib's oleans are tens of GB.")
    parser.add_argument("--build-timeout", type=float, default=7200)
    parser.add_argument("--keep", action="store_true", help="Leave the sandbox running.")
    parser.add_argument("--no-snapshot", action="store_true", help="Build and verify only.")
    args = parser.parse_args()

    mathlib_version = args.mathlib_version or args.lean_version

    for name in ("OPENSANDBOX_DOMAIN", "OPENSANDBOX_API_KEY"):
        if not os.environ.get(name):
            print(f"FAIL: ${name} is not set.", file=sys.stderr)
            return 2

    sandbox = AsyncSandbox(resolve_provider_config(provider_config()))
    print(f"==> starting a sandbox from {args.base_image}")
    await sandbox.start(
        SandboxSpec(
            image=args.base_image,
            ttl_s=args.build_timeout + 3600,
            ready_timeout_s=1800,
            workdir=LEAN_PROJECT_DIR,
            resources=SandboxResources.from_mapping(
                {"cpu": args.cpu, "memory_mib": args.memory_mib, "disk_gib": args.disk_gib}
            ),
            metadata={
                "purpose": "build-mathlib-environment",
                "lean-version": args.lean_version,
            },
        )
    )

    try:
        await run(
            sandbox,
            build_script(args.lean_version, mathlib_version),
            args.build_timeout,
            f"installing Lean {args.lean_version} and Mathlib {mathlib_version}",
        )

        # Prove the toolchain is the one the statements need before anything is snapshotted.
        output = await run(
            sandbox,
            f"cat > /tmp/probe.lean <<'EOF'\n{PROBE}\nEOF\nlake env lean /tmp/probe.lean",
            600,
            "probing the Lean version",
        )
        if args.lean_version.lstrip("v") not in output:
            print(output[-2000:], file=sys.stderr)
            raise SystemExit(f"FAIL: sandbox does not report Lean {args.lean_version}. Not snapshotting.")
        print(f"    Lean {args.lean_version} confirmed")

        if args.no_snapshot:
            print("--no-snapshot: stopping here.")
            return 0

        handle = await sandbox.serialize()
        sandbox_id = handle.get("sandbox_id") or handle.get("id")
        print(f"==> snapshotting sandbox {sandbox_id}")
        snapshot = _api(
            f"/sandboxes/{sandbox_id}/snapshots",
            method="POST",
            body={"metadata": {"lean-version": args.lean_version, "mathlib-version": mathlib_version}},
        )
        snapshot_id = snapshot.get("id") or snapshot.get("snapshotId")

        print(f"\nSnapshot: {snapshot_id}\n")
        print("Gate it before trusting any score -- for leancat, all 100 reference statements:")
        print(f"  python ../leancat/check_sandbox.py --snapshot-id {snapshot_id}")
        print("\nThen point runs at it:")
        print(f"  export LEANCAT_SANDBOX_SNAPSHOT_ID={snapshot_id}")
        return 0
    finally:
        if args.keep:
            print("--keep: sandbox left running; delete it when done.")
        else:
            await sandbox.stop()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
