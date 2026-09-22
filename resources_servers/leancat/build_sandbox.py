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

"""Build LeanCat's Lean 4.19.0 / Mathlib v4.19.0 environment and snapshot it.

**This is the recipe for the environment every LeanCat number depends on.** Run it once;
it prints a snapshot id that every later run boots from in seconds via
``LEANCAT_SANDBOX_SNAPSHOT_ID``. Nobody runs it again unless the snapshot is lost.

Why a snapshot and not an image: LeanCat's upstream publishes no container, and there is no
published image at Mathlib v4.19.0 -- ``leanprovercommunity/mathlib`` ships only
``latest``/``gitpod``/``debian``, and NeMo-Skills' sandbox image is pinned to v4.12.0 (a
sandbox on which 36 of the 100 reference statements fail to compile with their ``sorry``
still intact, silently capping any score at 64/100). Building inside a sandbox and
snapshotting needs no docker, no registry and no push.

The base image is NeMo-Skills' published sandbox: it already carries elan, lake and a
Mathlib project at /lean4/my_project, so this only has to move the pins and refetch the
cache. Any image with that layout works.

A snapshot is an opaque blob tied to one cell, so treat it as a cache and this script as the
source of truth: if the snapshot is lost or the cell is retired, re-run this.

Usage:
    export OPENSANDBOX_DOMAIN=https://<cell-endpoint>
    export OPENSANDBOX_API_KEY=<key>
    python build_sandbox.py                      # build, verify, snapshot
    python build_sandbox.py --keep               # leave the sandbox running for debugging
    python build_sandbox.py --no-snapshot        # build and verify only
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


# Published, and already Lean-shaped: elan + lake + a Mathlib project at LEAN_PROJECT_DIR.
BASE_IMAGE = "igitman/nemo-skills-sandbox:0.7.2"
LEAN_PROJECT_DIR = "/lean4/my_project"

# What LeanCat's statements are written against (upstream configs/evaluation_protocol.json).
LEAN_VERSION = "v4.19.0"
MATHLIB_VERSION = "v4.19.0"

# Repin the project and refetch Mathlib. `lake exe cache get` downloads prebuilt olean files,
# which is what keeps this minutes rather than the hours a from-source Mathlib build takes.
BUILD_SCRIPT = f"""set -euo pipefail
cd {LEAN_PROJECT_DIR}
elan toolchain install leanprover/lean4:{LEAN_VERSION}
elan override set leanprover/lean4:{LEAN_VERSION}
echo 'leanprover/lean4:{LEAN_VERSION}' > lean-toolchain
cat > lakefile.lean <<'LAKEFILE'
import Lake
open Lake DSL

package «my_project» where

require mathlib from git "https://github.com/leanprover-community/mathlib4" @ "{MATHLIB_VERSION}"

@[default_target]
lean_lib «MyProject» where
LAKEFILE
rm -f lake-manifest.json
lake update
lake exe cache get
lake build
"""

# The check that decides whether the environment is usable at all.
PROBE = "import Mathlib\\n#eval Lean.versionString\\n"


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
    parser.add_argument("--cpu", type=float, default=8, help="Build is CPU-bound; more is faster.")
    parser.add_argument("--memory-mib", type=int, default=32768)
    parser.add_argument("--disk-gib", type=int, default=60, help="Mathlib's oleans are tens of GB.")
    parser.add_argument("--build-timeout", type=float, default=7200)
    parser.add_argument("--keep", action="store_true", help="Leave the sandbox running.")
    parser.add_argument("--no-snapshot", action="store_true", help="Build and verify only.")
    args = parser.parse_args()

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
            metadata={"benchmark": "leancat", "purpose": "build-mathlib-environment"},
        )
    )

    try:
        await run(sandbox, BUILD_SCRIPT, args.build_timeout, f"building Mathlib {MATHLIB_VERSION}")

        # Prove the toolchain is the one the statements need before anything is snapshotted.
        output = await run(
            sandbox,
            f"cat > /tmp/probe.lean <<'EOF'\n{PROBE}\nEOF\nlake env lean /tmp/probe.lean",
            600,
            "probing the Lean version",
        )
        if LEAN_VERSION.lstrip("v") not in output:
            print(output[-2000:], file=sys.stderr)
            raise SystemExit(f"FAIL: sandbox does not report Lean {LEAN_VERSION}. Not snapshotting.")
        print(f"    Lean {LEAN_VERSION} confirmed")

        if args.no_snapshot:
            print("--no-snapshot: stopping here.")
            return 0

        handle = await sandbox.serialize()
        sandbox_id = handle.get("sandbox_id") or handle.get("id")
        print(f"==> snapshotting sandbox {sandbox_id}")
        snapshot = _api(f"/sandboxes/{sandbox_id}/snapshots", method="POST", body={})
        snapshot_id = snapshot.get("id") or snapshot.get("snapshotId")

        print(f"\nSnapshot: {snapshot_id}\n")
        print("Gate it before trusting any score -- all 100 reference statements must compile:")
        print(f"  python check_sandbox.py --snapshot-id {snapshot_id}")
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
