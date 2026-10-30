# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prove what an Atom shell sees in a task sandbox, without running a benchmark row.

Starts one sandbox from a task image through the configured provider, stages the NOOA runtime exactly as the agent
server does, applies the entrypoint's PATH policy, and reports what `command -v` resolves for the interpreter,
pip and rg. Acceptance: python3 and pip resolve to the image's own tools (or to nothing), rg resolves to
`<runtime>/shell-tools/rg`, and nothing from the runtime's `bin` is on PATH.

    set -a; source .env; set +a          # OPENSANDBOX_DOMAIN, OPENSANDBOX_API_KEY
    python recipes/nooa-atom/probe_runtime.py --image alexgshaw/code-from-image:20251031 \
        --wheelhouse "$NOOA_RUNTIME_WHEELHOUSE" [--wheelhouse-musl "$NOOA_RUNTIME_WHEELHOUSE_MUSL"]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from shlex import quote

from omegaconf import OmegaConf

from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.config import resolve_provider_config
from responses_api_agents.nooa_agent.sandbox_runtime import prepare_nooa_runtime


PROVIDER_CONFIG = "nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml"

# Runs under the staged interpreter with the entrypoint's PATH policy applied, then asks a shell what it sees.
PROBE = r"""
import json, os, subprocess, sys
from responses_api_agents.nooa_agent.sandbox_entrypoint import expose_runtime_scripts
path = expose_runtime_scripts()
shell = "for t in python3 python pip pip3 rg; do printf '%s=' \"$t\"; command -v \"$t\" || echo; done; " \
        "rg=$(command -v rg) && printf 'rg_target=%s\n' \"$(readlink -f \"$rg\")\"; " \
        "py=$(command -v python3) && printf 'python3_prefix=%s\n' \"$($py -c 'import sys; print(sys.prefix)')\"; " \
        "command -v pip >/dev/null && pip --version | sed 's/^/pip_version=/'; true"
out = subprocess.run(["sh", "-c", shell], capture_output=True, text=True, env={**os.environ, "PATH": path})
print(json.dumps({"runtime_python": sys.executable, "PATH": path, "shell": out.stdout, "stderr": out.stderr}))
"""


def verdict(report: dict[str, str], runtime_python: str) -> list[str]:
    runtime_root = str(Path(runtime_python).parent.parent)
    failures = []
    for tool in ("python3", "python", "pip", "pip3"):
        if report.get(tool, "").startswith(runtime_root + "/bin"):
            failures.append(f"{tool} resolves into the harness runtime: {report[tool]}")
    if report.get("rg") != f"{runtime_root}/shell-tools/rg":
        failures.append(f"rg does not resolve to {runtime_root}/shell-tools/rg: {report.get('rg') or '(absent)'}")
    if any(part == f"{runtime_root}/bin" for part in report.get("PATH", "").split(":")):
        failures.append("the runtime's bin is on PATH")
    return failures


async def probe(args: argparse.Namespace) -> int:
    config = OmegaConf.to_container(OmegaConf.load(args.provider_config), resolve=True)
    sandbox = AsyncSandbox(resolve_provider_config("sandbox", config))
    spec = SandboxSpec(
        image=args.image,
        ttl_s=1800,
        ready_timeout_s=args.ready_timeout_s,
        resources=SandboxResources.from_mapping({"cpu": 2, "memory_mib": 4096}),
        metadata={"nemo_gym_agent": "nooa-atom-probe"},
    )
    await sandbox.start(spec)
    try:
        python = await prepare_nooa_runtime(
            sandbox,
            wheelhouse_path=args.wheelhouse,
            musl_wheelhouse_path=args.wheelhouse_musl,
        )
        result = await sandbox.exec(f"{quote(python)} -I -c {quote(PROBE)}", cwd="/", timeout_s=300)
    finally:
        await sandbox.stop()
    if result.return_code != 0 or result.error_type:
        print(f"probe failed: rc={result.return_code} error={result.error_type}\n{result.stdout}\n{result.stderr}")
        return 2
    payload = json.loads((result.stdout or "").strip().splitlines()[-1])
    report = dict(line.split("=", 1) for line in payload["shell"].splitlines() if "=" in line)
    report["PATH"] = payload["PATH"]
    failures = verdict(report, payload["runtime_python"])
    print(json.dumps({"image": args.image, "runtime_python": payload["runtime_python"], **report}, indent=2))
    if payload["stderr"].strip():
        print("stderr:", payload["stderr"].strip(), file=sys.stderr)
    print("ACCEPTED" if not failures else "REJECTED: " + "; ".join(failures))
    return 0 if not failures else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, help="Task image, as the dataset row names it.")
    parser.add_argument(
        "--wheelhouse", type=Path, required=True, help="glibc runtime bundle (NOOA_RUNTIME_WHEELHOUSE)"
    )
    parser.add_argument("--wheelhouse-musl", type=Path, default=None, help="musl bundle for Alpine/BusyBox images")
    parser.add_argument("--provider-config", default=PROVIDER_CONFIG)
    parser.add_argument("--ready-timeout-s", type=int, default=1200)
    sys.exit(asyncio.run(probe(parser.parse_args())))


if __name__ == "__main__":
    main()
