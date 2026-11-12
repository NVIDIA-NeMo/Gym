# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check UID switching and optional model-only egress on a deployed OpenSandbox image."""

import argparse
import asyncio
import json
import shlex
from pathlib import Path
from uuid import uuid4

from omegaconf import OmegaConf

from nemo_gym.agents import restricted_network_policy
from nemo_gym.sandbox import AsyncSandbox, SandboxSpec, create_provider, resolve_provider_config
from resources_servers.pdf_analyst_qa.documents import SANDBOX_SCRIPTS_ROOT, sandbox_script_files


async def check(
    image: str, provider_config: str, model_url: str | None = None, node_cache_host_path: str | None = None
) -> dict:
    configuration = OmegaConf.to_container(OmegaConf.load(Path(provider_config)), resolve=True)
    provider = create_provider(resolve_provider_config("sandbox", configuration))
    sandbox = AsyncSandbox(provider)
    report = {
        "image": image,
        "root": None,
        "default": None,
        "model_network": None,
        "missing_view_mount": None,
        "stopped": False,
    }
    try:
        options = {"network_policy": restricted_network_policy("opensandbox", [model_url])} if model_url else {}
        if node_cache_host_path:
            view_path = f"{node_cache_host_path}/views/pdf-analyst-qa-mount-probe-{uuid4().hex}"
            report["view_host_path"] = view_path
            options["volumes"] = [
                {
                    "name": "pdf-analyst-qa-node-cache",
                    "host": {"path": node_cache_host_path},
                    "mountPath": "/var/lib/pdf-analyst-qa-cache",
                    "readOnly": False,
                },
                {
                    "name": "pdf-analyst-qa-documents",
                    "host": {"path": view_path},
                    "mountPath": "/workspace/docs",
                    "readOnly": True,
                },
            ]
        await sandbox.start(
            SandboxSpec(
                image=image,
                workdir="/workspace/task",
                ttl_s=600,
                ready_timeout_s=300,
                entrypoint=["/bin/bash", "-c", "exec sleep infinity"],
                provider_options=options,
                files=sandbox_script_files(),
            )
        )
        for name, user in [("root", 0), ("default", None)]:
            observed = await sandbox.exec("id -u", user=user, timeout_s=30)
            report[name] = {
                "return_code": observed.return_code,
                "error_type": observed.error_type,
                "stdout": (observed.stdout or "").strip(),
                "stderr": (observed.stderr or "").strip(),
            }
        if model_url:
            observed = await sandbox.exec(
                shlex.join(["python3", f"{SANDBOX_SCRIPTS_ROOT}/network_probe.py", model_url]), timeout_s=30
            )
            report["model_network"] = {
                "return_code": observed.return_code,
                "error_type": observed.error_type,
                "stdout": (observed.stdout or "").strip(),
                "stderr": (observed.stderr or "").strip(),
            }
        if node_cache_host_path:
            observed = await sandbox.exec(
                "test -d /workspace/docs && ! touch /workspace/docs/probe-file", timeout_s=30
            )
            report["missing_view_mount"] = {
                "return_code": observed.return_code,
                "error_type": observed.error_type,
                "stdout": (observed.stdout or "").strip(),
                "stderr": (observed.stderr or "").strip(),
            }
    finally:
        await sandbox.stop()
        report["stopped"] = True
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--provider-config", default="nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml")
    parser.add_argument("--model-url", help="Actual sandbox-reachable training-node model URL")
    parser.add_argument("--node-cache-host-path", help="Probe a fresh workspace-specific view bind under this cache")
    args = parser.parse_args()
    report = asyncio.run(check(args.image, args.provider_config, args.model_url, args.node_cache_host_path))
    print(json.dumps(report, indent=2))
    assert (
        report["root"]["error_type"] is None and report["root"]["return_code"] == 0 and report["root"]["stdout"] == "0"
    )
    assert (
        report["default"]["error_type"] is None
        and report["default"]["return_code"] == 0
        and report["default"]["stdout"] == "65534"
    )
    if args.model_url:
        assert report["model_network"]["error_type"] is None and report["model_network"]["return_code"] == 0
    if args.node_cache_host_path:
        assert report["missing_view_mount"]["error_type"] is None
        assert report["missing_view_mount"]["return_code"] == 0
