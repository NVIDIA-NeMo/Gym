# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Probe an OpenSandbox cell from a cluster node before running OfficeEnv sandbox episodes.

Starts one sandbox the way `configs/officeenv_opensandbox.yaml` would (image, resources, setup commands), then
measures what an episode depends on: setup time, the task tools, a call back to this node (where Gym's servers
would run), internet egress for Pi's runtime, and the time Pi's runtime install takes. Prints one JSON summary.

Run it on the node type that will host Gym's servers, with the cell's credentials in the environment
(OPENSANDBOX_DOMAIN, OPENSANDBOX_API_KEY, OPENSANDBOX_PROTOCOL), never on the command line:

    set -a; . <file with the cell's OPENSANDBOX_* variables>; set +a
    NEMO_GYM_RUN_ID=oe-probe-$(date +%s) .venv/bin/python resources_servers/officeenv/sandbox/probe_opensandbox.py

Leftover sandboxes carry the run id; `nemo_gym/sandbox/providers/opensandbox/cleanup_sandboxes.py` removes them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import OmegaConf


if TYPE_CHECKING:
    from nemo_gym.sandbox.api import AsyncSandbox
    from nemo_gym.sandbox.config import SandboxConfig


ROOT = Path(__file__).resolve().parents[3]
PROVIDER_CONFIG = ROOT / "nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml"
OFFICEENV_CONFIG = ROOT / "resources_servers/officeenv/configs/officeenv_opensandbox.yaml"
PI_INSTALLER = ROOT / "responses_api_agents/pi_agent/install_pi_runtime.sh"
EGRESS = ["https://pypi.org/simple/", "https://registry.npmjs.org/", "https://nodejs.org/dist/index.json"]
REPLY = b"officeenv-probe-ok"


class _Reply(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        # The caller's address as this node sees it: what an address filter on Gym's servers would match.
        body = REPLY + f" from {self.client_address[0]}".encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        pass


def node_address() -> str:
    """The address Gym's `use_absolute_ip` would advertise for this node."""
    return socket.gethostbyname(socket.gethostname())


def provider_block() -> dict:
    """The `sandbox` block as the OfficeEnv OpenSandbox config composes it, with the connection from the environment."""
    merged = OmegaConf.merge(
        OmegaConf.load(PROVIDER_CONFIG)["sandbox"], OmegaConf.load(OFFICEENV_CONFIG).get("sandbox", {})
    )
    block = OmegaConf.to_container(merged, resolve=True)
    if os.environ.get("OPENSANDBOX_PROTOCOL"):
        block["opensandbox"]["connection"]["protocol"] = os.environ["OPENSANDBOX_PROTOCOL"]
    return block


def sandbox_config() -> SandboxConfig:
    from nemo_gym.sandbox.config import SandboxConfig

    oe = OmegaConf.to_container(OmegaConf.load(OFFICEENV_CONFIG)["oe"], resolve=True)
    settings = oe["resources_servers"]["officeenv"]["sandbox_config"]
    return SandboxConfig(
        **{key: settings[key] for key in ("ttl_s", "resources", "env", "metadata") if key in settings}
    )


async def timed(sandbox: AsyncSandbox, command: str, timeout_s: float) -> dict:
    started = time.monotonic()
    result = await sandbox.exec(command, timeout_s=timeout_s)
    return {
        "seconds": round(time.monotonic() - started, 1),
        "return_code": result.return_code,
        "stdout": (result.stdout or "")[-1500:],
        "stderr": (result.stderr or "")[-1500:],
    }


async def probe(args: argparse.Namespace) -> dict:
    # Imported after argument parsing: Gym reads the process arguments as config overrides.
    from nemo_gym.sandbox.api import AsyncSandbox
    from nemo_gym.sandbox.config import resolve_provider_config

    block = provider_block()
    config = sandbox_config()
    if args.ttl_s:
        config = config.model_copy(update={"ttl_s": args.ttl_s})
    image = OmegaConf.load(OFFICEENV_CONFIG)["oe"]["resources_servers"]["officeenv"]["sandbox_config"]["image"]
    spec = config.spec(
        "sandbox",
        image=image,
        workdir="/workspace",
        metadata={"officeenv-probe": "1"},
        files={"/tmp/install_pi_runtime.sh": PI_INSTALLER.read_text()},
        named_configs={"sandbox": block},
    )
    host = args.host or node_address()
    server = ThreadingHTTPServer(("0.0.0.0", args.port), _Reply)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    summary: dict = {"node": socket.gethostname(), "node_address": host, "port": args.port, "image": image}
    sandbox = AsyncSandbox(resolve_provider_config("sandbox", {"sandbox": block}))
    started = time.monotonic()
    try:
        await sandbox.start(spec)
        summary["start_seconds_incl_setup"] = round(time.monotonic() - started, 1)
        summary["sandbox_id"] = sandbox.sandbox_id
        summary["system"] = await timed(sandbox, "nproc; free -m | sed -n 2p; df -h /tmp /workspace | tail -2", 60)
        summary["task_tools"] = await timed(
            sandbox, "python3 -c 'import pandas, openpyxl; print(pandas.__version__, openpyxl.__version__)'", 60
        )
        summary["callback_to_node"] = await timed(sandbox, f"curl -sS -m 15 http://{host}:{args.port}/probe", 60)
        summary["egress"] = await timed(
            sandbox,
            " ; ".join(
                f"curl -sS -o /dev/null -m 20 -w '%{{http_code}} %{{time_total}} {url}\\n' {url}" for url in EGRESS
            ),
            120,
        )
        if not args.skip_pi:
            runtime = f"/tmp/nemo-gym-pi-node-22.19.0-{args.pi_version}"
            summary["pi_install"] = await timed(
                sandbox, f"bash /tmp/install_pi_runtime.sh {runtime} {args.pi_version} && ls {runtime}", 900
            )
    finally:
        await sandbox.stop()
        server.shutdown()
    summary["callback_ok"] = REPLY.decode() in summary.get("callback_to_node", {}).get("stdout", "")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=5999, help="port of the callback server on this node")
    parser.add_argument("--host", help="address the sandbox calls back (default: this node's resolved address)")
    parser.add_argument("--pi-version", default="0.80.2")
    parser.add_argument("--skip-pi", action="store_true", help="skip timing Pi's runtime install")
    parser.add_argument("--ttl-s", type=int, default=1800, help="sandbox lifetime for the probe")
    args = parser.parse_args()
    sys.argv = sys.argv[:1]  # Gym parses the process arguments as Hydra overrides
    print(json.dumps(asyncio.run(probe(args)), indent=2))


if __name__ == "__main__":
    main()
