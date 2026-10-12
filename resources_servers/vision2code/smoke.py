# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed isolation smoke; run inside the intended Slurm worker allocation."""

import argparse
import asyncio
import json
import os
from pathlib import Path

from resources_servers.vision2code.render import render


def isolation_code(forbidden_paths: list[str]) -> str:
    """Build a trusted probe that never prints credential values."""
    return """
import json, os, socket
from PIL import Image, ImageDraw
allowed = {'PATH','HOME','LANG','TMPDIR','MPLBACKEND','MPLCONFIGDIR','OPENBLAS_NUM_THREADS',
           'OMP_NUM_THREADS','OUTPUT_PATH','TARGET_WIDTH','TARGET_HEIGHT'}
unexpected_env = sorted(set(os.environ) - allowed)
visible_forbidden = [p for p in FORBIDDEN if os.path.exists(p)]
network_blocked = False
try:
    with socket.create_connection(('1.1.1.1', 443), timeout=2):
        pass
except OSError:
    network_blocked = True
routes = open('/proc/net/route').read().splitlines()[1:]
processes = [name for name in os.listdir('/proc') if name.isdigit()]
report = dict(network_blocked=network_blocked, routes=routes,
              unexpected_env=unexpected_env, visible_forbidden=visible_forbidden,
              processes_visible=len(processes), environment_keys=sorted(os.environ))
print(json.dumps(report), flush=True)
assert network_blocked and not routes and not unexpected_env and not visible_forbidden, report
assert len(processes) == 1, 'Parent/host processes are visible'
im = Image.new('RGB', (TARGET_WIDTH, TARGET_HEIGHT), 'white')
ImageDraw.Draw(im).rectangle((10, 10, 50, 50), fill='black')
im.save(OUTPUT_PATH)
""".replace("FORBIDDEN", repr(forbidden_paths))


async def main_async(args: argparse.Namespace) -> None:
    forbidden = [str(Path.home()), str(Path.cwd()), "/root/.aws", "/root/.ssh", *args.forbid_path]
    provider = {
        "enroot": {
            "create": {"mount_point": "/workspace"},
            "probe": {"command": "test -d /workspace", "expected_stdout": None},
        }
    }
    result = await render(
        code=isolation_code(forbidden),
        width=80,
        height=60,
        image=str(args.image.resolve()),
        provider=provider,
        artifact_dir=args.artifacts.resolve(),
        timeout=60,
    )
    report = result.model_dump()
    report["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    report["image"] = str(args.image.resolve())
    (args.artifacts / "isolation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if result.status != "ok" or not report["slurm_job_id"]:
        raise SystemExit("Isolation smoke did not pass inside a Slurm allocation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument(
        "--forbid-path", action="append", default=[], help="Reference or credential path that must be absent"
    )
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
