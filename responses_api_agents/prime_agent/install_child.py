# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Install verified Prime release archives already uploaded by the Agent Server."""

import json
import subprocess
import sys
from pathlib import Path


def install(root: Path, packages: list[str]) -> None:
    """Install the SDK and its isolated Python kernel; requires Node 22 and Python 3.11+."""
    node = subprocess.run(["node", "--version"], check=True, capture_output=True, text=True).stdout.strip()
    if not node.startswith("v22.") or sys.version_info < (3, 11):
        raise RuntimeError("Prime child installation requires Node 22 and Python 3.11 or newer")
    names = {
        "prime-agent": "prime-agent",
        "prime-agent-ai": "@earendil-works/pi-ai",
        "prime-agent-core": "@earendil-works/pi-agent-core",
        "prime-agent-tui": "@earendil-works/pi-tui",
    }
    dependencies = {value: f"file:{root}/{key}-0.7.0.tgz" for key, value in names.items()}
    (root / "package.json").write_text(
        json.dumps(
            {
                "private": True,
                "type": "module",
                "dependencies": dependencies,
                "overrides": {name: f"${name}" for name in dependencies if name != "prime-agent"},
            }
        )
    )
    subprocess.run(["npm", "install", "--prefix", str(root), "--omit=dev", "--no-audit", "--no-fund"], check=True)
    subprocess.run(["python3", "-m", "venv", str(root / "kernel")], check=True)
    dist = root / "node_modules/prime-agent/dist"
    subprocess.run(
        [
            str(root / "kernel/bin/python"),
            "-m",
            "pip",
            "install",
            str(dist / "prime-agent-runtime"),
            str(dist / "skills/agent-message"),
            "ipykernel",
            "dill",
            "requests",
            "httpx",
            "pyyaml",
            "tomli",
            "python-dotenv",
            "pandas",
            "numpy",
            "scipy",
            "beautifulsoup4",
            "lxml",
            "pydantic",
            "tyro",
            *packages,
        ],
        check=True,
    )
    subprocess.run(
        [
            "node",
            "--input-type=module",
            "-e",
            f"import {{createAgentSession}} from '{dist}/index.js'; if(!createAgentSession) process.exit(1)",
        ],
        check=True,
    )


if __name__ == "__main__":
    install(Path(sys.argv[1]), json.loads(sys.argv[2]))
