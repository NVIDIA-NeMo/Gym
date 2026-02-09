#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Pre-build all NeMo Gym server virtual environments.

This script discovers all server directories (responses_api_models, resources_servers,
responses_api_agents) that have a pyproject.toml or requirements.txt, and creates a
venv for each in $NEMO_GYM_VENV_DIR.

Usage:
    python nemo_gym/prefetch_gym_venvs.py /opt/gym_venvs

    # Only prefetch specific servers:
    python nemo_gym/prefetch_gym_venvs.py /opt/gym_venvs --filter vllm_model simple_agent

    # Or via environment variable:
    NEMO_GYM_VENV_DIR=/opt/gym_venvs python nemo_gym/prefetch_gym_venvs.py

Environment variables:
    NEMO_GYM_VENV_DIR                 Fallback for venv_dir if not passed as argument.
    NEMO_GYM_FORCE_REBUILD_VENVS      Set to 'true' to force rebuild (same as --force-rebuild).
    UV_LINK_MODE                      Optional. Set to 'symlink' to share packages via UV cache (recommended).
    PYTHON_VERSION                    Optional. Python version for venvs (default: current interpreter version).
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


GYM_ROOT = Path(__file__).resolve().parent.parent
SERVER_TYPE_DIRS = ["responses_api_models", "resources_servers", "responses_api_agents"]
FORCE_REBUILD_ENV_VAR = "NEMO_GYM_FORCE_REBUILD_VENVS"


def get_head_server_deps() -> list[str]:
    """Get the head server dependency pins (ray and openai versions)."""
    deps = []
    try:
        import ray

        deps.append(f"ray[default]=={ray.__version__}")
    except ImportError:
        print("  WARNING: ray not importable, skipping ray pin")
    try:
        import openai

        deps.append(f"openai=={openai.__version__}")
    except ImportError:
        print("  WARNING: openai not importable, skipping openai pin")
    return deps


def discover_servers(filters: list[str] | None = None) -> list[tuple[str, Path]]:
    """Discover all server directories with pyproject.toml or requirements.txt."""
    servers = []
    for server_type in SERVER_TYPE_DIRS:
        type_dir = GYM_ROOT / server_type
        if not type_dir.exists():
            continue
        for server_dir in sorted(type_dir.iterdir()):
            if not server_dir.is_dir():
                continue
            has_deps = (server_dir / "pyproject.toml").exists() or (server_dir / "requirements.txt").exists()
            if not has_deps:
                continue

            server_name = server_dir.name
            if filters and server_name not in filters:
                continue

            servers.append((f"{server_type}/{server_name}", server_dir))
    return servers


def prefetch_venv(
    server_label: str,
    server_dir: Path,
    venv_dir: str,
    python_version: str,
    head_server_deps: list[str],
    force_rebuild: bool = False,
) -> bool:
    """Create and populate a venv for one server. Returns True on success."""
    server_name = server_dir.name
    venv_path = os.path.join(venv_dir, server_name)

    if force_rebuild and os.path.exists(venv_path):
        shutil.rmtree(venv_path)

    # Create venv
    result = subprocess.run(
        ["uv", "venv", "--seed", "--allow-existing", "--python", python_version, venv_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  ERROR creating venv: {result.stderr}")
        return False

    # Determine install command
    install_cmd = ["uv", "pip", "install", "--python", f"{venv_path}/bin/python"]
    if (server_dir / "pyproject.toml").exists():
        install_cmd.extend(["-e", str(server_dir)])
    elif (server_dir / "requirements.txt").exists():
        install_cmd.extend(["-r", str(server_dir / "requirements.txt")])
    install_cmd.extend(head_server_deps)

    # Install deps
    result = subprocess.run(
        install_cmd,
        capture_output=True,
        text=True,
        cwd=str(server_dir),
    )
    if result.returncode != 0:
        print(f"  ERROR installing deps: {result.stderr}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Pre-build NeMo Gym server venvs")
    parser.add_argument(
        "venv_dir",
        nargs="?",
        default=None,
        help="Directory where venvs will be created. Falls back to NEMO_GYM_VENV_DIR env var if not provided.",
    )
    parser.add_argument(
        "--filter",
        nargs="+",
        default=None,
        help="Only prefetch servers whose directory name exactly matches one of these strings (e.g., vllm_model simple_agent)",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        default=False,
        help="Force rebuild all venvs from scratch, even if they already exist. "
        f"Can also be set via {FORCE_REBUILD_ENV_VAR}=true",
    )
    args = parser.parse_args()

    venv_dir = args.venv_dir or os.environ.get("NEMO_GYM_VENV_DIR")
    if not venv_dir:
        parser.error("venv_dir is required. Pass as argument or set NEMO_GYM_VENV_DIR.")

    python_version = os.environ.get("PYTHON_VERSION", f"{sys.version_info.major}.{sys.version_info.minor}")
    force_rebuild = args.force_rebuild or os.environ.get(FORCE_REBUILD_ENV_VAR, "false").lower() == "true"

    os.makedirs(venv_dir, exist_ok=True)

    servers = discover_servers(args.filter)
    if not servers:
        print("No servers found to prefetch.")
        sys.exit(0)

    head_server_deps = get_head_server_deps()

    print(f"Prefetching {len(servers)} server venvs into {venv_dir}")

    prefetched = []
    failed = []
    total_start = time.time()

    for label, server_dir in servers:
        t0 = time.time()
        if prefetch_venv(label, server_dir, venv_dir, python_version, head_server_deps, force_rebuild=force_rebuild):
            print(f"  ✓ {label} ({time.time() - t0:.1f}s)")
            prefetched.append(label)
        else:
            print(f"  ✗ {label} ({time.time() - t0:.1f}s)")
            failed.append(label)

    print(f"Done: {len(prefetched)} ok, {len(failed)} failed in {time.time() - total_start:.1f}s")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
