#!/usr/bin/env python3
"""Install and smoke-test the RDKit server using Gym's production setup path."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from omegaconf import OmegaConf


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    gym_dir = Path(os.environ["GYM_DIR"]).resolve()
    scratch_base = Path(os.environ.get("SLURM_TMPDIR", "/tmp"))
    scratch_dir = scratch_base / f"rdkit-resource-preflight-{os.environ.get('SLURM_JOB_ID', 'local')}"
    venv_root = scratch_dir / "venvs"
    uv_cache_dir = scratch_dir / "uv-cache"
    uv_cache_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(gym_dir))
    from nemo_gym.cli_setup_command import setup_env_command
    from nemo_gym.global_config import (
        HEAD_SERVER_DEPS_KEY_NAME,
        PIP_INSTALL_VERBOSE_KEY_NAME,
        PYTHON_VERSION_KEY_NAME,
        SKIP_VENV_IF_PRESENT_KEY_NAME,
        UV_PIP_SET_PYTHON_KEY_NAME,
        UV_VENV_DIR_KEY_NAME,
    )

    setup_config = OmegaConf.create(
        {
            HEAD_SERVER_DEPS_KEY_NAME: [],
            PIP_INSTALL_VERBOSE_KEY_NAME: False,
            PYTHON_VERSION_KEY_NAME: f"{sys.version_info.major}.{sys.version_info.minor}",
            SKIP_VENV_IF_PRESENT_KEY_NAME: False,
            UV_PIP_SET_PYTHON_KEY_NAME: True,
            UV_VENV_DIR_KEY_NAME: str(venv_root),
        }
    )
    server_dir = gym_dir / "resources_servers/rdkit_chemistry"
    command = setup_env_command(server_dir, setup_config, "rdkit_chemistry")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(gym_dir)
    env["UV_CACHE_DIR"] = str(uv_cache_dir)
    subprocess.run(["/bin/bash", "-lc", command], check=True, env=env)

    server_python = venv_root / "resources_servers/rdkit_chemistry/.venv/bin/python"
    smoke_code = """
from resources_servers.rdkit_chemistry.app import compute_reward, extract_predicted_value
from responses_api_agents.simple_agent.app import SimpleAgent

assert extract_predicted_value(
    "Final Answer = 42", "count", answer_format="fmt_28"
) == 42.0
assert compute_reward(42.0, 42.0, property_type="count") == 1.0
assert SimpleAgent is not None
print("RDKit resource-server container preflight passed")
"""
    subprocess.run([str(server_python), "-c", smoke_code], check=True, env=env)

    container_path = gym_dir / "cluster/rdkit_no_tool_grpo/sqsh/nemo-rl-v0.6.0.sqsh"
    container_stat = container_path.stat()
    gym_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=gym_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    resource_files = {
        filename: sha256(server_dir / filename)
        for filename in ("app.py", "requirements.txt", "sandbox_launcher.py")
    }
    stamp = {
        "schema_version": 1,
        "completed_at": datetime.now(UTC).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gym_commit": gym_commit,
        "resource_files": resource_files,
        "container": {
            "path": str(container_path),
            "size": container_stat.st_size,
            "mtime_ns": container_stat.st_mtime_ns,
        },
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }
    stamp_path = gym_dir / "cluster/rdkit_no_tool_grpo/preflight/resource_server_container.json"
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_stamp = stamp_path.with_suffix(f".{os.environ.get('SLURM_JOB_ID', 'local')}.tmp")
    temporary_stamp.write_text(json.dumps(stamp, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_stamp, stamp_path)
    print(f"Wrote container preflight stamp: {stamp_path}")


if __name__ == "__main__":
    main()
