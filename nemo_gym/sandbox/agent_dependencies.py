# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provision the current Gym checkout and one harness in a sandbox-local venv."""

import asyncio
import shlex
import subprocess
import tarfile
from pathlib import Path

from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.agent_runtime_config import AgentRuntimeConfig


def build_source_archive(runtime: AgentRuntimeConfig, harness: str, destination: Path) -> str:
    """Snapshot selected source files, including edits, without copying the host environment."""
    deps = runtime.dependencies
    root = Path(deps.source_root).resolve() if deps.source_root else Path(__file__).resolve().parents[2]
    if not (root / "pyproject.toml").is_file():
        raise ValueError("Automatic dependencies need a Gym checkout: set runtime.dependencies.source_root")
    module = harness.split(":", 1)[0]
    if deps.harness_path:
        harness_path = Path(deps.harness_path)
    elif module.startswith("responses_api_agents."):
        harness_path = Path(*module.split(".")[:2])
    else:
        raise ValueError("Custom harnesses must set runtime.dependencies.harness_path relative to source_root")
    if harness_path.is_absolute() or ".." in harness_path.parts or not harness_path.parts:
        raise ValueError("runtime.dependencies.harness_path must be a relative directory inside source_root")
    harness_dir = root / harness_path
    if not harness_dir.resolve().is_relative_to(root):
        raise ValueError("Harness source must stay inside source_root")
    manifests = [name for name in ("requirements.txt", "pyproject.toml") if (harness_dir / name).is_file()]
    if len(manifests) != 1:
        raise ValueError("Harness must declare exactly one of requirements.txt or pyproject.toml")
    selectors = ["pyproject.toml", "README.md", "LICENSE", "MANIFEST.in", "nemo_gym", str(harness_path)]
    for parent in harness_path.parents:
        if parent != Path("."):
            selectors.append(str(parent / "__init__.py"))
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", *selectors],
        capture_output=True,
        check=True,
    )
    excluded = {"tests", "data", "outputs", "results", "cache", "__pycache__", ".venv"}
    with tarfile.open(destination, "w:gz") as archive:
        for name in sorted(set(result.stdout.decode(errors="replace").split("\0")) - {""}):
            path = Path(name)
            if any(part in excluded or part.startswith(".") for part in path.parts):
                continue
            source = root / path
            if source.is_symlink() or not source.is_file():
                continue
            archive.add(source, arcname=path.as_posix(), recursive=False)
    return harness_path.as_posix()


async def install_agent_dependencies(
    sandbox: AsyncSandbox, runtime: AgentRuntimeConfig, harness: str, remote_dir: str, local_dir: Path
) -> tuple[str, dict[str, str]]:
    """Return the installed Python and worker environment without modifying task dependencies."""
    archive = local_dir / "source.tar.gz"
    harness_path = await asyncio.to_thread(build_source_archive, runtime, harness, archive)
    await sandbox.upload(archive, f"{remote_dir}/source.tar.gz")
    quote = shlex.quote
    deps = runtime.dependencies
    source = f"{remote_dir}/source"
    venv = f"{remote_dir}/venv"
    python = f"{venv}/bin/python"
    env = runtime.env | {
        "UV_CACHE_DIR": f"{remote_dir}/uv-cache",
        "UV_PYTHON_INSTALL_DIR": f"{remote_dir}/python",
        "UV_UNMANAGED_INSTALL": f"{remote_dir}/uv",
    }
    command = f"""set -eu
if command -v uv >/dev/null 2>&1; then
    uv_bin=$(command -v uv)
else
    curl -fLsS --retry 3 https://astral.sh/uv/{deps.uv_version}/install.sh -o {quote(remote_dir + "/uv-install.sh")}
    sh {quote(remote_dir + "/uv-install.sh")}
    uv_bin={quote(remote_dir + "/uv/uv")}
fi
mkdir -p {quote(source + "/cache")}
tar -xzf {quote(remote_dir + "/source.tar.gz")} -C {quote(source)}
"$uv_bin" venv --seed --python {quote(deps.python_version)} {quote(venv)}
cd {quote(source + "/" + harness_path)}
if [ -f requirements.txt ]; then
    if [ -f overrides.txt ]; then
        "$uv_bin" pip install --python {quote(python)} -e {quote(source)} -r requirements.txt --override overrides.txt
    else
        "$uv_bin" pip install --python {quote(python)} -e {quote(source)} -r requirements.txt
    fi
else
    "$uv_bin" pip install --python {quote(python)} -e {quote(source)} -e .
fi
"""
    result = await sandbox.exec(command, env=env, timeout_s=runtime.setup_timeout_s)
    if result.error_type or result.return_code != 0:
        raise RuntimeError(
            f"Agent dependency installation failed (exit={result.return_code}, error={result.error_type}): "
            f"{(result.stderr or '')[-2000:]}"
        )
    # Keep both task cwd and its Python environment intact; only the worker uses this venv.
    return python, env
