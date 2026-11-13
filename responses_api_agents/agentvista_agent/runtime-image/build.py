# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the shared SVG evaluation SQSH using Enroot and explicit workspace inputs."""

import argparse
import hashlib
import json
import os
import pwd
import re
import shutil
import subprocess
from pathlib import Path


def digest(path: Path) -> str:
    """Hash a large image without reading it into memory."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    """Assemble and import-check a relocatable worker without mounting user homes."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-image", "python-distribution", "opencode", "uv", "source", "work-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--opencode-sha256", required=True)
    args = parser.parse_args()
    work, output = args.work_dir.resolve(), args.output.resolve()
    homes = {Path("/root"), Path.home(), Path(pwd.getpwuid(os.getuid()).pw_dir)}
    for path in (work, output):
        if any(path.is_relative_to(home) for home in homes):
            raise ValueError("Image build artifacts and caches must reside in the workspace")
    if output.exists():
        raise FileExistsError(output)
    work.mkdir(parents=True, exist_ok=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    pin = json.loads((Path(__file__).parents[1] / "runtime/source-pin.json").read_text())
    for name, expected in pin["files"].items():
        if digest(args.source / name) != expected:
            raise ValueError(f"Original benchmark source differs from pin: {name}")
    version = subprocess.check_output([str(args.opencode), "--version"], text=True).strip()
    if version != "1.17.11":
        raise ValueError("The shared image requires OpenCode CLI 1.17.11")
    if not re.fullmatch(r"[a-f0-9]{64}", args.opencode_sha256) or digest(args.opencode) != args.opencode_sha256:
        raise ValueError("OpenCode binary does not match the trusted input SHA-256")
    env = dict(os.environ, ENROOT_MOUNT_HOME="no", ENROOT_LOGIN_SHELL="no", NVIDIA_VISIBLE_DEVICES="void")
    for key in ("DATA", "RUNTIME", "CACHE", "CONFIG", "TEMP"):
        folder = work / ("enroot-" + key.lower())
        folder.mkdir()
        env["ENROOT_" + key + "_PATH"] = str(folder)
    # No cluster user-home or GPU hooks are needed to install the CPU worker.
    system = work / "enroot-system"
    system.mkdir()
    shutil.copytree("/etc/enroot/mounts.d", system / "mounts.d")
    env["ENROOT_SYSCONF_PATH"] = str(system)
    subprocess.run(["enroot", "create", "--name", "agentvista", str(args.base_image.resolve())], env=env, check=True)
    rootfs = work / "enroot-data/agentvista"
    source_target = rootfs / "opt/agentvista/source"
    for name in pin["files"]:
        target = source_target / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.source / name, target)
    shutil.copytree(args.python_distribution, rootfs / "opt/agentvista-python", symlinks=True)
    for src, relative in (
        (args.uv, "opt/agentvista/bin/uv"),
        (args.opencode, "opt/agentvista/bin/opencode"),
        (Path(__file__).parents[1] / "requirements-worker.txt", "opt/agentvista/requirements-worker.txt"),
    ):
        target = rootfs / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
    storage = work / "storage"
    variables = {}
    for key in ("UV_CACHE_DIR", "PIP_CACHE_DIR", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "TMPDIR"):
        folder = storage / key.lower()
        folder.mkdir(parents=True)
        variables[key] = str(folder)
    empty_home = work / "empty-home"
    empty_home.mkdir()
    install = rootfs / "opt/agentvista/install.sh"
    install.write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        "/opt/agentvista/bin/uv venv --python /opt/agentvista-python/bin/python3.13 /opt/agentvista-worker\n"
        "/opt/agentvista/bin/uv --no-config pip install --python /opt/agentvista-worker/bin/python "
        "--index-strategy unsafe-best-match --require-hashes -r /opt/agentvista/requirements-worker.txt\n"
        "/opt/agentvista/bin/uv pip freeze --python /opt/agentvista-worker/bin/python > /opt/agentvista/installed-packages.txt\n"
        "/opt/agentvista-worker/bin/python -c 'import qwen_vl_utils,torch,jupyter_client,ipykernel,mcp'\n"
        "PYTHONPATH=/opt/agentvista/source /opt/agentvista-worker/bin/python -c "
        "'import infer; from tools import list_tools; "
        'assert {"web_search","image_search","visit","code_interpreter"} <= set(list_tools())' + "'\n"
    )
    subprocess.run(
        [
            "enroot",
            "start",
            "--root",
            "--rw",
            "--mount",
            f"{work}:{work}:none:rbind,rw",
            *[
                argument
                for home in sorted(homes)
                for argument in ("--mount", f"{empty_home}:{home}:none:bind,ro,x-create=dir")
            ],
            "agentvista",
            "env",
            "PYTHONDONTWRITEBYTECODE=1",
            *[f"{key}={value}" for key, value in variables.items()],
            "bash",
            "/opt/agentvista/install.sh",
        ],
        env=env,
        check=True,
    )
    partial = output.with_suffix(output.suffix + ".partial")
    subprocess.run(["enroot", "export", "--output", str(partial), "agentvista"], env=env, check=True)
    partial.replace(output)
    output.chmod(0o644)
    receipt = {
        "image": str(output),
        "sha256": digest(output),
        "base_image_sha256": digest(args.base_image),
        "source_revision": pin["revision"],
        "opencode_version": version,
        "opencode_sha256": digest(args.opencode),
        "requirements_sha256": digest(Path(__file__).parents[1] / "requirements-worker.txt"),
        "installed_packages_sha256": digest(rootfs / "opt/agentvista/installed-packages.txt"),
        "installed_packages": (rootfs / "opt/agentvista/installed-packages.txt").read_text().splitlines(),
        "worker_python": "/opt/agentvista-worker/bin/python",
        "import_check_passed": True,
        "live_evaluation_validated": False,
    }
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
