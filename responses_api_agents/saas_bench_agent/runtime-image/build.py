# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build an SVG SaaS runtime from a staged Gym image without Docker/root access."""

import argparse
import hashlib
import json
import os
import pwd
import re
import shutil
import subprocess
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-image", required=True, type=Path)
    parser.add_argument("--python-distribution", required=True, type=Path)
    parser.add_argument("--opencode", required=True, type=Path)
    parser.add_argument("--uv", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true", help="Reuse this build directory after a failed installation")
    args = parser.parse_args()
    work = args.work_dir.resolve()
    try:
        home = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
    except KeyError:
        home = Path.home().resolve()
    if any(work == p or work.is_relative_to(p) for p in (Path("/root"), home)):
        raise ValueError("Build storage must be outside forbidden homes")
    work.mkdir(parents=True, exist_ok=args.resume)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(
        ENROOT_DATA_PATH=str(work / "enroot-data"),
        ENROOT_RUNTIME_PATH=str(work / "enroot-runtime"),
        ENROOT_CACHE_PATH=str(work / "enroot-cache"),
        ENROOT_CONFIG_PATH=str(work / "enroot-config"),
        ENROOT_TEMP_PATH=str(work / "tmp"),
        ENROOT_MOUNT_HOME="no",
        ENROOT_LOGIN_SHELL="no",
        NVIDIA_VISIBLE_DEVICES="void",
        UV_CACHE_DIR=str(work / "uv"),
        XDG_CACHE_HOME=str(work / "cache"),
        XDG_CONFIG_HOME=str(work / "config"),
        XDG_DATA_HOME=str(work / "data"),
        XDG_STATE_HOME=str(work / "state"),
        TMPDIR=str(work / "tmp"),
    )
    for key, value in env.items():
        if (
            key.startswith("ENROOT_")
            and key.endswith("PATH")
            or key
            in ("TMPDIR", "UV_CACHE_DIR", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME")
        ):
            Path(value).mkdir(parents=True, exist_ok=True)
    name = "saas_eval_runtime"
    if not args.resume:
        subprocess.run(["enroot", "create", "--name", name, str(args.base_image.resolve())], env=env, check=True)
    root = work / "enroot-data" / name
    if not (root / "etc/os-release").is_file():
        raise ValueError("Missing extracted base rootfs")
    system = work / "build-system"
    system.mkdir(exist_ok=True)
    shutil.copytree("/etc/enroot/mounts.d", system / "mounts.d", dirs_exist_ok=True)
    # Build without cluster hooks that populate a user home or require GPUs.
    env["ENROOT_SYSCONF_PATH"] = str(system)
    copied = {}

    def copy_file(source, target):
        source = Path(source)
        destination = root / str(target).lstrip("/")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.parent.resolve().is_relative_to(root.resolve()):
            raise ValueError("Image symlink escapes build root")
        if destination.is_symlink():
            destination.unlink()
        shutil.copy2(source, destination)
        copied[str(target)] = sha256(source)

    # The staged Gym image does not include the rootless application launcher.
    binaries = [Path(shutil.which("socat") or "MISSING_SOCAT"), *sorted(Path("/usr/bin").glob("enroot*"))]
    for binary in binaries:
        if not binary.is_file():
            raise FileNotFoundError(binary)
        copy_file(binary, str(binary))
        result = subprocess.run(["ldd", str(binary)], capture_output=True, text=True)
        for library in re.findall(r"(?:=>\s+)?(/[^\s]+)", result.stdout):
            src = Path(library)
            destination = root / str(src).lstrip("/")
            if src.is_file() and not destination.exists():
                copy_file(src, str(src))
    shutil.copytree("/usr/lib/enroot", root / "usr/lib/enroot", dirs_exist_ok=True)
    shutil.copytree("/etc/enroot", root / "etc/enroot", dirs_exist_ok=True)
    # A Python distribution, not a copied virtualenv with host-absolute links.
    shutil.copytree(args.python_distribution, root / "opt/saas-python", symlinks=True, dirs_exist_ok=True)
    copy_file(args.uv, "/usr/local/bin/uv")
    copy_file(args.opencode, "/opt/saas/bin/opencode")
    requirement = Path(__file__).with_name("requirements-worker.txt")
    copy_file(requirement, "/opt/saas/requirements-worker.txt")
    install = """set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
uv venv --allow-existing --python /opt/saas-python/bin/python3.13 /opt/saas-worker
uv pip install --python /opt/saas-worker/bin/python --requirement /opt/saas/requirements-worker.txt
/opt/saas/bin/opencode --version
/opt/saas-worker/bin/python -c 'import browser_use,playwright,fitz,mcp; print("SAAS_WORKER_IMPORTS_PASSED")'
"""
    copy_file_path = root / "opt/saas/install.sh"
    copy_file_path.write_text(install)
    subprocess.run(
        [
            "enroot",
            "start",
            "--root",
            "--rw",
            "--mount",
            str(work) + ":" + str(work),
            name,
            "bash",
            "/opt/saas/install.sh",
        ],
        env=env,
        check=True,
    )
    temporary = output.with_suffix(output.suffix + ".partial")
    subprocess.run(["enroot", "export", "--output", str(temporary), name], env=env, check=True)
    temporary.replace(output)
    output.chmod(0o644)
    receipt = {
        "output": str(output),
        "sha256": sha256(output),
        "base_image": str(args.base_image.resolve()),
        "base_image_sha256": sha256(args.base_image),
        "worker_requirements_sha256": sha256(requirement),
        "copied_files": copied,
        "opencode_version": "1.17.11",
        "worker_python": "/opt/saas-worker/bin/python",
        "completed_at": time.time(),
    }
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(receipt, indent=2) + "\n")
    output.with_suffix(output.suffix + ".json").chmod(0o644)
    print("SAAS_RUNTIME_IMAGE_READY", str(output), flush=True)


if __name__ == "__main__":
    main()
