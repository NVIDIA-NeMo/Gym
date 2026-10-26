# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Install the pinned Copilot CLI binary inside a task sandbox, outside the task repository.

Runs with the sandbox's own ``python3 -I`` (3.8+, standard library only). The npm platform package
carries a standalone binary; it is verified against the registry's sha512 integrity before use. The
binary unpacks its bundled package into ``$HOME/.cache``, so the installer runs it once with a private
HOME and records that directory for ``COPILOT_CLI_DIST_DIR``; the task's HOME is never written.
"""

import base64
import fcntl
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import uuid


MIN_GLIBC = (2, 28)


def package_name() -> str:
    """Return the npm platform package for this sandbox, or exit with the unsupported reason."""
    if sys.platform != "linux":
        sys.exit(f"Copilot sandbox sessions support Linux only, not {sys.platform}")
    arch = {"x86_64": "x64", "aarch64": "arm64", "arm64": "arm64"}.get(platform.machine())
    if arch is None:
        sys.exit(f"Copilot supports x86_64 and aarch64 sandboxes, not {platform.machine()}")
    try:
        glibc = os.confstr("CS_GNU_LIBC_VERSION")
    except (ValueError, OSError):
        glibc = None
    if glibc:
        version = tuple(int(part) for part in glibc.split()[-1].split(".")[:2])
        if version < MIN_GLIBC:
            sys.exit(f"Copilot requires glibc {'.'.join(map(str, MIN_GLIBC))} or newer; this sandbox has {glibc}")
        return f"copilot-linux-{arch}"
    if any(name.startswith("ld-musl-") for name in os.listdir("/lib")):
        return f"copilot-linuxmusl-{arch}"
    sys.exit("Copilot requires glibc or musl; could not identify the sandbox libc")


def fetch(url: str) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 - registry URL from the adapter
            return response.read()
    except Exception as error:
        hint = " (install ca-certificates in the task image)" if "CERTIFICATE" in str(error).upper() else ""
        sys.exit(f"Copilot download failed for {url}: {error}{hint}")


def install(runtime: str, version: str, registry: str) -> dict:
    record_path = os.path.join(runtime, "runtime.json")
    if os.path.exists(runtime):
        # A cancelled install may leave a published directory without a ready marker,
        # or a truncated record. Neither can be reused; the caller holds the install lock.
        try:
            with open(record_path) as stream:
                record = json.load(stream)
            if os.path.isfile(os.path.join(runtime, "ready")) and isinstance(record, dict) and check(record, version):
                return record
        except (OSError, ValueError):
            pass
        shutil.rmtree(runtime)
    package = package_name()
    metadata = json.loads(fetch(f"{registry.rstrip('/')}/@github/{package}/{version}"))
    tarball, integrity = metadata["dist"]["tarball"], metadata["dist"]["integrity"]
    if not integrity.startswith("sha512-"):
        sys.exit(f"Unsupported integrity format for {package}@{version}: {integrity}")
    data = fetch(tarball)
    digest = "sha512-" + base64.b64encode(hashlib.sha512(data).digest()).decode()
    if digest != integrity:
        sys.exit(f"Integrity mismatch for {package}@{version}: expected {integrity}, got {digest}")
    staging = f"{runtime}.partial-{uuid.uuid4().hex}"
    os.makedirs(os.path.join(staging, "home"))
    try:
        archive = os.path.join(staging, "package.tgz")
        with open(archive, "wb") as stream:
            stream.write(data)
        binary = os.path.join(staging, "copilot")
        with tarfile.open(archive, "r:gz") as tar:
            member = tar.getmember("package/copilot")
            if not member.isfile():
                sys.exit(f"{package}@{version} has no copilot executable")
            with tar.extractfile(member) as source, open(binary, "wb") as target:
                shutil.copyfileobj(source, target)
        os.chmod(binary, 0o755)
        os.remove(archive)
        # The first run unpacks the bundled package into the private HOME.
        probe = subprocess.run(
            [binary, "--version"],
            env={"HOME": os.path.join(staging, "home"), "PATH": "/usr/bin:/bin", "COPILOT_AUTO_UPDATE": "false"},
            capture_output=True,
            text=True,
            errors="replace",
            timeout=300,
        )
        if probe.returncode != 0 or version not in probe.stdout:
            sys.exit(f"Copilot health check failed (exit {probe.returncode}): {probe.stdout}\n{probe.stderr}")
        packages = os.path.join(staging, "home", ".cache", "copilot", "pkg")
        dist = [
            os.path.join(packages, name, version)
            for name in sorted(os.listdir(packages))
            if os.path.isdir(os.path.join(packages, name, version))
        ]
        if len(dist) != 1:
            sys.exit(f"Copilot did not unpack exactly one {version} package under {packages}: {dist}")
        os.rename(staging, runtime)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    record = {
        "binary": os.path.join(runtime, "copilot"),
        "dist_dir": dist[0].replace(staging, runtime, 1),
        "package": package,
        "version": version,
        "integrity": integrity,
    }
    with open(record_path, "w") as stream:
        json.dump(record, stream)
    if not check(record, version):
        sys.exit("Copilot runtime failed its health check after installation")
    with open(os.path.join(runtime, "ready"), "w"):
        pass
    return record


def check(record: dict, version: str) -> bool:
    """A runtime is reusable only when its binary reports the pinned version with its unpacked package."""
    if record.get("version") != version or not os.path.isdir(record.get("dist_dir", "")):
        return False
    try:
        probe = subprocess.run(
            [record["binary"], "--version"],
            env={
                "HOME": os.path.join(os.path.dirname(record["binary"]), "home"),
                "PATH": "/usr/bin:/bin",
                "COPILOT_AUTO_UPDATE": "false",
                "COPILOT_CLI_DIST_DIR": record["dist_dir"],
            },
            capture_output=True,
            text=True,
            errors="replace",
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0 and version in probe.stdout


def main() -> None:
    runtime, version, registry = sys.argv[1:4]
    os.makedirs(os.path.dirname(runtime), exist_ok=True)
    # Concurrent seeds in one sandbox share one installation.
    with open(f"{runtime}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        record = install(runtime, version, registry)
    print(json.dumps(record))


if __name__ == "__main__":
    main()
