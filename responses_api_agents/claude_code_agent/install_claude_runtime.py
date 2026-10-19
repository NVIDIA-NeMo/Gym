# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Install a pinned native Claude executable inside a sandbox, using stdlib Python."""

import hashlib
import json
import platform
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


def install(version: str, destination: Path, source: str, expected: str) -> None:
    """Copy a supplied binary or download the native release and verify its checksum."""
    if source:
        shutil.copyfile(source, destination)
    else:
        architecture = {"x86_64": "x64", "aarch64": "arm64"}.get(platform.machine())
        if platform.system() != "Linux" or architecture is None:
            raise RuntimeError("Claude sandbox installation requires Linux x64 or arm64")
        target = "linux-" + architecture
        if platform.libc_ver()[0] != "glibc" and Path("/etc/alpine-release").exists():
            target += "-musl"
        base = "https://downloads.claude.ai/claude-code-releases/" + version
        with urllib.request.urlopen(base + "/manifest.json", timeout=60) as response:
            manifest = json.load(response)
        expected = expected or manifest["platforms"][target]["checksum"]
        with urllib.request.urlopen(base + "/" + target + "/claude", timeout=120) as response:
            with destination.open("wb") as output:
                shutil.copyfileobj(response, output)
    if expected:
        with destination.open("rb") as binary:
            actual = hashlib.file_digest(binary, "sha256").hexdigest()
        if actual != expected:
            destination.unlink()
            raise ValueError("Claude executable SHA-256 mismatch")
    destination.chmod(0o700)
    subprocess.run([str(destination), "--version"], check=True, timeout=30)


if __name__ == "__main__":
    install(sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4])
