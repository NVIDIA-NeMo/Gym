# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Install/reuse a versioned native Claude runtime with Linux Python 3.8+ stdlib."""

import argparse
import fcntl
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as binary:
        for chunk in iter(lambda: binary.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version(executable: Path, version: str) -> str:
    actual = subprocess.run(
        [str(executable), "--version"],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=30,
    ).stdout.strip()
    if not actual.startswith(version + " "):
        raise ValueError("Claude version mismatch: expected %s, got %s" % (version, actual))
    return actual


def install(version: str, runtime: Path, *, source: str = "", expected: str = "") -> str:
    """Reuse a healthy verified binary, or atomically install it under a per-runtime lock.

    Downloads use the release manifest for corruption detection unless the caller supplies
    a trusted SHA-256. Offline source binaries require that caller-supplied checksum.
    """
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
        raise ValueError("Claude runtime requires an exact version")
    if source and not expected:
        raise ValueError("A preinstalled Claude binary requires a trusted SHA-256")
    expected = expected.lower()
    runtime.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(runtime) + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        runtime.mkdir(mode=0o700, exist_ok=True)
        executable = runtime / "claude"
        receipt = runtime / "installation.json"
        try:
            recorded = json.loads(receipt.read_text())
            checksum = expected or recorded["sha256"]
            if recorded["version"] == version and _sha256(executable) == checksum:
                return _version(executable, version)
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
            pass
        fd, name = tempfile.mkstemp(prefix=".claude-", dir=runtime)
        os.close(fd)
        temporary = Path(name)
        try:
            if source:
                shutil.copyfile(source, temporary)
            else:
                architecture = {"x86_64": "x64", "aarch64": "arm64"}.get(platform.machine())
                if platform.system() != "Linux" or architecture is None:
                    raise RuntimeError("Claude sandbox installation requires Linux x64 or arm64")
                target = "linux-" + architecture
                if platform.libc_ver()[0] != "glibc" and Path("/etc/alpine-release").exists():
                    target += "-musl"
                base = "https://downloads.claude.ai/claude-code-releases/" + version
                if not expected:
                    with urllib.request.urlopen(base + "/manifest.json", timeout=60) as response:
                        manifest = json.load(response)
                    expected = manifest["platforms"][target]["checksum"].lower()
                with urllib.request.urlopen(base + "/" + target + "/claude", timeout=120) as response:
                    with temporary.open("wb") as output:
                        shutil.copyfileobj(response, output)
            checksum = _sha256(temporary)
            if checksum != expected:
                raise ValueError("Claude executable SHA-256 mismatch")
            temporary.chmod(0o700)
            actual = _version(temporary, version)
            temporary.replace(executable)
            receipt.write_text(json.dumps({"version": version, "sha256": checksum}))
            return actual
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--source", default="")
    parser.add_argument("--sha256", default="")
    args = parser.parse_args()
    print(install(args.version, args.runtime, source=args.source, expected=args.sha256))
