# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify/install transferred Claude binaries with Linux Python 3.8+ stdlib; no network access."""

import argparse
import fcntl
import gzip
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Hash a native binary without requiring Python 3.11's file_digest."""
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


@contextmanager
def _runtime_lock(runtime: Path):
    runtime.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(runtime) + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _cached_version(version: str, runtime: Path, expected: str) -> str:
    if sha256_file(runtime / "claude") != expected.lower():
        raise ValueError("Claude executable SHA-256 mismatch")
    return _version(runtime / "claude", version)


def check_runtime(version: str, runtime: Path, *, expected: str) -> str:
    """Check bytes and health before the host decides whether an upload is needed."""
    with _runtime_lock(runtime):
        return _cached_version(version, runtime, expected)


def install(version: str, runtime: Path, *, source: str = "", expected: str = "", compressed: bool = False) -> str:
    """Reuse a verified binary or atomically install the supplied bytes under a lock."""
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
        raise ValueError("Claude runtime requires an exact version")
    if not expected:
        raise ValueError("A transferred Claude binary requires a trusted SHA-256")
    with _runtime_lock(runtime):
        try:
            return _cached_version(version, runtime, expected)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        if not source:
            raise ValueError("Claude runtime is absent; provide a transferred source binary")
        runtime.mkdir(mode=0o700, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".claude-", dir=runtime)
        os.close(fd)
        temporary = Path(name)
        try:
            if compressed:
                with gzip.open(source, "rb") as input_file, temporary.open("wb") as output:
                    shutil.copyfileobj(input_file, output)
            else:
                shutil.copyfile(source, temporary)
            if sha256_file(temporary) != expected.lower():
                raise ValueError("Claude executable SHA-256 mismatch")
            temporary.chmod(0o700)
            actual = _version(temporary, version)
            temporary.replace(runtime / "claude")
            return actual
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--source", default="")
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--gzip", action="store_true", help="Source is a gzip-compressed native binary")
    args = parser.parse_args()
    if args.check_only:
        try:
            print(check_runtime(args.version, args.runtime, expected=args.sha256))
        except (OSError, ValueError, subprocess.SubprocessError):
            raise SystemExit(3)
    else:
        print(install(args.version, args.runtime, source=args.source, expected=args.sha256, compressed=args.gzip))
