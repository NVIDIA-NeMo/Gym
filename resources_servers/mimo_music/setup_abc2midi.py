# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Find or install the external abcMIDI renderer (never vendored into Gym)."""

import hashlib
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlopen


# abc2midi 4.88, also used for the checked-in smoke evidence.
REVISION = "6441b478418350b338589cbfb1b4397fed4a490e"
SHA256 = "fc14d52edceaae4dcdf5380e7660a5c94b0f7e19918953ea8a5ad05834bd33b1"
PREFIX = Path(__file__).resolve().parent / ".abcmidi"


def ensure_abc2midi() -> str:
    """Return a renderer path; install a pinned Linux build when absent."""
    binary = shutil.which("abc2midi")
    if binary:
        return binary
    target = PREFIX / "bin" / "abc2midi"
    if not target.is_file():
        if sys.platform == "darwin":
            subprocess.run(["brew", "install", "abcmidi"], check=True, timeout=600)
            binary = shutil.which("abc2midi")
            if not binary:
                raise RuntimeError("brew installed abcmidi but abc2midi is not on PATH")
            return binary
        if sys.platform != "linux":
            raise RuntimeError("Install abc2midi on PATH; automatic installation supports Linux and macOS")
        for command in ("make", "cc"):
            if not shutil.which(command):
                raise RuntimeError(f"Install {command} to build abcMIDI, or install abc2midi on PATH")
        target.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(f"https://codeload.github.com/sshlien/abcmidi/tar.gz/{REVISION}", timeout=60) as response:
            archive = response.read()
        if hashlib.sha256(archive).hexdigest() != SHA256:
            raise RuntimeError("abcMIDI archive checksum mismatch")
        with tempfile.TemporaryDirectory(dir=PREFIX) as directory:
            with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
                tar.extractall(directory, filter="data")
            source = Path(directory) / f"abcmidi-{REVISION}"
            subprocess.run(["make", "CC=cc", "abc2midi"], cwd=source, check=True, timeout=300)
            # Atomic replace avoids exposing partially copied binaries to other workers.
            built = Path(directory) / "abc2midi-built"
            shutil.copy2(source / "abc2midi", built)
            subprocess.run([str(built), "-ver"], check=True, capture_output=True, timeout=10)
            built.replace(target)
    os.environ["PATH"] = str(target.parent) + os.pathsep + os.environ.get("PATH", "")
    return str(target)


if __name__ == "__main__":
    print(ensure_abc2midi())
