# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Standard-library installer for Linux task images without curl.

Executed inside the task sandbox, where only Python 3 is required. Cached
remote binaries remain the preferred installation path for large runs.
"""

import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlopen


def linux_target(machine, musl, cpu_flags):
    arch = {"x86_64": "x64", "aarch64": "arm64"}.get(machine)
    if arch is None:
        raise ValueError("Unsupported OpenCode architecture: " + machine)
    target = "linux-" + arch
    if arch == "x64" and "avx2" not in cpu_flags.split():
        target += "-baseline"
    if musl:
        target += "-musl"
    return target


def install(version):
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
        raise ValueError("Expected an exact OpenCode release version")
    if sys.platform != "linux":
        raise ValueError("This installer supports Linux sandboxes")
    musl = Path("/etc/alpine-release").exists() or bool(list(Path("/lib").glob("ld-musl-*.so*")))
    flags = Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else ""
    target = linux_target(platform.machine(), musl, flags)
    url = f"https://github.com/anomalyco/opencode/releases/download/v{version}/opencode-{target}.tar.gz"
    destination = Path.home() / ".opencode" / "bin"
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".install-", dir=destination) as directory:
        archive = Path(directory) / "opencode.tar.gz"
        with urlopen(url, timeout=120) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
        candidate = Path(directory) / "opencode"
        with tarfile.open(archive, "r:gz") as tar:
            members = [member for member in tar.getmembers() if member.name in ("opencode", "./opencode")]
            if len(members) != 1 or not members[0].isfile():
                raise ValueError("Release archive must contain one regular opencode binary")
            with tar.extractfile(members[0]) as source, candidate.open("wb") as output:
                shutil.copyfileobj(source, output)
        candidate.chmod(0o755)
        actual = subprocess.check_output([str(candidate), "--version"], text=True, timeout=30).strip()
        if actual != version:
            raise ValueError(f"OpenCode version mismatch: expected {version}, got {actual}")
        os.replace(candidate, destination / "opencode")
    print(f"Installed OpenCode {version} ({target})", flush=True)


if __name__ == "__main__":
    install(sys.argv[1])
