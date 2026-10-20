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
"""Family B's hermetic test runtime: Python 3.12, the hash-pinned test tools (pytest, mypy, ...) in
task_types/B/_vendor, and the pinned upstream archives in task_types/B/repos.

Family B runs each unit's hidden tests as `python -I -S -B swb_b_pytest_runner.py`: no site-packages, only
`task_types/B/_vendor` on the path. The types were built and calibrated on Python 3.12 (jinja 3.1.4's own suite
fails on 3.13), so the tests run on a uv-managed 3.12 interpreter (SWB_B_PYTHON), not on the server's.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


B_DIR = Path(__file__).parent / "task_types" / "B"
REPOS_DIR = B_DIR / "repos"
VENDOR_DIR = B_DIR / "_vendor"
REQUIREMENTS = B_DIR / "vendor_requirements.txt"


def ensure_family_b_runtime() -> Path:
    os.environ.setdefault("SWB_B_PYTHON", _python312())
    ensure_family_b_repos()
    if (VENDOR_DIR / "pytest").is_dir() and (VENDOR_DIR / "mypy").is_dir():
        return VENDOR_DIR
    tmp = VENDOR_DIR.with_name("_vendor.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    args = ["--no-deps", "--require-hashes", "--target", str(tmp), "-r", str(REQUIREMENTS)]
    uv = shutil.which("uv")
    cmd = (
        [uv, "pip", "install", "--no-config", "--python", sys.executable, *args]
        if uv
        else [sys.executable, "-m", "pip", "install", *args]
    )
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"installing family B test tools failed: {result.stderr[-2000:]}")
    tmp.rename(VENDOR_DIR)  # atomic, so a concurrent start never sees a half-installed directory
    return VENDOR_DIR


def ensure_family_b_repos() -> None:
    # The pinned upstream archives are not committed. Download each once and check it against SOURCES.json.
    # World generation needs them too, so prepare calls this without the test runtime.
    for name, source in json.loads((REPOS_DIR / "SOURCES.json").read_text()).items():
        path = REPOS_DIR / name
        if path.exists():
            continue
        data = urllib.request.urlopen(source["url"], timeout=300).read()
        if hashlib.sha256(data).hexdigest() != source["sha256"]:
            raise RuntimeError(f"{name}: sha256 does not match SOURCES.json")
        tmp = path.with_name(name + ".tmp")
        tmp.write_bytes(data)
        tmp.rename(path)


def _python312() -> str:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to provide Python 3.12 for family B's hidden tests")
    found = subprocess.run([uv, "python", "find", "--no-config", "3.12"], capture_output=True, text=True)
    if found.returncode != 0:
        subprocess.run([uv, "python", "install", "--no-config", "3.12"], check=True, capture_output=True)
        found = subprocess.run(
            [uv, "python", "find", "--no-config", "3.12"], capture_output=True, text=True, check=True
        )
    return found.stdout.strip()
