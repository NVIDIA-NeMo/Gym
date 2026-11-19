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
"""Build EternaFold, the folding engine every rna_design scorer calls, and point arnie at it."""

import fcntl
import logging
import os
import shutil
import subprocess
from pathlib import Path


logger = logging.getLogger(__name__)

_REPO = "https://github.com/eternagame/EternaFold.git"
_DEFAULT_DIR = Path(__file__).resolve().parent / ".eternafold"
# Upstream's Makefile adds x86-only SSE flags that arm64 compilers reject, so the whole line is replaced.
# Every -D is kept: they select which model the binary is built as.
_CXXFLAGS = "-std=c++11 -O3 -DEVIDENCE_SR -DEVIDENCE_PARS -DNDEBUG -pipe -Wno-everything"


def ensure_eternafold(base_dir: Path | str | None = None) -> Path:
    """Build EternaFold once, write the arnie config, set ARNIEFILE, and return the config path. Idempotent.

    An ARNIEFILE already set in the environment is used as is.
    """
    if os.environ.get("ARNIEFILE") and Path(os.environ["ARNIEFILE"]).is_file():
        return Path(os.environ["ARNIEFILE"])

    base_dir = Path(base_dir) if base_dir else _DEFAULT_DIR
    base_dir.mkdir(parents=True, exist_ok=True)
    src = base_dir / "EternaFold" / "src"
    binary = src / "contrafold"
    # Servers starting together from one checkout share a single build: the second waits for the first.
    with open(base_dir / ".build.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not binary.is_file():
            for tool in ("git", "make"):
                if shutil.which(tool) is None:
                    raise RuntimeError(f"building EternaFold needs `{tool}` on PATH")
            if not src.is_dir():
                logger.info("Cloning EternaFold into %s", base_dir)
                subprocess.run(["git", "clone", "--depth=1", _REPO, str(base_dir / "EternaFold")], check=True)
            logger.info("Building EternaFold")
            subprocess.run(["make", "-C", str(src), f"-j{os.cpu_count() or 2}", f"CXXFLAGS={_CXXFLAGS}"], check=True)
            if not binary.is_file():
                raise RuntimeError(f"EternaFold build finished but {binary} is missing")

    params = base_dir / "EternaFold" / "parameters" / "EternaFoldParams.v1"
    tmp = base_dir / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    arniefile = base_dir / "arniefile.txt"
    arniefile.write_text(f"eternafold: {src}\neternafold_params: {params}\nTMP: {tmp}\n")
    os.environ["ARNIEFILE"] = str(arniefile)
    return arniefile
