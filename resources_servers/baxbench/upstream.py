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
"""Fetch the pinned upstream BaxBench source and import its environments and scenarios.

Prompts, test lists and framework images all come from upstream at ``UPSTREAM_SHA``; nothing is vendored.
The HF dataset (LogicStar/BaxBench) is older than this commit, so it is not used.
"""

import importlib
import importlib.util
import io
import re
import sys
import tarfile
import types
import urllib.request
from pathlib import Path

from resources_servers.baxbench import docker_shim


UPSTREAM_REPO = "https://github.com/logic-star-ai/baxbench"
# main after v1.0.0 (a67abd94), with upstream's test fixes; the leaderboard's latest harness.
UPSTREAM_SHA = "de885cd93d561682e203a5a0d57c33b08aac6f5e"
_TARBALL_URL = f"https://codeload.github.com/logic-star-ai/baxbench/tar.gz/{UPSTREAM_SHA}"
# Imported by scenario modules but used only while a test runs, which never happens on this side.
_TEST_ONLY_MODULES = ("imageio", "pdfplumber")


def image_slug(env_id: str) -> str:
    """Framework image slug: ``Python-Flask`` -> ``python-flask``, ``Go-net/http`` -> ``go-net-http``."""
    return re.sub(r"[^a-z0-9]+", "-", env_id.lower()).strip("-")


def fetch_upstream(cache_dir: Path) -> Path:
    """Download upstream at ``UPSTREAM_SHA`` once and return its ``src`` directory."""
    root = cache_dir / f"baxbench-{UPSTREAM_SHA}"
    if not (root / "src" / "scenarios" / "__init__.py").exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(_TARBALL_URL, timeout=120) as response:
            payload = response.read()
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
            tar.extractall(cache_dir, filter="data")
    return root / "src"


def import_upstream(src_dir: Path) -> tuple[types.ModuleType, types.ModuleType]:
    """Return upstream's ``env`` and ``scenarios`` packages (``all_envs``, ``all_scenarios``)."""
    docker_shim.install()
    for name in _TEST_ONLY_MODULES:
        if importlib.util.find_spec(name) is None:
            sys.modules.setdefault(name, types.ModuleType(name))
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return importlib.import_module("env"), importlib.import_module("scenarios")
