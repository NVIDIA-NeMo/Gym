# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""run any gym responses() agent inside a sandbox, harness chosen by config.

load_runner_source returns the script injected into the container (``sandbox_agent_runner.py``),
which imports the configured agent and calls responses() so the agent edits files with its own
tools. harvest pulls produced files back out by glob.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


_RUNNER_SOURCE_PATH = Path(__file__).with_name("sandbox_agent_runner.py")


def agent_key(agent_server_module: str) -> str:
    """responses_api_agents.hermes_agent.app maps to hermes_agent, the deps-script key."""
    parts = agent_server_module.split(".")
    return parts[-2] if len(parts) >= 2 else agent_server_module


def load_runner_source() -> str:
    """return the injected runner script verbatim.

    the runner is a plain module (``sandbox_agent_runner.py``) rather than a templated string, so
    it is diffable and syntax-checked with the rest of the package. it reads the agent
    module/class from ``NV_AGENT_*`` env vars set by the caller, so no per-agent rendering is needed.
    """
    return _RUNNER_SOURCE_PATH.read_text(encoding="utf-8")


def deps_recipe_key(*paths: Path) -> str:
    """stable hash of the deps-install inputs so a prefix is reused until its recipe changes."""
    blob = b"".join(p.read_bytes() for p in paths if p.exists()) or b"no-script"
    return hashlib.sha256(blob).hexdigest()


def harvest(workdir: Path, globs: list[str], *, seeded: dict[str, str] | None = None) -> dict[str, str]:
    """collect files the agent produced under workdir that match any glob.

    returns {relative_posix_path: text_content}. files identical to a seeded input are skipped
    so unchanged context files are not reported as produced. unreadable or binary files are
    skipped. point it at e.g. ["rtl/**/*.sv", "rtl/**/*.v"].
    """
    workdir = Path(workdir)
    seeded = seeded or {}
    produced: dict[str, str] = {}
    for pattern in globs:
        for fpath in sorted(workdir.glob(pattern)):
            if not fpath.is_file():
                continue
            rel = fpath.relative_to(workdir).as_posix()
            if rel in produced:
                continue
            try:
                content = fpath.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if seeded.get(rel) == content:
                continue  # unchanged context file
            produced[rel] = content
    return produced
