# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load the reference converter itself as a hash-checked runtime dependency.

Protocol authority:
Revision: 35d35a7c38b356ff1549feda11572a4bceda4e49

No implementation from that repository is copied into this module. Its prompt,
schema, coordinate validator, observation renderer, and history planner remain
owned by the pinned dependency rather than a second local implementation.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Any

from .reference_lock import REFERENCE_LOCK


_LOAD_LOCK = threading.RLock()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Reference:
    """One immutable producer profile, used directly for live rendering."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.lock = copy.deepcopy(REFERENCE_LOCK)
        location = root or os.environ.get("OSWORLD_V4_PROMPT_FORMAT_REPO")
        if not location:
            raise ValueError("Set OSWORLD_V4_PROMPT_FORMAT_REPO to the pinned reference checkout")
        self.root = Path(location).resolve()
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.root, text=True).strip()
        if commit != self.lock["commit"]:
            raise ValueError(f"Reference commit differs: {commit} != {self.lock['commit']}")
        for name, identity in self.lock["runtime_files"].items():
            if file_sha256(self.root / name) != identity["sha256"]:
                raise ValueError(f"Reference file differs from the pinned source: {name}")
        with _LOAD_LOCK:
            self.coordinates = self._load("sft_coordinate_normalization", "sft_coordinate_normalization.py")
            self.tools = self._load("sft_tool_prompt", "sft_tool_prompt.py")
            self.converter = self._load("_nemotron_v4_reference_converter", "convert_procua_sft.py")
        self.profile = self.converter.resolve_agent_profile(self.lock["agent_profile"])
        self.profile_sha256 = self.converter.prompt_profile_sha256(
            self.profile.prompt_format_variant,
            wait_tool_contract=self.profile.wait_tool_contract,
        )
        if self.profile_sha256 != self.lock["prompt_profile_sha256"]:
            raise ValueError("Reference prompt-profile identity differs")

    def _load(self, name: str, relative: str) -> ModuleType:
        path = self.root / relative
        existing = sys.modules.get(name)
        if existing is not None:
            if Path(existing.__file__).resolve() != path:
                raise RuntimeError(f"A different {name} is already loaded; use an isolated v4 process")
            return existing
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        return module

    def system_prompt(self, password: str = "password") -> str:
        return self.converter.osworld_eval_system_prompt(password, self.profile.wait_tool_contract)

    def schemas(self) -> list[dict[str, Any]]:
        return self.tools.canonical_cua_tool_schemas(wait_tool_contract=self.profile.wait_tool_contract)

    def identity(self) -> dict[str, Any]:
        return {
            "repository": self.lock["repository"],
            "commit": self.lock["commit"],
            "agent_profile": self.profile.name,
            "prompt_format_variant": self.profile.prompt_format_variant,
            "execution_feedback_contract": self.profile.execution_feedback_contract,
            "wait_tool_contract": self.profile.wait_tool_contract,
            "prompt_profile_sha256": self.profile_sha256,
            "runtime_files": self.lock["runtime_files"],
        }
