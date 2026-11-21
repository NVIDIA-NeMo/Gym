# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve Hermes's generic tools package within its installed distribution."""

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType


class _HermesToolsFinder(importlib.abc.MetaPathFinder):
    def __init__(self, root: Path) -> None:
        self._root = str(root)

    def find_spec(
        self, fullname: str, path: Sequence[str] | None = None, target: ModuleType | None = None
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname == "tools":
            return importlib.machinery.PathFinder.find_spec(fullname, [self._root], target)
        return None


def import_model_tools() -> None:
    """Import Hermes tools while preserving the caller's other dependency paths.

    Hermes imports ``tools.registry`` through a top-level package name also used
    by training repositories. A scoped finder selects its own package during
    initialization; subsequent imports use that package's normal search path.
    """
    if "model_tools" in sys.modules:
        importlib.import_module("model_tools")
        return
    spec = importlib.util.find_spec("model_tools")
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError("Hermes model_tools is not installed")
    root = Path(spec.origin).resolve().parent
    expected = root / "tools" / "__init__.py"
    if not expected.is_file() or not (root / "tools" / "registry.py").is_file():
        raise ModuleNotFoundError(f"Hermes installation has no tools.registry beside {spec.origin}")
    existing = sys.modules.get("tools")
    if existing is not None:
        loaded = existing.__spec__
        if loaded is None or loaded.origin is None or Path(loaded.origin).resolve() != expected:
            raise ImportError("Another tools package is already loaded; run the Hermes server in its own process")
    finder = _HermesToolsFinder(root)
    sys.meta_path.insert(0, finder)
    try:
        importlib.import_module("model_tools")
    finally:
        sys.meta_path.remove(finder)
