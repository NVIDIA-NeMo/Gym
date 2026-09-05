# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Built-in NeMo Gym agent harness implementations and legacy import aliases."""

import importlib
import importlib.abc
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from nemo_gym import component_search_roots


# Keep this regular package extensible. A plain ``__init__.py`` would make Python ignore harness
# directories in higher-priority plugin roots even though Gym's registry discovers them there.
# Legacy directories are fallback package locations so canonical imports also work for old plugins.
__path__ = [
    str(harness_dir)
    for root in component_search_roots(sys_path=[Path(entry) for entry in sys.path if entry])
    for subdir in ("harnesses", "responses_api_agents")
    if (harness_dir := root / subdir).is_dir()
]


class _LegacyHarnessAliasLoader(importlib.abc.Loader):
    """Load a legacy-qualified module as the identical canonical module object."""

    def __init__(self, canonical_name: str) -> None:
        self.canonical_name = canonical_name
        self._canonical_metadata = None

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        module = importlib.import_module(self.canonical_name)
        self._canonical_metadata = (module.__spec__, module.__loader__, module.__package__)
        return module

    def exec_module(self, module: ModuleType) -> None:
        # Import machinery temporarily writes the legacy spec onto the shared module. Restore the
        # canonical metadata so relative imports and reload() continue to use the public package name.
        module.__spec__, module.__loader__, module.__package__ = self._canonical_metadata


class _LegacyHarnessAliasFinder(importlib.abc.MetaPathFinder):
    """Map ``responses_api_agents.*`` imports onto ``harnesses.*`` without duplicate modules."""

    legacy_prefix = "responses_api_agents."

    def find_spec(self, fullname: str, path=None, target=None):
        if not fullname.startswith(self.legacy_prefix):
            return None
        canonical_name = f"harnesses.{fullname.removeprefix(self.legacy_prefix)}"
        canonical_spec = importlib.util.find_spec(canonical_name)
        if canonical_spec is None:
            return None
        return importlib.util.spec_from_loader(
            fullname,
            _LegacyHarnessAliasLoader(canonical_name),
            is_package=canonical_spec.submodule_search_locations is not None,
        )


if not any(isinstance(finder, _LegacyHarnessAliasFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _LegacyHarnessAliasFinder())

# If the canonical package is imported first, make the package-level legacy name an exact alias too.
sys.modules.setdefault("responses_api_agents", sys.modules[__name__])
