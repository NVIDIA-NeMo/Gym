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
"""Shared fixtures for the telemetry unit tests."""

import importlib
import importlib.util
import os
import sys
from contextlib import contextmanager
from types import ModuleType

import pytest

from nemo_gym.telemetry import metrics as telemetry_metrics
from nemo_gym.telemetry import setup as telemetry_setup


_LENS_PREFIX = "nemo.lens"


#: True when the telemetry extra is installed. Tests that exercise the *enabled* path need
#: it; `test_fallbacks.py` deliberately does not carry this marker, because covering the
#: nemo-lens-absent path is the whole point of that file.
def _lens_installed() -> bool:
    """True when nemo-lens is importable.

    `find_spec("nemo.lens")` imports the parent `nemo` package to find it, so it *raises*
    rather than returning None when nothing provides `nemo` at all — which is exactly the
    no-telemetry-extra case this predicate exists to detect.
    """
    try:
        return importlib.util.find_spec("nemo.lens") is not None
    except (ImportError, ValueError):
        return False


LENS_INSTALLED = _lens_installed()

requires_lens = pytest.mark.skipif(
    not LENS_INSTALLED,
    reason="requires the telemetry extra: uv sync --extra telemetry",
)


class _BlockNemoLens:
    """A ``sys.meta_path`` finder that makes ``import nemo.lens`` fail.

    This is how the tests reach the configuration nobody runs by accident: nemo-lens
    absent. Raising from ``find_spec`` (rather than returning ``None``) stops the real
    package being found even though it is installed in the test environment.
    """

    def find_spec(self, fullname, path=None, target=None):
        if fullname == _LENS_PREFIX or fullname.startswith(_LENS_PREFIX + "."):
            raise ImportError(f"nemo-lens is blocked for this test: {fullname}")
        return None


def import_without_lens(module_name: str) -> ModuleType:
    """Import *module_name* fresh with nemo-lens unavailable.

    Restores ``sys.meta_path`` and ``sys.modules`` afterwards so the blocked import does
    not leak into other tests.
    """
    blocker = _BlockNemoLens()
    saved = {name: mod for name, mod in sys.modules.items() if name == module_name or name.startswith(_LENS_PREFIX)}

    # importlib.import_module also rebinds the submodule as an attribute of its parent
    # package, and restoring sys.modules alone does not undo that — a later
    # `from nemo_gym.telemetry import _fallbacks` would keep resolving to the blocked
    # copy. Save the parent attribute too.
    parent_name, _, child_name = module_name.rpartition(".")
    parent = sys.modules.get(parent_name) if parent_name else None
    had_attr = parent is not None and hasattr(parent, child_name)
    saved_attr = getattr(parent, child_name, None) if had_attr else None

    for name in saved:
        del sys.modules[name]
    sys.meta_path.insert(0, blocker)
    try:
        return importlib.import_module(module_name)
    finally:
        sys.meta_path.remove(blocker)
        for name in list(sys.modules):
            if name == module_name or name.startswith(_LENS_PREFIX):
                del sys.modules[name]
        sys.modules.update(saved)
        if had_attr:
            setattr(parent, child_name, saved_attr)


def _reset_all_telemetry_state():
    """Reset Gym's process-globals and nemo-lens's one-shot init guard.

    ``nemo.lens.handle.setup_telemetry`` raises on a second call in the same process (by
    design — production code initialises once at startup). Tests need to initialise many
    times, so the guard is cleared here rather than by weakening the production check.
    """
    telemetry_setup._reset_for_testing()
    telemetry_metrics._reset_verify_tally_for_testing()
    try:
        from nemo.lens import handle as lens_handle
        from nemo.lens.state import set_enabled_span_groups
    except ImportError:
        return
    lens_handle._INITIALIZED = False
    set_enabled_span_groups(frozenset())


@contextmanager
def no_lens():
    """Make nemo-lens unimportable for the duration of the block.

    ``import_without_lens`` only covers a module's *import-time* imports.
    ``nemo_gym.telemetry.setup`` imports lens lazily inside ``init_telemetry``, so
    testing that path needs the blocker live while the function runs.
    """
    blocker = _BlockNemoLens()
    saved = {name: mod for name, mod in sys.modules.items() if name.startswith(_LENS_PREFIX)}
    for name in saved:
        del sys.modules[name]
    sys.meta_path.insert(0, blocker)
    try:
        yield
    finally:
        sys.meta_path.remove(blocker)
        for name in list(sys.modules):
            if name.startswith(_LENS_PREFIX):
                del sys.modules[name]
        sys.modules.update(saved)


@pytest.fixture(autouse=True)
def reset_telemetry_state():
    """Keep process-global telemetry state from leaking between tests."""
    _reset_all_telemetry_state()
    yield
    _reset_all_telemetry_state()


@pytest.fixture
def clean_otel_env(monkeypatch):
    """Remove every telemetry env var so a test starts from a known environment."""
    for name in list(os.environ):
        if name.startswith(("NEMO_GYM_OTEL_", "NEMO_LENS_", "OTEL_")):
            monkeypatch.delenv(name, raising=False)
    return monkeypatch
