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
"""Pytest fixtures and skip markers for the gym_v resources server tests.

The helper-level tests under this directory only need pydantic + Pillow + the
nemo_gym schemas. The integration tests additionally need `gym_v` (and, for
some envs, `reasoning_gym`). Until the resources-server container image
pip-installs those extras, the integration tests must skip cleanly.

Note on the top-level package-name collision (deliberate non-`__init__`).
The resources-server package is mounted at `resources_servers/gym_v/`. If
this tests directory also had an `__init__.py`, pytest's default conftest
discovery would walk upward to the first non-`__init__` ancestor and prepend
that ancestor (`resources_servers/`) to `sys.path`, which would then make
a bare `import gym_v` resolve to OUR local package instead of the upstream
pip-installed Gym-V. We deliberately leave this directory without an
`__init__.py` and rely on pytest's rootdir-based discovery.
"""

from __future__ import annotations

import importlib
import importlib.util

import pytest


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _has_upstream_gym_v() -> bool:
    # Defense in depth: actually import `gym_v` and check for attributes
    # that only the real upstream package exposes (`make`, `Env`). The
    # local resources_servers.gym_v package has neither at top level.
    try:
        gym_v = importlib.import_module("gym_v")
    except ImportError:
        return False
    return hasattr(gym_v, "make") and hasattr(gym_v, "Env")


requires_gym_v = pytest.mark.skipif(
    not _has_upstream_gym_v(),
    reason="upstream gym_v not installed; rebuild the resources-server container.",
)

requires_reasoning_gym = pytest.mark.skipif(
    not _has_module("reasoning_gym"),
    reason="reasoning_gym not installed; rebuild the resources-server container.",
)
