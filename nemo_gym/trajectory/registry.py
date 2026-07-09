# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""The trajectory-builder plugin registry.

A trajectory builder turns a rollout's captured token entries into masked token
chains. All builders emit the same Trajectory schema; they differ only in HOW
they reconstruct the chains (e.g. per_request vs prefix_merging). Builders are
pluggable so users can add their own without editing Gym, in lookup precedence:

  1. register_builder(name, fn) — explicit in-process registration (and the
     built-in @register decorator used by the shipped builders).
  2. Python entry points in the "nemo_gym.trajectory_builders" group, so a
     separate package publishes a builder that becomes available on install:

         [project.entry-points."nemo_gym.trajectory_builders"]
         my_builder = "my_pkg.builders:build_my_trajectory"
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Callable


if TYPE_CHECKING:
    from nemo_gym.observability.records import TokenEntry
    from nemo_gym.trajectory.strategies import BuildOutput

# A builder maps a rollout's token entries to a BuildOutput (chains + notes).
TrajectoryBuilder = Callable[[list["TokenEntry"]], "BuildOutput"]

ENTRY_POINT_GROUP = "nemo_gym.trajectory_builders"

_REGISTRY: dict[str, TrajectoryBuilder] = {}
_ENTRY_POINT_BUILDERS: dict[str, Callable[[], TrajectoryBuilder]] | None = None


def register(name: str):
    """Decorator used by the built-in builders (per_request, prefix_merging)."""

    def deco(fn: TrajectoryBuilder) -> TrajectoryBuilder:
        _REGISTRY[name] = fn
        return fn

    return deco


def register_builder(name: str, fn: TrajectoryBuilder, *, override: bool = False) -> None:
    """Register a custom trajectory builder in-process (public plugin API)."""
    if not name:
        raise ValueError("builder name must be non-empty")
    if not override and name in _REGISTRY:
        raise ValueError(f"trajectory builder {name!r} is already registered")
    _REGISTRY[name] = fn


def _entry_point_builders() -> dict[str, Callable[[], TrajectoryBuilder]]:
    """Discover builder loaders from installed entry points (cached)."""
    global _ENTRY_POINT_BUILDERS
    if _ENTRY_POINT_BUILDERS is None:
        loaders: dict[str, Callable[[], TrajectoryBuilder]] = {}
        for ep in entry_points(group=ENTRY_POINT_GROUP):
            if ep.name in loaders:
                raise ValueError(f"duplicate trajectory builder entry point {ep.name!r}; rename one.")
            loaders[ep.name] = ep.load
        _ENTRY_POINT_BUILDERS = loaders
    return _ENTRY_POINT_BUILDERS


def get_builder(name: str) -> TrajectoryBuilder:
    """Return a trajectory builder by name (explicit/built-in > entry point)."""
    if name in _REGISTRY:
        return _REGISTRY[name]
    loader = _entry_point_builders().get(name)
    if loader is not None:
        return loader()
    raise KeyError(f"unknown trajectory builder {name!r}; have {list_builders()}")


def list_builders() -> list[str]:
    """List available trajectory builder names from all sources."""
    return sorted({*_REGISTRY, *_entry_point_builders()})
