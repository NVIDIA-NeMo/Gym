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

"""Name-to-harness registry for dispatching tasks to their SWE harness."""

from __future__ import annotations

from responses_api_agents.swe_env.harness import SweTaskHarness


_HARNESSES: dict[str, SweTaskHarness] = {}


def register_harness(harness: SweTaskHarness, *, override: bool = False) -> None:
    """Register a harness under its ``name``.

    Args:
        harness (SweTaskHarness): The harness to register. Its ``name`` must be
            non-empty.
        override (bool): If ``True``, replace an existing harness with the same
            name instead of raising.

    Raises:
        ValueError: If the harness name is empty, or a harness with the same name
            is already registered and ``override`` is ``False``.
    """
    if not harness.name:
        raise ValueError("Harness must define a non-empty 'name'")
    if not override and harness.name in _HARNESSES:
        raise ValueError(f"Harness {harness.name!r} is already registered")
    _HARNESSES[harness.name] = harness


def get_harness(name: str) -> SweTaskHarness:
    """Look up a registered harness by name.

    Args:
        name (str): The registry key of the harness.

    Returns:
        SweTaskHarness: The registered harness.

    Raises:
        KeyError: If no harness is registered under ``name``.
    """
    try:
        return _HARNESSES[name]
    except KeyError as exc:
        available = ", ".join(sorted(_HARNESSES)) or "(none)"
        raise KeyError(f"Unknown SWE harness {name!r}. Registered: {available}") from exc


def list_harnesses() -> list[str]:
    """List the names of all registered harnesses.

    Returns:
        list[str]: The registered harness names, sorted alphabetically.
    """
    return sorted(_HARNESSES)
