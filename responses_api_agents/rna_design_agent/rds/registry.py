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
"""The tool registry, the single place a tool is declared."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from .spec import Cost, ToolSpec


_REGISTRY: dict[str, ToolSpec] = {}


def tool(
    name: str, summary: str, cost: Cost, model: type, needs: Iterable[str] = (), steps: Iterable[str] = ()
) -> Callable[[Callable], Callable]:
    """Register a tool. `name` is "group.verb"."""
    if "." not in name:
        raise ValueError(f"tool name must be 'group.verb', got {name!r}")

    def decorate(fn: Callable) -> Callable:
        if name in _REGISTRY:
            raise ValueError(f"tool already registered: {name}")
        _REGISTRY[name] = ToolSpec(
            name=name, summary=summary, cost=cost, model=model, fn=fn, needs=frozenset(needs), steps=tuple(steps)
        )
        return fn

    return decorate


def get(name: str) -> ToolSpec:
    """Look up a tool by its registry name or by its dotless model-facing name (`fold_eternafold`)."""
    if name in _REGISTRY:
        return _REGISTRY[name]
    for spec in _REGISTRY.values():
        if spec.llm_name == name:
            return spec
    raise KeyError(f"unknown tool: {name} (have: {', '.join(sorted(_REGISTRY))})")


def all_tools() -> list[ToolSpec]:
    """Every registered tool, name-sorted. The tool list is a prompt-cache prefix, so the order must be stable."""
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


def llm_tools() -> list[dict[str, Any]]:
    """Tool definitions for a controller, in registry order."""
    return [t.llm_tool() for t in all_tools()]


def dispatch(name: str, arguments: dict[str, Any]) -> Any:
    """Run a tool by name with LLM-supplied arguments."""
    return get(name).call(**arguments)
