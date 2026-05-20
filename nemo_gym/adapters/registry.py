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
"""Interceptor registry — maps short names to interceptor classes.

Resolution happens at config-validation time (fail-fast), not at first
request.  Unknown names raise immediately.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Type

from nemo_gym.adapters.types import (
    RequestInterceptor,
    RequestToResponseInterceptor,
    ResponseInterceptor,
)

logger = logging.getLogger(__name__)

InterceptorClass = Type[RequestInterceptor | RequestToResponseInterceptor | ResponseInterceptor]

# Built-in interceptors — each module must expose a class named ``Interceptor``.
_BUILTIN: dict[str, str] = {
    # NB: ``endpoint`` runs an upstream HTTP call itself. Post-Phase-1.5 the
    # adapter middleware handles upstream forwarding via ``call_next`` on the
    # model server's own route, so enabling ``endpoint`` in a middleware-hosted
    # chain duplicates the model server's upstream call. Only enable it when
    # running the framework as a NEL-style standalone library (no middleware
    # host) where the pipeline itself must drive the upstream request.
    "endpoint": "nemo_gym.adapters.interceptors.endpoint",
    "drop_params": "nemo_gym.adapters.interceptors.drop_params",
    "modify_tools": "nemo_gym.adapters.interceptors.modify_tools",
    "turn_counter": "nemo_gym.adapters.interceptors.turn_counter",
    "raise_client_errors": "nemo_gym.adapters.interceptors.raise_client_errors",
    "system_message": "nemo_gym.adapters.interceptors.system_message",
    "payload_modifier": "nemo_gym.adapters.interceptors.payload_modifier",
    "consolidate_system": "nemo_gym.adapters.interceptors.consolidate_system",
    "caching": "nemo_gym.adapters.interceptors.caching",
    "log_tokens": "nemo_gym.adapters.interceptors.log_tokens",
    "response_stats": "nemo_gym.adapters.interceptors.response_stats",
    "reasoning": "nemo_gym.adapters.interceptors.reasoning",
    "progress_tracking": "nemo_gym.adapters.interceptors.progress_tracking",
    "logging": "nemo_gym.adapters.interceptors.request_logging",
}

# External / plugin registrations at runtime.
_EXTRA: dict[str, str] = {}


class InterceptorRegistry:
    """Resolve interceptor names to classes and instantiate them."""

    @staticmethod
    def register(name: str, module_path: str) -> None:
        """Register a custom interceptor at runtime."""
        _EXTRA[name] = module_path

    @staticmethod
    def resolve_class(name: str) -> InterceptorClass:
        """Import and return the ``Interceptor`` class for *name*.

        Raises ``ValueError`` immediately if the name is unknown or the
        module cannot be imported (fail-fast).
        """
        module_path = _EXTRA.get(name) or _BUILTIN.get(name)
        if module_path is None:
            available = sorted(set(_BUILTIN) | set(_EXTRA))
            raise ValueError(f"Unknown interceptor {name!r}. Available: {available}")
        try:
            mod = importlib.import_module(module_path)
        except ImportError as exc:
            raise ValueError(f"Cannot import interceptor module {module_path!r} for {name!r}: {exc}") from exc
        cls = getattr(mod, "Interceptor", None)
        if cls is None:
            raise ValueError(f"Module {module_path!r} does not expose an 'Interceptor' class")
        return cls

    @staticmethod
    def create(
        name: str, config: dict[str, Any] | None = None
    ) -> RequestInterceptor | RequestToResponseInterceptor | ResponseInterceptor:
        """Resolve *name*, instantiate with *config* kwargs, and return."""
        cls = InterceptorRegistry.resolve_class(name)
        instance = cls(**(config or {}))
        instance._registry_name = name
        return instance

    @staticmethod
    def available() -> list[str]:
        return sorted(set(_BUILTIN) | set(_EXTRA))
