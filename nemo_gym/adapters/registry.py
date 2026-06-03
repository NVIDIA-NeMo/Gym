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
"""Interceptor registry — maps short names to interceptor classes."""

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

# Each module must expose a class named ``Interceptor``.
#
# This dict holds only the framework-level builtins. Interceptor families
# (observability / caching / request-rewriting) add themselves via plain
# ``_BUILTIN[<name>] = <module_path>`` assignments below the dict literal,
# so each follow-on PR touches its own anchor lines and merges cleanly with
# the others.
_BUILTIN: dict[str, str] = {
    # ``endpoint`` drives the upstream HTTP call from inside the pipeline.
    # Required for ``start_adapter_proxy`` (standalone host mode); forbidden
    # inside ``install_middleware`` because the host server already forwards.
    "endpoint": "nemo_gym.adapters.interceptors.endpoint",
    # ``logging`` is the canonical "did the chain fire?" probe used by
    # framework-level tests. Lightweight; logs request keys + response status.
    "logging": "nemo_gym.adapters.interceptors.request_logging",
}

# Family extensions — follow-on PRs append entries to ``_BUILTIN`` here.
# Each family adds entries under its own family-named comment marker so
# different families don't fight for the same diff context.

# Observability family — passive (or mildly-normalizing) interceptors.
_BUILTIN["log_tokens"] = "nemo_gym.adapters.interceptors.log_tokens"
_BUILTIN["response_stats"] = "nemo_gym.adapters.interceptors.response_stats"
_BUILTIN["progress_tracking"] = "nemo_gym.adapters.interceptors.progress_tracking"
_BUILTIN["reasoning"] = "nemo_gym.adapters.interceptors.reasoning"
_BUILTIN["raise_client_errors"] = "nemo_gym.adapters.interceptors.raise_client_errors"

# External / plugin registrations at runtime.
_EXTRA: dict[str, str] = {}


class InterceptorRegistry:
    @staticmethod
    def register(name: str, module_path: str) -> None:
        _EXTRA[name] = module_path

    @staticmethod
    def resolve_class(name: str) -> InterceptorClass:
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
        cls = InterceptorRegistry.resolve_class(name)
        instance = cls(**(config or {}))
        instance._registry_name = name
        return instance

    @staticmethod
    def available() -> list[str]:
        return sorted(set(_BUILTIN) | set(_EXTRA))
