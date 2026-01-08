# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import annotations

import inspect
from typing import Any, Dict, List

from omegaconf import OmegaConf

from .tool_manager import Tool
from .utils import hydra_config_connector_factory, locate


class MCPClientTool(Tool):
    """Base Tool that delegates to an MCP client (stdio or streamable HTTP).

    Config keys (overridable via tool_overrides[provider_id]):
      - transport: "stdio" | "streamable_http" (default: stdio)
      - command, args: for stdio transport
      - base_url: for streamable_http transport
      - hide_args, disabled_tools, enabled_tools
      - output_formatter: dotted path or callable
      - init_hook: dotted path or callable (receives client for side effects)
    """

    provider_id: str = ""

    def __init__(self) -> None:
        self._config: Dict[str, Any] = {
            "client": None,  # dotted path or class
            "client_params": {},  # kwargs for client constructor
            "hide_args": {},
            "disabled_tools": [],
            "enabled_tools": [],
            "output_formatter": None,
            "init_hook": None,
        }
        self._client = None

    # Centralized helper to update internal config with a single entry point.
    # Subclasses should use this instead of mutating _config directly.
    def apply_config_updates(self, updates: Dict[str, Any] | None) -> None:
        if not updates:
            return
        self._config.update(updates)
        # In the future, side-effects or validations for specific keys can be handled here

    def default_config(self) -> Dict[str, Any]:
        return dict(self._config)

    def _resolve_maybe_callable(self, value: Any):
        if value is None:
            return None
        if isinstance(value, str):
            try:
                return locate(value)
            except Exception:
                # If not resolvable, pass through; client wrapper may handle
                return value
        return value

    def post_configure(self) -> None:
        pass

    def configure(self, overrides: Dict[str, Any] | None = None, context: Dict[str, Any] | None = None) -> None:
        cfg = dict(self._config)
        if overrides:
            cfg.update(overrides)

        output_formatter = self._resolve_maybe_callable(cfg.get("output_formatter"))
        init_hook = self._resolve_maybe_callable(cfg.get("init_hook"))

        # Explicit config-connector activation: init_hook == "hydra" builds from full context
        if init_hook == "hydra":
            init_hook = hydra_config_connector_factory(OmegaConf.create(context or {}))

        # Construct client (do not pass init_hook here; we will invoke it ourselves to allow context)
        custom_client_cls = cfg.get("client")
        if not custom_client_cls:
            raise ValueError("MCPClientTool requires 'client' (class path or class) to be specified")
        if isinstance(custom_client_cls, str):
            custom_client_cls = locate(custom_client_cls)
        client_params = dict(cfg.get("client_params", {}))

        # User-specified client class and params
        self._client = custom_client_cls(**client_params)

        # Attach common behaviors post-init (recognized by MCP metaclass wrappers)
        self._client._hide_args = cfg.get("hide_args", {})
        self._client._disabled_tools = set(cfg.get("disabled_tools", []))
        self._client._enabled_tools = set(cfg.get("enabled_tools", []))
        self._client.output_formatter = output_formatter

        # Invoke init_hook with context support if provided
        if callable(init_hook):
            try:
                sig = inspect.signature(init_hook)
                if len(sig.parameters) >= 2:
                    init_hook(self._client, context)
                else:
                    init_hook(self._client)
            except Exception:
                # Fall back to best-effort single-arg invocation
                try:
                    init_hook(self._client)
                except Exception:
                    raise

        self._config = cfg

    async def list_tools(self) -> List[Dict[str, Any]]:
        return await self._client.list_tools()

    async def execute(self, tool_name: str, arguments: Dict[str, Any], extra_args: Dict[str, Any] | None = None):
        return await self._client.call_tool(tool=tool_name, args=arguments, extra_args=extra_args)
