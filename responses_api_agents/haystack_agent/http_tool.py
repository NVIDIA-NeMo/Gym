# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Haystack tools that dispatch directly to a Gym Resources Server HTTP route."""

from collections.abc import Mapping
from typing import Any

from haystack.tools import Tool

from nemo_gym.server_utils import ServerClient
from responses_api_agents.haystack_agent import chat_generator


def _normalize_parameters(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Drop placeholder ``null`` properties used by Workplace Assistant schemas.

    Workplace Assistant publishes one broad parameter object per tool and represents fields that
    do not apply to a particular tool with ``null``. ``null`` is not a JSON Schema, however, and
    Haystack validates schemas on ``Tool`` construction. Removing those fields preserves the
    tool's actual callable signature and produces a valid schema for the model.
    """
    normalized = dict(schema)
    properties = normalized.get("properties")
    if isinstance(properties, Mapping):
        normalized["properties"] = {
            name: _normalize_parameters(value) if isinstance(value, Mapping) else value
            for name, value in properties.items()
            if value is not None
        }
        required = normalized.get("required")
        if isinstance(required, list):
            normalized["required"] = [name for name in required if name in normalized["properties"]]
    return normalized


class HTTPTool(Tool):
    """A request-scoped function tool backed by ``POST /{name}`` on a Resources Server."""

    def __init__(self, schema: Mapping[str, Any], server_client: ServerClient, resources_server_name: str) -> None:
        if schema.get("type") != "function":
            raise ValueError(f"HTTP environment tools must have type 'function', got {schema.get('type')!r}.")

        name = schema.get("name")
        parameters = schema.get("parameters")
        if not isinstance(name, str) or not name:
            raise ValueError("HTTP environment function tools require a non-empty name.")
        if not isinstance(parameters, dict):
            raise ValueError(f"HTTP environment tool '{name}' requires an object parameters schema.")

        self._server_client = server_client
        self._resources_server_name = resources_server_name
        super().__init__(
            name=name,
            description=str(schema.get("description") or ""),
            parameters=_normalize_parameters(parameters),
            async_function=self._invoke,
        )

    async def _invoke(self, **arguments: Any) -> str:
        state = chat_generator._current_run_state.get()
        if state is None:
            raise RuntimeError("HTTP environment tools can only be invoked during a Haystack rollout.")

        response = await self._server_client.post(
            server_name=self._resources_server_name,
            url_path=f"/{self.name}",
            json=arguments,
            cookies=state.resources_server_cookies,
        )
        state.resources_server_cookies = response.cookies
        return (await response.content.read()).decode()
