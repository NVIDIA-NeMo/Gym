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

from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any

from jsonschema import Draft202012Validator, ValidationError
from pydantic import BaseModel

from nemo_gym.server_utils import ServerClient


def _as_tool_dict(tool: Any) -> dict[str, Any]:
    return tool.model_dump(mode="json", exclude_none=True) if isinstance(tool, BaseModel) else dict(tool)


def _annotation(schema: dict[str, Any]) -> Any:
    schema_type = schema.get("type")
    if schema_type == "string":
        return str
    if schema_type == "integer":
        return int
    if schema_type == "number":
        return float
    if schema_type == "boolean":
        return bool
    if schema_type == "array":
        return list
    if schema_type == "object":
        return dict
    return Any


class ResourceToolDispatcher:
    """Private per-rollout transport for methods attached to a NOOA agent."""

    def __init__(
        self,
        *,
        server_client: ServerClient,
        resources_server_name: str,
        cookies: dict[str, str],
    ) -> None:
        self._server_client = server_client
        self._resources_server_name = resources_server_name
        self._cookies = cookies
        self._lock = asyncio.Lock()

    async def call(
        self,
        *,
        name: str,
        arguments: dict[str, Any],
        validator: Draft202012Validator,
    ) -> Any:
        async with self._lock:
            return await self._call(name=name, arguments=arguments, validator=validator)

    async def _call(
        self,
        *,
        name: str,
        arguments: dict[str, Any],
        validator: Draft202012Validator,
    ) -> Any:
        try:
            validator.validate(arguments)
        except ValidationError as error:
            output: Any = {"error": f"Invalid arguments for {name}: {error.message}"}
        else:
            response = await self._server_client.post(
                server_name=self._resources_server_name,
                url_path=f"/{name}",
                json=arguments,
                cookies=self._cookies,
            )
            self._cookies.update({key: morsel.value for key, morsel in response.cookies.items()})
            body = (await response.content.read()).decode(errors="replace")
            try:
                output = json.loads(body)
            except json.JSONDecodeError:
                output = body
        return output


def _make_method(
    *,
    agent_class: type[Any],
    dispatcher: ResourceToolDispatcher,
    name: str,
    description: str,
    schema: dict[str, Any],
) -> Any:
    validator = Draft202012Validator(schema)
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    ordered_names = [
        *(parameter_name for parameter_name in properties if parameter_name in required),
        *(parameter_name for parameter_name in properties if parameter_name not in required),
    ]
    parameters = [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    explicit_defaults: dict[str, Any] = {}
    for parameter_name in ordered_names:
        parameter_schema = properties[parameter_name]
        if not parameter_name.isidentifier() or parameter_name.startswith("_"):
            raise ValueError(f"resource tool {name!r} has invalid parameter name {parameter_name!r}")
        if parameter_name in required:
            default = inspect.Parameter.empty
        else:
            default = parameter_schema.get("default")
            if "default" in parameter_schema:
                explicit_defaults[parameter_name] = default
        parameters.append(
            inspect.Parameter(
                parameter_name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=default,
                annotation=_annotation(parameter_schema),
            )
        )
    signature = inspect.Signature(parameters, return_annotation=Any)

    async def invoke(agent: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            bound = signature.bind(agent, *args, **kwargs)
        except TypeError as error:
            raise TypeError(f"Invalid arguments for {name}: {error}") from error
        arguments = {key: value for key, value in bound.arguments.items() if key != "self"}
        for parameter_name, default in explicit_defaults.items():
            arguments.setdefault(parameter_name, default)
        return await dispatcher.call(name=name, arguments=arguments, validator=validator)

    invoke.__name__ = name
    invoke.__qualname__ = f"{agent_class.__name__}.{name}"
    invoke.__doc__ = description or f"Call the Resources server's {name} tool."
    invoke.__signature__ = signature
    return invoke


def create_agent_class_with_resource_methods(
    agent_class: type[Any],
    *,
    dispatcher: ResourceToolDispatcher,
    tools: list[Any],
) -> type[Any]:
    """Create a per-rollout agent subclass exposing resource methods directly."""

    seen: set[str] = set()
    methods: dict[str, Any] = {"__module__": agent_class.__module__}
    for raw_tool in tools:
        tool = _as_tool_dict(raw_tool)
        if tool.get("type") != "function":
            raise ValueError(f"resource methods support function tools only, received {tool.get('type')!r}")
        name = tool.get("name")
        if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
            raise ValueError(f"resource tool name must be a public Python identifier, received {name!r}")
        if name in seen or hasattr(agent_class, name):
            raise ValueError(f"duplicate or conflicting agent method name {name!r}")
        seen.add(name)

        schema = tool.get("parameters") or {"type": "object", "properties": {}}
        try:
            Draft202012Validator.check_schema(schema)
        except Exception as error:
            raise ValueError(f"resource tool {name!r} has an invalid JSON Schema: {error}") from error
        if schema.get("type") != "object":
            raise ValueError(f"resource tool {name!r} parameters must use an object JSON Schema")

        method = _make_method(
            agent_class=agent_class,
            dispatcher=dispatcher,
            name=name,
            description=tool.get("description") or "",
            schema=schema,
        )
        methods[name] = method

    methods["__gym_resource_method_names__"] = tuple(name for name in methods if name != "__module__")
    return type(f"{agent_class.__name__}WithResources", (agent_class,), methods)


def validate_agent_resource_method_bindings(agent: Any) -> None:
    """Reject instance fields created by __init__ that hide resource methods."""

    agent_class = type(agent)
    method_names = inspect.getattr_static(agent_class, "__gym_resource_method_names__", ())
    instance_fields = vars(agent) if hasattr(agent, "__dict__") else {}
    for name in method_names:
        expected = vars(agent_class)[name]
        actual = inspect.getattr_static(agent, name)
        if name in instance_fields or actual is not expected:
            raise ValueError(f"agent instance field conflicts with resource method name {name!r}")
