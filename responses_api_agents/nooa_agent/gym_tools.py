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

import inspect
import json
from dataclasses import dataclass
from time import perf_counter, time
from types import MethodType
from typing import Any
from uuid import uuid4

from jsonschema import Draft202012Validator, ValidationError
from pydantic import BaseModel

from nemo_gym.server_utils import ServerClient


@dataclass(slots=True)
class GymToolExecution:
    tool_call_id: str
    name: str
    arguments: dict[str, Any]
    output: Any
    status: str
    started_at: float
    completed_at: float
    duration_ms: float
    error_type: str | None = None


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


class GymTools:
    """Per-rollout namespace of Resources server methods available to generated NOOA code."""

    def __init__(
        self,
        *,
        server_client: ServerClient,
        resources_server_name: str,
        tools: list[Any],
        cookies: dict[str, str],
        observations: list[GymToolExecution],
    ) -> None:
        self._server_client = server_client
        self._resources_server_name = resources_server_name
        self._cookies = cookies
        self._observations = observations
        self._install(tools)

    def _install(self, tools: list[Any]) -> None:
        seen: set[str] = set()
        for raw_tool in tools:
            tool = _as_tool_dict(raw_tool)
            if tool.get("type") != "function":
                raise ValueError(f"gym_tools supports function tools only, received {tool.get('type')!r}")
            name = tool.get("name")
            if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
                raise ValueError(f"resource tool name must be a public Python identifier, received {name!r}")
            if name in seen or hasattr(self, name):
                raise ValueError(f"duplicate or reserved resource tool name {name!r}")
            seen.add(name)

            schema = tool.get("parameters") or {"type": "object", "properties": {}}
            try:
                Draft202012Validator.check_schema(schema)
            except Exception as error:
                raise ValueError(f"resource tool {name!r} has an invalid JSON Schema: {error}") from error
            if schema.get("type") != "object":
                raise ValueError(f"resource tool {name!r} parameters must use an object JSON Schema")

            method = self._make_method(name, tool.get("description") or "", schema)
            setattr(self, name, MethodType(method, self))

    @staticmethod
    def _make_method(name: str, description: str, schema: dict[str, Any]) -> Any:
        validator = Draft202012Validator(schema)

        async def invoke(namespace: GymTools, **kwargs: Any) -> Any:
            started_at = time()
            started_monotonic = perf_counter()
            call_id = f"gym_tool_{uuid4().hex}"
            status = "completed"
            error_type = None
            try:
                validator.validate(kwargs)
            except ValidationError as error:
                output: Any = {"error": f"Invalid arguments for {name}: {error.message}"}
                status = "failed"
                error_type = "invalid_arguments"
            else:
                response = await namespace._server_client.post(
                    server_name=namespace._resources_server_name,
                    url_path=f"/{name}",
                    json=kwargs,
                    cookies=namespace._cookies,
                )
                namespace._cookies.update({key: morsel.value for key, morsel in response.cookies.items()})
                body = (await response.content.read()).decode(errors="replace")
                try:
                    output = json.loads(body)
                except json.JSONDecodeError:
                    output = body
                if not 200 <= response.status < 400:
                    status = "failed"
                    error_type = f"http_{response.status}"

            completed_at = max(started_at, time())
            namespace._observations.append(
                GymToolExecution(
                    tool_call_id=call_id,
                    name=name,
                    arguments=kwargs,
                    output=output,
                    status=status,
                    started_at=started_at,
                    completed_at=completed_at,
                    duration_ms=(perf_counter() - started_monotonic) * 1000,
                    error_type=error_type,
                )
            )
            return output

        invoke.__name__ = name
        invoke.__qualname__ = f"GymTools.{name}"
        invoke.__doc__ = description or f"Call the Gym Resources server's {name} tool."
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        parameters = [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        for parameter_name, parameter_schema in properties.items():
            if not parameter_name.isidentifier() or parameter_name.startswith("_"):
                raise ValueError(f"resource tool {name!r} has invalid parameter name {parameter_name!r}")
            default = inspect.Parameter.empty if parameter_name in required else None
            parameters.append(
                inspect.Parameter(
                    parameter_name,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=default,
                    annotation=_annotation(parameter_schema),
                )
            )
        invoke.__signature__ = inspect.Signature(parameters, return_annotation=Any)
        return invoke
