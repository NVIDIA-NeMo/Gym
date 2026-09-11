# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import inspect
import json
import keyword
from dataclasses import dataclass, make_dataclass
from time import perf_counter, time
from typing import Any
from uuid import uuid4

from jsonschema import Draft202012Validator, ValidationError
from pydantic import BaseModel

from nemo_gym.base_resources_server import RESERVED_MCP_TOOL_NAMES
from nemo_gym.server_utils import ServerClient
from responses_api_agents.nooa_agent.observability import GymTraceHooks


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
    invocation_id: str = "root"
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


def _validate_identifier(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.isidentifier() or keyword.iskeyword(value) or value.startswith("_"):
        raise ValueError(f"{label} must be a public Python identifier, received {value!r}")
    return value


def _validate_tool(tool: dict[str, Any], *, seen: set[str]) -> tuple[str, dict[str, Any]]:
    if tool.get("type") != "function":
        raise ValueError(f"resource tools support function tools only, received {tool.get('type')!r}")
    name = _validate_identifier(tool.get("name"), label="resource tool name")
    if name in RESERVED_MCP_TOOL_NAMES:
        raise ValueError(f"resource tool name {name!r} is reserved by Gym")
    if name in seen:
        raise ValueError(f"duplicate resource tool name {name!r}")
    seen.add(name)

    schema = tool.get("parameters") or {"type": "object", "properties": {}, "additionalProperties": False}
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as error:
        raise ValueError(f"resource tool {name!r} has an invalid JSON Schema: {error}") from error
    if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        raise ValueError(
            f"resource tool {name!r} must use a closed object JSON Schema with additionalProperties=false"
        )
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    if not isinstance(properties, dict) or not isinstance(required, list):
        raise ValueError(f"resource tool {name!r} has invalid properties or required fields")
    invalid = [
        parameter
        for parameter in properties
        if not isinstance(parameter, str)
        or not parameter.isidentifier()
        or keyword.iskeyword(parameter)
        or parameter == "self"
    ]
    unknown_required = set(required) - set(properties)
    if invalid or unknown_required:
        raise ValueError(
            f"resource tool {name!r} cannot form a Python signature: "
            f"invalid={invalid}, unknown_required={sorted(unknown_required)}"
        )
    return name, schema


class _GymToolDispatcher:
    """Host-only transport, cookie, and observation state for one rollout."""

    def __init__(
        self,
        *,
        server_client: ServerClient,
        resources_server_name: str,
        cookies: dict[str, str],
        observations: list[GymToolExecution],
        trace_hooks: GymTraceHooks | None,
    ) -> None:
        self.server_client = server_client
        self.resources_server_name = resources_server_name
        self.cookies = cookies
        self.observations = observations
        self.trace_hooks = trace_hooks
        self.lock = asyncio.Lock()

    async def invoke(self, name: str, arguments: dict[str, Any], validator: Draft202012Validator) -> Any:
        started_at = time()
        started_monotonic = perf_counter()
        call_id = f"gym_tool_{uuid4().hex}"
        status = "completed"
        error_type = None
        output: Any = None
        try:
            try:
                validator.validate(arguments)
            except ValidationError as error:
                output = {"error": f"Invalid arguments for {name}: {error.message}"}
                status = "failed"
                error_type = "invalid_arguments"
            else:
                # Serialize resource calls because one rollout shares one stateful
                # cookie jar and resource session.
                async with self.lock:
                    response = await self.server_client.post(
                        server_name=self.resources_server_name,
                        url_path=f"/{name}",
                        json=arguments,
                        cookies=self.cookies,
                    )
                    self.cookies.update({key: morsel.value for key, morsel in response.cookies.items()})
                    body = (await response.content.read()).decode(errors="replace")
                try:
                    output = json.loads(body)
                except json.JSONDecodeError:
                    output = body
                if not 200 <= response.status < 400:
                    status = "failed"
                    error_type = f"http_{response.status}"
            return output
        except asyncio.CancelledError:
            status = "cancelled"
            error_type = "CancelledError"
            raise
        except Exception as error:
            status = "failed"
            error_type = type(error).__name__
            raise
        finally:
            completed_at = max(started_at, time())
            execution = GymToolExecution(
                tool_call_id=call_id,
                name=name,
                arguments=arguments,
                output=output,
                status=status,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=(perf_counter() - started_monotonic) * 1000,
                invocation_id=self.trace_hooks.invocation_id if self.trace_hooks else "root",
                error_type=error_type,
            )
            self.observations.append(execution)
            if self.trace_hooks is not None:
                self.trace_hooks.record_tool_execution(execution)


def _make_method(dispatcher: _GymToolDispatcher, name: str, description: str, schema: dict[str, Any]) -> Any:
    validator = Draft202012Validator(schema)

    async def invoke(namespace: Any, *args: Any, **kwargs: Any) -> Any:
        bound = inspect.signature(invoke).bind(namespace, *args, **kwargs)
        arguments = dict(bound.arguments)
        arguments.pop("self", None)
        return await dispatcher.invoke(name, arguments, validator)

    invoke.__name__ = name
    invoke.__qualname__ = name
    invoke.__doc__ = description or f"Call the Gym Resources server's {name} tool."
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    ordered_names = [key for key in properties if key in required] + [key for key in properties if key not in required]
    parameters = [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    for parameter_name in ordered_names:
        parameter_schema = properties[parameter_name]
        default = inspect.Parameter.empty if parameter_name in required else parameter_schema.get("default")
        parameters.append(
            inspect.Parameter(
                parameter_name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=default,
                annotation=_annotation(parameter_schema),
            )
        )
    invoke.__signature__ = inspect.Signature(parameters, return_annotation=Any)  # type: ignore[attr-defined]
    return invoke


def build_tool_namespace(
    *,
    namespace_name: str,
    server_client: ServerClient,
    resources_server_name: str,
    tools: list[Any],
    allowed_tools: frozenset[str],
    cookies: dict[str, str],
    observations: list[GymToolExecution],
    trace_hooks: GymTraceHooks | None = None,
) -> Any:
    """Build a semantic public tool object backed by a private dispatcher."""

    _validate_identifier(namespace_name, label="tool_namespace")
    requested_names = {str(_as_tool_dict(tool).get("name")) for tool in tools}
    unauthorized = requested_names - allowed_tools
    if unauthorized:
        raise ValueError(f"resource tools are not authorized by nooa.allowed_tools: {sorted(unauthorized)}")
    dispatcher = _GymToolDispatcher(
        server_client=server_client,
        resources_server_name=resources_server_name,
        cookies=cookies,
        observations=observations,
        trace_hooks=trace_hooks,
    )
    seen: set[str] = set()
    method_definitions: dict[str, Any] = {}
    for raw_tool in tools:
        tool = _as_tool_dict(raw_tool)
        name, schema = _validate_tool(tool, seen=seen)
        method_definitions[name] = _make_method(dispatcher, name, tool.get("description") or "", schema)

    class_name = "".join(part.capitalize() for part in namespace_name.split("_")) + "Tools"
    namespace_type = make_dataclass(
        class_name,
        [],
        namespace={
            "__doc__": f"Tools provided by the {resources_server_name} Gym resource server.",
            **method_definitions,
        },
    )
    return namespace_type()
