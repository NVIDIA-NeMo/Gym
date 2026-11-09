# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tools that an Environment Server serves to the agent for one episode."""

import functools
import inspect
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Protocol, get_type_hints

from openai.types.responses import FunctionToolParam
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, field_validator


# The names a tool grant accepts (``nemo_gym.server_runtime.tool_access``); no slash, so a name is one path segment.
TOOL_NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
_EPISODE_TOOL_ATTRIBUTE = "__nemo_gym_episode_tool__"


class ToolSpec(BaseModel):
    """One tool as the agent sees it."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=TOOL_NAME_PATTERN)
    description: str = Field(min_length=1)
    parameters: dict[str, JsonValue] = Field(description="JSON schema of the arguments object.")

    @field_validator("parameters")
    @classmethod
    def validate_object_schema(cls, parameters: dict[str, JsonValue]) -> dict[str, JsonValue]:
        # Model APIs and MCP clients take only an object schema; some MCP clients refuse the whole tool list.
        if parameters.get("type") != "object":
            raise ValueError('A tool\'s parameters must be a JSON schema with "type": "object"')
        return parameters

    def function_tool(self) -> FunctionToolParam:
        """Declare this tool to the model, for an agent that calls it over direct HTTP.

        The parameters are the full schema, with its ``$defs``. The declaration is not strict, because a strict
        schema must make every property required.
        """
        return FunctionToolParam(
            type="function",
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            strict=False,
        )


class ToolResult(BaseModel):
    """What one tool call returns to the agent. ``is_error`` marks an error the agent should see and handle."""

    model_config = ConfigDict(extra="forbid")

    output: JsonValue
    is_error: bool = False


class ToolCallRecord(BaseModel):
    """One tool call of an episode, in the order the server ran it."""

    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: JsonValue
    output: JsonValue
    is_error: bool
    duration_seconds: float


class ToolArgumentsError(ValueError):
    """The agent sent arguments that the tool does not accept. The message is shown to the agent."""


class EpisodeToolset(Protocol):
    """The tools of one episode.

    ``specs`` is fixed for the episode. ``call`` runs one tool by name; the server checks the name against
    ``specs`` first. It raises ``ToolArgumentsError`` for arguments the tool does not accept.
    """

    def specs(self) -> Sequence[ToolSpec]: ...

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult: ...


def episode_tool(method: Callable[[Any, Any], Awaitable[Any]]) -> Callable[[Any, Any], Awaitable[Any]]:
    """Mark an async method as a tool for ``TypedToolset``. Its docstring is the tool's description."""
    # A plain function would run its side effects and then fail when the server awaits its result.
    if not inspect.iscoroutinefunction(method):
        raise TypeError(f"Episode tool {method.__name__!r} must be an async method")
    setattr(method, _EPISODE_TOOL_ATTRIBUTE, True)
    return method


_POSITIONAL = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)


@functools.cache
def _tool_table(owner_type: type) -> dict[str, tuple[type[BaseModel], ToolSpec]]:
    """The arguments model and spec of each ``@episode_tool`` method of a class.

    They depend only on the class, so each class builds them once, not once for each episode.
    """
    tools: dict[str, tuple[type[BaseModel], ToolSpec]] = {}
    for name, function in inspect.getmembers(owner_type, predicate=inspect.isfunction):
        if not getattr(function, _EPISODE_TOOL_ATTRIBUTE, False):
            continue
        parameters = list(inspect.signature(function).parameters.values())[1:]
        arguments_model = (
            get_type_hints(function).get(parameters[0].name)
            if len(parameters) == 1 and parameters[0].kind in _POSITIONAL
            else None
        )
        if not (isinstance(arguments_model, type) and issubclass(arguments_model, BaseModel)):
            raise TypeError(f"Episode tool {name!r} must take exactly one positional Pydantic model of arguments")
        description = inspect.getdoc(function)
        if not description:
            raise TypeError(f"Episode tool {name!r} needs a docstring, which is its description")
        spec = ToolSpec(name=name, description=description, parameters=arguments_model.model_json_schema())
        tools[name] = (arguments_model, spec)
    return tools


class TypedToolset:
    """An ``EpisodeToolset`` built from the ``@episode_tool`` methods of one object, such as an episode's state.

    Each method takes one Pydantic model of arguments and returns a ``ToolResult``, a Pydantic model or a JSON
    value. The tool's name is the method's name, and its parameters are the model's JSON schema.
    """

    def __init__(self, owner: object) -> None:
        self._owner = owner
        self._tools = _tool_table(type(owner))

    def specs(self) -> list[ToolSpec]:
        return [spec for _, spec in self._tools.values()]

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        arguments_model, _ = self._tools[name]
        try:
            parsed = arguments_model.model_validate(arguments)
        except ValidationError as error:
            raise ToolArgumentsError(str(error)) from error
        result = await getattr(self._owner, name)(parsed)
        if isinstance(result, ToolResult):
            return result
        if isinstance(result, BaseModel):
            return ToolResult(output=result.model_dump(mode="json"))
        return ToolResult(output=result)
