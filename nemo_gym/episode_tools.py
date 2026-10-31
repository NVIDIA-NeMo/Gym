# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tools that an Environment Server serves to the agent for one episode."""

import inspect
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Protocol, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError


# The same names a tool grant accepts (``nemo_gym.tool_access``); no slash, so a name is one path segment.
TOOL_NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
_EPISODE_TOOL_ATTRIBUTE = "__nemo_gym_episode_tool__"


class ToolSpec(BaseModel):
    """One tool as the agent sees it."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=TOOL_NAME_PATTERN)
    description: str = Field(min_length=1)
    parameters: dict[str, JsonValue] = Field(description="JSON schema of the arguments object.")


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
    setattr(method, _EPISODE_TOOL_ATTRIBUTE, True)
    return method


class TypedToolset:
    """An ``EpisodeToolset`` built from the ``@episode_tool`` methods of one object, such as an episode's state.

    Each method takes one Pydantic model of arguments and returns a ``ToolResult``, a Pydantic model or a JSON
    value. The tool's name is the method's name, and its parameters are the model's JSON schema.
    """

    def __init__(self, owner: object) -> None:
        self._tools: dict[str, tuple[Callable[[Any], Awaitable[Any]], type[BaseModel], ToolSpec]] = {}
        for name, function in inspect.getmembers(type(owner), predicate=inspect.isfunction):
            if not getattr(function, _EPISODE_TOOL_ATTRIBUTE, False):
                continue
            parameters = list(inspect.signature(function).parameters)[1:]
            arguments_model = get_type_hints(function).get(parameters[0]) if len(parameters) == 1 else None
            if not (isinstance(arguments_model, type) and issubclass(arguments_model, BaseModel)):
                raise TypeError(f"Episode tool {name!r} must take exactly one Pydantic model of arguments")
            description = inspect.getdoc(function)
            if not description:
                raise TypeError(f"Episode tool {name!r} needs a docstring, which is its description")
            spec = ToolSpec(name=name, description=description, parameters=arguments_model.model_json_schema())
            self._tools[name] = (getattr(owner, name), arguments_model, spec)

    def specs(self) -> list[ToolSpec]:
        return [spec for _, _, spec in self._tools.values()]

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        method, arguments_model, _ = self._tools[name]
        try:
            parsed = arguments_model.model_validate(arguments)
        except ValidationError as error:
            raise ToolArgumentsError(str(error)) from error
        result = await method(parsed)
        if isinstance(result, ToolResult):
            return result
        if isinstance(result, BaseModel):
            return ToolResult(output=result.model_dump(mode="json"))
        return ToolResult(output=result)
