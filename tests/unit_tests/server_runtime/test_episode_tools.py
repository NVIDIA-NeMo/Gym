# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
from pydantic import BaseModel, ValidationError

from nemo_gym.server_runtime.episode_tools import ToolArgumentsError, ToolResult, ToolSpec, TypedToolset, episode_tool


class _Amount(BaseModel):
    amount: int


class _Total(BaseModel):
    total: int


class _Counter:
    def __init__(self) -> None:
        self.total = 0

    @episode_tool
    async def add(self, arguments: _Amount) -> _Total:
        """Add an amount to the total."""
        self.total += arguments.amount
        return _Total(total=self.total)

    @episode_tool
    async def overdraw(self, arguments: _Amount) -> ToolResult:
        """Refuse to go below zero."""
        return ToolResult(output="The total cannot go below zero", is_error=True)

    async def reset(self, arguments: _Amount) -> None:
        """Not a tool: it has no decorator."""
        self.total = 0


def test_the_tools_are_the_decorated_methods_with_their_docstrings_and_schemas() -> None:
    specs = {spec.name: spec for spec in TypedToolset(_Counter()).specs()}

    assert sorted(specs) == ["add", "overdraw"]
    assert specs["add"].description == "Add an amount to the total."
    assert specs["add"].parameters == _Amount.model_json_schema()


def test_a_call_runs_on_the_owner_and_returns_its_result_as_json() -> None:
    counter = _Counter()
    toolset = TypedToolset(counter)

    first = asyncio.run(toolset.call("add", {"amount": 2}))
    second = asyncio.run(toolset.call("add", {"amount": 3}))

    assert (first.output, second.output, counter.total) == ({"total": 2}, {"total": 5}, 5)
    assert not second.is_error


def test_a_tool_returns_its_own_error_result_unchanged() -> None:
    result = asyncio.run(TypedToolset(_Counter()).call("overdraw", {"amount": 1}))

    assert result == ToolResult(output="The total cannot go below zero", is_error=True)


def test_arguments_the_model_rejects_raise_an_error_the_agent_sees() -> None:
    with pytest.raises(ToolArgumentsError, match="amount"):
        asyncio.run(TypedToolset(_Counter()).call("add", {"amount": "two"}))


def test_a_tool_without_a_docstring_is_refused() -> None:
    class _Undocumented:
        @episode_tool
        async def add(self, arguments: _Amount) -> int:
            return arguments.amount

    with pytest.raises(TypeError, match="docstring"):
        TypedToolset(_Undocumented())


@pytest.mark.parametrize("signature", ["no_arguments", "two_arguments", "not_a_model", "keyword_only"])
def test_a_tool_must_take_one_model_of_arguments(signature: str) -> None:
    """The server passes the arguments model positionally, so a keyword-only parameter is refused when built."""

    async def no_arguments(self) -> None:
        """Doc."""

    async def keyword_only(self, *, arguments: _Amount) -> None:
        """Doc."""

    async def two_arguments(self, first: _Amount, second: _Amount) -> None:
        """Doc."""

    async def not_a_model(self, arguments: dict) -> None:
        """Doc."""

    owner_type = type("_Owner", (), {"tool": episode_tool(locals()[signature])})

    with pytest.raises(TypeError, match="one positional Pydantic model"):
        TypedToolset(owner_type())


def test_a_tool_must_be_async() -> None:
    def add(self, arguments: _Amount) -> int:
        """Doc."""
        return arguments.amount

    with pytest.raises(TypeError, match="must be an async method"):
        episode_tool(add)


def test_a_tool_name_must_be_one_path_segment() -> None:
    with pytest.raises(ValidationError):
        ToolSpec(name="a/b", description="d", parameters={"type": "object"})


@pytest.mark.parametrize("parameters", [{}, {"type": "string"}, {"properties": {}}])
def test_a_tools_parameters_must_be_an_object_schema(parameters: dict) -> None:
    with pytest.raises(ValidationError, match="type"):
        ToolSpec(name="add", description="d", parameters=parameters)


class _Line(BaseModel):
    start: _Amount
    end: _Amount


def test_a_specs_declaration_keeps_the_nested_schema() -> None:
    spec = ToolSpec(name="draw", description="Draw a line.", parameters=_Line.model_json_schema())

    declaration = spec.function_tool()

    assert declaration == {
        "type": "function",
        "name": "draw",
        "description": "Draw a line.",
        "parameters": _Line.model_json_schema(),
        "strict": False,
    }
    assert "_Amount" in declaration["parameters"]["$defs"]


def test_each_class_builds_its_tools_once() -> None:
    """Every episode of a server builds a toolset; the specs come from the class, not the episode."""
    first, second = TypedToolset(_Counter()), TypedToolset(_Counter())

    assert [spec is other for spec, other in zip(first.specs(), second.specs())] == [True, True]
