# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import copy
import importlib
import inspect
import json
import textwrap
from contextlib import suppress

import pytest
from pydantic import ValidationError

from responses_api_agents.apex_agent import stirrup_runtime


# The exact schema shape Archipelago MCP tools present: the property site is a
# bare $ref with no inline "type", which GLM-family vLLM parsers mistype.
ARCHIPELAGO_SCHEMA = {
    "type": "object",
    "properties": {"request": {"$ref": "#/$defs/DynamicModel"}},
    "required": ["request"],
    "$defs": {
        "DynamicModel": {
            "type": "object",
            "title": "DynamicModel",
            "properties": {"code": {"type": "string", "title": "Code"}},
            "required": ["code"],
        }
    },
}


def test_archipelago_bare_ref_property_is_inlined_and_defs_dropped() -> None:
    result = stirrup_runtime.inline_schema_refs(ARCHIPELAGO_SCHEMA)

    request = result["properties"]["request"]
    assert request["type"] == "object"
    assert "code" in request["properties"]
    assert "$defs" not in result
    assert "$ref" not in json.dumps(result)


def test_ref_inside_anyof_is_inlined() -> None:
    schema = {
        "type": "object",
        "properties": {"request": {"anyOf": [{"$ref": "#/$defs/DynamicModel"}, {"type": "null"}], "default": None}},
        "$defs": {"DynamicModel": {"type": "object", "properties": {"code": {"type": "string"}}}},
    }

    result = stirrup_runtime.inline_schema_refs(schema)

    variants = result["properties"]["request"]["anyOf"]
    assert variants[0]["type"] == "object"
    assert "code" in variants[0]["properties"]
    assert variants[1] == {"type": "null"}
    assert "$defs" not in result


def test_nested_refs_are_fully_inlined() -> None:
    schema = {
        "type": "object",
        "properties": {"a": {"$ref": "#/$defs/A"}},
        "$defs": {
            "A": {"type": "object", "properties": {"b": {"$ref": "#/$defs/B"}}},
            "B": {"type": "object", "properties": {"leaf": {"type": "integer"}}},
        },
    }

    result = stirrup_runtime.inline_schema_refs(schema)

    inner = result["properties"]["a"]["properties"]["b"]
    assert inner["type"] == "object"
    assert inner["properties"]["leaf"] == {"type": "integer"}
    assert "$ref" not in json.dumps(result)
    assert "$defs" not in result


def test_self_referential_schema_keeps_remaining_ref_and_defs() -> None:
    schema = {
        "type": "object",
        "properties": {"tree": {"$ref": "#/$defs/Node"}},
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {"children": {"type": "array", "items": {"$ref": "#/$defs/Node"}}},
            }
        },
    }

    result = stirrup_runtime.inline_schema_refs(schema)

    tree = result["properties"]["tree"]
    assert tree["type"] == "object"
    assert tree["properties"]["children"]["items"] == {"$ref": "#/$defs/Node"}
    assert "$defs" in result


def test_sibling_keys_are_preserved_and_win_on_conflict() -> None:
    schema = {
        "type": "object",
        "properties": {"request": {"$ref": "#/$defs/D", "description": "site description"}},
        "$defs": {"D": {"type": "object", "description": "definition description", "properties": {}}},
    }

    result = stirrup_runtime.inline_schema_refs(schema)

    request = result["properties"]["request"]
    assert request["type"] == "object"
    assert request["properties"] == {}
    assert request["description"] == "site description"


def test_ref_free_schema_is_structurally_equal() -> None:
    schema = {
        "type": "object",
        "properties": {"code": {"type": "string"}, "flags": {"type": "array", "items": {"type": "string"}}},
        "required": ["code"],
    }

    assert stirrup_runtime.inline_schema_refs(schema) == schema


def test_input_schema_is_never_mutated() -> None:
    snapshot = copy.deepcopy(ARCHIPELAGO_SCHEMA)

    stirrup_runtime.inline_schema_refs(ARCHIPELAGO_SCHEMA)

    assert ARCHIPELAGO_SCHEMA == snapshot


def test_multiple_inline_sites_of_one_definition_are_independent_copies() -> None:
    schema = {
        "type": "object",
        "properties": {"first": {"$ref": "#/$defs/D"}, "second": {"$ref": "#/$defs/D"}},
        "$defs": {"D": {"type": "object", "properties": {"code": {"type": "string"}}}},
    }

    result = stirrup_runtime.inline_schema_refs(schema)

    result["properties"]["first"]["properties"]["code"]["type"] = "integer"
    assert result["properties"]["second"]["properties"]["code"] == {"type": "string"}
    # The source definition table is untouched as well.
    assert schema["$defs"]["D"]["properties"]["code"] == {"type": "string"}


def test_exponential_ref_chain_is_capped_by_the_expansion_budget() -> None:
    # A doubling chain (each def referencing the previous twice) is acyclic but
    # multiplies inline copies exponentially; the node budget must bail out and
    # return the original schema instead of stalling every completion request.
    depth = 24
    defs = {"D0": {"type": "object", "properties": {"leaf": {"type": "string"}}}}
    for level in range(1, depth + 1):
        previous = f"#/$defs/D{level - 1}"
        defs[f"D{level}"] = {
            "type": "object",
            "properties": {"left": {"$ref": previous}, "right": {"$ref": previous}},
        }
    schema = {"type": "object", "properties": {"root": {"$ref": f"#/$defs/D{depth}"}}, "$defs": defs}

    assert stirrup_runtime.inline_schema_refs(schema) == schema


def test_garbage_input_returns_original_and_never_raises() -> None:
    assert stirrup_runtime.inline_schema_refs(None) is None
    assert stirrup_runtime.inline_schema_refs("not a schema") == "not a schema"
    assert stirrup_runtime.inline_schema_refs([1, 2]) == [1, 2]

    dangling = {"type": "object", "properties": {"request": {"$ref": "#/$defs/missing"}}, "$defs": {}}
    result = stirrup_runtime.inline_schema_refs(dangling)
    assert result["properties"]["request"] == {"$ref": "#/$defs/missing"}
    assert "$defs" in result

    non_local = {"type": "object", "properties": {"request": {"$ref": "https://example.com/schema.json"}}}
    assert stirrup_runtime.inline_schema_refs(non_local) == non_local

    hostile = {"$defs": "not a table", "properties": {"request": {"$ref": 42}}}
    assert stirrup_runtime.inline_schema_refs(hostile)["properties"] == hostile["properties"]


def test_inlined_schema_is_semantically_equivalent_on_the_generated_model() -> None:
    j2p = pytest.importorskip("json_schema_to_pydantic")

    model = j2p.create_model(ARCHIPELAGO_SCHEMA)
    generated = model.model_json_schema()
    # The generated schema reproduces the bug shape: no inline type at the property site.
    assert generated["properties"]["request"].get("type") is None
    inlined = stirrup_runtime.inline_schema_refs(generated)

    valid_instance = {"request": {"code": "ls"}}
    invalid_instance = {"request": "str"}
    try:
        import jsonschema
    except ImportError:
        jsonschema = None
    if jsonschema is not None:
        for candidate in (generated, inlined):
            validator = jsonschema.Draft202012Validator(candidate)
            assert not list(validator.iter_errors(valid_instance))
            assert list(validator.iter_errors(invalid_instance))
    else:  # pragma: no cover - jsonschema is installed in the dev venv
        assert inlined["properties"]["request"]["type"] == "object"
        assert "code" in inlined["properties"]["request"]["properties"]
    # The source model agrees with both schemas on the same instances.
    model.model_validate(valid_instance)
    with pytest.raises(ValidationError):
        model.model_validate(invalid_instance)


def _stirrup_client_modules() -> list:
    modules = []
    for name in ("stirrup.clients.utils", "stirrup.clients.chat_completions_client", "stirrup.clients.litellm_client"):
        with suppress(Exception):
            modules.append(importlib.import_module(name))
    return modules


def test_installed_patch_inlines_the_wire_schema_in_the_caller_namespace() -> None:
    pytest.importorskip("stirrup")
    j2p = pytest.importorskip("json_schema_to_pydantic")
    import stirrup.clients.chat_completions_client as chat_completions_client_module
    from stirrup.core.models import Tool, ToolResult, ToolUseCountMetadata

    wrapper_model = j2p.create_model(ARCHIPELAGO_SCHEMA)

    async def executor(params) -> ToolResult[ToolUseCountMetadata]:  # pragma: no cover - never called
        return ToolResult(content="ok", metadata=ToolUseCountMetadata())

    tool = Tool(name="execute_code", description="Run code.", parameters=wrapper_model, executor=executor)

    originals = [(module, module.to_openai_tools) for module in _stirrup_client_modules()]
    try:
        stirrup_runtime.install_tool_schema_inlining()
        # ChatCompletionsClient calls the binding in ITS OWN namespace (from-import),
        # so that binding is the one that must produce the inlined wire payload.
        payload = chat_completions_client_module.to_openai_tools({"execute_code": tool})
        parameters = payload[0]["function"]["parameters"]
        assert parameters["properties"]["request"]["type"] == "object"
        assert "code" in parameters["properties"]["request"]["properties"]
        assert "$ref" not in json.dumps(payload)
    finally:
        for module, original in originals:
            module.to_openai_tools = original


def test_double_install_leaves_the_same_wrapper_bound() -> None:
    pytest.importorskip("stirrup")
    import stirrup.clients.chat_completions_client as chat_completions_client_module
    import stirrup.clients.utils as stirrup_client_utils

    originals = [(module, module.to_openai_tools) for module in _stirrup_client_modules()]
    try:
        stirrup_runtime.install_tool_schema_inlining()
        wrapper = stirrup_client_utils.to_openai_tools
        assert getattr(wrapper, "_apex_schema_inline_patch", False)
        stirrup_runtime.install_tool_schema_inlining()
        assert stirrup_client_utils.to_openai_tools is wrapper
        assert chat_completions_client_module.to_openai_tools is wrapper
    finally:
        for module, original in originals:
            module.to_openai_tools = original


def _call_linenos(func: object, callee: str) -> list[int]:
    """Line numbers of real (non-commented) `callee(...)` call nodes inside func."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == callee
    ]


def test_run_stirrup_rollout_installs_the_schema_inlining_patch() -> None:
    # The sandbox runs stirrup_runtime.py standalone; the fix is inert unless
    # run_stirrup_rollout installs the patch before constructing the Agent.
    # AST-based so a commented-out call cannot satisfy the assertion.
    install_calls = _call_linenos(stirrup_runtime.run_stirrup_rollout, "install_tool_schema_inlining")
    agent_calls = _call_linenos(stirrup_runtime.run_stirrup_rollout, "Agent")
    assert install_calls
    assert agent_calls
    assert min(install_calls) < min(agent_calls)


def test_inspect_tool_shows_the_dereferenced_schema() -> None:
    # inspect_tool is what the model reads before emitting arguments; it must
    # present the same dereferenced shape that goes over the wire. AST-based so
    # a commented-out call cannot satisfy the assertion.
    tree = ast.parse(textwrap.dedent(inspect.getsource(stirrup_runtime.run_stirrup_rollout)))
    inline_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "inline_schema_refs"
    ]
    assert any(
        isinstance(call.args[0], ast.Call)
        and isinstance(call.args[0].func, ast.Attribute)
        and call.args[0].func.attr == "model_json_schema"
        for call in inline_calls
        if call.args
    )
