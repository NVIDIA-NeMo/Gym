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


def test_archipelago_bare_ref_property_gains_type_and_keeps_ref_and_defs() -> None:
    result = stirrup_runtime.annotate_schema_ref_types(ARCHIPELAGO_SCHEMA)

    request = result["properties"]["request"]
    assert request["$ref"] == "#/$defs/DynamicModel"
    assert request["type"] == "object"
    assert result["$defs"] == ARCHIPELAGO_SCHEMA["$defs"]


def test_ref_inside_anyof_gains_type() -> None:
    schema = {
        "type": "object",
        "properties": {"request": {"anyOf": [{"$ref": "#/$defs/DynamicModel"}, {"type": "null"}], "default": None}},
        "$defs": {"DynamicModel": {"type": "object", "properties": {"code": {"type": "string"}}}},
    }

    result = stirrup_runtime.annotate_schema_ref_types(schema)

    variants = result["properties"]["request"]["anyOf"]
    assert variants[0] == {"$ref": "#/$defs/DynamicModel", "type": "object"}
    assert variants[1] == {"type": "null"}
    # The property SITE stays untyped for anyOf shapes (there is no single type
    # to annotate it with); install_tool_argument_coercion covers that shape.
    assert "type" not in result["properties"]["request"]
    assert result["$defs"] == schema["$defs"]


def test_bare_ref_chain_resolves_to_the_final_definition_type() -> None:
    schema = {
        "type": "object",
        "properties": {"request": {"$ref": "#/$defs/A"}},
        "$defs": {"A": {"$ref": "#/$defs/B"}, "B": {"type": "object", "properties": {}}},
    }

    result = stirrup_runtime.annotate_schema_ref_types(schema)

    request = result["properties"]["request"]
    assert request["$ref"] == "#/$defs/A"
    assert request["type"] == "object"


def test_bare_ref_chain_cycle_terminates_without_annotation() -> None:
    schema = {
        "type": "object",
        "properties": {"request": {"$ref": "#/$defs/A"}},
        "$defs": {"A": {"$ref": "#/$defs/B"}, "B": {"$ref": "#/$defs/A"}},
    }

    assert stirrup_runtime.annotate_schema_ref_types(schema) == schema


def test_ref_sites_inside_the_defs_table_are_annotated_and_tables_preserved() -> None:
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

    result = stirrup_runtime.annotate_schema_ref_types(schema)

    assert result["properties"]["tree"] == {"$ref": "#/$defs/Node", "type": "object"}
    items = result["$defs"]["Node"]["properties"]["children"]["items"]
    assert items == {"$ref": "#/$defs/Node", "type": "object"}
    assert result["$defs"]["Node"]["type"] == "object"


def test_conflicting_sibling_constraints_stay_conjunctive() -> None:
    # The regression case that rules out full inlining: under JSON Schema
    # 2020-12, sibling keys next to $ref apply CONJUNCTIVELY with the resolved
    # definition, so an instance must satisfy the def's maxLength 1 AND the
    # site's maxLength 3. Merging the def into the site (siblings winning)
    # would loosen the schema to maxLength 3 alone.
    jsonschema = pytest.importorskip("jsonschema")
    schema = {
        "type": "object",
        "properties": {"request": {"$ref": "#/$defs/Short", "maxLength": 3}},
        "$defs": {"Short": {"type": "string", "maxLength": 1}},
    }

    result = stirrup_runtime.annotate_schema_ref_types(schema)

    request = result["properties"]["request"]
    assert request == {"$ref": "#/$defs/Short", "maxLength": 3, "type": "string"}
    assert result["$defs"] == schema["$defs"]
    for candidate in (schema, result):
        validator = jsonschema.Draft202012Validator(candidate)
        assert not list(validator.iter_errors({"request": "a"}))
        assert list(validator.iter_errors({"request": "ab"}))


def test_site_with_its_own_type_is_left_untouched() -> None:
    schema = {
        "type": "object",
        "properties": {"request": {"$ref": "#/$defs/D", "type": "integer"}},
        "$defs": {"D": {"type": "object", "properties": {}}},
    }

    result = stirrup_runtime.annotate_schema_ref_types(schema)

    assert result["properties"]["request"] == {"$ref": "#/$defs/D", "type": "integer"}


def test_only_the_type_is_copied_from_the_definition() -> None:
    schema = {
        "type": "object",
        "properties": {"request": {"$ref": "#/$defs/D"}},
        "$defs": {"D": {"type": "string", "maxLength": 5, "description": "definition description"}},
    }

    result = stirrup_runtime.annotate_schema_ref_types(schema)

    assert result["properties"]["request"] == {"$ref": "#/$defs/D", "type": "string"}


def test_data_position_dicts_are_never_annotated() -> None:
    # Values of const/enum/default/examples are instance DATA; a data dict that
    # happens to carry a "$ref" key must pass through byte-identical.
    data_value = {"$ref": "#/$defs/A"}
    schema = {
        "$defs": {"A": {"type": "object"}},
        "properties": {
            "c": {"const": {"$ref": "#/$defs/A"}},
            "e": {"enum": [{"$ref": "#/$defs/A"}, "plain"]},
            "d": {"type": "object", "default": {"$ref": "#/$defs/A"}},
            "x": {"examples": [{"$ref": "#/$defs/A"}]},
        },
    }

    result = stirrup_runtime.annotate_schema_ref_types(schema)

    assert result["properties"]["c"]["const"] == data_value
    assert result["properties"]["e"]["enum"][0] == data_value
    assert result["properties"]["d"]["default"] == data_value
    assert result["properties"]["x"]["examples"][0] == data_value


def test_definition_names_are_pointer_escaped() -> None:
    # A def literally named "a/b" must not shadow the pointer path $defs->a->b;
    # per RFC 6901 it is addressed as "#/$defs/a~1b".
    schema = {
        "properties": {
            "escaped": {"$ref": "#/$defs/a~1b"},
            "path": {"$ref": "#/$defs/a/b"},
        },
        "$defs": {"a/b": {"type": "string"}, "a": {"type": "object"}},
    }

    result = stirrup_runtime.annotate_schema_ref_types(schema)

    assert result["properties"]["escaped"]["type"] == "string"
    # The slash form is a nested pointer path this resolver does not model;
    # it must pass through un-annotated rather than copy the wrong type.
    assert "type" not in result["properties"]["path"]


def test_ref_free_schema_is_structurally_equal() -> None:
    schema = {
        "type": "object",
        "properties": {"code": {"type": "string"}, "flags": {"type": "array", "items": {"type": "string"}}},
        "required": ["code"],
    }

    assert stirrup_runtime.annotate_schema_ref_types(schema) == schema


def test_input_is_never_mutated_and_output_shares_no_mutable_state() -> None:
    snapshot = copy.deepcopy(ARCHIPELAGO_SCHEMA)

    result = stirrup_runtime.annotate_schema_ref_types(ARCHIPELAGO_SCHEMA)

    assert ARCHIPELAGO_SCHEMA == snapshot
    result["properties"]["request"]["type"] = "integer"
    result["$defs"]["DynamicModel"]["properties"]["code"]["type"] = "integer"
    result["required"].append("mutated")
    assert ARCHIPELAGO_SCHEMA == snapshot


def test_garbage_input_returns_original_and_never_raises() -> None:
    assert stirrup_runtime.annotate_schema_ref_types(None) is None
    assert stirrup_runtime.annotate_schema_ref_types("not a schema") == "not a schema"
    assert stirrup_runtime.annotate_schema_ref_types([1, 2]) == [1, 2]

    dangling = {"type": "object", "properties": {"request": {"$ref": "#/$defs/missing"}}, "$defs": {}}
    assert stirrup_runtime.annotate_schema_ref_types(dangling) == dangling

    non_local = {"type": "object", "properties": {"request": {"$ref": "https://example.com/schema.json"}}}
    assert stirrup_runtime.annotate_schema_ref_types(non_local) == non_local

    hostile = {"$defs": "not a table", "properties": {"request": {"$ref": 42}}}
    assert stirrup_runtime.annotate_schema_ref_types(hostile) == hostile


def test_annotated_schema_is_semantically_equivalent_on_the_generated_model() -> None:
    j2p = pytest.importorskip("json_schema_to_pydantic")
    jsonschema = pytest.importorskip("jsonschema")

    model = j2p.create_model(ARCHIPELAGO_SCHEMA)
    generated = model.model_json_schema()
    # The generated schema reproduces the bug shape: no inline type at the property site.
    assert generated["properties"]["request"].get("type") is None
    annotated = stirrup_runtime.annotate_schema_ref_types(generated)
    assert annotated["properties"]["request"]["type"] == "object"

    valid_instance = {"request": {"code": "ls"}}
    invalid_instance = {"request": "str"}
    for candidate in (generated, annotated):
        validator = jsonschema.Draft202012Validator(candidate)
        assert not list(validator.iter_errors(valid_instance))
        assert list(validator.iter_errors(invalid_instance))
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


def test_installed_patch_annotates_the_wire_schema_in_the_caller_namespace() -> None:
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
        stirrup_runtime.install_tool_schema_type_annotation()
        # ChatCompletionsClient calls the binding in ITS OWN namespace (from-import),
        # so that binding is the one that must produce the annotated wire payload.
        payload = chat_completions_client_module.to_openai_tools({"execute_code": tool})
        parameters = payload[0]["function"]["parameters"]
        request = parameters["properties"]["request"]
        assert request["type"] == "object"
        assert "$ref" in request
        assert "$defs" in json.dumps(parameters)
    finally:
        for module, original in originals:
            module.to_openai_tools = original


def test_double_install_leaves_the_same_wrapper_bound() -> None:
    pytest.importorskip("stirrup")
    import stirrup.clients.chat_completions_client as chat_completions_client_module
    import stirrup.clients.utils as stirrup_client_utils

    originals = [(module, module.to_openai_tools) for module in _stirrup_client_modules()]
    try:
        stirrup_runtime.install_tool_schema_type_annotation()
        wrapper = stirrup_client_utils.to_openai_tools
        assert getattr(wrapper, "_apex_schema_type_patch", False)
        stirrup_runtime.install_tool_schema_type_annotation()
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


def test_run_stirrup_rollout_installs_the_schema_type_annotation_patch() -> None:
    # The sandbox runs stirrup_runtime.py standalone; the fix is inert unless
    # run_stirrup_rollout installs the patch before constructing the Agent.
    # AST-based so a commented-out call cannot satisfy the assertion.
    install_calls = _call_linenos(stirrup_runtime.run_stirrup_rollout, "install_tool_schema_type_annotation")
    agent_calls = _call_linenos(stirrup_runtime.run_stirrup_rollout, "Agent")
    assert install_calls
    assert agent_calls
    assert min(install_calls) < min(agent_calls)


def test_inspect_tool_shows_the_annotated_schema() -> None:
    # inspect_tool is what the model reads before emitting arguments; it must
    # present the same annotated shape that goes over the wire. AST-based so
    # a commented-out call cannot satisfy the assertion.
    tree = ast.parse(textwrap.dedent(inspect.getsource(stirrup_runtime.run_stirrup_rollout)))
    annotate_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "annotate_schema_ref_types"
    ]
    assert any(
        isinstance(call.args[0], ast.Call)
        and isinstance(call.args[0].func, ast.Attribute)
        and call.args[0].func.attr == "model_json_schema"
        for call in annotate_calls
        if call.args
    )
