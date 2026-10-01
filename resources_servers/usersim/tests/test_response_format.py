# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from resources_servers.usersim.response_format import responses_json_schema


def _assert_strict_objects(node: Any) -> None:
    if isinstance(node, dict):
        assert "default" not in node
        properties = node.get("properties")
        if node.get("type") == "object" and isinstance(properties, dict):
            assert node["additionalProperties"] is False
            assert node["required"] == list(properties)
        for value in node.values():
            _assert_strict_objects(value)
    elif isinstance(node, list):
        for value in node:
            _assert_strict_objects(value)


@pytest.mark.parametrize("scorer", ["safety_agentic", "tool_use"])
def test_pinned_usersim_native_scorer_schema_is_strict_responses_compatible(scorer: str) -> None:
    if scorer == "safety_agentic":
        module = pytest.importorskip("usersim.engine.evaluator.scorers.safety_agentic")
        schema = module._AgenticJudgment.model_json_schema()
    else:
        module = pytest.importorskip("usersim.engine.evaluator.scorers.tool_use")
        schema = module._build_schema().model_json_schema()

    _assert_strict_objects(responses_json_schema(schema, strict=True))


def test_strict_schema_recursively_closes_native_tool_scorer_objects() -> None:
    original = {
        "$defs": {
            "axis": {
                "type": "object",
                "properties": {
                    "score": {"enum": [1, 3, 5], "type": "integer"},
                    "reasoning": {"type": "string"},
                },
                "required": ["reasoning", "score"],
            }
        },
        "type": "object",
        "properties": {"overall": {"$ref": "#/$defs/axis"}},
        "required": ["overall"],
    }

    adapted = responses_json_schema(original, strict=True)

    assert adapted["additionalProperties"] is False
    assert adapted["$defs"]["axis"]["additionalProperties"] is False
    assert adapted["$defs"]["axis"]["required"] == ["score", "reasoning"]
    assert "additionalProperties" not in original
    assert "additionalProperties" not in original["$defs"]["axis"]


def test_strict_schema_wraps_ref_siblings_without_losing_descriptions() -> None:
    original = {
        "$defs": {"score": {"enum": [1, 3, 5], "type": "integer"}},
        "type": "object",
        "properties": {
            "score": {
                "$ref": "#/$defs/score",
                "description": "Native scorer guidance",
            }
        },
    }

    adapted = responses_json_schema(original, strict=True)

    score = adapted["properties"]["score"]
    assert score == {
        "anyOf": [{"$ref": "#/$defs/score"}],
        "description": "Native scorer guidance",
    }
    assert original["properties"]["score"]["$ref"] == "#/$defs/score"


def test_non_strict_schema_keeps_native_shape() -> None:
    original = {
        "type": "object",
        "properties": {"optional": {"type": "string", "default": "value"}},
    }

    adapted = responses_json_schema(original, strict=False)

    assert adapted == original
    assert adapted is not original
