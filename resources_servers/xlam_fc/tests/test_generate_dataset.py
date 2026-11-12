# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from generate_dataset import convert_parameter_type, convert_tool_to_openai_format, parameter_schema


# xlam type strings are Python-style annotations with free-text qualifiers. The declared JSON-schema
# type must agree with the native JSON values the dataset keeps in expected_answers, or a model that
# follows the schema can never score 1 (NVBug 6900453).
#
# All 30 distinct annotation strings in Salesforce/xlam-function-calling-60k (60,000 rows), most frequent first.
DATASET_ANNOTATIONS = [
    ("str", {"type": "string"}),
    ("str, optional", {"type": "string"}),
    ("int", {"type": "integer"}),
    ("int, optional", {"type": "integer"}),
    ("float", {"type": "number"}),
    ("List[int]", {"type": "array", "items": {"type": "integer"}}),
    ("List[float]", {"type": "array", "items": {"type": "number"}}),
    ("bool, optional", {"type": "boolean"}),
    ("float, optional", {"type": "number"}),
    ("list", {"type": "array"}),
    ("List[List[int]]", {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}}),
    ("List", {"type": "array"}),
    ("List[Union[int, float]]", {"type": "array"}),
    ("Tuple[float, float]", {"type": "array", "items": {"type": "number"}}),
    ("List[str]", {"type": "array", "items": {"type": "string"}}),
    ("Dict", {"type": "object"}),
    ("List[List[float]]", {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}),
    ("Callable[[float], float]", {}),
    ("set", {"type": "array"}),
    ("List[Tuple[float, float]]", {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}),
    ("List[List[str]]", {"type": "array", "items": {"type": "array", "items": {"type": "string"}}}),
    ("List[Tuple[int, int]]", {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}}),
    ("bool", {"type": "boolean"}),
    ("str, default='elonmusk'", {"type": "string"}),
    ("str, optional, default 'London'", {"type": "string"}),
    ("str, optional, default='20'", {"type": "string"}),
    ("str, optional, default='fr-FR'", {"type": "string"}),
    ("int, optional, default=100", {"type": "integer"}),
    ("str, optional, default='0'", {"type": "string"}),
    ("str, optional, default='25'", {"type": "string"}),
]


@pytest.mark.parametrize(("xlam_type", "schema"), DATASET_ANNOTATIONS)
def test_parameter_schema_follows_every_annotation_in_the_dataset(xlam_type: str, schema: dict) -> None:
    assert parameter_schema(xlam_type) == schema


def test_the_dataset_annotation_list_is_complete() -> None:
    assert len(DATASET_ANNOTATIONS) == 30
    assert len({xlam_type for xlam_type, _ in DATASET_ANNOTATIONS}) == 30


# Annotations the dataset does not use but the parser must still handle: separators alone (","), which the
# first version of the parser raised IndexError on, and the Optional / Union / lowercase-generic spellings.
@pytest.mark.parametrize(
    ("xlam_type", "schema"),
    [
        ("", {}),
        (",", {}),
        (" , ", {}),
        ("Any", {}),
        ("Optional[int]", {"type": "integer"}),
        ("int or None", {"type": "integer"}),
        ("Optional[List[int]]", {"type": "array", "items": {"type": "integer"}}),
        ("Union[int, str]", {}),
        ("Union[int, int]", {"type": "integer"}),
        ("Tuple[int, str]", {"type": "array"}),
        ("list[str]", {"type": "array", "items": {"type": "string"}}),
        ("Set[str]", {"type": "array", "items": {"type": "string"}}),
        ("dict", {"type": "object"}),
        ("Dict[str, int]", {"type": "object"}),
        ("str, optional, default='json'", {"type": "string"}),
    ],
)
def test_parameter_schema_handles_annotations_outside_the_dataset(xlam_type: str, schema: dict) -> None:
    assert parameter_schema(xlam_type) == schema


@pytest.mark.parametrize(
    ("xlam_type", "json_type"),
    [("int", "integer"), ("int, optional", "integer"), ("List[int]", "array"), ("Union[int, str]", None)],
)
def test_convert_parameter_type_returns_the_top_level_json_type(xlam_type: str, json_type) -> None:
    assert convert_parameter_type(xlam_type) == json_type


def test_tool_conversion_keeps_defaults_and_required_and_types_compound_parameters() -> None:
    tool = {
        "name": "tema",
        "description": "Triple EMA",
        "parameters": {
            "symbol": {"description": "Ticker", "type": "str"},
            "outputsize": {"description": "Points", "type": "int, optional", "default": 30},
            "nums": {"description": "Numbers", "type": "List[int]"},
            "anything": {"description": "Unknown", "type": "Union[int, str]"},
        },
    }
    converted = convert_tool_to_openai_format(tool)
    properties = converted["parameters"]["properties"]
    assert properties["symbol"] == {"type": "string", "description": "Ticker"}
    assert properties["outputsize"] == {"type": "integer", "description": "Points", "default": 30}
    assert properties["nums"] == {"type": "array", "items": {"type": "integer"}, "description": "Numbers"}
    # An unparseable type declares no type at all instead of silently claiming "string".
    assert properties["anything"] == {"description": "Unknown"}
    assert converted["parameters"]["required"] == ["symbol", "nums", "anything"]
    assert converted["strict"] is False
