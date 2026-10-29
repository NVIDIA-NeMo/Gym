# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
from generate_dataset import convert_parameter_type, convert_tool_to_openai_format, parameter_schema


# xlam type strings are Python-style annotations with free-text qualifiers. The declared JSON-schema
# type must agree with the native JSON values the dataset keeps in expected_answers, or a model that
# follows the schema can never score 1 (NVBug 6900453).
@pytest.mark.parametrize(
    ("xlam_type", "schema"),
    [
        ("int", {"type": "integer"}),
        ("str", {"type": "string"}),
        ("int, optional", {"type": "integer"}),
        ("float, optional", {"type": "number"}),
        ("bool, optional", {"type": "boolean"}),
        ("str, default='x'", {"type": "string"}),
        ("str, optional, default='json'", {"type": "string"}),
        ("List[int]", {"type": "array", "items": {"type": "integer"}}),
        ("List[float]", {"type": "array", "items": {"type": "number"}}),
        ("list[str]", {"type": "array", "items": {"type": "string"}}),
        ("Tuple[float, float]", {"type": "array", "items": {"type": "number"}}),
        ("Tuple[int, str]", {"type": "array"}),
        ("List[List[int]]", {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}}),
        ("List[Union[int, float]]", {"type": "array"}),
        ("Set[str]", {"type": "array", "items": {"type": "string"}}),
        ("Dict[str, int]", {"type": "object"}),
        ("dict", {"type": "object"}),
        ("Optional[int]", {"type": "integer"}),
        ("int or None", {"type": "integer"}),
        ("Optional[List[int]]", {"type": "array", "items": {"type": "integer"}}),
        ("Union[int, str]", {}),
        ("Union[int, int]", {"type": "integer"}),
        ("Any", {}),
        ("", {}),
    ],
)
def test_parameter_schema_follows_the_xlam_type_string(xlam_type: str, schema: dict) -> None:
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
