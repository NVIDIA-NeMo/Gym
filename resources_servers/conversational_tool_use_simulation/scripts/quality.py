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

"""Validation helpers for materializing conversational tool-use datasets."""

from typing import Any

import jsonschema.validators
from jsonschema.exceptions import SchemaError


SCHEMA_MARKERS = {"type", "properties", "required", "items", "enum", "anyOf", "oneOf", "allOf", "$ref", "title"}
DSML_LEAK_MARKERS = (
    "DSML",
    "\\uff5cDSML\\uff5c",
    "｜DSML｜",
    "<｜DSML｜function_calls",
    "</｜DSML｜function_calls>",
    "<｜DSML｜invoke",
    "</｜DSML｜invoke>",
    "<｜DSML｜parameter",
    "</｜DSML｜parameter>",
)
SPECIAL_TOKEN_LEAK_MARKERS = (
    "<｜begin▁of▁sentence｜>",
    "<｜end▁of▁sentence｜>",
    "<｜User｜>",
    "<｜Assistant｜>",
    "<think>",
    "</think>",
    "<function_results>",
    "</function_results>",
    "<result>",
    "</result>",
)


class ArtifactValidationError(ValueError):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def detect_leak(text: str) -> str | None:
    if any(marker in text for marker in DSML_LEAK_MARKERS):
        return "dsml_leak"
    if any(marker in text for marker in SPECIAL_TOKEN_LEAK_MARKERS):
        return "special_token_leak"
    return None


def validate_tool_schema(value: Any, *, tool_name: str, field_name: str) -> None:
    reason = f"invalid_tool_{field_name}_schema"
    if not isinstance(value, dict):
        raise ArtifactValidationError(reason, f"tool {tool_name} has non-dict {field_name} schema")
    if not value or not SCHEMA_MARKERS.intersection(value):
        raise ArtifactValidationError(reason, f"tool {tool_name} has non-schema {field_name}")
    try:
        validator_class = jsonschema.validators.validator_for(value)
        validator_class.check_schema(value)
    except SchemaError as exc:
        raise ArtifactValidationError(
            reason,
            f"tool {tool_name} has invalid {field_name} JSON Schema at {exc.json_path}: {exc.message}",
        ) from exc
