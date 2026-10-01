# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapt native UserSim JSON schemas to strict Responses API schemas."""

from collections.abc import Mapping
from copy import deepcopy
from typing import Any


def responses_json_schema(schema: Mapping[str, Any], *, strict: bool) -> dict[str, Any]:
    """Return a defensive schema copy accepted by strict Responses decoders.

    Strict Responses schemas require every object property to be present and
    reject undeclared properties. Nullable branches preserve UserSim's optional
    value semantics while required fields make the wire shape deterministic.
    Pydantic ``default`` annotations do not affect validation and are omitted
    because strict decoders reject that unsupported keyword.
    """
    adapted = deepcopy(dict(schema))
    if strict:
        _make_strict(adapted)
    return adapted


def _make_strict(node: Any) -> None:
    if isinstance(node, dict):
        node.pop("default", None)
        if "$ref" in node and len(node) > 1:
            reference = node.pop("$ref")
            node["anyOf"] = [{"$ref": reference}]
        properties = node.get("properties")
        if node.get("type") == "object" and isinstance(properties, dict):
            node["additionalProperties"] = False
            node["required"] = list(properties)
        for value in node.values():
            _make_strict(value)
    elif isinstance(node, list):
        for value in node:
            _make_strict(value)
