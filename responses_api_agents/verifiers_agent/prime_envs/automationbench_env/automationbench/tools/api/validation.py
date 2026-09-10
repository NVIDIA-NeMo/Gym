# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""JSON Schema validation for structured API requests."""

import json
import re
from collections.abc import Mapping
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError


def check_schema(schema: Mapping[str, Any]) -> None:
    """Fail fast when a catalog entry is not valid JSON Schema Draft 2020-12."""
    Draft202012Validator.check_schema(schema)


def _location(label: str, path: list[str | int]) -> str:
    """Render a validator path using the names exposed by the API tool."""
    location = label
    for part in path:
        location += f"[{part}]" if isinstance(part, int) else f".{part}"
    return location


def _additional_properties(error: ValidationError) -> list[str]:
    schema = error.schema
    if not isinstance(error.instance, Mapping) or not isinstance(schema, Mapping):
        return []

    properties = schema.get("properties", {})
    patterns = schema.get("patternProperties", {})
    return sorted(
        name
        for name in error.instance
        if name not in properties and not any(re.search(pattern, name) for pattern in patterns)
    )


def _format_error(error: ValidationError, label: str) -> str:
    """Translate common validator messages into a stable model-facing vocabulary."""
    location = _location(label, list(error.absolute_path))
    if error.validator == "type":
        expected = error.validator_value
        if isinstance(expected, list):
            return f"{location} must be one of these types: {', '.join(expected)}"
        return f"{location} must be {expected}"
    if error.validator == "required":
        missing = error.message.removesuffix(" is a required property").strip("'")
        return f"missing {location} field: {missing}"
    if error.validator == "additionalProperties":
        unexpected = _additional_properties(error)
        if unexpected:
            fields = ", ".join(unexpected)
            noun = "field" if len(unexpected) == 1 else "fields"
            return f"unknown {location} {noun}: {fields}"
    if error.validator == "enum":
        options = ", ".join(json.dumps(option, sort_keys=True) for option in error.validator_value)
        return f"{location} must be one of: {options}"
    if error.validator == "const":
        return f"{location} must equal {json.dumps(error.validator_value, sort_keys=True)}"
    if error.validator == "minLength" and error.validator_value == 1:
        return f"{location} must not be empty"
    if error.validator == "anyOf":
        return f"{location} does not match any allowed schema"
    if error.validator == "oneOf":
        return f"{location} must match exactly one allowed schema"
    return f"{location}: {error.message}"


def validation_errors(schema: Mapping[str, Any], value: Any, label: str) -> list[str]:
    """Return deterministic errors from the full JSON Schema Draft 2020-12 validator.

    Catalog schemas are checked once by the contract loader. This function only
    validates request values, keeping the execution path small while still
    supporting every standard keyword used by the catalog.
    """
    validator = Draft202012Validator(schema)
    errors = sorted(
        validator.iter_errors(value),
        key=lambda error: (
            tuple(f"{type(part).__name__}:{part}" for part in error.absolute_path),
            str(error.validator),
            error.message,
        ),
    )
    return sorted(_format_error(error, label) for error in errors)


__all__ = ["check_schema", "validation_errors"]
