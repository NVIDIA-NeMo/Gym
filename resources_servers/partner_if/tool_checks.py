# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic checks of new calls against the task's advertised functions."""

from typing import TYPE_CHECKING

from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for
from referencing import Registry
from referencing.exceptions import Unresolvable

from resources_servers.partner_if.evidence import EvidenceUnavailable


if TYPE_CHECKING:
    from resources_servers.partner_if.checks import CheckContext


def single_advertised_tool_call(context: "CheckContext") -> tuple[bool, str]:
    """Require one call to an advertised function with schema-valid object arguments.

    The adapter keeps supplied history in context and newly emitted calls in trajectory.
    Code strings are data: neither their syntax nor their execution is assessed here.
    """
    evidence = context.evidence
    if evidence.trajectory_error:
        raise EvidenceUnavailable(evidence.trajectory_error)
    if not evidence.tools:
        raise EvidenceUnavailable("The advertised function registry is missing")
    validators = {}
    for tool in evidence.tools:
        name = tool.get("name")
        schema = tool.get("parameters")
        if (
            tool.get("type") != "function"
            or not isinstance(name, str)
            or not name
            or name in validators
            or not isinstance(schema, dict)
        ):
            raise EvidenceUnavailable("Expected unique named functions with JSON parameter schemas")
        try:
            validator = validator_for(schema)
            validator.check_schema(schema)
        except SchemaError as error:
            raise EvidenceUnavailable("The advertised function has an invalid parameter schema") from error
        # Dataset schemas may use local references, but must never fetch remote resources while grading.
        validators[name] = validator(schema, registry=Registry())
    calls = [event for event in evidence.trajectory if event.kind == "tool_call"]
    if len(calls) != 1:
        return False, f"Expected exactly one new function call; received {len(calls)}"
    call = calls[0]
    if call.tool_name not in validators:
        return False, "The new function call names a function absent from the advertised registry"
    if not isinstance(call.arguments, dict):
        return False, "Function arguments must be a JSON object"
    try:
        valid = validators[call.tool_name].is_valid(call.arguments)
    except Unresolvable as error:
        raise EvidenceUnavailable("The advertised parameter schema contains an unresolved reference") from error
    if not valid:
        return False, "Function arguments do not satisfy the advertised JSON parameter schema"
    return True, "Exactly one new call to an advertised function with schema-valid arguments"
