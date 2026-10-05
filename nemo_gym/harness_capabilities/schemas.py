# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JSON Schema fragments for additional RFC requirements at the saved locations.

Native Gym validation still runs first. These fragments add requiredness and
protocol structure; they do not certify payload completeness or execution coverage.
"""

NONBLANK = {"type": "string", "pattern": r"\S"}
TIMESTAMP = {"type": "number", "minimum": 0}
HTTP_STATUS = {"type": "integer", "minimum": 100, "maximum": 599}
NONEMPTY = {"type": "array", "minItems": 1}
OBJECT_ARRAY = {"type": "array", "items": {"type": "object"}}


def required_object(**properties: dict) -> dict:
    """Require the named properties without restricting unrelated fields."""
    return {"type": "object", "required": list(properties), "properties": properties}


MODEL_REF = required_object(type={"const": "responses_api_models"}, name=NONBLANK)
DIALECT = {"enum": ["chat", "responses", "messages"]}
RETURNED_OUTCOME = {
    "allOf": [
        required_object(status_code={"type": "integer", "minimum": 200, "maximum": 299}),
        {
            "anyOf": [
                required_object(dialect={"enum": ["chat", "messages"]}, finish_reason=NONBLANK),
                required_object(dialect={"const": "responses"}, response_status={"enum": ["completed", "incomplete"]}),
            ]
        },
    ]
}
OUTCOME = {
    "type": "object",
    "properties": {"status_code": {"anyOf": [HTTP_STATUS, {"type": "null"}]}},
    "anyOf": [required_object(error_category=NONBLANK), RETURNED_OUTCOME],
}
RESPONSE_ID = {
    "if": required_object(error_category=NONBLANK),
    "then": {"properties": {"response_id": {"anyOf": [NONBLANK, {"type": "null"}]}}},
    "else": required_object(response_id=NONBLANK),
}


def protocol_branch(dialect: str, **properties: dict) -> dict:
    """Select a body shape using the saved model-call dialect."""
    return required_object(response_metadata=required_object(dialect={"const": dialect}), **properties)


REQUEST = {
    "anyOf": [
        protocol_branch(
            "responses",
            request=required_object(input={"anyOf": [{"type": "string"}, OBJECT_ARRAY]}),
        ),
        protocol_branch("chat", request=required_object(messages=OBJECT_ARRAY)),
        protocol_branch("messages", request=required_object(messages=OBJECT_ARRAY)),
    ]
}
RESPONSE = {
    **required_object(response={"type": ["object", "string", "null"]}),
    "if": required_object(
        response_metadata={
            **required_object(status_code={"type": "integer", "minimum": 200, "maximum": 299}),
            "not": required_object(error_category=NONBLANK),
        },
        response={"not": {"type": "null"}},
    ),
    "then": {
        "anyOf": [
            protocol_branch("responses", response=required_object(output=OBJECT_ARRAY)),
            protocol_branch(
                "chat",
                response=required_object(
                    choices={
                        "type": "array",
                        "items": required_object(message=required_object(role={"const": "assistant"})),
                    }
                ),
            ),
            protocol_branch("messages", response=required_object(content=OBJECT_ARRAY)),
        ]
    },
}
RESPONSE_PRESENCE = {
    "anyOf": [
        required_object(response_metadata=required_object(status_code=HTTP_STATUS)),
        required_object(
            response_metadata=required_object(status_code={"type": "null"}, error_category=NONBLANK),
            response={"type": "null"},
        ),
    ]
}
TOOL_OUTPUT = required_object(output={"type": ["string", "array", "object"]})
SANDBOX_PRESENCE = required_object(records={"type": "array", "contains": required_object(kind={"const": "sandbox"})})
SANDBOX_OUTCOME = required_object(
    outcome={"enum": ["completed", "failed", "timeout", "oom", "sandbox_error", "cancelled"]}
)
SANDBOX_ERROR = {
    "if": required_object(outcome={"enum": ["failed", "oom", "sandbox_error"]}),
    "then": required_object(error_type=NONBLANK),
}
TURN_CALLS = required_object(model_calls={**NONEMPTY, "items": required_object(model_call_id=NONBLANK)})
