# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JSON Schema construction helper and reference grammar for relationship resolution."""


def required_object(**properties: dict) -> dict:
    """Require the named properties without restricting unrelated fields."""
    return {"type": "object", "required": list(properties), "properties": properties}


MODEL_CALL_REF = {
    "type": "object",
    "properties": {
        "model_call_id": {"anyOf": [{"type": "string", "pattern": "\\S"}, {"type": "null"}]},
        "model_ref": {
            "anyOf": [
                {
                    "type": "object",
                    "required": ["type", "name"],
                    "properties": {
                        "type": {"const": "responses_api_models"},
                        "name": {"type": "string", "pattern": "\\S"},
                    },
                },
                {"type": "null"},
            ]
        },
        "response_id": {"anyOf": [{"type": "string", "pattern": "\\S"}, {"type": "null"}]},
    },
    "anyOf": [
        {
            "type": "object",
            "required": ["model_call_id"],
            "properties": {"model_call_id": {"type": "string", "pattern": "\\S"}},
        },
        {
            "type": "object",
            "required": ["model_ref", "response_id"],
            "properties": {
                "model_ref": {
                    "type": "object",
                    "required": ["type", "name"],
                    "properties": {
                        "type": {"const": "responses_api_models"},
                        "name": {"type": "string", "pattern": "\\S"},
                    },
                },
                "response_id": {"type": "string", "pattern": "\\S"},
            },
        },
    ],
}
