# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persisted P0 trajectory evidence, independent of health and live qualification."""

SCHEMA_VERSION = "gym-p0-evidence/v1"
TEXT = {"type": "string", "minLength": 1}
TOKENS = {"type": ["integer", "null"], "minimum": 0}
TOKEN_FIELDS = ("tokens_in", "tokens_out", "tokens_reasoning", "tokens_total", "cached_tokens")


def object_schema(properties: dict, required: tuple | list) -> dict:
    return {"type": "object", "properties": properties, "required": list(required)}


MODEL_REF = object_schema({"type": TEXT, "name": TEXT}, ("type", "name"))
CALL_REF = {
    **object_schema({"model_call_id": TEXT, "model_ref": MODEL_REF, "response_id": TEXT}, ()),
    "anyOf": [{"required": ["model_call_id"]}, {"required": ["model_ref", "response_id"]}],
}
CALL = object_schema(
    {
        "model_call_id": TEXT,
        "model_ref": MODEL_REF,
        "dialect": {"enum": ["responses", "chat", "messages"]},
        "status_code": {"type": ["integer", "null"], "minimum": 100, "maximum": 599},
        "response_status": {"type": ["string", "null"]},
        "finish_reason": {"type": ["string", "null"]},
        "error_category": {"type": ["string", "null"]},
    },
    ("model_call_id", "model_ref", "dialect"),
)
USAGE = object_schema({key: TOKENS for key in TOKEN_FIELDS}, ())
INVOCATION = object_schema(
    {"invocation_id": TEXT, "model_calls": {"type": "array", "items": {"type": "object"}}},
    ("invocation_id", "model_calls"),
)
TOOL = object_schema(
    {
        "invocation_id": TEXT,
        "tool_call_id": TEXT,
        "tool_name": TEXT,
        "status": {"enum": ["completed", "failed", "timeout", "cancelled"]},
    },
    ("invocation_id", "tool_call_id", "tool_name", "status"),
)
TURN = object_schema(
    {
        "invocation_id": TEXT,
        "task_id": TEXT,
        "rollout_id": TEXT,
        "turn_no": {"type": "integer", "minimum": 1},
        "timestamp": {"type": "number"},
        "question": {"not": {"type": "null"}},
        "resolved": {"type": ["boolean", "null"]},
    },
    ("invocation_id", "task_id", "rollout_id", "turn_no", "timestamp", "question", "resolved"),
)
SCHEMAS = {"TE-1": CALL, "TE-2": USAGE, "TE-3": TURN, "TE-5": TOOL, "TE-8": INVOCATION}
