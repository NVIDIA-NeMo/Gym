# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned, intentionally stricter subsets of Gym's optional contracts.

Pinned to Gym 26688f7d (TrajectoryRecord 1.0). Additional producer fields are
allowed: these are capability requirements, not copies of Gym's whole schema.
"""

SCHEMA_VERSION = "gym-artifact-contracts/v1"
TEXT = {"type": "string", "minLength": 1}
NUMBER = {"type": "number"}
NONNEGATIVE = {"type": "number", "minimum": 0}
TOKENS = {"type": "integer", "minimum": 0}
TOKEN_FIELDS = (
    "tokens_in",
    "tokens_out",
    "tokens_reasoning",
    "tokens_total",
    "cached_tokens",
)


def object_schema(properties: dict, required: tuple | list) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
    }


MODEL_REF = object_schema({"type": TEXT, "name": TEXT}, ("type", "name"))
CALL_REF = {
    **object_schema(
        {"model_call_id": TEXT, "model_ref": MODEL_REF, "response_id": TEXT},
        (),
    ),
    "anyOf": [
        {"required": ["model_call_id"]},
        {"required": ["model_ref", "response_id"]},
    ],
}
CALL = {
    **object_schema(
        {
            "model_call_id": TEXT,
            "model_ref": MODEL_REF,
            # Native captures may lack a provider model on failed calls. The
            # required model_ref still identifies the configured model server.
            "model": {"anyOf": [TEXT, {"type": "null"}]},
            "dialect": {"enum": ["responses", "chat", "messages"]},
            "started_at": NUMBER,
            "completed_at": NUMBER,
            "latency_total_ms": NONNEGATIVE,
            "status_code": {"type": ["integer", "null"]},
            "error_category": {"type": ["string", "null"]},
        },
        (
            "model_call_id",
            "model_ref",
            "dialect",
            "started_at",
            "completed_at",
            "latency_total_ms",
        ),
    ),
    "anyOf": [
        {
            "required": ["status_code"],
            "properties": {
                "status_code": {
                    "type": "integer",
                    "minimum": 100,
                    "maximum": 599,
                }
            },
        },
        {
            "required": ["error_category"],
            "properties": {"error_category": TEXT},
        },
    ],
}
USAGE = object_schema({key: TOKENS for key in TOKEN_FIELDS}, TOKEN_FIELDS)
INVOCATION = object_schema(
    {
        "invocation_id": TEXT,
        "model_calls": {"type": "array", "items": {"type": "object"}},
        "conversation": {"type": "array", "items": {"type": "object"}},
    },
    ("invocation_id", "model_calls"),
)
TOOL = object_schema(
    {
        "invocation_id": TEXT,
        "tool_call_id": TEXT,
        "tool_name": TEXT,
        "started_at": NUMBER,
        "completed_at": NUMBER,
        "duration_ms": NONNEGATIVE,
        "timing_source": {"enum": ["executor", "artifact", "harness"]},
        "status": {"enum": ["completed", "failed", "timeout", "cancelled"]},
    },
    (
        "invocation_id",
        "tool_call_id",
        "tool_name",
        "started_at",
        "completed_at",
        "duration_ms",
        "timing_source",
        "status",
    ),
)
TURN = object_schema(
    {
        "invocation_id": TEXT,
        "task_id": TEXT,
        "rollout_id": TEXT,
        "turn_no": {"type": "integer", "minimum": 1},
        "timestamp": NUMBER,
        "question": {"not": {"type": "null"}},
        "answer": {"not": {"type": "null"}},
        "step_count": {"type": "integer", "minimum": 0},
        "resolved": {"type": ["boolean", "null"]},
        "model_calls": {"type": "array", "minItems": 1},
    },
    (
        "invocation_id",
        "task_id",
        "rollout_id",
        "turn_no",
        "timestamp",
        "question",
        "answer",
        "step_count",
        "resolved",
        "model_calls",
    ),
)

SCHEMAS = {"C1": CALL, "C2": USAGE, "C3": TURN, "C5": TOOL, "C10": INVOCATION}
