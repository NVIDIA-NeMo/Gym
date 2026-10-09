# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused evidence checks. TE labels are reporting metadata, not dispatch keys."""

import base64
import binascii
from collections import Counter
from dataclasses import dataclass

from jsonschema import Draft202012Validator

from .checks import SchemaCheck, SemanticCheck
from .results import Results, gate_passes


NAMES = {
    "TE-1": "model_call_status",
    "TE-2": "token_counts",
    "TE-3": "steps",
    "TE-4": "history",
    "TE-5": "tool_record",
    "TE-6": "verifier_outcome",
    "TE-7": "payloads",
    "TE-8": "run_join",
    "TE-9": "step_join",
}
PROFILE = "gym-p0/v1"
TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens", "cached_tokens")


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


@dataclass(frozen=True)
class EvidenceScope:
    """Declared applicability; absent artifacts never imply an exemption."""

    tools: bool = True
    verifier: bool = True
    steps: bool = True
    require_sandbox: bool = False


def required_object(**properties: dict) -> dict:
    """Require the named properties without restricting unrelated fields."""
    return {"type": "object", "required": list(properties), "properties": properties}


def _mapping(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _objects(value: object) -> list[dict]:
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


def _missing_content(value: object) -> bool:
    """Reject unresolved media/opaque references; validate inline image data."""
    if isinstance(value, list):
        return any(_missing_content(item) for item in value)
    if isinstance(value, dict):
        unavailable = any(value.get(key) for key in ("file_id", "file_url", "encrypted_content"))
        if value.get("image_url"):
            image = value["image_url"]
            image = image.get("url") if isinstance(image, dict) else image
            if not isinstance(image, str) or ";base64," not in image or not image.startswith("data:"):
                unavailable = True
            else:
                try:
                    if not base64.b64decode(image.split(";base64,", 1)[1], validate=True):
                        unavailable = True
                except (ValueError, binascii.Error):
                    unavailable = True
        return bool(unavailable) or any(_missing_content(item) for item in value.values())
    return False


def _resolve(reference: dict, calls: list[dict]) -> list[int]:
    if not Draft202012Validator(MODEL_CALL_REF).is_valid(reference):
        return []
    return [
        i
        for i, call in enumerate(calls)
        if all(
            (call.get(key) if key == "model_call_id" else _mapping(call.get("response_metadata")).get(key)) == value
            for key, value in reference.items()
            if key in {"model_call_id", "model_ref", "response_id"} and value is not None
        )
    ]


class Inspector:
    """Run independent predicates with focused prerequisites and explicit outcomes."""

    def __init__(self, record: dict | None, scope: EvidenceScope) -> None:
        self.available = record is not None
        self.record = record or {}
        self.scope = scope
        self.trajectory = _mapping(self.record.get("ng_trajectory"))
        self.results = Results()
        self.calls = _objects(self.trajectory.get("model_calls"))
        self.invocations = _objects(self.trajectory.get("invocations"))
        self.turns = _objects(self.trajectory.get("turns"))
        self.tools = _objects(self.trajectory.get("tool_calls"))
        self.observations = _objects(_mapping(self.record.get("ng_agent_observations")).get("records"))

    def model_calls(self) -> None:
        self.results.run(
            SchemaCheck(
                id="model_calls.present",
                tier="P0",
                evidence=("TE-1", "TE-2", "TE-4", "TE-7", "TE-8", "TE-9"),
                location="$.ng_trajectory.model_calls",
                reason="expected a nonempty collection of saved model_calls",
                value=self.trajectory.get("model_calls"),
                schema={"type": "array", "minItems": 1, "items": {"type": "object"}},
                depends_on=(),
                applies=True,
                available=self.available,
            )
        )
        self.results.run(
            SchemaCheck(
                id="calls.identity",
                tier="P0",
                evidence=("TE-1",),
                location="$.ng_trajectory.model_calls",
                reason="required evidence does not match its schema",
                value=self.trajectory.get("model_calls"),
                schema={
                    "type": "array",
                    "items": required_object(model_call_id={"type": "string", "pattern": "\\S"}),
                },
                depends_on=("model_calls.present",),
                applies=True,
                available=self.available,
            )
        )
        self.results.run(
            SchemaCheck(
                id="calls.model",
                tier="P0",
                evidence=("TE-1",),
                location="$.ng_trajectory.model_calls",
                reason="required evidence does not match its schema",
                value=self.trajectory.get("model_calls"),
                schema={
                    "type": "array",
                    "items": required_object(
                        response_metadata=required_object(
                            model_ref=required_object(
                                type={"const": "responses_api_models"}, name={"type": "string", "pattern": "\\S"}
                            )
                        )
                    ),
                },
                depends_on=("model_calls.present",),
                applies=True,
                available=self.available,
            )
        )
        self.results.run(
            SchemaCheck(
                id="calls.protocol",
                tier="P0",
                evidence=("TE-1",),
                location="$.ng_trajectory.model_calls",
                reason="required evidence does not match its schema",
                value=self.trajectory.get("model_calls"),
                schema={
                    "type": "array",
                    "items": required_object(
                        response_metadata=required_object(dialect={"enum": ["chat", "responses", "messages"]})
                    ),
                },
                depends_on=("model_calls.present",),
                applies=True,
                available=self.available,
            )
        )
        self.results.run(
            SchemaCheck(
                id="calls.timing",
                tier="P0",
                evidence=("TE-1",),
                location="$.ng_trajectory.model_calls",
                reason="required evidence does not match its schema",
                value=self.trajectory.get("model_calls"),
                schema={
                    "type": "array",
                    "items": required_object(
                        started_at={"type": "number", "minimum": 0}, completed_at={"type": "number", "minimum": 0}
                    ),
                },
                depends_on=("model_calls.present",),
                applies=True,
                available=self.available,
            )
        )
        self.results.run(
            SchemaCheck(
                id="calls.outcome",
                tier="P0",
                evidence=("TE-1",),
                location="$.ng_trajectory.model_calls",
                reason="required evidence does not match its schema",
                value=self.trajectory.get("model_calls"),
                schema={
                    "type": "array",
                    "items": required_object(
                        response_metadata={
                            "type": "object",
                            "properties": {
                                "status_code": {
                                    "anyOf": [{"type": "integer", "minimum": 100, "maximum": 599}, {"type": "null"}]
                                },
                                "error_category": {"anyOf": [{"type": "string", "pattern": "\\S"}, {"type": "null"}]},
                                "response_status": {"anyOf": [{"type": "string", "pattern": "\\S"}, {"type": "null"}]},
                                "finish_reason": {"anyOf": [{"type": "string", "pattern": "\\S"}, {"type": "null"}]},
                            },
                            "anyOf": [
                                required_object(error_category={"type": "string", "pattern": "\\S"}),
                                {
                                    "allOf": [
                                        required_object(
                                            status_code={"type": "integer", "minimum": 200, "maximum": 299}
                                        ),
                                        {
                                            "anyOf": [
                                                required_object(
                                                    dialect={"enum": ["chat", "messages"]},
                                                    finish_reason={"type": "string", "pattern": "\\S"},
                                                ),
                                                required_object(
                                                    dialect={"const": "responses"},
                                                    response_status={"enum": ["completed", "incomplete"]},
                                                ),
                                            ]
                                        },
                                    ]
                                },
                            ],
                        }
                    ),
                },
                depends_on=("model_calls.present",),
                applies=True,
                available=self.available,
            )
        )
        self.results.run(
            SchemaCheck(
                id="calls.response_id",
                tier="P0",
                evidence=("TE-1",),
                location="$.ng_trajectory.model_calls",
                reason="required evidence does not match its schema",
                value=self.trajectory.get("model_calls"),
                schema={
                    "type": "array",
                    "items": required_object(
                        response_metadata={
                            "if": required_object(error_category={"type": "string", "pattern": "\\S"}),
                            "then": {
                                "properties": {
                                    "response_id": {"anyOf": [{"type": "string", "pattern": "\\S"}, {"type": "null"}]}
                                }
                            },
                            "else": required_object(response_id={"type": "string", "pattern": "\\S"}),
                        }
                    ),
                },
                depends_on=("model_calls.present",),
                applies=True,
                available=self.available,
            )
        )
        self.results.run(
            SchemaCheck(
                id="calls.request",
                tier="P0",
                evidence=("TE-4", "TE-7"),
                location="$.ng_trajectory.model_calls",
                reason="required evidence does not match its schema",
                value=self.trajectory.get("model_calls"),
                schema={
                    "type": "array",
                    "items": {
                        "anyOf": [
                            required_object(
                                response_metadata=required_object(dialect={"const": "responses"}),
                                request=required_object(
                                    input={
                                        "anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "object"}}]
                                    }
                                ),
                            ),
                            required_object(
                                response_metadata=required_object(dialect={"const": "chat"}),
                                request=required_object(messages={"type": "array", "items": {"type": "object"}}),
                            ),
                            required_object(
                                response_metadata=required_object(dialect={"const": "messages"}),
                                request=required_object(messages={"type": "array", "items": {"type": "object"}}),
                            ),
                        ]
                    },
                },
                depends_on=("model_calls.present",),
                applies=True,
                available=self.available,
            )
        )
        self.results.run(
            SchemaCheck(
                id="calls.response",
                tier="P0",
                evidence=("TE-4", "TE-7"),
                location="$.ng_trajectory.model_calls",
                reason="required evidence does not match its schema",
                value=self.trajectory.get("model_calls"),
                schema={
                    "type": "array",
                    "items": {
                        "anyOf": [
                            {
                                "type": "object",
                                "required": ["response", "response_metadata"],
                                "properties": {
                                    "response": {"type": ["object", "string"]},
                                    "response_metadata": required_object(
                                        status_code={"type": "integer", "minimum": 100, "maximum": 599}
                                    ),
                                },
                                "if": required_object(
                                    response_metadata={
                                        "type": "object",
                                        "required": ["status_code"],
                                        "properties": {
                                            "status_code": {"type": "integer", "minimum": 200, "maximum": 299}
                                        },
                                        "not": required_object(error_category={"type": "string", "pattern": "\\S"}),
                                    }
                                ),
                                "then": {
                                    "anyOf": [
                                        required_object(
                                            response_metadata=required_object(dialect={"const": "responses"}),
                                            response=required_object(
                                                output={"type": "array", "items": {"type": "object"}}
                                            ),
                                        ),
                                        required_object(
                                            response_metadata=required_object(dialect={"const": "chat"}),
                                            response=required_object(
                                                choices={
                                                    "type": "array",
                                                    "items": required_object(
                                                        message=required_object(role={"const": "assistant"})
                                                    ),
                                                }
                                            ),
                                        ),
                                        required_object(
                                            response_metadata=required_object(dialect={"const": "messages"}),
                                            response=required_object(
                                                content={"type": "array", "items": {"type": "object"}}
                                            ),
                                        ),
                                    ]
                                },
                            },
                            required_object(
                                response={"type": "null"},
                                response_metadata=required_object(
                                    status_code={"type": "null"}, error_category={"type": "string", "pattern": "\\S"}
                                ),
                            ),
                        ]
                    },
                },
                depends_on=("model_calls.present",),
                applies=True,
                available=self.available,
            )
        )
        for field in TOKEN_FIELDS:
            # Optional metrics retain missing/null as unavailable; no normalization recheck.
            self.results.run(
                SchemaCheck(
                    id="tokens." + field,
                    tier="P0",
                    evidence=("TE-2",),
                    location="$.ng_trajectory.model_calls",
                    reason="required evidence does not match its schema",
                    value=self.trajectory.get("model_calls"),
                    schema={
                        "type": "array",
                        "items": required_object(
                            token_stats={
                                "type": "object",
                                "properties": {field: {"type": ["integer", "null"], "minimum": 0}},
                            }
                        ),
                    },
                    depends_on=("model_calls.present",),
                    applies=True,
                    available=self.available,
                )
            )
        self.results.run(
            SemanticCheck(
                id="content.media",
                tier="P0",
                evidence=("TE-4", "TE-7"),
                location="$.ng_trajectory.model_calls",
                reason="external, encrypted or invalid media is unavailable to this reader",
                predicate=lambda: not any(
                    (_missing_content(c.get(k)) for c in self.calls for k in ("request", "response"))
                ),
                available=self.available,
                depends_on=("model_calls.present", "calls.request", "calls.response"),
            )
        )
        valid = True
        for index, call in enumerate(self.calls):
            previous = _mapping(call.get("request")).get("previous_response_id")
            if previous:
                matches = [
                    c
                    for c in self.calls[:index]
                    if _mapping(c.get("response_metadata")).get("response_id") == previous
                    and _mapping(c.get("response_metadata")).get("model_ref")
                    == _mapping(call.get("response_metadata")).get("model_ref")
                ]
                valid &= (
                    len(matches) == 1
                    and matches[0].get("request") is not None
                    and matches[0].get("response") is not None
                )
        self.results.run(
            SemanticCheck(
                id="content.previous_response",
                tier="P0",
                evidence=("TE-4",),
                location="$.ng_trajectory.model_calls",
                reason="previous response has no unique retained history",
                predicate=lambda: valid,
                available=self.available,
                depends_on=("model_calls.present", "calls.request"),
            )
        )

    def structure(self) -> None:
        for field in ("task_id", "rollout_id"):
            self.results.run(
                SchemaCheck(
                    id="trajectory." + field,
                    tier="P0",
                    evidence=("TE-3", "TE-8"),
                    location="$.ng_trajectory",
                    reason="required evidence does not match its schema",
                    value=self.trajectory,
                    schema=required_object(**{field: {"type": "string", "pattern": "\\S"}}),
                    depends_on=(),
                    applies=True,
                    available=self.available,
                )
            )
        self.results.run(
            SchemaCheck(
                id="invocations.present",
                tier="P0",
                evidence=("TE-8",),
                location="$.ng_trajectory.invocations",
                reason="expected a nonempty collection of saved invocations",
                value=self.trajectory.get("invocations"),
                schema={"type": "array", "minItems": 1, "items": {"type": "object"}},
                depends_on=(),
                applies=True,
                available=self.available,
            )
        )
        self.results.run(
            SchemaCheck(
                id="invocations.identity",
                tier="P0",
                evidence=("TE-8",),
                location="$.ng_trajectory.invocations",
                reason="required evidence does not match its schema",
                value=self.trajectory.get("invocations"),
                schema={
                    "type": "array",
                    "items": required_object(invocation_id={"type": "string", "pattern": "\\S"}),
                },
                depends_on=("invocations.present",),
                applies=True,
                available=self.available,
            )
        )
        self.results.run(
            SchemaCheck(
                id="invocations.references",
                tier="P0",
                evidence=("TE-8",),
                location="$.ng_trajectory.invocations",
                reason="required evidence does not match its schema",
                value=self.trajectory.get("invocations"),
                schema={
                    "type": "array",
                    "items": required_object(
                        model_calls={
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "model_call_id": {
                                        "anyOf": [{"type": "string", "pattern": "\\S"}, {"type": "null"}]
                                    },
                                    "model_ref": {
                                        "anyOf": [
                                            required_object(
                                                type={"const": "responses_api_models"},
                                                name={"type": "string", "pattern": "\\S"},
                                            ),
                                            {"type": "null"},
                                        ]
                                    },
                                    "response_id": {"anyOf": [{"type": "string", "pattern": "\\S"}, {"type": "null"}]},
                                },
                                "anyOf": [
                                    required_object(model_call_id={"type": "string", "pattern": "\\S"}),
                                    required_object(
                                        model_ref=required_object(
                                            type={"const": "responses_api_models"},
                                            name={"type": "string", "pattern": "\\S"},
                                        ),
                                        response_id={"type": "string", "pattern": "\\S"},
                                    ),
                                ],
                            },
                        }
                    ),
                },
                depends_on=("invocations.present",),
                applies=True,
                available=self.available,
            )
        )
        self.results.run(
            SchemaCheck(
                id="turns.present",
                tier="P0",
                evidence=("TE-3", "TE-9"),
                location="$.ng_trajectory.turns",
                reason="expected a nonempty collection of saved turns",
                value=self.trajectory.get("turns"),
                schema={"type": "array", "minItems": 1, "items": {"type": "object"}},
                depends_on=(),
                applies=self.scope.steps,
                available=self.available,
            )
        )
        self.results.run(
            SchemaCheck(
                id="steps.invocation",
                tier="P0",
                evidence=("TE-3", "TE-9"),
                location="$.ng_trajectory.turns",
                reason="required evidence does not match its schema",
                value=self.trajectory.get("turns"),
                schema={
                    "type": "array",
                    "items": required_object(invocation_id={"type": "string", "pattern": "\\S"}),
                },
                depends_on=("turns.present",),
                applies=self.scope.steps,
                available=self.available,
            )
        )
        self.results.run(
            SchemaCheck(
                id="steps.number",
                tier="P0",
                evidence=("TE-3", "TE-9"),
                location="$.ng_trajectory.turns",
                reason="required evidence does not match its schema",
                value=self.trajectory.get("turns"),
                schema={"type": "array", "items": required_object(turn_no={"type": "integer", "minimum": 1})},
                depends_on=("turns.present",),
                applies=self.scope.steps,
                available=self.available,
            )
        )
        self.results.run(
            SchemaCheck(
                id="steps.timestamp",
                tier="P0",
                evidence=("TE-3",),
                location="$.ng_trajectory.turns",
                reason="required evidence does not match its schema",
                value=self.trajectory.get("turns"),
                schema={"type": "array", "items": required_object(timestamp={"type": "number", "minimum": 0})},
                depends_on=("turns.present",),
                applies=self.scope.steps,
                available=self.available,
            )
        )
        self.results.run(
            SchemaCheck(
                id="steps.resolution",
                tier="P0",
                evidence=("TE-3",),
                location="$.ng_trajectory.turns",
                reason="required evidence does not match its schema",
                value=self.trajectory.get("turns"),
                schema={"type": "array", "items": required_object(resolved={"type": ["boolean", "null"]})},
                depends_on=("turns.present",),
                applies=self.scope.steps,
                available=self.available,
            )
        )
        self.results.run(
            SchemaCheck(
                id="steps.references",
                tier="P0",
                evidence=("TE-9",),
                location="$.ng_trajectory.turns",
                reason="required evidence does not match its schema",
                value=self.trajectory.get("turns"),
                schema={
                    "type": "array",
                    "items": required_object(
                        model_calls={
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "model_call_id": {
                                        "anyOf": [{"type": "string", "pattern": "\\S"}, {"type": "null"}]
                                    },
                                    "model_ref": {
                                        "anyOf": [
                                            required_object(
                                                type={"const": "responses_api_models"},
                                                name={"type": "string", "pattern": "\\S"},
                                            ),
                                            {"type": "null"},
                                        ]
                                    },
                                    "response_id": {"anyOf": [{"type": "string", "pattern": "\\S"}, {"type": "null"}]},
                                },
                                "anyOf": [
                                    required_object(model_call_id={"type": "string", "pattern": "\\S"}),
                                    required_object(
                                        model_ref=required_object(
                                            type={"const": "responses_api_models"},
                                            name={"type": "string", "pattern": "\\S"},
                                        ),
                                        response_id={"type": "string", "pattern": "\\S"},
                                    ),
                                ],
                            },
                        }
                    ),
                },
                depends_on=("turns.present",),
                applies=self.scope.steps,
                available=self.available,
            )
        )
        self.results.run(
            SchemaCheck(
                id="tool_calls.present",
                tier="P0",
                evidence=("TE-5",),
                location="$.ng_trajectory.tool_calls",
                reason="expected a nonempty collection of saved tool_calls",
                value=self.trajectory.get("tool_calls"),
                schema={"type": "array", "minItems": 1, "items": {"type": "object"}},
                depends_on=(),
                applies=self.scope.tools,
                available=self.available,
            )
        )
        self.results.run(
            SchemaCheck(
                id="tools.identity",
                tier="P0",
                evidence=("TE-5",),
                location="$.ng_trajectory.tool_calls",
                reason="required evidence does not match its schema",
                value=self.trajectory.get("tool_calls"),
                schema={
                    "type": "array",
                    "items": required_object(tool_call_id={"type": "string", "pattern": "\\S"}),
                },
                depends_on=("tool_calls.present",),
                applies=self.scope.tools,
                available=self.available,
            )
        )
        self.results.run(
            SchemaCheck(
                id="tools.name",
                tier="P0",
                evidence=("TE-5",),
                location="$.ng_trajectory.tool_calls",
                reason="required evidence does not match its schema",
                value=self.trajectory.get("tool_calls"),
                schema={"type": "array", "items": required_object(tool_name={"type": "string", "pattern": "\\S"})},
                depends_on=("tool_calls.present",),
                applies=self.scope.tools,
                available=self.available,
            )
        )
        self.results.run(
            SchemaCheck(
                id="tools.invocation",
                tier="P0",
                evidence=("TE-5",),
                location="$.ng_trajectory.tool_calls",
                reason="required evidence does not match its schema",
                value=self.trajectory.get("tool_calls"),
                schema={
                    "type": "array",
                    "items": required_object(invocation_id={"type": "string", "pattern": "\\S"}),
                },
                depends_on=("tool_calls.present",),
                applies=self.scope.tools,
                available=self.available,
            )
        )
        self.results.run(
            SchemaCheck(
                id="tools.status",
                tier="P0",
                evidence=("TE-5",),
                location="$.ng_trajectory.tool_calls",
                reason="required evidence does not match its schema",
                value=self.trajectory.get("tool_calls"),
                schema={
                    "type": "array",
                    "items": required_object(status={"enum": ["completed", "failed", "timeout", "cancelled"]}),
                },
                depends_on=("tool_calls.present",),
                applies=self.scope.tools,
                available=self.available,
            )
        )
        self.results.run(
            SchemaCheck(
                id="tools.output",
                tier="P0",
                evidence=("TE-5",),
                location="$.ng_trajectory.tool_calls",
                reason="required evidence does not match its schema",
                value=self.trajectory.get("tool_calls"),
                schema={"type": "array", "items": required_object(output={"type": ["string", "array", "object"]})},
                depends_on=("tool_calls.present",),
                applies=self.scope.tools,
                available=self.available,
            )
        )
        self.results.run(
            SemanticCheck(
                id="steps.invocation_target",
                tier="P0",
                evidence=("TE-3", "TE-9"),
                location="$.ng_trajectory.turns",
                reason="invocation reference does not resolve to exactly one saved invocation",
                predicate=lambda: all(
                    (
                        sum((i.get("invocation_id") == r.get("invocation_id") for i in self.invocations)) == 1
                        for r in self.turns
                    )
                ),
                available=self.available,
                applies=self.scope.steps,
                depends_on=("turns.present", "steps.invocation", "invocations.present", "invocations.identity"),
            )
        )
        self.results.run(
            SemanticCheck(
                id="tools.invocation_target",
                tier="P0",
                evidence=("TE-5",),
                location="$.ng_trajectory.tool_calls",
                reason="invocation reference does not resolve to exactly one saved invocation",
                predicate=lambda: all(
                    (
                        sum((i.get("invocation_id") == r.get("invocation_id") for i in self.invocations)) == 1
                        for r in self.tools
                    )
                ),
                available=self.available,
                applies=self.scope.tools,
                depends_on=("tool_calls.present", "tools.invocation", "invocations.present", "invocations.identity"),
            )
        )
        # Arguments currently live only in the invocation conversation. Tool output has its own authority above.
        requests = [
            [
                item
                for inv in self.invocations
                if inv.get("invocation_id") == tool.get("invocation_id")
                for item in _objects(inv.get("conversation"))
                if item.get("type") == "function_call" and item.get("call_id") == tool.get("tool_call_id")
            ]
            for tool in self.tools
        ]
        self.results.run(
            SemanticCheck(
                id="tools.request",
                tier="P0",
                evidence=("TE-5",),
                location="$.ng_trajectory.invocations[*].conversation",
                reason="tool execution lacks a unique saved request with name and arguments",
                predicate=lambda: all(
                    (
                        len(items) == 1
                        and isinstance(items[0].get("arguments"), str)
                        and (items[0].get("name") == tool.get("tool_name"))
                        for items, tool in zip(requests, self.tools)
                    )
                ),
                available=self.available,
                applies=self.scope.tools,
                depends_on=("tool_calls.present", "tools.identity", "tools.name", "tools.invocation_target"),
            )
        )

    def ownership(self) -> None:
        parent_valid = True
        for invocation in self.invocations:
            current = invocation
            seen = {id(current)}
            while current.get("parent_invocation_id") is not None:
                matches = [i for i in self.invocations if i.get("invocation_id") == current["parent_invocation_id"]]
                if len(matches) != 1 or id(matches[0]) in seen:
                    parent_valid = False
                    break
                current = matches[0]
                seen.add(id(current))
        self.results.run(
            SemanticCheck(
                id="invocations.parent",
                tier="P0",
                evidence=("TE-8",),
                location="$.ng_trajectory.invocations[*].parent_invocation_id",
                reason="parent invocation is missing, ambiguous or cyclic",
                predicate=lambda: parent_valid,
                available=self.available,
                depends_on=("invocations.present", "invocations.identity"),
            )
        )
        owners: dict[int, list[str]] = {i: [] for i in range(len(self.calls))}
        valid = True
        for inv in self.invocations:
            for reference in _objects(inv.get("model_calls")):
                matches = _resolve(reference, self.calls)
                valid &= len(matches) == 1
                if len(matches) == 1:
                    owners[matches[0]].append(inv.get("invocation_id"))

        self.results.run(
            SemanticCheck(
                id="ownership.call_target",
                tier="P0",
                evidence=("TE-8",),
                location="$.ng_trajectory.invocations[*].model_calls",
                reason="call reference does not resolve uniquely with all supplied identifiers",
                predicate=lambda: valid,
                available=self.available,
                depends_on=(
                    "model_calls.present",
                    "calls.identity",
                    "invocations.present",
                    "invocations.references",
                    "invocations.identity",
                ),
            )
        )
        self.results.run(
            SemanticCheck(
                id="ownership.call_owner",
                tier="P0",
                evidence=("TE-8",),
                location="$.ng_trajectory.invocations[*].model_calls",
                reason="each saved call must have exactly one invocation owner",
                predicate=lambda: all((len(v) == 1 for v in owners.values())),
                available=self.available,
                depends_on=("ownership.call_target",),
            )
        )
        helpers = set()
        auxiliary_valid = True
        for observation in self.observations:
            if observation.get("kind") == "context_compaction":
                refs = observation.get("model_calls")
                auxiliary_valid &= Draft202012Validator({"type": "array", "items": MODEL_CALL_REF}).is_valid(refs)
                for reference in _objects(refs):
                    matches = _resolve(reference, self.calls)
                    auxiliary_valid &= len(matches) == 1
                    helpers.update(matches)
        self.results.run(
            SemanticCheck(
                id="steps.compaction_target",
                tier="P0",
                evidence=("TE-9",),
                location="$.ng_agent_observations.records",
                reason="compaction helper reference is invalid or unresolved",
                predicate=lambda: auxiliary_valid,
                available=self.available,
                applies=self.scope.steps,
                depends_on=("model_calls.present",),
            )
        )
        refs = Counter()
        turn_valid, owner_valid = True, True
        for turn in self.turns:
            for reference in _objects(turn.get("model_calls")):
                matches = _resolve(reference, self.calls)
                turn_valid &= len(matches) == 1
                if len(matches) == 1:
                    index = matches[0]
                    refs[index] += 1
                    owner_valid &= not owners[index] or owners[index] == [turn.get("invocation_id")]

        self.results.run(
            SemanticCheck(
                id="steps.call_target",
                tier="P0",
                evidence=("TE-9",),
                location="$.ng_trajectory.turns[*].model_calls",
                reason="step call reference does not resolve uniquely with all supplied identifiers",
                predicate=lambda: turn_valid,
                available=self.available,
                applies=self.scope.steps,
                depends_on=(
                    "model_calls.present",
                    "calls.identity",
                    "turns.present",
                    "steps.references",
                    "steps.invocation_target",
                ),
            )
        )
        self.results.run(
            SemanticCheck(
                id="steps.call_owner",
                tier="P0",
                evidence=("TE-9",),
                location="$.ng_trajectory.turns[*].model_calls",
                reason="call ownership contradicts its step invocation",
                predicate=lambda: owner_valid,
                available=self.available,
                applies=self.scope.steps,
                depends_on=("steps.call_target", "ownership.call_target"),
            )
        )
        policy = set(range(len(self.calls))) - helpers
        # The collector retains HTTP failures as invocation-owned attempts, not
        # model turns (#4045). A producer may still link retries to a persisted
        # assistant message, but absence of that optional link is not data loss.
        http_errors = {
            i
            for i in policy
            if type(status := _mapping(self.calls[i].get("response_metadata")).get("status_code")) is int
            and status >= 400
        }
        self.results.run(
            SemanticCheck(
                id="steps.attempt_accounting",
                tier="P0",
                evidence=("TE-9",),
                location="$.ng_trajectory.turns[*].model_calls",
                reason="every policy response needs one step; HTTP errors may be unbound and compaction calls stay separate",
                predicate=lambda: bool(policy)
                and all((refs[i] == 1 for i in policy - http_errors))
                and all((refs[i] <= 1 for i in http_errors))
                and (not any((refs[i] for i in helpers))),
                available=self.available,
                applies=self.scope.steps,
                depends_on=("steps.call_target", "steps.compaction_target"),
            )
        )

    def evaluation(self) -> None:
        self.results.run(
            SchemaCheck(
                id="evaluation.reward",
                tier="P0",
                evidence=("TE-6",),
                location="$",
                reason="required evidence does not match its schema",
                value=self.record,
                schema=required_object(reward={"type": "number"}),
                depends_on=(),
                applies=self.scope.verifier,
                available=self.available,
            )
        )
        self.results.run(
            SchemaCheck(
                id="evaluation.evaluation_completed",
                tier="P0",
                evidence=("TE-6",),
                location="$",
                reason="required evidence does not match its schema",
                value=self.record,
                schema=required_object(evaluation_completed={"type": "boolean"}),
                depends_on=(),
                applies=self.scope.verifier,
                available=self.available,
            )
        )
        self.results.run(
            SchemaCheck(
                id="evaluation.mask_sample",
                tier="P0",
                evidence=("TE-6",),
                location="$",
                reason="required evidence does not match its schema",
                value=self.record,
                schema=required_object(mask_sample={"type": "boolean"}),
                depends_on=(),
                applies=self.scope.verifier,
                available=self.available,
            )
        )
        failed = self.record.get("mask_sample") is True or self.record.get("evaluation_completed") is False
        for field in ("failure_kind", "failure_reason"):
            self.results.run(
                SchemaCheck(
                    id="evaluation." + field,
                    tier="P0",
                    evidence=("TE-6",),
                    location="$",
                    reason="required evidence does not match its schema",
                    value=self.record,
                    schema=required_object(**{field: {"type": "string", "pattern": "\\S"}}),
                    depends_on=(),
                    applies=self.scope.verifier and failed,
                    available=self.available,
                )
            )
        sandbox = [r for r in self.observations if r.get("kind") == "sandbox"]
        self.results.run(
            SchemaCheck(
                id="sandbox.present",
                tier="P0",
                evidence=("TE-6",),
                location="$.ng_agent_observations.records",
                reason="required evidence does not match its schema",
                value=sandbox,
                schema={"type": "array", "minItems": 1},
                depends_on=(),
                applies=self.scope.require_sandbox,
                available=self.available,
            )
        )
        self.results.run(
            SchemaCheck(
                id="sandbox.identity",
                tier="P0",
                evidence=("TE-6",),
                location="$.ng_agent_observations.records",
                reason="required evidence does not match its schema",
                value=sandbox,
                schema={"type": "array", "items": required_object(sandbox_id={"type": "string", "pattern": "\\S"})},
                depends_on=("sandbox.present",) if self.scope.require_sandbox else (),
                applies=bool(sandbox) or self.scope.require_sandbox,
                available=self.available,
            )
        )
        self.results.run(
            SchemaCheck(
                id="sandbox.outcome",
                tier="P0",
                evidence=("TE-6",),
                location="$.ng_agent_observations.records",
                reason="required evidence does not match its schema",
                value=sandbox,
                schema={
                    "type": "array",
                    "items": required_object(
                        outcome={"enum": ["completed", "failed", "timeout", "oom", "sandbox_error", "cancelled"]}
                    ),
                },
                depends_on=("sandbox.present",) if self.scope.require_sandbox else (),
                applies=bool(sandbox) or self.scope.require_sandbox,
                available=self.available,
            )
        )
        self.results.run(
            SchemaCheck(
                id="sandbox.error",
                tier="P0",
                evidence=("TE-6",),
                location="$.ng_agent_observations.records",
                reason="required evidence does not match its schema",
                value=sandbox,
                schema={
                    "type": "array",
                    "items": {
                        "if": required_object(outcome={"enum": ["failed", "oom", "sandbox_error"]}),
                        "then": required_object(error_type={"type": "string", "pattern": "\\S"}),
                    },
                },
                depends_on=("sandbox.present",) if self.scope.require_sandbox else (),
                applies=bool(sandbox) or self.scope.require_sandbox,
                available=self.available,
            )
        )

    def gaps(self) -> None:
        # Canonical projection retains producer gaps. Duplicate attachment gaps are not compared.
        codes = [str(g.get("code", "")) for g in _objects(self.trajectory.get("gaps"))]
        self.results.run(
            SemanticCheck(
                id="calls.capture_gap",
                tier="P0",
                evidence=("TE-1", "TE-4", "TE-7", "TE-8", "TE-9"),
                location="$.ng_trajectory.gaps",
                reason="producer explicitly reports unavailable evidence",
                predicate=lambda: not any(
                    (c.startswith("model_call_capture") or c == "agent_observation_join_failed" for c in codes)
                ),
                available=self.available,
                applies=True,
                depends_on=("model_calls.present",),
            )
        )
        self.results.run(
            SemanticCheck(
                id="ownership.gap",
                tier="P0",
                evidence=("TE-8",),
                location="$.ng_trajectory.gaps",
                reason="producer explicitly reports unavailable evidence",
                predicate=lambda: not any(
                    (c.startswith("model_call_reference") or c == "model_call_ownership_unavailable" for c in codes)
                ),
                available=self.available,
                applies=True,
                depends_on=("model_calls.present",),
            )
        )
        self.results.run(
            SemanticCheck(
                id="steps.gap",
                tier="P0",
                evidence=("TE-3",),
                location="$.ng_trajectory.gaps",
                reason="producer explicitly reports unavailable evidence",
                predicate=lambda: not any(
                    (
                        c in {"turns_unavailable", "turn_evidence_incomplete", "trajectory_projection_failed"}
                        for c in codes
                    )
                ),
                available=self.available,
                applies=self.scope.steps,
                depends_on=("model_calls.present",),
            )
        )
        self.results.run(
            SemanticCheck(
                id="steps.accounting_gap",
                tier="P0",
                evidence=("TE-9",),
                location="$.ng_trajectory.gaps",
                reason="producer explicitly reports unavailable evidence",
                predicate=lambda: not "turn_model_call_scope_incomplete" in codes,
                available=self.available,
                applies=self.scope.steps,
                depends_on=("model_calls.present",),
            )
        )

    def result(self, source: str) -> dict:
        checks = self.results.dump()
        evidence = {}
        for key, name in NAMES.items():
            statuses = {c["status"] for c in checks if key in c["evidence"]}
            if (
                (key in {"TE-3", "TE-9"} and not self.scope.steps)
                or (key == "TE-5" and not self.scope.tools)
                or (key == "TE-6" and not self.scope.verifier)
            ):
                statuses = {"not_applicable"}
            verdict = (
                "not_fulfilled" if "fail" in statuses else "fulfilled" if "pass" in statuses else "not_applicable"
            )
            evidence[key] = {"name": name, "verdict": verdict, "basis": "retained_artifacts"}
        return {
            "source": source,
            "checks": checks,
            "evidence": evidence,
            "verdict": "fulfilled" if gate_passes(checks) else "not_fulfilled",
            "token_availability": {
                key: {
                    "available": sum(_mapping(c.get("token_stats")).get(key) is not None for c in self.calls),
                    "calls": len(self.calls),
                }
                for key in TOKEN_FIELDS
            },
            "is_behavioral_qualification": False,
            "scope_closure": "not_independently_witnessed",
            "findings": [
                {"evidence": te, "assertion": c["id"], "location": source + ":" + path, "reason": reason}
                for c in checks
                if c["status"] == "fail"
                for te in c["evidence"]
                for path in c["locations"]
                for reason in c["reasons"]
            ],
        }


def inspect_record(record: dict | None, *, source: str = "record", scope: EvidenceScope = EvidenceScope()) -> dict:
    """Validate designated fields; absent required input fails applicable checks."""
    inspector = Inspector(record, scope)
    inspector.model_calls()
    inspector.structure()
    inspector.ownership()
    inspector.evaluation()
    inspector.gaps()
    return inspector.result(source)
