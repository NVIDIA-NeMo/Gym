# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Stable fingerprints for the task surface used by an evaluation rollout."""

from __future__ import annotations

import hashlib
import json
from typing import Any


TASK_CONTRACT_SCHEMA = "automationbench.task-contract.v1"

# These verifier parameters treat a list as an unordered collection: every
# positive needle must be present, every negative needle must be absent, and
# recipient/attendee order is immaterial.  Keep this list deliberately narrow;
# other assertion arrays can encode an order that is part of the contract.
_ORDER_INSENSITIVE_ASSERTION_KEYS = {
    "attendees",
    "body_contains",
    "body_not_contains",
    "content_contains",
    "content_not_contains",
    "exact_bcc",
    "exact_cc",
    "exact_to",
    "subject_contains",
    "subject_not_contains",
    "text_contains",
    "text_contains_any",
    "text_contains_nonnegated_any",
    "text_not_contains",
}


def _plain_message(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        message = message.model_dump()
    if not isinstance(message, dict):
        raise TypeError("task prompt messages must be objects")
    return {"role": message.get("role"), "content": message.get("content")}


def _canonical_assertion_value(value: Any, *, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {
            child_key: _canonical_assertion_value(child_value, key=child_key)
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        normalized = [_canonical_assertion_value(item) for item in value]
        if key in _ORDER_INSENSITIVE_ASSERTION_KEYS:
            return sorted(
                normalized,
                key=lambda item: json.dumps(
                    item, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                ),
            )
        return normalized
    return value


def canonical_assertion_json(assertion: Any) -> str:
    """Serialize an assertion while ignoring verifier-irrelevant list order."""
    normalized = _canonical_assertion_value(assertion)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def task_contract_payload(
    *, example_id: Any, prompt: list[Any], info: dict[str, Any]
) -> dict[str, Any]:
    """Return the canonical task inputs whose changes require a fresh rollout."""
    assertions = info.get("assertions")
    initial_state = info.get("initial_state")
    tools = info.get("zapier_tools")
    if not isinstance(assertions, list):
        raise TypeError("task info.assertions must be a list")
    if not isinstance(initial_state, dict):
        raise TypeError("task info.initial_state must be an object")
    if not isinstance(tools, list) or any(not isinstance(tool, str) for tool in tools):
        raise TypeError("task info.zapier_tools must be a list of strings")

    canonical_assertions = sorted(canonical_assertion_json(assertion) for assertion in assertions)
    return {
        "schema": TASK_CONTRACT_SCHEMA,
        "example_id": example_id,
        "task_name": info.get("task_name"),
        "prompt": [_plain_message(message) for message in prompt],
        "assertions_multiset": canonical_assertions,
        "initial_state": initial_state,
        "zapier_tools": sorted(tools),
    }


def task_contract_sha256(*, example_id: Any, prompt: list[Any], info: dict[str, Any]) -> str:
    """Hash the exact task inputs used by a rollout."""
    payload = task_contract_payload(example_id=example_id, prompt=prompt, info=info)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()
