# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Normalize episode boundaries from captured requests without changing agent history."""

import hashlib
import json
from copy import deepcopy

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming


SCHEMA = "mini-swe-gym-episode-v2"
# Explicit generation settings only. Transport credentials/headers and arbitrary metadata are excluded.
REQUEST_FIELDS = frozenset(
    {
        "model",
        "input",
        "instructions",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "max_output_tokens",
        "max_tool_calls",
        "temperature",
        "top_p",
        "top_logprobs",
        "reasoning",
        "text",
        "truncation",
        "store",
        "stream",
        "include",
        "service_tier",
        "background",
        "context_management",
        "moderation",
    }
)


def request_snapshot(parameters):
    return deepcopy({key: value for key, value in parameters.items() if key in REQUEST_FIELDS})


def snapshot_hash(snapshot):
    if snapshot is None:
        return None
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def canonical_input(items):
    """Apply the same SDK defaults to captured request and stored trajectory items."""
    return NeMoGymResponseCreateParamsNonStreaming(input=deepcopy(items)).model_dump(mode="json")["input"]


def episode_export(trajectory, capture):
    """Return normalized request/output plus explicit validity for incomplete captures.

    Unobserved or inconsistent requests produce no guessed training items. The complete raw
    trajectory remains in the self-contained audit archive and grading still runs.
    """
    request = capture["first_request"]
    metadata = {
        "schema": SCHEMA,
        "capture_scope": "agent_to_gym_model_request",
        "exact_downstream_wire_capture": False,
        "flattened_history_is_exact_context_replay": False,
        "state": capture["state"],
        "explicit_request_parameters": sorted(request) if request is not None else [],
        "unsent_parameter_values": "SDK serialization defaults; not observed model parameters",
        "first_request_sha256": capture["first_request_sha256"],
        "prompt_boundary": None,
        "normalization_valid": False,
        "error": None,
    }
    empty_request = NeMoGymResponseCreateParamsNonStreaming(input=[], tools=[]).model_dump(mode="json")
    if request is None:
        return empty_request, [], metadata
    try:
        initial = canonical_input(request["input"])
        boundary = len(initial)
        if canonical_input(trajectory[:boundary]) != initial:
            raise ValueError("Captured first request does not match the full-trajectory prefix")
        normalized = NeMoGymResponseCreateParamsNonStreaming.model_validate(request).model_dump(mode="json")
    except (ValueError, TypeError, KeyError) as error:
        metadata.update(state="boundary_error", error=str(error))
        return empty_request, [], metadata
    metadata.update(prompt_boundary=boundary, normalization_valid=True)
    return normalized, list(trajectory[boundary:]), metadata
