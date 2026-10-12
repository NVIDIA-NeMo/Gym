# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native evidence paths and their authoritative Gym models.

Validate retained JSON strictly before applying the stronger TE semantics. Missing
collections are handled by the capability checks and explicit applicability.
"""

from collections.abc import Iterator

from pydantic import TypeAdapter, ValidationError

from nemo_gym.base_responses_api_model import ModelCallRecord
from nemo_gym.rollout_observability import (
    AgentInvocation,
    AgentObservationRecord,
    TrajectoryModelCall,
    TrajectoryToolCall,
    TrajectoryTurn,
)


SCHEMA_VERSION = "gym-p0-evidence/v2"
TOKEN_FIELDS = ("tokens_in", "tokens_out", "tokens_reasoning", "tokens_total", "cached_tokens")
PATH_MODELS = {
    "ng_model_call_capture.calls": TypeAdapter(list[ModelCallRecord]),
    "ng_trajectory.model_calls": TypeAdapter(list[TrajectoryModelCall]),
    "ng_trajectory.turns": TypeAdapter(list[TrajectoryTurn]),
    "ng_trajectory.invocations": TypeAdapter(list[AgentInvocation]),
    "ng_trajectory.tool_calls": TypeAdapter(list[TrajectoryToolCall]),
    "ng_agent_observations.records": TypeAdapter(list[AgentObservationRecord]),
}


def model_errors(record: dict) -> Iterator[tuple[str, str]]:
    """Yield JSON pointers and error codes without copying source payloads."""
    for path, adapter in PATH_MODELS.items():
        parent, field = path.split(".")
        if parent not in record:
            continue
        owner = record[parent]
        location = f"/{parent}"
        if not isinstance(owner, dict):
            yield location, "path.object"
            continue
        if field not in owner:
            continue
        location += f"/{field}"
        try:
            adapter.validate_python(owner[field], strict=True)
        except ValidationError as exc:
            # Neither messages nor input/context are safe: validators can embed payloads.
            for error in exc.errors(include_url=False, include_context=False, include_input=False):
                parts = error["loc"]
                if path == "ng_agent_observations.records" and len(parts) > 1:
                    # Pydantic inserts the discriminated union variant after the index;
                    # it is not a field in the persisted object.
                    parts = (parts[0], *parts[2:])
                pointer = "".join("/" + str(part).replace("~", "~0").replace("/", "~1") for part in parts)
                yield location + pointer, "model." + error["type"]
