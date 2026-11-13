# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed model-outcome events for a scored Harbor episode whose agent run ended on a limit that is part of the task.

Rollout health flags a rollout whose last model call failed, unless the row names why with a typed event
(``rollout_ended_on_failed_model_call``). Two endings are the model's own result and are scored here like any other
answer: the harness stopped at the task's ``[agent].timeout_sec`` (the model call in flight is cancelled), and the
model server refused a prompt over its image limit with a 400 (the agent kept more images in context than the
deployment serves). This module reads both from the agent's response and observations and returns the row's
``timeouts`` (a ``TimeoutEvent`` with ``model_outcome: true``) and ``model_limit_rejections`` entries.
"""

import re
from typing import Any

from pydantic import ValidationError

from nemo_gym.agents import AGENT_OUTCOME_METADATA_KEY, AgentOutcome
from nemo_gym.core import NeMoGymResponse
from nemo_gym.trajectory import AgentInvocation, AgentObservationBundle


# vLLM's refusal of a prompt over --limit-mm-per-prompt.
IMAGE_LIMIT_MESSAGE = re.compile(r"At most \d+ image\(s\) may be provided in one prompt")


def _budget_cut(response: NeMoGymResponse, observations: AgentObservationBundle | None) -> bool:
    """Whether the harness stopped the run at its time budget: Stirrup's ``stop_reason: wall_time`` on the response,
    or OpenCode's top-level invocation with ``error_type: timeout``. A continued session's observations hold every
    activation; only the last one's invocation counts."""
    if response.status != "incomplete":
        return False
    if (response.metadata or {}).get("stop_reason") == "wall_time":
        return True
    roots = [
        record
        for record in (observations.records if observations is not None else [])
        if isinstance(record, AgentInvocation) and record.parent_invocation_id is None
    ]
    last = max((root.activation_id or 0 for root in roots), default=0)
    return any(root.error_type == "timeout" for root in roots if (root.activation_id or 0) == last)


def _image_limit_message(response: NeMoGymResponse) -> str | None:
    """The model server's image-limit refusal the response failed on, when its harness classified it as a 400."""
    if response.status != "failed" or response.error is None:
        return None
    raw = (response.metadata or {}).get(AGENT_OUTCOME_METADATA_KEY)
    try:
        outcome = AgentOutcome.model_validate_json(raw) if raw is not None else None
    except ValidationError:
        return None
    message = response.error.message or ""
    if outcome is None or outcome.status_code != 400 or not IMAGE_LIMIT_MESSAGE.search(message):
        return None
    return message


def model_outcome_events(
    response: NeMoGymResponse | None,
    observations: AgentObservationBundle | None,
    *,
    event_id: str,
    harness: str,
    agent_timeout_sec: float | None,
) -> dict[str, list[dict[str, Any]]]:
    """The ``timeouts`` and ``model_limit_rejections`` entries for a scored episode's last agent run, keyed by the
    row field; empty when the run ended otherwise. ``event_id`` names the episode, ``harness`` the agent server."""
    if response is None:
        return {}
    if _budget_cut(response, observations):
        event: dict[str, Any] = {
            "event_id": f"{event_id}:agent-budget",
            "limit": "agent.timeout_sec",
            "enforcer": harness,
            "evidence": "internal",
            "model_outcome": True,
        }
        if agent_timeout_sec is not None:
            event["configured_s"] = float(agent_timeout_sec)
        return {"timeouts": [event]}
    message = _image_limit_message(response)
    if message is not None:
        rejection = {
            "event_id": f"{event_id}:image-limit",
            "limit": "limit_mm_per_prompt.image",
            "enforcer": "model_server",
            "status_code": 400,
            "message": message,
        }
        return {"model_limit_rejections": [rejection]}
    return {}
