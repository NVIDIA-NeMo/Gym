# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kilo 7.4's OpenCode-compatible persisted observations and token accounting."""

import json
import sqlite3
from pathlib import Path

from nemo_gym.openai_utils import (
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.rollout_observability import AgentInvocation, AgentObservationBundle, ObservationGap
from responses_api_agents.opencode_sandboxed_agent.app import parse_opencode_observations


def read_kilo_observations(
    database: Path, *, fallback_invocation_id: str
) -> tuple[AgentObservationBundle, NeMoGymResponseUsage | None]:
    """Reuse the pinned session-tree contract; sum usage over persisted provider steps once."""
    observations = parse_opencode_observations(database, fallback_invocation_id)
    observations.source = "kilocode"
    usages = []
    incomplete = False
    started: set[str] = set()
    finished: set[str] = set()
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        roots = connection.execute("select id from session where parent_id is null").fetchall()
        if len(roots) != 1:
            raise ValueError(f"Expected one Kilo root session, found {len(roots)}")
        message_rows = connection.execute("select session_id, data from message").fetchall()
        # Step parts distinguish actual model usage from synthetic assistant bookkeeping.
        rows = connection.execute("select message_id, data from part order by time_created, id").fetchall()
    invocations = {
        record.invocation_id: record for record in observations.records if isinstance(record, AgentInvocation)
    }
    for session_id, data in message_rows:
        try:
            message = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            continue
        error = message.get("error") if isinstance(message, dict) else None
        name = error.get("name") if isinstance(error, dict) else None
        if session_id in invocations and isinstance(name, str) and name:
            invocations[session_id].error_type = name
    for message_id, data in rows:
        try:
            part = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            incomplete = True
            continue
        if not isinstance(part, dict):
            incomplete = True
            continue
        if part.get("type") == "step-start":
            started.add(message_id)
        if part.get("type") != "step-finish":
            continue
        finished.add(message_id)
        tokens = part.get("tokens")
        if not isinstance(tokens, dict):
            incomplete = True
            continue
        cache = tokens.get("cache")
        cache = cache if isinstance(cache, dict) else {}

        def count(value: object) -> int | None:
            return value if type(value) is int and value >= 0 else None

        input_tokens, output_tokens = count(tokens.get("input")), count(tokens.get("output"))
        if input_tokens is None or output_tokens is None:
            incomplete = True
            continue
        cache_read, cache_write = count(cache.get("read")), count(cache.get("write"))
        reasoning = count(tokens.get("reasoning"))
        prompt = input_tokens + (cache_read or 0) + (cache_write or 0)
        completion = output_tokens + (reasoning or 0)
        usages.append(
            NeMoGymResponseUsage(
                input_tokens=prompt,
                output_tokens=completion,
                total_tokens=prompt + completion,
                # Kilo defaults missing backend counters to zero before persistence. A stored
                # zero cannot prove a reported zero; positive counters do establish details.
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=cache_read or None),
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=reasoning or None),
            )
        )
    usage = NeMoGymResponseUsage.sum_from_list(usages) if usages else None
    if incomplete or started - finished:
        observations.gaps.append(ObservationGap(code="usage_totals_incomplete"))
        observations.gaps.append(ObservationGap(code="turn_model_call_scope_incomplete"))
        if usage is not None:
            usage.input_tokens_details.cached_tokens = None
            usage.output_tokens_details.reasoning_tokens = None
    if (
        usage is None
        or usage.input_tokens_details.cached_tokens is None
        or usage.output_tokens_details.reasoning_tokens is None
    ):
        observations.gaps.append(
            ObservationGap(
                code="usage_details_unavailable", detail="Persisted optional counters are absent or defaulted zeros"
            )
        )
    return observations, usage
