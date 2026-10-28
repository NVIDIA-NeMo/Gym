# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep each model call's token metadata once in a rollout record."""

from typing import Any, Mapping

from nemo_gym.openai_utils import TOKEN_METADATA_FIELDS
from nemo_gym.rollout_observability import NG_AGENT_OBSERVATIONS_KEY, NG_TRAJECTORY_KEY


def _present_token_metadata(item: Mapping[str, Any]) -> set[str]:
    return {field for field in TOKEN_METADATA_FIELDS if item.get(field) is not None}


def _strip_matching_items(value: Any, canonical: Mapping[str, set[str]]) -> int:
    """Remove the token metadata a canonical output item already holds from every item that repeats it."""
    removed = 0
    if isinstance(value, dict):
        item_id = value.get("id")
        for field in canonical.get(item_id, ()) if isinstance(item_id, str) else ():
            if field in value:
                del value[field]
                removed += 1
        for child in value.values():
            removed += _strip_matching_items(child, canonical)
    elif isinstance(value, list):
        for child in value:
            removed += _strip_matching_items(child, canonical)
    return removed


def strip_duplicate_token_metadata(result: dict[str, Any]) -> int:
    """Keep token metadata only on ``response.output`` and drop the copies of it elsewhere in the record.

    Copies appear in the trajectory (conversation, turns, captured requests and responses) and in the agent
    observations. A copy is stripped only when ``response.output`` holds an item with the same ``id`` that
    carries the same field, so metadata is never removed without a canonical twin. Item ids stay, so a
    copy still joins to its canonical item. Returns the number of fields removed.
    """
    response = result.get("response")
    output = response.get("output") if isinstance(response, Mapping) else None
    canonical = {
        item["id"]: present
        for item in output or []
        if isinstance(item, Mapping) and isinstance(item.get("id"), str) and (present := _present_token_metadata(item))
    }
    if not canonical:
        return 0
    return sum(
        _strip_matching_items(result.get(key), canonical) for key in (NG_TRAJECTORY_KEY, NG_AGENT_OBSERVATIONS_KEY)
    )
