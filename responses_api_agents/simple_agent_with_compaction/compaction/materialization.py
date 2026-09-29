# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize policy plans into Responses-API request views."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from responses_api_agents.simple_agent_with_compaction.compaction.history import (
    HistoryViewPlan,
    MaterializedHistoryView,
    SemanticHistory,
)


def _marker_part_for(image_part: Mapping[str, Any], text: str) -> dict[str, str]:
    part_type = image_part.get("type")
    if part_type == "input_image":
        return {"type": "input_text", "text": text}
    return {"type": "text", "text": text}


def materialize_history_view(history: SemanticHistory, plan: HistoryViewPlan) -> MaterializedHistoryView:
    retained = plan.retained_part_ids
    artifacts_by_anchor = {artifact.anchor_part_id: artifact for artifact in plan.artifacts}
    items: list[Mapping[str, Any]] = []
    media_ids: list[str] = []
    descriptor: list[str] = []

    for event in history.events:
        item = deepcopy(dict(event.item))
        content = item.get("content")
        if isinstance(content, list):
            materialized_content: list[Any] = []
            for part in event.parts:
                assert part.content_index is not None
                source_part = content[part.content_index]
                artifact = artifacts_by_anchor.get(part.part_id)
                if artifact is not None:
                    if not isinstance(source_part, Mapping):
                        raise TypeError(f"Cannot anchor omission marker at non-mapping part {part.part_id}")
                    materialized_content.append(_marker_part_for(source_part, artifact.text))
                    descriptor.append(f"artifact:{artifact.artifact_id}")

                if part.part_id not in retained:
                    continue
                if part.kind == "image":
                    if part.media_id is None:
                        raise ValueError(f"Image part {part.part_id} has no media ID")
                    materialized_content.append(deepcopy(dict(history.media_arena.resolve(part.media_id))))
                    media_ids.append(part.media_id)
                else:
                    materialized_content.append(deepcopy(source_part))
                descriptor.append(f"part:{part.part_id}")

            if not materialized_content:
                continue
            item["content"] = materialized_content
            items.append(item)
            continue

        part = event.parts[0]
        artifact = artifacts_by_anchor.get(part.part_id)
        if artifact is not None:
            items.append(
                {
                    "role": event.role if event.role != "unknown" else "user",
                    "type": "message",
                    "content": artifact.text,
                }
            )
            descriptor.append(f"artifact:{artifact.artifact_id}")
        if part.part_id in retained:
            items.append(item)
            descriptor.append(f"part:{part.part_id}")

    return MaterializedHistoryView(
        items=tuple(items),
        media_ids=tuple(media_ids),
        descriptor=tuple(descriptor),
        decision=plan.decision,
    )


def descriptor_is_append_compatible(
    previous_completed_descriptor: Sequence[str] | None,
    current_descriptor: Sequence[str],
) -> bool:
    """Return whether the current semantic view only appends to the prior one."""

    if previous_completed_descriptor is None:
        return False
    prefix = tuple(previous_completed_descriptor)
    current = tuple(current_descriptor)
    return len(current) >= len(prefix) and current[: len(prefix)] == prefix


def ordered_media_is_append_compatible(
    previous_media_ids: Sequence[str] | None,
    current_media_ids: Sequence[str],
) -> bool:
    if previous_media_ids is None:
        return False
    prefix = tuple(previous_media_ids)
    current = tuple(current_media_ids)
    return len(current) >= len(prefix) and current[: len(prefix)] == prefix
