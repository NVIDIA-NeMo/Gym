# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lineage hashing scales with appended content, not repeated policy evaluations."""

import pytest

import nemo_gym.agents.context_management.history as history_module
from nemo_gym.agents.context_management.config import RecencyHistoryPolicyConfig
from nemo_gym.agents.context_management.controller import HistoryController
from nemo_gym.agents.context_management.history import SemanticHistory
from nemo_gym.agents.context_management.policies import IdentityHistoryPolicy, RecencyHistoryPolicy


@pytest.mark.parametrize("recency", [False, True])
def test_source_digests_are_computed_once_and_preserve_lineage(monkeypatch, recency):
    canonical = history_module.canonical_digest
    hashed_parts = []

    def observed(value):
        if isinstance(value, dict) and set(value) == {"kind", "payload", "observation_group_id"}:
            hashed_parts.append(value)
        return canonical(value)

    monkeypatch.setattr(history_module, "canonical_digest", observed)
    history = SemanticHistory("digest-regression")
    for turn in range(5):
        history.append_items(
            [
                {
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": "data:image/png;base64," + "A" * 100_000}],
                }
            ],
            turn_id=turn,
        )
    assert len(hashed_parts) == 5
    policy = (
        RecencyHistoryPolicy(
            RecencyHistoryPolicyConfig.model_validate({"images": {"enabled": True, "keep_last_groups": 1}})
        )
        if recency
        else IdentityHistoryPolicy()
    )
    controller = HistoryController(history, policy)
    first = controller.prepare(applies_to_step=6)
    assert controller.prepare(applies_to_step=6) == first
    assert len(hashed_parts) == 5  # Including omitted images: no old payload is hashed again.
    expected = tuple(
        canonical(
            {
                "kind": part.kind,
                "payload": event.item["content"][part.content_index],
                "observation_group_id": part.observation_group_id,
            }
        )
        for event, part in history.parts
    )
    assert tuple(record.source_digest for record in first.view.decision.lineage.unit_records) == expected
    history.append_items([{"role": "user", "content": "new observation"}], turn_id=6)
    assert len(hashed_parts) == 6
    controller.prepare(applies_to_step=7)
    assert len(hashed_parts) == 6
