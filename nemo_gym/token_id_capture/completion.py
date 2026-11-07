# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visible output fingerprints shared with framework-owned call selection.

Uses Gym main's completion helper with this branch's existing fingerprint and
exact-reasoning implementations.
"""

from typing import Any

from nemo_gym.token_id_capture.fingerprint import assistant_fingerprint
from nemo_gym.token_id_capture.replay import replay_context


def output_item_fingerprint(item: dict[str, Any]) -> str:
    """Reuse dialect-neutral visible fingerprints and exact reasoning evidence."""
    if item.get("type") == "reasoning":
        return replay_context([item]).items[0].digest
    result = assistant_fingerprint([item])
    if not result:
        raise ValueError("Expected a model-authored output item")
    return result
