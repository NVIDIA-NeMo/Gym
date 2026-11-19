# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The record of one message that a harness adds to a request in the model's name."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HarnessNote:
    """One note a harness writes as an assistant message, with the harness releases checked to send this exact text.

    ``harness`` names the agent harness (for example ``opencode``) and ``name`` the note within it. A release goes into
    ``checked_versions`` only after its text was compared with ``text``, so a new default release of the harness cannot
    silently break capture: the harness's own tests check that its default release is listed.
    """

    harness: str
    name: str
    text: str
    checked_versions: frozenset[str]
