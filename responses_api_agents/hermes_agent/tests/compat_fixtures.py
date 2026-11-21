# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Restore the real class/registry together; this fixture never bypasses a pin."""

import pytest
from run_agent import AIAgent

from responses_api_agents.hermes_agent import length_transcript


@pytest.fixture(autouse=True)
def restore_length_transcript_state():
    native_method = AIAgent.run_conversation
    registry = dict(length_transcript._installed)
    try:
        yield
    finally:
        AIAgent.run_conversation = native_method
        length_transcript._installed.clear()
        length_transcript._installed.update(registry)
