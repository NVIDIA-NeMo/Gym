# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable rollout processors provided by NeMo Gym."""

from nemo_gym.processors.base import BaseProcessor, BaseProcessorConfig
from nemo_gym.processors.single_agent_turn import (
    SingleAgentTurnProcessor,
    SingleAgentTurnProcessorConfig,
)
from nemo_gym.processors.user_assistant import (
    EpisodeEvent,
    EpisodeStatus,
    ParticipantTurn,
    UserAssistantProcessor,
    UserAssistantProcessorConfig,
    UserAssistantRunRequest,
    UserAssistantVerifyRequest,
    UserAssistantVerifyResponse,
)


__all__ = [
    "BaseProcessor",
    "BaseProcessorConfig",
    "SingleAgentTurnProcessor",
    "SingleAgentTurnProcessorConfig",
    "EpisodeEvent",
    "EpisodeStatus",
    "ParticipantTurn",
    "UserAssistantProcessor",
    "UserAssistantProcessorConfig",
    "UserAssistantRunRequest",
    "UserAssistantVerifyRequest",
    "UserAssistantVerifyResponse",
]
