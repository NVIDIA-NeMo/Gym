# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable rollout processors provided by NeMo Gym."""

from nemo_gym.processors.base import BaseProcessor, BaseProcessorConfig
from nemo_gym.processors.multi_agent import (
    EpisodeEvent,
    EpisodeStatus,
    MultiAgentEpisodeSpec,
    MultiAgentProcessor,
    MultiAgentProcessorConfig,
    MultiAgentRunRequest,
    MultiAgentVerifyRequest,
    MultiAgentVerifyResponse,
    ParticipantTurn,
)
from nemo_gym.processors.single_agent_turn import (
    SingleAgentTurnProcessor,
    SingleAgentTurnProcessorConfig,
)


__all__ = [
    "BaseProcessor",
    "BaseProcessorConfig",
    "SingleAgentTurnProcessor",
    "SingleAgentTurnProcessorConfig",
    "EpisodeEvent",
    "EpisodeStatus",
    "MultiAgentEpisodeSpec",
    "MultiAgentProcessor",
    "MultiAgentProcessorConfig",
    "MultiAgentRunRequest",
    "MultiAgentVerifyRequest",
    "MultiAgentVerifyResponse",
    "ParticipantTurn",
]
