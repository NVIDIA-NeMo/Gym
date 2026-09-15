# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Episode processor extension points."""

from nemo_gym.processors.base import BaseProcessor, BaseProcessorConfig
from nemo_gym.processors.contracts import (
    AgentTurn,
    EpisodeFailure,
    EpisodeId,
    EpisodeRequest,
    EpisodeResponse,
    EpisodeSeedRequest,
    EpisodeSeedResponse,
    EpisodeVerification,
    EpisodeVerifyRequest,
    ResourcesSessionCloseRequest,
    ResourcesSessionCloseResponse,
    TaskIdentity,
)


__all__ = [
    "AgentTurn",
    "BaseProcessor",
    "BaseProcessorConfig",
    "EpisodeFailure",
    "EpisodeId",
    "EpisodeRequest",
    "EpisodeResponse",
    "EpisodeSeedRequest",
    "EpisodeSeedResponse",
    "EpisodeVerifyRequest",
    "EpisodeVerification",
    "ResourcesSessionCloseRequest",
    "ResourcesSessionCloseResponse",
    "TaskIdentity",
]
