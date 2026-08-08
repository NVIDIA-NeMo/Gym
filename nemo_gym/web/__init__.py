# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-light contracts shared by NeMo Gym web agents and runtimes."""

from nemo_gym.web.actions import ActionParseError, parse_model_action
from nemo_gym.web.models import (
    WebAction,
    WebActionProfile,
    WebArtifactRef,
    WebBenchmark,
    WebImage,
    WebObservation,
    WebObservationProfile,
    WebRuntimeProfile,
    WebStepResult,
    WebTab,
    WebTask,
    WebVerifierResult,
)


__all__ = [
    "ActionParseError",
    "WebAction",
    "WebActionProfile",
    "WebArtifactRef",
    "WebBenchmark",
    "WebImage",
    "WebObservation",
    "WebObservationProfile",
    "WebRuntimeProfile",
    "WebStepResult",
    "WebTab",
    "WebTask",
    "WebVerifierResult",
    "parse_model_action",
]
