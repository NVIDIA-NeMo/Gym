# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-light contracts shared by NeMo Gym web agents and runtimes."""

from nemo_gym.web.actions import ActionParseError, parse_model_action
from nemo_gym.web.api_models import (
    WebCloseResponse,
    WebEvaluateRequest,
    WebEvaluateResponse,
    WebResetRequest,
    WebSeedSessionRequest,
    WebSeedSessionResponse,
    WebSessionStatusResponse,
    WebStepRequest,
    WebStepResponse,
    WebVerifyRequest,
    WebVerifyResponse,
)
from nemo_gym.web.artifacts import WebArtifactStore
from nemo_gym.web.composed_backend import ComposedWebBackend, WebBrowserDriver, WebTaskEvaluator
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
from nemo_gym.web.operation_runner import (
    DirectWebOperationRunner,
    ThreadAffineWebOperationRunner,
    WebOperationRunner,
)
from nemo_gym.web.session import (
    BenchmarkPreconditionError,
    CapacityUnavailableError,
    EvaluatorConfigurationError,
    EvaluatorInfrastructureError,
    SessionConflictError,
    SessionNotFoundError,
    WebSessionState,
)
from nemo_gym.web.site_pool import LocalSiteLockPool, SiteLease, SitePool, UnmanagedSitePool


__all__ = [
    "ActionParseError",
    "BenchmarkPreconditionError",
    "CapacityUnavailableError",
    "ComposedWebBackend",
    "DirectWebOperationRunner",
    "EvaluatorConfigurationError",
    "EvaluatorInfrastructureError",
    "LocalSiteLockPool",
    "SiteLease",
    "SitePool",
    "SessionConflictError",
    "SessionNotFoundError",
    "ThreadAffineWebOperationRunner",
    "UnmanagedSitePool",
    "WebCloseResponse",
    "WebEvaluateRequest",
    "WebEvaluateResponse",
    "WebAction",
    "WebActionProfile",
    "WebArtifactRef",
    "WebArtifactStore",
    "WebBenchmark",
    "WebBrowserDriver",
    "WebImage",
    "WebObservation",
    "WebObservationProfile",
    "WebOperationRunner",
    "WebResetRequest",
    "WebRuntimeProfile",
    "WebSessionState",
    "WebSeedSessionRequest",
    "WebSeedSessionResponse",
    "WebSessionStatusResponse",
    "WebStepRequest",
    "WebStepResponse",
    "WebStepResult",
    "WebTab",
    "WebTask",
    "WebTaskEvaluator",
    "WebVerifierResult",
    "WebVerifyRequest",
    "WebVerifyResponse",
    "parse_model_action",
]
