# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend-neutral session state and lifecycle errors for web runtimes."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from nemo_gym.web.api_models import WebStepResponse
from nemo_gym.web.models import WebObservation, WebTask, WebVerifierResult
from nemo_gym.web.operation_runner import WebOperationRunner
from nemo_gym.web.protocol import WebEnvironmentBackend
from nemo_gym.web.site_pool import SiteLease


class SessionNotFoundError(KeyError):
    pass


class SessionConflictError(RuntimeError):
    pass


class CapacityUnavailableError(RuntimeError):
    pass


class BenchmarkPreconditionError(RuntimeError):
    """A deterministic task/environment setup failure for the deployment."""


class EvaluatorConfigurationError(RuntimeError):
    """A required model-backed evaluator is not configured for a task."""


class EvaluatorInfrastructureError(RuntimeError):
    """A benchmark evaluator or post-action environment operation failed."""


@dataclass
class WebSessionState:
    """Process-local state shared by BrowserGym and native web backends."""

    session_id: str
    task: WebTask
    backend: WebEnvironmentBackend
    site_lease: SiteLease
    observation: WebObservation
    seed_info: dict[str, Any]
    created_at: float
    last_access_at: float
    operation_runner: WebOperationRunner
    status: str = "ready"
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    operations: OrderedDict[str, WebStepResponse] = field(default_factory=OrderedDict)
    verifier_result: WebVerifierResult | None = None
