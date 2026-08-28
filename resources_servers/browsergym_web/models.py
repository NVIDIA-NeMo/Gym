# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility exports for the backend-neutral web HTTP contract."""

from __future__ import annotations

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


__all__ = [
    "WebCloseResponse",
    "WebEvaluateRequest",
    "WebEvaluateResponse",
    "WebResetRequest",
    "WebSeedSessionRequest",
    "WebSeedSessionResponse",
    "WebSessionStatusResponse",
    "WebStepRequest",
    "WebStepResponse",
    "WebVerifyRequest",
    "WebVerifyResponse",
]
