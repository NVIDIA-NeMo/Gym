# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime protocol implemented by BrowserGym and optional legacy backends."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from nemo_gym.web.models import WebAction, WebObservation, WebStepResult, WebTask, WebVerifierResult


@runtime_checkable
class WebEnvironmentBackend(Protocol):
    def reset(self, task: WebTask) -> tuple[WebObservation, dict[str, Any]]: ...

    def observe(self) -> WebObservation: ...

    def step(self, action: WebAction) -> WebStepResult: ...

    def evaluate(self, final_answer: str | None = None) -> WebVerifierResult: ...

    def close(self) -> None: ...
