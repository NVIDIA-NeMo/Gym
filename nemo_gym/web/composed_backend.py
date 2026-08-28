# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend-neutral composition of a web browser driver and task evaluator."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from nemo_gym.web.models import (
    WebAction,
    WebObservation,
    WebStepResult,
    WebTask,
    WebVerifierResult,
)


@runtime_checkable
class WebBrowserDriver(Protocol):
    """Own the live browser only; model prompting and scoring live elsewhere."""

    def reset(self, task: WebTask) -> tuple[WebObservation, dict[str, Any]]: ...

    def observe(self) -> WebObservation: ...

    def step(self, action: WebAction) -> WebStepResult: ...

    def evaluation_context(self) -> Any:
        """Return process-local state required by a colocated evaluator."""
        ...

    def close(self) -> None: ...


@runtime_checkable
class WebTaskEvaluator(Protocol):
    """Score a completed task without owning the browser lifecycle."""

    def prepare(
        self,
        *,
        task: WebTask,
        observation: WebObservation,
        browser_context: Any,
    ) -> None:
        """Capture evaluator state required before the policy starts acting."""
        ...

    def evaluate(
        self,
        *,
        task: WebTask,
        observation: WebObservation,
        final_answer: str | None,
        browser_context: Any,
    ) -> WebVerifierResult: ...

    def close(self) -> None: ...


class ComposedWebBackend:
    """Implement the common environment protocol from independent roles.

    The driver owns Chromium/Playwright/PyAutoGUI mechanics. The evaluator
    owns benchmark scoring. The agent remains an HTTP client of this backend
    and never imports either implementation.
    """

    def __init__(self, driver: WebBrowserDriver, evaluator: WebTaskEvaluator) -> None:
        self.driver = driver
        self.evaluator = evaluator
        self._task: WebTask | None = None
        self._observation: WebObservation | None = None
        self._closed = False

    def reset(self, task: WebTask) -> tuple[WebObservation, dict[str, Any]]:
        if self._closed:
            raise RuntimeError("a closed web backend cannot be reset")
        if self._task is not None:
            # A session reset starts a new evaluator lifecycle. In particular,
            # WebArena-family before-state snapshots must never survive into a
            # retry of the same task. Invalidate the active state before
            # cleanup so a close failure cannot expose the old lifecycle.
            self._task = None
            self._observation = None
            self.evaluator.close()
        observation, info = self.driver.reset(task)
        try:
            self.evaluator.prepare(
                task=task,
                observation=observation,
                browser_context=self.driver.evaluation_context(),
            )
        except BaseException:
            # A failed before-state capture leaves no valid rollout. Release
            # the live browser and any partially prepared evaluator state
            # immediately instead of retaining partial state. Cleanup errors
            # must not hide the deterministic prepare failure.
            for cleanup in (self.driver.close, self.evaluator.close):
                try:
                    cleanup()
                except BaseException:
                    pass
            raise
        self._task = task
        self._observation = observation
        return observation, info

    def observe(self) -> WebObservation:
        self._require_active()
        observation = self.driver.observe()
        self._observation = observation
        return observation

    def step(self, action: WebAction) -> WebStepResult:
        self._require_active()
        result = self.driver.step(action)
        self._observation = result.observation
        return result

    def evaluate(self, final_answer: str | None = None) -> WebVerifierResult:
        task, observation = self._require_active()
        return self.evaluator.evaluate(
            task=task,
            observation=observation,
            final_answer=final_answer,
            browser_context=self.driver.evaluation_context(),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        first_error: BaseException | None = None
        try:
            self.driver.close()
        except BaseException as exc:  # Cleanup must still close the evaluator.
            first_error = exc
        try:
            self.evaluator.close()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
        finally:
            self._task = None
            self._observation = None
        if first_error is not None:
            raise first_error

    def _require_active(self) -> tuple[WebTask, WebObservation]:
        if self._closed:
            raise RuntimeError("web backend is closed")
        if self._task is None or self._observation is None:
            raise RuntimeError("web backend must be reset before use")
        return self._task, self._observation
