# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from nemo_gym.web.composed_backend import ComposedWebBackend, WebBrowserDriver, WebTaskEvaluator
from nemo_gym.web.models import (
    WebAction,
    WebBenchmark,
    WebObservation,
    WebRuntimeProfile,
    WebStepResult,
    WebTask,
    WebVerifierResult,
)
from nemo_gym.web.protocol import WebEnvironmentBackend


class _Driver:
    def __init__(self) -> None:
        self.task: WebTask | None = None
        self.observation = WebObservation()
        self.closed = False

    def reset(self, task: WebTask) -> tuple[WebObservation, dict[str, Any]]:
        self.task = task
        self.observation = WebObservation(url="https://example.test/start")
        return self.observation, {"driver": "fake-native-visual"}

    def observe(self) -> WebObservation:
        return self.observation

    def step(self, action: WebAction) -> WebStepResult:
        self.observation = WebObservation(
            url="https://example.test/done",
            last_action=action.script,
        )
        return WebStepResult(observation=self.observation, execution_ok=True)

    def evaluation_context(self) -> Any:
        return {"task_id": self.task.task_id if self.task else None}

    def close(self) -> None:
        self.closed = True


class _Evaluator:
    def __init__(self) -> None:
        self.prepared: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    def prepare(
        self,
        *,
        task: WebTask,
        observation: WebObservation,
        browser_context: Any,
    ) -> None:
        self.prepared.append(
            {
                "task": task,
                "observation": observation,
                "browser_context": browser_context,
            }
        )

    def evaluate(
        self,
        *,
        task: WebTask,
        observation: WebObservation,
        final_answer: str | None,
        browser_context: Any,
    ) -> WebVerifierResult:
        self.calls.append(
            {
                "task": task,
                "observation": observation,
                "final_answer": final_answer,
                "browser_context": browser_context,
            }
        )
        success = observation.url.endswith("/done") and final_answer == "done"
        return WebVerifierResult(
            reward=float(success),
            raw_score=float(success),
            task_success=success,
            verifier_version="fake-evaluator-v1",
        )

    def close(self) -> None:
        self.closed = True


def _task() -> WebTask:
    return WebTask(
        benchmark=WebBenchmark.WEBARENA,
        task_id="0",
        runtime_profile=WebRuntimeProfile.NATIVE_VISUAL,
    )


def test_composed_backend_is_protocol_compatible_and_keeps_roles_separate() -> None:
    driver = _Driver()
    evaluator = _Evaluator()
    backend = ComposedWebBackend(driver, evaluator)

    assert isinstance(driver, WebBrowserDriver)
    assert isinstance(evaluator, WebTaskEvaluator)
    assert isinstance(backend, WebEnvironmentBackend)

    observation, info = backend.reset(_task())
    assert observation.url.endswith("/start")
    assert info == {"driver": "fake-native-visual"}
    assert evaluator.prepared[0]["browser_context"] == {"task_id": "0"}
    assert backend.observe() == observation

    result = backend.step(WebAction(name="click", script="click(10, 20)"))
    assert result.observation.last_action == "click(10, 20)"

    score = backend.evaluate("done")
    assert score.task_success
    assert evaluator.calls[0]["browser_context"] == {"task_id": "0"}

    backend.close()
    backend.close()
    assert driver.closed
    assert evaluator.closed


def test_composed_backend_fails_closed_before_reset_and_after_close() -> None:
    backend = ComposedWebBackend(_Driver(), _Evaluator())

    with pytest.raises(RuntimeError, match="reset before use"):
        backend.observe()
    backend.close()
    with pytest.raises(RuntimeError, match="closed"):
        backend.reset(_task())


def test_close_attempts_both_roles_when_driver_cleanup_fails() -> None:
    driver = _Driver()
    evaluator = _Evaluator()

    def fail_close() -> None:
        driver.closed = True
        raise RuntimeError("driver cleanup failed")

    driver.close = fail_close  # type: ignore[method-assign]
    backend = ComposedWebBackend(driver, evaluator)

    with pytest.raises(RuntimeError, match="driver cleanup failed"):
        backend.close()
    assert driver.closed
    assert evaluator.closed


def test_close_propagates_evaluator_cleanup_failure_and_clears_state() -> None:
    driver = _Driver()
    evaluator = _Evaluator()

    def fail_close() -> None:
        evaluator.closed = True
        raise RuntimeError("evaluator cleanup failed")

    evaluator.close = fail_close  # type: ignore[method-assign]
    backend = ComposedWebBackend(driver, evaluator)
    backend.reset(_task())

    with pytest.raises(RuntimeError, match="evaluator cleanup failed"):
        backend.close()
    assert driver.closed
    assert evaluator.closed
    with pytest.raises(RuntimeError, match="backend is closed"):
        backend.observe()


def test_prepare_failure_releases_live_driver_before_propagating() -> None:
    driver = _Driver()
    evaluator = _Evaluator()

    def fail_prepare(**_kwargs) -> None:
        raise RuntimeError("before-state capture failed")

    evaluator.prepare = fail_prepare  # type: ignore[method-assign]
    backend = ComposedWebBackend(driver, evaluator)

    with pytest.raises(RuntimeError, match="before-state capture failed"):
        backend.reset(_task())
    assert driver.closed


def test_reset_starts_a_fresh_evaluator_lifecycle() -> None:
    driver = _Driver()
    evaluator = _Evaluator()
    close_count = 0

    def track_close() -> None:
        nonlocal close_count
        close_count += 1

    evaluator.close = track_close  # type: ignore[method-assign]
    backend = ComposedWebBackend(driver, evaluator)

    backend.reset(_task())
    backend.reset(_task())

    assert close_count == 1
    assert len(evaluator.prepared) == 2
    assert evaluator.prepared[-1]["browser_context"] == {"task_id": "0"}


def test_reset_cleanup_failure_invalidates_previous_lifecycle() -> None:
    evaluator = _Evaluator()
    backend = ComposedWebBackend(_Driver(), evaluator)
    backend.reset(_task())

    def fail_close() -> None:
        raise RuntimeError("evaluator cleanup failed")

    evaluator.close = fail_close  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="evaluator cleanup failed"):
        backend.reset(_task())
    with pytest.raises(RuntimeError, match="reset before use"):
        backend.step(WebAction(name="noop", script="noop()"))
    with pytest.raises(RuntimeError, match="reset before use"):
        backend.evaluate("done")
