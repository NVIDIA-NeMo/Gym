# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime alerts that can be adapted to any interactive agent harness."""

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


@dataclass(frozen=True)
class AlertContext:
    """Clock state supplied to alert conditions and delivery callbacks."""

    elapsed_seconds: float
    deadline_seconds: float

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.deadline_seconds - self.elapsed_seconds)


AlertCondition = Callable[[AlertContext], bool | Awaitable[bool]]
AlertFunc = Callable[[AlertContext], None | Awaitable[None]]


class TimeRemainingAlertConfig(BaseModel):
    """One user-visible warning relative to the agent deadline."""

    model_config = ConfigDict(extra="forbid")

    remaining_seconds: float = Field(gt=0)
    message: str = Field(min_length=1)


class AlertScheduleConfig(BaseModel):
    """Serializable configuration compiled into runtime alert callables."""

    model_config = ConfigDict(extra="forbid")

    deadline_seconds: float = Field(gt=0)
    poll_interval_seconds: float = Field(default=5.0, gt=0)
    retry_interval_seconds: float = Field(default=15.0, gt=0)
    alerts: list[TimeRemainingAlertConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_alerts(self) -> "AlertScheduleConfig":
        thresholds = [alert.remaining_seconds for alert in self.alerts]
        if len(thresholds) != len(set(thresholds)):
            raise ValueError("alert remaining_seconds values must be unique")
        if any(threshold >= self.deadline_seconds for threshold in thresholds):
            raise ValueError("alert remaining_seconds must be less than deadline_seconds")
        return self


@dataclass
class _RegisteredAlert:
    name: str
    alert_func: AlertFunc
    condition: AlertCondition
    delivered: bool = False
    attempts: int = 0
    last_attempt_elapsed_seconds: float | None = None
    delivered_elapsed_seconds: float | None = None
    last_error: str | None = None


async def _resolve(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class AlertScheduler:
    """Run one operation while evaluating registered alerts concurrently."""

    def __init__(
        self,
        *,
        deadline_seconds: float,
        poll_interval_seconds: float = 5.0,
        retry_interval_seconds: float = 15.0,
    ) -> None:
        if deadline_seconds <= 0:
            raise ValueError("deadline_seconds must be greater than zero")
        if poll_interval_seconds <= 0 or retry_interval_seconds <= 0:
            raise ValueError("alert polling and retry intervals must be greater than zero")
        self.deadline_seconds = deadline_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.retry_interval_seconds = retry_interval_seconds
        self._alerts: list[_RegisteredAlert] = []

    def register_alert(
        self,
        alert_func: AlertFunc,
        condition: AlertCondition,
        *,
        name: str,
    ) -> None:
        """Register a delivery callback and the condition that enables it."""
        if not name:
            raise ValueError("alert name must not be empty")
        if any(alert.name == name for alert in self._alerts):
            raise ValueError(f"alert name {name!r} is already registered")
        self._alerts.append(_RegisteredAlert(name=name, alert_func=alert_func, condition=condition))

    async def run_until_complete(self, operation: Awaitable[Any]) -> Any:
        """Return the operation result after delivering any due alerts."""
        started_at = time.monotonic()
        operation_task = asyncio.ensure_future(operation)
        try:
            while not operation_task.done():
                elapsed = time.monotonic() - started_at
                context = AlertContext(
                    elapsed_seconds=elapsed,
                    deadline_seconds=self.deadline_seconds,
                )
                for alert in self._alerts:
                    if alert.delivered:
                        continue
                    if (
                        alert.last_attempt_elapsed_seconds is not None
                        and elapsed - alert.last_attempt_elapsed_seconds < self.retry_interval_seconds
                    ):
                        continue
                    if not await _resolve(alert.condition(context)):
                        continue

                    alert.attempts += 1
                    alert.last_attempt_elapsed_seconds = elapsed
                    try:
                        await _resolve(alert.alert_func(context))
                    except Exception as error:  # noqa: BLE001 - alerts must not kill an episode
                        alert.last_error = f"{type(error).__name__}: {error}"
                    else:
                        alert.delivered = True
                        alert.delivered_elapsed_seconds = elapsed
                        alert.last_error = None

                if operation_task.done():
                    break
                await asyncio.wait(
                    {operation_task},
                    timeout=self.poll_interval_seconds,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            return await operation_task
        finally:
            if not operation_task.done():
                operation_task.cancel()

    def status(self) -> list[dict[str, Any]]:
        """Return JSON-serializable delivery diagnostics."""
        return [
            {
                "name": alert.name,
                "delivered": alert.delivered,
                "attempts": alert.attempts,
                "last_attempt_elapsed_seconds": alert.last_attempt_elapsed_seconds,
                "delivered_elapsed_seconds": alert.delivered_elapsed_seconds,
                "last_error": alert.last_error,
            }
            for alert in self._alerts
        ]


def build_time_remaining_scheduler(
    config: AlertScheduleConfig,
    deliver: Callable[[str, AlertContext], None | Awaitable[None]],
) -> AlertScheduler:
    """Compile typed time-remaining alerts into callable registrations."""
    scheduler = AlertScheduler(
        deadline_seconds=config.deadline_seconds,
        poll_interval_seconds=config.poll_interval_seconds,
        retry_interval_seconds=config.retry_interval_seconds,
    )
    for alert in sorted(
        config.alerts,
        key=lambda item: item.remaining_seconds,
        reverse=True,
    ):
        threshold = alert.remaining_seconds
        message = alert.message

        async def alert_func(
            context: AlertContext,
            *,
            message: str = message,
        ) -> None:
            await _resolve(deliver(message, context))

        def condition(
            context: AlertContext,
            *,
            threshold: float = threshold,
        ) -> bool:
            return context.remaining_seconds <= threshold

        scheduler.register_alert(
            alert_func,
            condition,
            name=f"remaining_{threshold:g}s",
        )
    return scheduler
