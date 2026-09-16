import asyncio

import pytest
from pydantic import ValidationError

from responses_api_agents.harbor_agent_general.alerts import (
    AlertScheduleConfig,
    build_time_remaining_scheduler,
)


@pytest.mark.asyncio
async def test_time_remaining_alert_delivers_once() -> None:
    delivered = []
    operation_done = asyncio.Event()
    config = AlertScheduleConfig.model_validate(
        {
            "deadline_seconds": 0.05,
            "poll_interval_seconds": 0.001,
            "retry_interval_seconds": 0.001,
            "alerts": [{"remaining_seconds": 0.04, "message": "finish now"}],
        }
    )

    async def deliver(message, context):
        delivered.append((message, context.remaining_seconds))
        operation_done.set()

    scheduler = build_time_remaining_scheduler(config, deliver)
    await scheduler.run_until_complete(operation_done.wait())

    assert len(delivered) == 1
    assert delivered[0][0] == "finish now"
    assert scheduler.status()[0]["delivered"] is True
    assert scheduler.status()[0]["attempts"] == 1


@pytest.mark.asyncio
async def test_failed_alert_delivery_is_retried() -> None:
    attempts = 0
    operation_done = asyncio.Event()
    config = AlertScheduleConfig.model_validate(
        {
            "deadline_seconds": 0.05,
            "poll_interval_seconds": 0.001,
            "retry_interval_seconds": 0.001,
            "alerts": [{"remaining_seconds": 0.04, "message": "finish now"}],
        }
    )

    async def deliver(_message, _context):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("session not ready")
        operation_done.set()

    scheduler = build_time_remaining_scheduler(config, deliver)
    await scheduler.run_until_complete(operation_done.wait())

    assert attempts == 2
    assert scheduler.status()[0]["delivered"] is True
    assert scheduler.status()[0]["last_error"] is None


def test_alert_threshold_must_precede_deadline() -> None:
    with pytest.raises(ValidationError, match="less than deadline_seconds"):
        AlertScheduleConfig.model_validate(
            {
                "deadline_seconds": 300,
                "alerts": [{"remaining_seconds": 300, "message": "too late"}],
            }
        )
