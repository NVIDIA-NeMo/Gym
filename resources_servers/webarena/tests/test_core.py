# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import io
from typing import Any

import pytest
from PIL import Image
from pydantic import ValidationError

from resources_servers.webarena import browser as browser_module
from resources_servers.webarena.actions import (
    decode_tool_arguments,
    execute_computer_actions,
    validate_tool_arguments,
)
from resources_servers.webarena.actor_pool import BrowserActorPool
from resources_servers.webarena.app import SingleFlightDeduplicator
from resources_servers.webarena.observations import build_observation_message
from resources_servers.webarena.schemas import (
    WebArenaAgentRunRequest,
    WebArenaResourcesServerConfig,
    WebArenaTaskRow,
)


def _config(**overrides: Any) -> WebArenaResourcesServerConfig:
    values = {
        "host": "0.0.0.0",
        "port": 8080,
        "entrypoint": "app.py",
        "name": "webarena",
        "num_workers": 1,
        "browser_startup_timeout_seconds": 1,
        "browser_pool_wait_timeout_seconds": 1,
    }
    values.update(overrides)
    return WebArenaResourcesServerConfig(**values)


def _task(images: list[dict[str, Any]] | None = None) -> WebArenaTaskRow:
    return WebArenaTaskRow(
        task_id="webarena-test",
        sites=["reddit"],
        start_urls=["__REDDIT__"],
        intent="Open the task.",
        eval={"eval_types": ["string_match"], "reference_answers": {"exact_match": "done"}},
        task_input_images=images or [],
    )


def _png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (8, 8), color="red").save(output, format="PNG")
    return output.getvalue()


def test_resources_server_rejects_multiple_uvicorn_workers() -> None:
    with pytest.raises(ValidationError, match="num_workers=1"):
        _config(num_workers=2)


def test_benchmark_credentials_are_python_defaults() -> None:
    config = _config()
    assert config.credentials["shopping"]["username"] == "emma.lopez@gmail.com"
    assert config.credentials["gitlab"]["password"] == "hello1234"


def test_initial_browser_workers_defaults_to_full_pool() -> None:
    config = _config(max_browser_workers=23)
    assert config.initial_browser_workers == 23
    assert config.post_action_screenshot_delay_seconds == 4
    assert _config(max_browser_workers=23, initial_browser_workers=0).initial_browser_workers == 0


def test_run_request_loads_stable_inline_task_from_verifier_metadata() -> None:
    request = WebArenaAgentRunRequest.model_validate(
        {
            "responses_create_params": {"input": [], "parallel_tool_calls": False},
            "verifier_metadata": {"task": _task().model_dump(mode="json")},
        }
    )
    assert request.task_payload.task_id == "webarena-test"
    assert len(request.task_payload.dataset_hash) == 64


def test_task_validation_rejects_non_classic_evaluator() -> None:
    with pytest.raises(ValidationError, match="Unsupported non-Classic"):
        WebArenaTaskRow.model_validate(
            {
                **_task().model_dump(mode="json"),
                "eval": {"eval_types": ["webarena_verified_har"]},
            }
        )


@pytest.mark.asyncio
async def test_initial_task_navigation_retries_three_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    class FlakyPage:
        calls = 0

        async def goto(self, url: str, *, wait_until: str, timeout: int) -> None:
            self.calls += 1
            assert timeout == 35_000
            if self.calls < 3:
                raise TimeoutError(f"transient failure for {url}")

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(browser_module.asyncio, "sleep", no_sleep)
    page = FlakyPage()
    await browser_module._navigate_initial_page(page, "http://example")
    assert page.calls == 3


def test_tool_arguments_validate_relative_coordinates() -> None:
    args = decode_tool_arguments('{"actions":[{"action":"left_click","coordinate":[0.5,0.25]}]}')
    assert validate_tool_arguments("computer", args) == args
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_tool_arguments(
            "computer",
            {"actions": [{"action": "left_click", "coordinate": [2, 0]}]},
        )


def test_computer_actions_convert_relative_coordinates() -> None:
    class FakePyAutoGUI:
        calls: list[tuple[Any, ...]] = []

        def click(self, *args: Any, **kwargs: Any) -> None:
            self.calls.append((*args, kwargs))

    fake = FakePyAutoGUI()
    execute_computer_actions(
        fake,
        {"actions": [{"action": "left_click", "coordinate": [0.5, 0.25]}]},
        width=1000,
        height=800,
    )
    assert fake.calls == [(500, 200, {"button": "left", "clicks": 1})]


def test_observation_matches_source_turn_order_and_keeps_task_image() -> None:
    task_image = {
        "type": "input_image",
        "image_url": "data:image/png;base64,dGFzaw==",
        "detail": "auto",
    }
    message = build_observation_message(
        screenshot=_png_bytes(),
        task=_task([task_image]),
        step=1,
        tab_context={"current_tab_id": 0, "tab_count": 1, "tabs": []},
        image_format="JPEG",
        jpeg_quality=85,
    )
    assert [part["type"] for part in message.content] == [
        "input_image",
        "input_text",
        "input_image",
        "input_text",
    ]
    assert message.content[0]["image_url"].startswith("data:image/jpeg;base64,")
    assert message.content[2]["image_url"] == task_image["image_url"]


@pytest.mark.asyncio
async def test_single_flight_deduplicates_concurrent_requests() -> None:
    dedup = SingleFlightDeduplicator(ttl_seconds=60)
    calls = 0

    async def operation() -> str:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        return "ok"

    first, second = await asyncio.gather(
        dedup.execute("step", "same-id", {"x": 1}, operation),
        dedup.execute("step", "same-id", {"x": 1}, operation),
    )
    assert (first, second) == ("ok", "ok")
    assert calls == 1

    await dedup.execute("step", "next-id", {"x": 2}, operation)
    assert len(dedup._entries) == 1
    assert dedup._entries["step"].request_id == "next-id"


class _RemoteMethod:
    def __init__(self, function: Any):
        self.function = function

    async def remote(self, *args: Any, **kwargs: Any) -> Any:
        return self.function(*args, **kwargs)


class _FakeActor:
    def __init__(self):
        self.health = _RemoteMethod(lambda: {"healthy": True})
        self.close_episode = _RemoteMethod(lambda: True)
        self.shutdown = _RemoteMethod(lambda: None)


@pytest.mark.asyncio
async def test_pool_capacity_is_global_and_waits_for_release() -> None:
    pool = BrowserActorPool(_config(max_browser_workers=1), worker_factory=_FakeActor)
    first = await pool.checkout("one")
    pending = asyncio.create_task(pool.checkout("two"))
    await asyncio.sleep(0)
    assert not pending.done()
    await pool.release("one")
    second = await pending
    assert second is first
    await pool.release("two")


@pytest.mark.asyncio
async def test_pool_staggers_browser_actor_readiness() -> None:
    pool = BrowserActorPool(
        _config(
            max_browser_workers=2,
            browser_startup_concurrency=1,
            browser_startup_stagger_seconds=0.03,
        ),
        worker_factory=_FakeActor,
    )
    loop = asyncio.get_running_loop()

    async def checkout(env_id: str) -> float:
        await pool.checkout(env_id)
        return loop.time()

    ready_times = sorted(await asyncio.gather(checkout("one"), checkout("two")))
    assert ready_times[1] - ready_times[0] >= 0.02
    await asyncio.gather(pool.release("one"), pool.release("two"))
