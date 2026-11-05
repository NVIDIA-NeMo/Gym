# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session images reach the CLI unchanged and share the activation lifecycle."""

import asyncio
import base64
import json
from pathlib import Path
from time import monotonic

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from nemo_gym.agents import AGENT_ACTIVATION_HEADER, AgentCloseSessionRequest, AgentContinuationRequirements
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.opencode_agent.tests.test_continuation import (
    activate,
    close,
    install_artifact_runner,
    roots,
    start,
)
from responses_api_agents.opencode_agent.tests.test_sandbox_sessions import seed, setup  # noqa: F401


PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=")
GIF = base64.b64decode("R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7")


def image_part(content: bytes = PNG, *, media_type: str = "image/png") -> dict[str, str]:
    return {
        "type": "input_image",
        "image_url": f"data:{media_type};base64,{base64.b64encode(content).decode()}",
        "detail": "auto",
    }


def image_input(*parts: dict[str, str]) -> list[dict]:
    return [{"role": "user", "content": [{"type": "input_text", "text": "Compare these images."}, *parts]}]


def capture_images(sandbox) -> list[tuple[str, bytes]]:
    """Keep the fixture's text artifacts while observing actual binary upload contents."""
    images = []
    upload = sandbox.upload

    async def upload_file(source, destination):
        if Path(destination).name.startswith("image-"):
            images.append((destination, Path(source).read_bytes()))
        else:
            await upload(source, destination)

    sandbox.upload = upload_file
    return images


def attached_paths(payload: dict) -> list[str]:
    command = payload["command"]
    return [command[index + 1] for index, argument in enumerate(command) if argument == "-f"]


def assert_vision_configuration(payload: dict) -> None:
    config = json.loads(payload["env"]["OPENCODE_CONFIG_CONTENT"])
    assert config["provider"]["nemo_gym"]["models"]["dummy_model"]["modalities"] == {
        "input": ["text", "image"],
        "output": ["text"],
    }
    assert config["attachment"]["image"] == {
        "max_width": 2**31 - 1,
        "max_height": 2**31 - 1,
        "max_base64_bytes": 2**31 - 1,
    }


def test_session_images_upload_original_bytes_and_cli_attachments_in_order(setup):  # noqa: F811
    agent, sandbox = setup
    images = capture_images(sandbox)
    request = seed()
    prompt = image_input(
        image_part(), {"type": "input_text", "text": "And this one."}, image_part(GIF, media_type="image/gif")
    )
    with TestClient(agent.setup_webserver()) as client:
        assert client.post("/v1/agent_sessions", json=request.model_dump(mode="json")).status_code == 200
        response = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": prompt})
        assert response.status_code == 200 and response.json()["status"] == "completed", response.text
        payload = sandbox.params()
        assert payload["cwd"] == "/app"
        assert payload["prompt"] == "Compare these images.\nAnd this one."
        assert attached_paths(payload) == [destination for destination, _ in images]
        assert [content for _, content in images] == [PNG, GIF]
        assert [Path(destination).name for destination, _ in images] == ["image-1.png", "image-2.gif"]
        assert all(Path(destination).parent == Path(sandbox.directory) for destination, _ in images)
        assert all(not Path(destination).is_relative_to("/app") for destination, _ in images)
        assert_vision_configuration(payload)
        close(client, request)
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


@pytest.mark.parametrize(
    "part",
    [
        {"image_url": "https://example.invalid/image.png"},
        {"file_id": "file-image"},
        {"image_url": "data:image/svg+xml;base64,PHN2Zy8+"},
        {"image_url": "data:image/png;base64,not-base64!"},
        {"image_url": "data:image/png,plain-text"},
        {"image_url": "data:image/png;base64,"},
        {"image_url": "data:image/png;base64," + base64.b64encode(GIF).decode()},
    ],
)
def test_invalid_image_does_not_consume_activation(setup, tmp_path, part):  # noqa: F811
    agent, sandbox = setup
    images = capture_images(sandbox)
    install_artifact_runner(sandbox, tmp_path)
    with TestClient(agent.setup_webserver()) as client:
        request = start(client)
        invalid = activate(client, 1, image_input({"type": "input_image", "detail": "auto", **part}))
        assert invalid.status_code == 422, invalid.text
        assert images == []
        sandbox.launch.assert_not_awaited()
        valid = activate(client, 1, image_input(image_part()))
        assert valid.status_code == 200 and valid.json()["status"] == "completed", valid.text
        assert [content for _, content in images] == [PNG]
        close(client, request)


@pytest.mark.parametrize("role", ["system", "developer"])
def test_system_image_rejected_before_activation(setup, tmp_path, role):  # noqa: F811
    agent, sandbox = setup
    install_artifact_runner(sandbox, tmp_path)
    with TestClient(agent.setup_webserver()) as client:
        request = start(client)
        prompt = [{"role": role, "content": [image_part()]}, {"role": "user", "content": "Inspect"}]
        response = activate(client, 1, prompt)
        assert response.status_code == 422, response.text
        sandbox.launch.assert_not_awaited()
        assert activate(client, 1).status_code == 200
        close(client, request)


def test_text_only_session_does_not_enable_vision_or_stage_attachments(setup):  # noqa: F811
    agent, sandbox = setup
    images = capture_images(sandbox)
    request = seed()
    with TestClient(agent.setup_webserver()) as client:
        assert client.post("/v1/agent_sessions", json=request.model_dump(mode="json")).status_code == 200
        response = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "Inspect"})
        assert response.status_code == 200 and response.json()["status"] == "completed", response.text
        payload = sandbox.params()
        config = json.loads(payload["env"]["OPENCODE_CONFIG_CONTENT"])
        assert "modalities" not in config["provider"]["nemo_gym"]["models"]["dummy_model"]
        assert "attachment" not in config
        assert attached_paths(payload) == [] and images == []
        assert payload["prompt"] == "Inspect"
        close(client, request)


def test_continuation_uses_distinct_image_paths_and_keeps_vision_for_retained_history(setup, tmp_path):  # noqa: F811
    agent, sandbox = setup
    images = capture_images(sandbox)
    payloads = install_artifact_runner(sandbox, tmp_path)
    with TestClient(agent.setup_webserver()) as client:
        request = start(client)
        first = activate(client, 1, image_input(image_part()))
        second = activate(client, 2, image_input(image_part(GIF, media_type="image/gif")))
        retry = activate(client, 2, image_input(image_part(GIF, media_type="image/gif")))
        third = activate(client, 3, "Summarize both images.")
        assert all(response.status_code == 200 for response in (first, second, retry, third))
        assert retry.json() == second.json()
        assert len(images) == 2 and len(payloads) == 3
        assert attached_paths(payloads[0]) == [images[0][0]]
        assert attached_paths(payloads[1]) == [images[1][0]]
        assert attached_paths(payloads[2]) == []
        assert images[0][0].endswith("/activation-1/image-1.png")
        assert images[1][0].endswith("/activation-2/image-1.gif")
        assert [content for _, content in images] == [PNG, GIF]
        for payload in payloads:
            assert_vision_configuration(payload)
        assert [root["activation_id"] for root in roots(close(client, request)["agent_observations"])] == [1, 2, 3]


def test_image_staging_spends_activation_deadline_and_next_activation_recovers(setup, tmp_path):  # noqa: F811
    agent, sandbox = setup
    images = capture_images(sandbox)
    install_artifact_runner(sandbox, tmp_path)
    upload = sandbox.upload

    async def slow_upload(source, destination):
        if Path(destination).name.startswith("image-"):
            await asyncio.sleep(2)
        await upload(source, destination)

    with TestClient(agent.setup_webserver()) as client:
        request = start(client)
        sandbox.upload = slow_upload
        started = monotonic()
        response = activate(client, 1, image_input(image_part()), timeout=0.01)
        elapsed = monotonic() - started
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["status"] == "incomplete" and result["output"] == []
        assert roots(result["_ng_agent_observations"])[0]["error_type"] == "timeout"
        assert elapsed < 1, f"image staging ignored the activation deadline: {elapsed:.2f} s"
        assert images == []
        sandbox.launch.assert_not_awaited()
        second = activate(client, 2, "Continue without an image.")
        assert second.status_code == 200 and second.json()["status"] == "completed", second.text
        assert [root["status"] for root in roots(close(client, request)["agent_observations"])] == [
            "incomplete",
            "completed",
        ]
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


async def test_close_cancels_image_upload_before_releasing_borrowed_sandbox(setup):  # noqa: F811
    agent, sandbox = setup
    entered, cancelled = asyncio.Event(), asyncio.Event()
    upload = sandbox.upload
    local_paths = []

    async def blocked_upload(source, destination):
        if Path(destination).name.startswith("image-"):
            local_paths.append(Path(source))
            assert Path(source).read_bytes() == PNG
            entered.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()
        await upload(source, destination)

    sandbox.upload = blocked_upload
    request = Request(
        {"type": "http", "session": {}, "headers": [(AGENT_ACTIVATION_HEADER.encode(), b"1")], "path_params": {}}
    )
    body = seed().model_copy(update={"continuation": AgentContinuationRequirements()})
    await agent.seed_agent_session(request, body)
    running = asyncio.create_task(
        agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input=image_input(image_part())))
    )
    await asyncio.wait_for(entered.wait(), 2)
    receipt = await agent.close_agent_session(
        request, AgentCloseSessionRequest(agent_session_id=body.agent_session_id, episode_id=body.episode_id)
    )
    with pytest.raises(asyncio.CancelledError):
        await running
    assert cancelled.is_set()
    assert local_paths and all(not path.exists() for path in local_paths)
    [root] = roots(receipt.agent_observations.model_dump(mode="json"))
    assert root["status"] == "incomplete" and root["error_type"] == "cancelled"
    sandbox.launch.assert_not_awaited()
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()
