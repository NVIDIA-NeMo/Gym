# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from nemo_gym.core import MediaFile, split_text_and_media
from responses_api_agents.openclaw_agent.app import parse_openclaw_session_items
from responses_api_agents.openclaw_agent.tests.test_sandbox import (  # noqa: F401
    close_body,
    events,
    seed,
    setup,
    staged,
)


IMAGE_URI = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l9sAAAAASUVORK5CYII="
)
IMAGE = MediaFile.from_data_uri(IMAGE_URI)
SECOND_IMAGE = MediaFile(
    media_type="image/gif",
    data=b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;",
)


def image_body(*, image_only=False, distinct=False):
    content = [] if image_only else [{"type": "input_text", "text": "Describe the image"}]
    content.extend(
        {"type": "input_image", "image_url": uri, "detail": "high"}
        for uri in (IMAGE_URI, SECOND_IMAGE.to_data_uri() if distinct else IMAGE_URI)
    )
    return {"input": [{"role": "user", "content": content}]}


def native_image_event(*, distinct=False):
    return {
        "type": "message",
        "message": {
            "role": "user",
            "content": [
                {"type": "text", "text": "Native prompt"},
                {"type": "image", "data": IMAGE_URI.split(",", 1)[1], "mimeType": "image/png"},
                {
                    "type": "image",
                    "data": (SECOND_IMAGE if distinct else IMAGE).to_data_uri().split(",", 1)[1],
                    "mimeType": "image/gif" if distinct else "image/png",
                },
            ],
        },
    }


@pytest.mark.parametrize("image_only", [False, True])
@pytest.mark.parametrize("distinct", [False, True])
def test_session_stages_prompt_images_and_replays_once(setup, monkeypatch, image_only, distinct):
    agent, sandbox = setup
    original_upload = sandbox.upload
    uploads = []

    async def upload(source, destination):
        if "/media/" in destination:
            uploads.append(destination)
            sandbox.files[destination] = Path(source).read_bytes()
        else:
            await original_upload(source, destination)

    monkeypatch.setattr(sandbox, "upload", upload)
    sandbox.events = json.dumps(native_image_event(distinct=distinct)) + "\n" + sandbox.events
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        path = "/ng-rollout/openclaw-smoke-a2/v1/responses"
        with patch(
            "responses_api_agents.openclaw_agent.app.split_text_and_media", wraps=split_text_and_media
        ) as decode:
            first = client.post(path, json=image_body(image_only=image_only, distinct=distinct))
            assert first.status_code == 200, first.text
            replay = client.post(path, json=image_body(image_only=image_only, distinct=distinct))
            assert replay.json() == first.json()
            decode.assert_called_once()
        assert len(uploads) == 2
        assert sandbox.files[uploads[0]] == IMAGE.data
        assert sandbox.files[uploads[1]] == (SECOND_IMAGE.data if distinct else IMAGE.data)
        assert uploads[0].startswith(f"{sandbox.directory}/home/.openclaw/media/")
        prompt = staged(sandbox)["prompt"]
        assert prompt.count(f"[Image: source: {uploads[0]}]") == 1
        assert prompt.count(f"[Image: source: {uploads[1]}]") == 1
        assert prompt.index(uploads[0]) < prompt.index(uploads[1])
        assert ("Describe the image" in prompt) is not image_only
        config = json.loads(sandbox.files[f"{sandbox.directory}/home/.openclaw/openclaw.json"])
        assert config["models"]["providers"]["nemo"]["models"][0]["input"] == ["text", "image"]
        assert config["agents"]["defaults"]["imageMaxDimensionPx"] == 2**31 - 1
        assert client.post(path, json={"input": "different"}).status_code == 409
        sandbox.launch.assert_awaited_once()
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        assert closed.status_code == 200
        user = next(
            item
            for item in closed.json()["agent_observations"]["records"][0]["conversation"]
            if item.get("role") == "user"
        )
        assert user["content"][1]["image_url"] == IMAGE_URI
        assert user["content"][2]["image_url"] == (SECOND_IMAGE.to_data_uri() if distinct else IMAGE_URI)
        sandbox.disconnect.assert_awaited_once()
        sandbox.stop.assert_not_awaited()


@pytest.mark.parametrize("image_input", [False, True])
def test_image_capability_is_opt_in_for_text_prompts(setup, image_input):
    agent, sandbox = setup
    agent.config.image_input = image_input
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        result = client.post("/ng-rollout/openclaw-smoke-a2/v1/responses", json={"input": "task"})
        assert result.status_code == 200, result.text
        config = json.loads(sandbox.files[f"{sandbox.directory}/home/.openclaw/openclaw.json"])
        assert config["models"]["providers"]["nemo"]["models"][0]["input"] == (
            ["text", "image"] if image_input else ["text"]
        )
        assert config["agents"]["defaults"].get("imageMaxDimensionPx") == (2**31 - 1 if image_input else None)
        assert staged(sandbox)["prompt"] == "task"
        assert not any("/media/" in path for path in sandbox.files)
        assert client.post("/v1/agent_sessions/close", json=close_body(session_id)).status_code == 200


@pytest.mark.parametrize(
    "role,image",
    [
        ("user", {"image_url": "https://example.com/a.png"}),
        ("user", {"image_url": "data:image/png;base64,%%%"}),
        ("user", {"image_url": "data:image/png;base64,é"}),
        ("user", {"image_url": "data:image/png;base64,aGVsbG8="}),
        ("user", {"image_url": "data:image/svg+xml;base64,PHN2Zy8+"}),
        ("user", {"file_id": "file-image"}),
        ("system", {"image_url": IMAGE_URI}),
        ("assistant", {"image_url": IMAGE_URI}),
    ],
)
def test_rejects_unsupported_images_before_activation_and_allows_recovery(setup, role, image):
    agent, sandbox = setup
    items = [{"role": role, "content": [{"type": "input_image", "detail": "auto", **image}]}]
    if role != "user":
        items.append({"role": "user", "content": "task"})
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        path = "/ng-rollout/openclaw-smoke-a2/v1/responses"
        rejected = client.post(path, json={"input": items})
        assert rejected.status_code == 422
        if image.get("image_url") in {"data:image/png;base64,%%%", "data:image/png;base64,é"}:
            assert "invalid base64" in rejected.text
        sandbox.launch.assert_not_awaited()
        assert client.post(path, json={"input": "task"}).status_code == 200
        assert client.post("/v1/agent_sessions/close", json=close_body(session_id)).status_code == 200


@pytest.mark.parametrize("stop_reason", ["stop", "length"])
@pytest.mark.parametrize("retained", [0, 1, 2])
@pytest.mark.parametrize("distinct", [False, True])
def test_gradable_responses_require_native_prompt_images(setup, monkeypatch, stop_reason, retained, distinct):
    agent, sandbox = setup
    sandbox.events = events(stop_reason=stop_reason)
    if retained:
        native = native_image_event(distinct=distinct)
        native["message"]["content"] = native["message"]["content"][: retained + 1]
        sandbox.events = json.dumps(native) + "\n" + sandbox.events
    original_upload = sandbox.upload

    async def upload(source, destination):
        if "/media/" not in destination:
            await original_upload(source, destination)

    monkeypatch.setattr(sandbox, "upload", upload)
    with TestClient(agent.setup_webserver(), raise_server_exceptions=False) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        result = client.post("/ng-rollout/openclaw-smoke-a2/v1/responses", json=image_body(distinct=distinct))
        if retained == 2:
            assert result.status_code == 200, result.text
            assert result.json()["status"] == ("completed" if stop_reason == "stop" else "incomplete")
            assert result.json()["output"][-1]["content"][0]["text"] == "Fixed"
        else:
            assert result.status_code >= 500
        closed = client.post("/v1/agent_sessions/close", json=close_body(session_id))
        assert closed.status_code == 200
        if retained < 2:
            assert closed.json()["agent_observations"]["records"][0]["status"] == "failed"


@pytest.mark.parametrize("before_model_call", [False, True])
def test_wall_timeout_does_not_require_native_prompt_images(setup, monkeypatch, before_model_call):
    agent, sandbox = setup
    sandbox.result.update(timed_out=True, return_code=-15)
    sandbox.events = "" if before_model_call else events(stop_reason="aborted")
    original_upload = sandbox.upload

    async def upload(source, destination):
        if "/media/" not in destination:
            await original_upload(source, destination)

    monkeypatch.setattr(sandbox, "upload", upload)
    with TestClient(agent.setup_webserver()) as client:
        session_id = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).json()["agent_session_id"]
        result = client.post("/ng-rollout/openclaw-smoke-a2/v1/responses", json=image_body())
        assert result.status_code == 200, result.text
        assert result.json()["status"] == "incomplete"
        if not before_model_call:
            assert result.json()["output"][-1]["content"][0]["text"] == "Fixed"
        assert client.post("/v1/agent_sessions/close", json=close_body(session_id)).status_code == 200


@pytest.mark.parametrize("ending", ["close", "deadline", "failure"])
async def test_image_staging_is_fenced_and_spends_activation_deadline(setup, monkeypatch, ending):
    agent, sandbox = setup
    state = await agent._seed_agent_session_state(seed())
    entered, cancelled = asyncio.Event(), asyncio.Event()
    original_upload = sandbox.upload

    async def upload(source, destination):
        if "/media/" not in destination:
            await original_upload(source, destination)
            return
        entered.set()
        try:
            if ending == "failure":
                raise RuntimeError("image upload failed")
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(sandbox, "upload", upload)
    payload = {"config": {}, "prompt": "task", "cwd": "/app", "command": [], "env": {}}
    task = asyncio.create_task(
        state.execute(payload, media=[IMAGE], timeout=0.05 if ending == "deadline" else 5, close_timeout=1)
    )
    await asyncio.wait_for(entered.wait(), 1)
    if ending == "close":
        await state.close(1)
    # Check completion separately so an outer TimeoutError cannot satisfy the expected deadline error.
    done, _ = await asyncio.wait({task}, timeout=2)
    try:
        assert task in done, "image staging did not stop at its deadline or close fence"
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await state.close(1)
    with pytest.raises(
        asyncio.CancelledError if ending == "close" else TimeoutError if ending == "deadline" else RuntimeError
    ):
        await task
    assert cancelled.is_set()
    sandbox.launch.assert_not_awaited()
    await state.close(1)
    sandbox.disconnect.assert_awaited_once()
    sandbox.stop.assert_not_awaited()


def test_native_transcript_image_bytes_and_order_are_preserved():
    user = parse_openclaw_session_items([native_image_event()], include_input=True)[0]
    assert user.model_dump()["content"] == [
        {"type": "input_text", "text": "Native prompt"},
        {"type": "input_image", "image_url": IMAGE_URI, "detail": "auto"},
        {"type": "input_image", "image_url": IMAGE_URI, "detail": "auto"},
    ]


def test_local_image_prompt_is_rejected_instead_of_dropped(setup):
    agent, sandbox = setup
    with TestClient(agent.setup_webserver()) as client:
        response = client.post("/v1/responses", json=image_body())
        assert response.status_code == 422
        assert "agent session" in response.json()["detail"]
        sandbox.launch.assert_not_awaited()
