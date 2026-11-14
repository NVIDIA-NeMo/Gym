# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The image tools on the resources server: the typed seed loads the task images, and each tool returns its image."""

import asyncio
import base64
import threading
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import anyio
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from nemo_gym.client import ServerClient
from nemo_gym.core import TOOL_OUTPUT_IMAGES_KEY, EpisodeId, TaskId
from nemo_gym.server_runtime import (
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
    ResourcesSessionLostError,
    check_resources_session_contract,
)
from resources_servers.agentic_mme.app import AgenticMMEConfig, AgenticMMEServer
from resources_servers.agentic_mme.image_tools import IMAGE_TOOLS, ImageWorkspace
from resources_servers.agentic_mme.tests.test_app import response


def _png(size: tuple[int, int], color: str) -> bytes:
    stream = BytesIO()
    Image.new("RGB", size, color).save(stream, format="PNG")
    return stream.getvalue()


RED = _png((40, 20), "red")
BLUE = _png((10, 10), "blue")


@pytest.fixture
def image_root(tmp_path) -> Path:
    image_root = tmp_path / "images"
    (image_root / "task-1").mkdir(parents=True)
    (image_root / "task-1" / "0.png").write_bytes(RED)
    (image_root / "task-1" / "1.png").write_bytes(BLUE)
    (image_root / "task-1" / "broken.png").write_bytes(b"not an image")
    return image_root


def _server(image_root: Path, **extra) -> AgenticMMEServer:
    config = AgenticMMEConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="agentic_mme",
        serve_image_tools=True,
        image_root=str(image_root),
        **extra,
    )
    return AgenticMMEServer(config=config, server_client=MagicMock(spec=ServerClient))


def _seed(session_id: str = "session", sources: tuple[str, ...] = ("0.png", "1.png")) -> ResourcesSeedSessionRequest:
    return ResourcesSeedSessionRequest(
        resources_session_id=session_id,
        episode_id=EpisodeId(rollout_id=f"rollout-{session_id}"),
        task_id=TaskId(taskset="agentic_mme:benchmark", task_id="task-1"),
        task_data={
            "verifier_metadata": {"golden_answer": {"value": "red"}},
            "image_files": [
                {"source": f"task-1/{source}", "target": f"images/{index}.png"} for index, source in enumerate(sources)
            ],
        },
    )


def _seeded(server: AgenticMMEServer, seed: ResourcesSeedSessionRequest) -> TestClient:
    client = TestClient(server.setup_webserver())
    seeded = client.post("/seed_session", json=seed.model_dump(mode="json"))
    assert seeded.status_code == 200, seeded.text
    return client


def _image(data_url: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(data_url.partition(",")[2])))


def test_tools_match_the_agentic_mme_agent_and_return_the_new_image(image_root) -> None:
    server = _server(image_root)
    client = TestClient(server.setup_webserver())
    seeded = client.post("/seed_session", json=_seed().model_dump(mode="json"))
    assert seeded.status_code == 200, seeded.text
    # No sandbox provider, so no sandbox is lent.
    assert seeded.json().get("sandbox_access") is None
    # The agent loads the prompt images as images 0, 1, ... and runs the same operations on its own workspace.
    reference = ImageWorkspace()
    for data in (RED, BLUE):
        reference.load("data:image/png;base64," + base64.b64encode(data).decode())

    for name, arguments in [
        ("crop", {"image_index": 0, "bbox_2d": [0, 0, 500, 1000], "zoom_scale": 2.0}),
        ("grayscale", {"image_index": 2}),
    ]:
        reply = client.post(f"/{name}", json=arguments)
        assert reply.status_code == 200, reply.text
        expected = reference.apply(name, arguments)
        expected_url = expected.pop("image_url")
        label = f"Image {expected['new_image_index']}"
        assert reply.json() == {**expected, TOOL_OUTPUT_IMAGES_KEY: [{"image_url": expected_url, "label": label}]}

    # The crop of the left half, zoomed twice, then made gray, is image 3.
    assert expected["new_image_index"] == 3
    assert _image(expected_url).size == (40, 40)


def test_each_session_has_its_own_workspace(image_root) -> None:
    server = _server(image_root)
    two_images = _seeded(server, _seed("a"))
    one_image = _seeded(server, _seed("b", sources=("1.png",)))

    assert two_images.post("/invert", json={"image_index": 1}).json()["new_image_index"] == 2
    reply = one_image.post("/invert", json={"image_index": 0}).json()
    assert reply["new_image_index"] == 1
    assert _image(reply[TOOL_OUTPUT_IMAGES_KEY][0]["image_url"]).getpixel((0, 0)) == (255, 255, 0)


def test_a_failed_tool_returns_the_agent_s_error_to_the_model(image_root) -> None:
    client = _seeded(_server(image_root), _seed())

    reply = client.post("/crop", json={"image_index": 9, "bbox_2d": [0, 0, 10, 10]})

    assert reply.status_code == 200
    assert reply.json() == {"ok": False, "error": "image_index does not exist in this rollout"}


def test_a_tool_call_needs_a_seeded_workspace_and_answers_404_after_close(image_root) -> None:
    server = _server(image_root)
    assert TestClient(server.setup_webserver()).post("/invert", json={"image_index": 0}).status_code == 409

    client = _seeded(server, _seed())
    close = ResourcesCloseSessionRequest(resources_session_id="session", episode_id=_seed().episode_id)
    assert client.post("/close_session", json=close.model_dump(mode="json")).status_code == 200

    assert server._sessions == {}
    late = client.post("/invert", json={"image_index": 0})
    # The harness records a failed tool call that names the lost session, and nothing is recreated.
    assert late.status_code == 404
    assert late.json()["detail"]["failure_kind"] == "session_lost"
    assert server._sessions == {}


class _GatedApply:
    """Hold each image operation's worker thread until ``release`` is set, and count overlapping workers."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0
        self.running = 0
        self.most_running = 0
        self._count = threading.Lock()
        self._apply = ImageWorkspace.apply

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Replace ImageWorkspace.apply with this gate, as a function so that it binds to each workspace."""

        def apply(workspace: ImageWorkspace, name: str, args: dict) -> dict:
            return self(workspace, name, args)

        monkeypatch.setattr(ImageWorkspace, "apply", apply)

    def __call__(self, workspace: ImageWorkspace, name: str, args: dict) -> dict:
        with self._count:
            self.calls += 1
            self.running += 1
            self.most_running = max(self.most_running, self.running)
        self.started.set()
        try:
            assert self.release.wait(5)
            return self._apply(workspace, name, args)
        finally:
            with self._count:
                self.running -= 1


def _invert(server: AgenticMMEServer):
    """The /invert handler, called in the test's event loop, and a request seeded for "session"."""
    route = next(route for route in server.setup_webserver().routes if getattr(route, "path", None) == "/invert")
    return route.endpoint, IMAGE_TOOLS["invert"][0](image_index=0)


async def test_close_waits_for_a_running_tool_call_and_a_call_queued_behind_it_sees_the_fence(
    image_root, monkeypatch
) -> None:
    server = _server(image_root)
    invert, args = _invert(server)
    request = SimpleNamespace(session={})
    await server.seed_session(request, _seed())
    gate = _GatedApply()
    gate.install(monkeypatch)

    running = asyncio.create_task(invert(request, args))
    assert await asyncio.to_thread(gate.started.wait, 5)
    close = asyncio.create_task(
        server.close_resources_session(
            ResourcesCloseSessionRequest(resources_session_id="session", episode_id=_seed().episode_id)
        )
    )
    queued = asyncio.create_task(invert(request, args))
    await asyncio.sleep(0.1)
    assert not close.done()

    gate.release.set()
    assert (await running)["new_image_index"] == 2
    await close
    with pytest.raises(ResourcesSessionLostError):
        await queued
    # The queued call never reached the workspace that close dropped.
    assert gate.calls == 1


async def test_a_disconnected_tool_call_keeps_the_session_until_its_worker_finishes(image_root, monkeypatch) -> None:
    """A disconnect cancels the handler (an anyio cancel scope, as the disconnect middleware uses), not its thread."""
    server = _server(image_root)
    invert, args = _invert(server)
    request = SimpleNamespace(session={})
    await server.seed_session(request, _seed())
    gate = _GatedApply()
    gate.install(monkeypatch)

    async def disconnected_call() -> None:
        async with anyio.create_task_group() as group:
            group.start_soon(invert, request, args)
            assert await anyio.to_thread.run_sync(gate.started.wait, 5)
            group.cancel_scope.cancel()

    disconnected = asyncio.create_task(disconnected_call())
    assert await asyncio.to_thread(gate.started.wait, 5)
    await asyncio.sleep(0.05)
    following = asyncio.create_task(invert(request, args))
    await asyncio.sleep(0.1)
    assert not following.done()
    assert gate.calls == 1

    gate.release.set()
    await disconnected
    # The next call ran after the first worker finished, on the image list that worker extended.
    assert (await following)["new_image_index"] == 3
    assert gate.most_running == 1


def test_seed_refuses_an_image_it_cannot_load(image_root) -> None:
    client = TestClient(_server(image_root).setup_webserver())

    refused = client.post("/seed_session", json=_seed(sources=("0.png", "broken.png")).model_dump(mode="json"))

    assert refused.status_code == 422
    assert "images/1.png" in refused.json()["detail"]


def test_server_tools_follow_the_session_contract(image_root) -> None:
    server = _server(image_root)
    check_resources_session_contract(
        server.setup_webserver(),
        _seed("contract"),
        keeps_state=True,
        holds_state=lambda resources_session_id: resources_session_id in server._sessions,
        verify_body={
            "responses_create_params": {"input": "What colour is image 0?"},
            "response": response("<answer>red</answer>").model_dump(mode="json"),
            "verifier_metadata": {"golden_answer": {"value": "red"}},
        },
    )


def test_without_serve_image_tools_the_server_has_no_tool_routes(image_root) -> None:
    server = AgenticMMEServer(
        config=AgenticMMEConfig(host="0.0.0.0", port=8080, entrypoint="", name="agentic_mme"),
        server_client=MagicMock(spec=ServerClient),
    )
    assert TestClient(server.setup_webserver()).post("/invert", json={"image_index": 0}).status_code == 404


def test_serve_image_tools_needs_an_image_root() -> None:
    config = AgenticMMEConfig(host="0.0.0.0", port=8080, entrypoint="", name="agentic_mme", serve_image_tools=True)
    with pytest.raises(ValueError, match="image_root"):
        AgenticMMEServer(config=config, server_client=MagicMock(spec=ServerClient))


def test_mcp_lists_only_the_image_tools_and_returns_each_image_as_image_content(image_root) -> None:
    pytest.importorskip("mcp")
    from nemo_gym.server_runtime import TOKEN_HEADER, maybe_auto_expose

    server = _server(image_root, expose_tools_over_mcp=True)
    app = server.setup_webserver()
    maybe_auto_expose(server, app)
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    with TestClient(app) as client:
        seeded = client.post("/seed_session", json=_seed().model_dump(mode="json"))
        assert seeded.status_code == 200, seeded.text
        headers[TOKEN_HEADER] = seeded.json()["resources_tools"]["headers"][TOKEN_HEADER]

        def rpc(method: str, params: dict) -> dict:
            body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
            return client.post("/mcp", headers=headers, json=body).json()["result"]

        rpc(
            "initialize",
            {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}},
        )
        assert {tool["name"] for tool in rpc("tools/list", {})["tools"]} == set(IMAGE_TOOLS)
        result = rpc("tools/call", {"name": "flip", "arguments": {"image_index": 1, "direction": "horizontal"}})

    assert result.get("isError") is not True, result
    # Without structured content, a client such as OpenCode keeps the image parts.
    assert result.get("structuredContent") is None
    text, label, image = result["content"]
    assert label == {"type": "text", "text": "Image 2"}
    assert image["type"] == "image" and image["mimeType"] == "image/png"
    assert _image("data:image/png;base64," + image["data"]).getpixel((0, 0)) == (0, 0, 255)
    assert TOOL_OUTPUT_IMAGES_KEY not in text["text"]
