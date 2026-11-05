# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The lent sandbox: typed seed and close, run on the local sandbox provider in a temporary directory."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from nemo_gym.client import ServerClient
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.server_runtime import ResourcesCloseSessionRequest, ResourcesSeedSessionRequest
from nemo_gym.testing.session_conformance import check_resources_session_contract
from resources_servers.agentic_mme import app as agentic_mme_app
from resources_servers.agentic_mme.app import AgenticMMEConfig, AgenticMMEServer


PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR-not-a-real-image-but-bytes"


@pytest.fixture
def root(tmp_path, monkeypatch) -> Path:
    """The sandbox working directory on the host; the local provider runs commands there."""
    root = tmp_path / "workspace"
    monkeypatch.setattr(agentic_mme_app, "WORKDIR", str(root))
    monkeypatch.setattr(
        "nemo_gym.server_runtime.sandbox_lending.get_global_config_dict", lambda: {"sandbox": {"local": {}}}
    )
    # The local provider cannot be reattached from another process, so it has no descriptor to hand out.
    monkeypatch.setattr(AsyncSandbox, "serialize", AsyncMock(return_value={"sandbox_id": "local"}))
    return root


@pytest.fixture
def stopped(monkeypatch) -> list[AsyncSandbox]:
    stopped: list[AsyncSandbox] = []
    real_stop = AsyncSandbox.stop

    async def stop(self: AsyncSandbox) -> None:
        stopped.append(self)
        await real_stop(self)

    monkeypatch.setattr(AsyncSandbox, "stop", stop)
    return stopped


@pytest.fixture
def image_root(tmp_path) -> Path:
    image_root = tmp_path / "images"
    (image_root / "task-1").mkdir(parents=True)
    (image_root / "task-1" / "0.png").write_bytes(PNG)
    (image_root / "secret.txt").write_text("not a task image")
    return image_root


def _server(image_root: Path, **extra) -> AgenticMMEServer:
    config = AgenticMMEConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="agentic_mme",
        sandbox_provider="sandbox",
        sandbox_config={"image": "unused-by-the-local-provider"},
        image_root=str(image_root),
        **extra,
    )
    return AgenticMMEServer(config=config, server_client=MagicMock(spec=ServerClient))


def _seed(session_id: str = "session", image_files: list | None = None) -> ResourcesSeedSessionRequest:
    return ResourcesSeedSessionRequest(
        resources_session_id=session_id,
        episode_id=EpisodeId(rollout_id=f"rollout-{session_id}"),
        task_id=TaskId(taskset="agentic_mme:benchmark", task_id="task-1"),
        task_data={
            "verifier_metadata": {"golden_answer": {"value": "red"}},
            "image_files": image_files
            if image_files is not None
            else [{"source": "task-1/0.png", "target": "images/0.png"}],
        },
    )


def test_seed_lends_a_sandbox_that_holds_the_task_images(root, image_root, stopped) -> None:
    server = _server(image_root)
    client = TestClient(server.setup_webserver())

    seeded = client.post("/seed_session", json=_seed().model_dump(mode="json"))

    assert seeded.status_code == 200, seeded.text
    access = seeded.json()["sandbox_access"]
    assert access["workdir"] == str(root)
    assert access["connection"] == {
        "kind": "direct",
        "provider_config_ref": "sandbox",
        "descriptor": {"sandbox_id": "local"},
    }
    assert (root / "images" / "0.png").read_bytes() == PNG
    # A repeated seed returns the same session and starts no second sandbox.
    again = client.post("/seed_session", json=_seed().model_dump(mode="json"))
    assert again.status_code == 200
    assert len(server._sessions) == 1

    close = ResourcesCloseSessionRequest(resources_session_id="session", episode_id=_seed().episode_id)
    assert client.post("/close_session", json=close.model_dump(mode="json")).status_code == 200
    assert len(stopped) == 1
    assert server._sessions == {}
    # A repeated close succeeds and stops nothing more.
    assert client.post("/close_session", json=close.model_dump(mode="json")).status_code == 200
    assert len(stopped) == 1


def test_seed_can_lend_an_empty_sandbox_next_to_the_server_tools(root, image_root, stopped) -> None:
    """With the tools on the server, the lent sandbox holds no task images, so the harness must use the tools."""
    # The tool workspace decodes the image, so this test needs a real one.
    Image.new("RGB", (4, 4), "red").save(image_root / "task-1" / "0.png")
    server = _server(image_root, serve_image_tools=True, task_images_in_sandbox=False)
    client = TestClient(server.setup_webserver())

    seeded = client.post("/seed_session", json=_seed().model_dump(mode="json"))

    assert seeded.status_code == 200, seeded.text
    assert seeded.json()["sandbox_access"]["workdir"] == str(root)
    assert root.is_dir() and list(root.iterdir()) == []
    # The server still loaded the task image into the tool workspace.
    assert server._sessions["session"].workspace is not None

    close = ResourcesCloseSessionRequest(resources_session_id="session", episode_id=_seed().episode_id)
    assert client.post("/close_session", json=close.model_dump(mode="json")).status_code == 200
    assert len(stopped) == 1 and server._sessions == {}


def test_an_empty_sandbox_without_server_tools_reads_no_task_images(root, stopped) -> None:
    config = AgenticMMEConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="agentic_mme",
        sandbox_provider="sandbox",
        sandbox_config={"image": "unused-by-the-local-provider"},
        task_images_in_sandbox=False,
    )
    server = AgenticMMEServer(config=config, server_client=MagicMock(spec=ServerClient))
    client = TestClient(server.setup_webserver())

    # No image_root is set, and a row without image_files still gets its sandbox.
    seed = _seed()
    seed.task_data.pop("image_files")
    seeded = client.post("/seed_session", json=seed.model_dump(mode="json"))

    assert seeded.status_code == 200, seeded.text
    assert seeded.json()["sandbox_access"] is not None
    assert list(root.iterdir()) == []


def test_lent_sandbox_follows_the_session_contract(root, image_root, stopped) -> None:
    server = _server(image_root)
    check_resources_session_contract(
        server.setup_webserver(),
        _seed("contract"),
        keeps_state=True,
        holds_state=lambda resources_session_id: resources_session_id in server._sessions,
    )


@pytest.mark.parametrize(
    "image_files, message",
    [
        ([], "empty"),
        ([{"source": "task-1/missing.png", "target": "images/0.png"}], "not under image_root"),
        ([{"source": "../outside.png", "target": "images/0.png"}], "outside image_root"),
        ([{"source": "task-1/0.png", "target": "../escape.png"}], "working directory"),
        ([{"source": "task-1/0.png"}], "malformed"),
    ],
)
def test_seed_refuses_bad_image_files_without_a_retry(root, image_root, stopped, image_files, message) -> None:
    client = TestClient(_server(image_root).setup_webserver())

    refused = client.post("/seed_session", json=_seed(image_files=image_files).model_dump(mode="json"))

    # 422 is terminal for the Environment Server, so a bad row fails once instead of being dispatched again.
    assert refused.status_code == 422
    assert message in refused.json()["detail"]
    assert stopped == []


def test_without_a_sandbox_provider_the_server_stays_stateless(image_root) -> None:
    server = AgenticMMEServer(
        config=AgenticMMEConfig(host="0.0.0.0", port=8080, entrypoint="", name="agentic_mme"),
        server_client=MagicMock(spec=ServerClient),
    )
    client = TestClient(server.setup_webserver())

    seeded = client.post("/seed_session", json=_seed().model_dump(mode="json"))

    assert seeded.status_code == 200
    assert seeded.json().get("sandbox_access") is None
    assert server._sessions == {}
    check_resources_session_contract(server.setup_webserver(), _seed("stateless"), keeps_state=False)


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"sandbox_config": {}, "image_root": "/images"}, "sandbox_config.image"),
        ({"sandbox_config": {"image": "image"}}, "image_root"),
    ],
)
def test_a_sandbox_provider_needs_an_image_and_an_image_root(fields, message) -> None:
    config = AgenticMMEConfig(
        host="0.0.0.0", port=8080, entrypoint="", name="agentic_mme", sandbox_provider="sandbox", **fields
    )
    with pytest.raises(ValueError, match=message):
        AgenticMMEServer(config=config, server_client=MagicMock(spec=ServerClient))


def test_task_images_in_sandbox_off_needs_a_sandbox_provider(image_root) -> None:
    config = AgenticMMEConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="agentic_mme",
        serve_image_tools=True,
        image_root=str(image_root),
        task_images_in_sandbox=False,
    )
    with pytest.raises(ValueError, match="needs sandbox_provider"):
        AgenticMMEServer(config=config, server_client=MagicMock(spec=ServerClient))
