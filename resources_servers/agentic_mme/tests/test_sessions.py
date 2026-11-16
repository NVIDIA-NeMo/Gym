# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The lent sandbox: typed seed and close, run on the local sandbox provider in a temporary directory."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from nemo_gym.client import ServerClient
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.server_runtime import (
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
    check_resources_session_contract,
)
from resources_servers.agentic_mme import app as agentic_mme_app
from resources_servers.agentic_mme.app import AgenticMMEConfig, AgenticMMEServer
from resources_servers.agentic_mme.tests.test_app import response


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


def _verify_body() -> dict:
    """A verify for the seeded task, as single_agent_turn sends it."""
    return {
        "responses_create_params": {"input": "What colour is the image?"},
        "response": response("<answer>red</answer>").model_dump(mode="json"),
        "verifier_metadata": {"golden_answer": {"value": "red"}},
    }


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
        verify_body=_verify_body(),
    )


async def test_close_releases_the_lent_sandbox_by_session_id(root, image_root, stopped) -> None:
    """Close stops the sandbox lent under the closed resources_session_id, whatever else the server recorded."""
    server = _server(image_root)
    await server._lent_sandboxes.lend("session", _seed().task_id)
    assert server._sessions == {}

    await server.close_resources_session(
        ResourcesCloseSessionRequest(resources_session_id="session", episode_id=_seed().episode_id)
    )

    assert len(stopped) == 1
    assert server._lent_sandboxes.get("session") is None


def test_shutdown_stops_every_sandbox_still_lent(root, image_root, stopped) -> None:
    server = _server(image_root)
    with TestClient(server.setup_webserver()) as client:
        for session in ("a", "b"):
            assert client.post("/seed_session", json=_seed(session).model_dump(mode="json")).status_code == 200
        assert stopped == []

    assert len(stopped) == 2
    assert server._lent_sandboxes.get("a") is None and server._lent_sandboxes.get("b") is None


def test_a_relative_image_root_is_relative_to_the_gym_checkout(root, stopped, monkeypatch, tmp_path) -> None:
    """The server process runs in its own directory, so a relative image_root must not depend on it."""
    checkout = tmp_path / "gym"
    (checkout / "data" / "task-1").mkdir(parents=True)
    (checkout / "data" / "task-1" / "0.png").write_bytes(PNG)
    monkeypatch.setattr(agentic_mme_app, "PARENT_DIR", checkout)
    monkeypatch.chdir(tmp_path)
    client = TestClient(_server(Path("data")).setup_webserver())

    seeded = client.post("/seed_session", json=_seed().model_dump(mode="json"))

    assert seeded.status_code == 200, seeded.text
    assert (root / "images" / "0.png").read_bytes() == PNG


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


def test_seed_refuses_an_unreadable_image_without_a_retry(root, image_root, stopped, monkeypatch) -> None:
    """A task image that exists but cannot be read fails the same way on every retry, so the seed gets 422."""
    read_bytes = Path.read_bytes

    def unreadable(path: Path) -> bytes:
        if path.name == "0.png":
            raise PermissionError(13, "Permission denied", str(path))
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", unreadable)
    server = _server(image_root)
    client = TestClient(server.setup_webserver())

    refused = client.post("/seed_session", json=_seed().model_dump(mode="json"))

    assert refused.status_code == 422
    assert "'task-1/0.png' cannot be read" in refused.json()["detail"]
    assert server._sessions == {}
    assert server._lent_sandboxes.get("session") is None
    assert stopped == []


async def test_retire_releases_the_lent_sandbox_without_close_s_fence(root, image_root, stopped, monkeypatch) -> None:
    """A checkpoint retire frees what close frees. A failed stop keeps both for a retry, and no fence is left."""
    server = _server(image_root)
    await server.seed_session(SimpleNamespace(session={}), _seed())
    assert server._lent_sandboxes.get("session") is not None
    recorded_stop = AsyncSandbox.stop
    failures = [RuntimeError("stop failed")]

    async def stop_failing_once(self: AsyncSandbox) -> None:
        if failures:
            raise failures.pop()
        await recorded_stop(self)

    monkeypatch.setattr(AsyncSandbox, "stop", stop_failing_once)

    with pytest.raises(RuntimeError, match="stop failed"):
        await server.retire_session_state("session")
    assert "session" in server._sessions
    assert server._lent_sandboxes.get("session") is not None

    await server.retire_session_state("session")
    assert server._sessions == {}
    assert server._lent_sandboxes.get("session") is None
    assert len(stopped) == 1
    # The checkpoint participant refuses the retired attempt; a fence would also refuse a later restore of the ID.
    assert server._fences.get("session") is None


# A malformed typed seed: it names a resources session but has no episode, task or task_data.
_ID_ONLY_SEED = {"resources_session_id": "s"}
# A cookie seed, as an agent's own /run posts: a dataset row.
_COOKIE_SEED = {"verifier_metadata": {"golden_answer": {"value": "red"}}}


@pytest.mark.parametrize("server_tools", [False, True])
@pytest.mark.parametrize("body", [_ID_ONLY_SEED, _COOKIE_SEED])
def test_a_server_with_episode_state_takes_only_a_typed_episode_seed(
    root, image_root, stopped, server_tools, body
) -> None:
    extra = {"serve_image_tools": True} if server_tools else {}
    server = _server(image_root, **extra)
    client = TestClient(server.setup_webserver())

    refused = client.post("/seed_session", json=body)

    assert refused.status_code == 422, refused.text
    assert server._sessions == {} and stopped == []


def test_a_stateless_server_refuses_a_malformed_typed_seed_and_accepts_a_cookie_seed() -> None:
    server = AgenticMMEServer(
        config=AgenticMMEConfig(host="0.0.0.0", port=8080, entrypoint="", name="agentic_mme"),
        server_client=MagicMock(spec=ServerClient),
    )
    client = TestClient(server.setup_webserver())

    assert client.post("/seed_session", json=_ID_ONLY_SEED).status_code == 422
    assert client.post("/seed_session", json=_COOKIE_SEED).status_code == 200


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


@pytest.mark.parametrize(
    ("network_access", "extra", "tool_servers"),
    [
        ("model_only", {}, []),
        ("model_and_tools", {"expose_tools_over_mcp": True}, ["agentic_mme"]),
    ],
)
def test_the_lent_sandbox_gets_the_configured_network(
    image_root, monkeypatch, network_access, extra, tool_servers
) -> None:
    made = []
    monkeypatch.setattr(agentic_mme_app, "LentSandboxes", lambda *args, **kwargs: made.append(kwargs) or MagicMock())

    _server(
        image_root,
        sandbox_network_access=network_access,
        sandbox_model_server={"type": "responses_api_models", "name": "policy_model"},
        **extra,
    )

    (kwargs,) = made
    assert kwargs["network_access"] == network_access
    assert kwargs["model_server"] == "policy_model"
    assert list(kwargs["tool_servers"]) == tool_servers


def test_model_and_tools_needs_the_tools_over_mcp(image_root) -> None:
    with pytest.raises(ValueError, match="needs expose_tools_over_mcp"):
        _server(
            image_root,
            sandbox_network_access="model_and_tools",
            sandbox_model_server={"type": "responses_api_models", "name": "policy_model"},
        )


def test_a_network_rule_needs_a_lent_sandbox(image_root) -> None:
    config = AgenticMMEConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="agentic_mme",
        image_root=str(image_root),
        sandbox_network_access="model_only",
    )
    with pytest.raises(ValueError, match="applies to a lent sandbox"):
        AgenticMMEServer(config=config, server_client=MagicMock(spec=ServerClient))
