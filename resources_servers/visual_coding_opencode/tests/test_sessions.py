# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The lent sandbox: typed seed and close, run on the local sandbox provider in a temporary directory."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.server_runtime import (
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
    check_resources_session_contract,
)
from resources_servers.visual_coding_opencode.app import VisualCodingOpenCodeConfig, VisualCodingOpenCodeServer


@pytest.fixture
def root(tmp_path, monkeypatch) -> Path:
    """The sandbox working directory on the host; the local provider runs commands there."""
    monkeypatch.setattr(
        "nemo_gym.server_runtime.sandbox_lending.get_global_config_dict", lambda: {"sandbox": {"local": {}}}
    )
    # The local provider cannot be reattached from another process, so it has no descriptor to hand out.
    monkeypatch.setattr(AsyncSandbox, "serialize", AsyncMock(return_value={"sandbox_id": "local"}))
    return tmp_path / "workspace"


@pytest.fixture
def stopped(monkeypatch) -> list[AsyncSandbox]:
    stopped: list[AsyncSandbox] = []
    real_stop = AsyncSandbox.stop

    async def stop(self: AsyncSandbox) -> None:
        stopped.append(self)
        await real_stop(self)

    monkeypatch.setattr(AsyncSandbox, "stop", stop)
    return stopped


def _config(**fields) -> VisualCodingOpenCodeConfig:
    return VisualCodingOpenCodeConfig(
        name="visual_coding_opencode",
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        **fields,
    )


def _server(root: Path, **fields) -> VisualCodingOpenCodeServer:
    config = _config(
        sandbox_provider="sandbox",
        sandbox_config={"image": "unused-by-the-local-provider"},
        sandbox_workdir=str(root),
        **fields,
    )
    return VisualCodingOpenCodeServer(config=config, server_client=MagicMock(spec=ServerClient))


def _seed(session_id: str = "session") -> ResourcesSeedSessionRequest:
    return ResourcesSeedSessionRequest(
        resources_session_id=session_id,
        episode_id=EpisodeId(rollout_id=f"rollout-{session_id}"),
        task_id=TaskId(taskset="babyvision_opencode:benchmark", task_id="task-1"),
        task_data={"expected_answer": "(4,7)", "question": "Which cell?"},
    )


def test_seed_lends_a_sandbox_and_close_stops_it(root, stopped) -> None:
    server = _server(root)
    client = TestClient(server.setup_webserver())

    seeded = client.post("/seed_session", json=_seed().model_dump(mode="json"))

    assert seeded.status_code == 200, seeded.text
    assert seeded.json()["sandbox_access"] == {
        "connection": {"kind": "direct", "provider_config_ref": "sandbox", "descriptor": {"sandbox_id": "local"}},
        "workdir": str(root),
    }
    # A repeated seed returns the same session and starts no second sandbox.
    again = client.post("/seed_session", json=_seed().model_dump(mode="json"))
    assert again.status_code == 200
    assert again.json()["sandbox_access"] == seeded.json()["sandbox_access"]
    assert server._lent_sandboxes.get("session") is not None

    close = ResourcesCloseSessionRequest(resources_session_id="session", episode_id=_seed().episode_id)
    assert client.post("/close_session", json=close.model_dump(mode="json")).status_code == 200
    assert len(stopped) == 1
    assert server._lent_sandboxes.get("session") is None
    # A repeated close succeeds and stops nothing more.
    assert client.post("/close_session", json=close.model_dump(mode="json")).status_code == 200
    assert len(stopped) == 1


def test_a_failed_stop_leaves_the_session_open_and_a_retried_close_stops_it(root, monkeypatch) -> None:
    stops: list[AsyncSandbox] = []
    real_stop = AsyncSandbox.stop

    async def stop(self: AsyncSandbox) -> None:
        stops.append(self)
        if len(stops) == 1:
            raise RuntimeError("provider unavailable")
        await real_stop(self)

    monkeypatch.setattr(AsyncSandbox, "stop", stop)
    server = _server(root)
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)
    assert client.post("/seed_session", json=_seed().model_dump(mode="json")).status_code == 200
    close = ResourcesCloseSessionRequest(resources_session_id="session", episode_id=_seed().episode_id)

    assert client.post("/close_session", json=close.model_dump(mode="json")).status_code >= 500
    # The stop raised before the fence, so the session keeps its sandbox and is not fenced: a close can retry.
    assert "session" not in server._fences
    assert server._lent_sandboxes.get("session") is not None and "session" in server._sessions

    assert client.post("/close_session", json=close.model_dump(mode="json")).status_code == 200
    assert len(stops) == 2 and stops[0] is stops[1]
    assert server._lent_sandboxes.get("session") is None and "session" in server._fences


def test_lent_sandbox_follows_the_session_contract(root, stopped) -> None:
    server = _server(root)
    check_resources_session_contract(
        server.setup_webserver(),
        _seed("contract"),
        keeps_state=True,
        holds_state=lambda resources_session_id: server._lent_sandboxes.get(resources_session_id) is not None,
    )


def test_shutdown_stops_every_sandbox_still_lent(root, stopped) -> None:
    server = _server(root)
    with TestClient(server.setup_webserver()) as client:
        for session in ("a", "b"):
            assert client.post("/seed_session", json=_seed(session).model_dump(mode="json")).status_code == 200
        assert stopped == []

    assert len(stopped) == 2
    assert server._lent_sandboxes.get("a") is None and server._lent_sandboxes.get("b") is None


def test_model_only_lends_a_sandbox_that_reaches_only_the_policy_model(root, monkeypatch) -> None:
    """The config's model_only setting reaches the lent sandbox's spec as an egress rule for the model's host."""
    monkeypatch.setattr(
        "nemo_gym.server_runtime.sandbox_lending.get_global_config_dict", lambda: {"sandbox": {"opensandbox": {}}}
    )
    monkeypatch.setattr("nemo_gym.server_runtime.sandbox_network.get_server_url", lambda name: "http://10.1.2.3:9000")
    specs = []

    async def start(self: AsyncSandbox, spec) -> None:
        specs.append(spec)

    monkeypatch.setattr(AsyncSandbox, "__init__", lambda self, provider: None)
    monkeypatch.setattr(AsyncSandbox, "start", start)
    server = _server(
        root,
        sandbox_network_access="model_only",
        sandbox_model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
    )
    client = TestClient(server.setup_webserver())

    assert client.post("/seed_session", json=_seed().model_dump(mode="json")).status_code == 200

    (spec,) = specs
    assert spec.image == "unused-by-the-local-provider"
    assert spec.provider_options["network_policy"] == {
        "defaultAction": "deny",
        "egress": [{"action": "allow", "target": "10.1.2.3"}],
    }


def test_without_a_sandbox_provider_the_server_stays_stateless() -> None:
    server = VisualCodingOpenCodeServer(config=_config(), server_client=MagicMock(spec=ServerClient))
    client = TestClient(server.setup_webserver())

    seeded = client.post("/seed_session", json=_seed().model_dump(mode="json"))

    assert seeded.status_code == 200
    assert seeded.json()["sandbox_access"] is None
    check_resources_session_contract(server.setup_webserver(), _seed("stateless"), keeps_state=False)


# A malformed typed seed: it names a resources session but has no episode, task or task_data.
_ID_ONLY_SEED = {"resources_session_id": "s"}


@pytest.mark.parametrize(
    "body",
    [
        _ID_ONLY_SEED,
        {**_seed().model_dump(mode="json"), "task_data": None},
        # A cookie seed, as an agent's own /run posts: a dataset row with a scalar task_id.
        {"task_id": "task-1", "expected_answer": "(4,7)", "question": "Which cell?"},
    ],
)
def test_a_lending_server_lends_only_to_a_typed_episode_seed(root, stopped, body) -> None:
    server = _server(root)
    client = TestClient(server.setup_webserver())

    refused = client.post("/seed_session", json=body)

    assert refused.status_code == 422, refused.text
    assert server._lent_sandboxes.get("s") is None and server._lent_sandboxes.get("session") is None
    assert stopped == []


def test_a_stateless_server_refuses_a_malformed_typed_seed_and_accepts_a_cookie_seed() -> None:
    server = VisualCodingOpenCodeServer(config=_config(), server_client=MagicMock(spec=ServerClient))
    client = TestClient(server.setup_webserver())

    assert client.post("/seed_session", json=_ID_ONLY_SEED).status_code == 422
    assert client.post("/seed_session", json={"task_id": "task-1"}).status_code == 200


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"sandbox_provider": "sandbox"}, "Set sandbox_config.image"),
        ({"sandbox_network_access": "model_only"}, "set sandbox_provider"),
        (
            {"sandbox_provider": "sandbox", "sandbox_config": {"image": "i"}, "sandbox_network_access": "model_only"},
            "needs model_server",
        ),
    ],
)
def test_sandbox_settings_that_cannot_work_are_refused_at_startup(fields, message) -> None:
    with pytest.raises(ValueError, match=message):
        VisualCodingOpenCodeServer(config=_config(**fields), server_client=MagicMock(spec=ServerClient))
