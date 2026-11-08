# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Episode sessions and the agent-run seed, through the server's HTTP routes with fake sandboxes."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.client import ServerClient
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.server_runtime import ResourcesSeedSessionRequest, check_resources_session_contract
from resources_servers.deepswe.app import VerifierResult
from resources_servers.deepswe.validate_golden import _empty_response
from resources_servers.deepswe_external1.app import (
    DeepsweExternal1ResourcesServer,
    DeepsweExternal1ResourcesServerConfig,
)
from resources_servers.deepswe_external1.inline_task import InlineTask
from resources_servers.deepswe_external1.prepare_examples import task_row


CLOSE = {"resources_session_id": "resources-session", "episode_id": {"rollout_id": "rollout"}}


def make_server(*, mode: str = "agent") -> tuple[DeepsweExternal1ResourcesServer, list[AsyncMock], list[str]]:
    """A server whose sandboxes are fakes, with patch collection and grading stubbed."""
    config = DeepsweExternal1ResourcesServerConfig(
        host="127.0.0.1",
        port=8000,
        entrypoint="app.py",
        name="deepswe_external1",
        is_verifying_golden_patch=mode == "golden",
        is_verifying_null_patch=mode == "null",
        sandbox_provider="sandbox",
        sandbox_config={},
        logs_dir=Path.cwd() / "logs",
    )
    server = DeepsweExternal1ResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    sandboxes: list[AsyncMock] = []
    phases: list[str] = []

    async def create(_task: InlineTask, *, phase: str, files=None) -> AsyncMock:
        box = AsyncMock()
        box.serialize.return_value = {"sandbox_id": f"sb-{len(sandboxes) + 1}"}
        sandboxes.append(box)
        phases.append(phase)
        return box

    server._create_sandbox = AsyncMock(side_effect=create)
    server._collect_model_patch = AsyncMock(return_value=b"committed patch")
    server._run_verifier = AsyncMock(return_value=VerifierResult(evaluation_completed=True, reward=1.0))
    return server, sandboxes, phases


def seed(task: InlineTask, session_id: str = "resources-session") -> ResourcesSeedSessionRequest:
    return ResourcesSeedSessionRequest(
        resources_session_id=session_id,
        episode_id=EpisodeId(rollout_id="rollout"),
        task_id=TaskId(taskset="deepswe_external1", task_id=task.data.task_id),
        task_data=task_row(task.data, "original instruction\n"),
    )


def verify_body(task: InlineTask) -> dict:
    return task_row(task.data, "original instruction\n") | {"response": _empty_response()}


def test_episode_session_lends_the_agent_sandbox_grades_it_once_and_closes(task: InlineTask) -> None:
    server, sandboxes, phases = make_server()
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        seeded = client.post("/seed_session", json=seed(task).model_dump(mode="json"))
        assert seeded.status_code == 200, seeded.text
        assert seeded.json()["sandbox_access"] == {
            "connection": {"kind": "direct", "provider_config_ref": "sandbox", "descriptor": {"sandbox_id": "sb-1"}},
            "workdir": "/app",
        }
        # The agent gets a lease that cannot destroy the sandbox verify still has to collect from.
        sandboxes[0].serialize.assert_any_await(scope="operate")
        assert client.post("/seed_session", json=seed(task).model_dump(mode="json")).json() == seeded.json()
        assert len(sandboxes) == 1, "a repeated seed returns the same sandbox"
        assert client.post("/seed_session", json=task_row(task.data, "x")).status_code == 409

        verified = client.post("/verify", json=verify_body(task))
        assert verified.status_code == 200, verified.text
        result = verified.json()
        assert result["reward"] == 1.0 and result["evaluation_completed"] and not result["mask_sample"]
        assert result["agent_sandbox_id"] == "sb-1" and result["verifier_sandbox_id"] == "sb-2"
        assert result["model_patch"] == "committed patch" and result["cleanup_errors"] == []
        assert phases == ["agent", "verifier"]
        sandboxes[0].stop.assert_awaited_once()
        sandboxes[1].stop.assert_awaited_once()

        repeated = client.post("/verify", json=verify_body(task))
        assert repeated.status_code == 200 and repeated.json() == result
        assert len(sandboxes) == 2, "an identical retry replays the verdict"

        assert client.post("/close_session", json=CLOSE).status_code == 200
        assert client.post("/seed_session", json=seed(task).model_dump(mode="json")).status_code == 409
        late = client.post("/verify", json=verify_body(task)).json()
        assert late["mask_sample"] and late["failure_kind"] == "session_lost" and late["reward"] == 0

    sandboxes[0].stop.assert_awaited_once()
    assert not server._session_id_to_sandbox and not server._seeded_agent_sandboxes
    assert not server._verifier_sandboxes


def test_close_without_verify_stops_the_agent_sandbox_and_a_failed_stop_is_retried(task: InlineTask) -> None:
    server, sandboxes, _ = make_server()
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=seed(task).model_dump(mode="json")).status_code == 200
        sandboxes[0].stop.side_effect = [RuntimeError("delete unavailable"), None]
        assert client.post("/close_session", json=CLOSE).status_code == 500
        assert server._session_id_to_sandbox, "a failed stop keeps the sandbox for the retried close"
        assert client.post("/close_session", json=CLOSE).status_code == 200
    assert sandboxes[0].stop.await_count == 2
    assert not server._session_id_to_sandbox and not server._seeded_agent_sandboxes


def test_verify_of_another_task_is_refused_and_keeps_the_session(task: InlineTask) -> None:
    server, sandboxes, _ = make_server()
    other = verify_body(task) | {"image": "public.example/tasks/other@sha256:" + "c" * 64}
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=seed(task).model_dump(mode="json")).status_code == 200
        assert client.post("/verify", json=other).status_code == 409
        sandboxes[0].stop.assert_not_awaited()
        assert client.post("/verify", json=verify_body(task)).json()["reward"] == 1.0


def test_failed_agent_cleanup_masks_the_sample_and_close_retries_the_stop(task: InlineTask) -> None:
    server, sandboxes, phases = make_server()
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=seed(task).model_dump(mode="json")).status_code == 200
        sandboxes[0].stop.side_effect = [RuntimeError("delete unavailable"), None]
        result = client.post("/verify", json=verify_body(task)).json()
        assert result["mask_sample"] and not result["evaluation_completed"]
        assert result["failure_stage"] == "agent_cleanup" and result["cleanup_errors"] == ["agent"]
        assert phases == ["agent"], "no verifier starts while the agent sandbox is still held"
        assert client.post("/close_session", json=CLOSE).status_code == 200
    assert sandboxes[0].stop.await_count == 2 and not server._session_id_to_sandbox


@pytest.mark.parametrize("mode", ["golden", "null"])
def test_validation_modes_refuse_episode_seeds(task: InlineTask, mode: str) -> None:
    server, sandboxes, _ = make_server(mode=mode)
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        refused = client.post("/seed_session", json=seed(task).model_dump(mode="json"))
    assert refused.status_code == 422 and "golden/null-validation" in refused.text
    assert not sandboxes


def test_agent_run_seed_returns_the_handle_and_verify_consumes_it(task: InlineTask) -> None:
    server, sandboxes, _ = make_server()
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        seeded = client.post("/seed_session", json=task_row(task.data, "original instruction\n"))
        assert seeded.status_code == 200, seeded.text
        assert seeded.json() == {"sandbox_handle": "sb-1", "sandbox_descriptor": {"sandbox_id": "sb-1"}}
        verified = client.post("/verify", json=verify_body(task) | {"sandbox_handle": "sb-1"}).json()
        assert verified["reward"] == 1.0 and not verified["mask_sample"]
        sandboxes[0].stop.assert_awaited_once()
        again = client.post("/verify", json=verify_body(task)).json()
        assert again["mask_sample"] and again["failure_kind"] == "session_lost"
    assert len(sandboxes) == 2


def test_session_conformance(task: InlineTask) -> None:
    server, _, _ = make_server()
    check_resources_session_contract(
        server.setup_webserver(),
        seed(task, "conformance-session"),
        keeps_state=True,
        holds_state=lambda session_id: session_id in server._session_id_to_sandbox,
        verify_body=verify_body(task),
    )
