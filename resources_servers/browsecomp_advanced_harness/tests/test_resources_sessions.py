# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The episode sessions an Environment Server seeds and closes, next to the agent-run seed."""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.client import ServerClient
from nemo_gym.core import (
    SESSION_LOST,
    EpisodeId,
    NeMoGymResponse,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    TaskId,
)
from nemo_gym.server_runtime import (
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
    check_resources_session_contract,
)
from resources_servers.browsecomp_advanced_harness import app as harness_app
from resources_servers.browsecomp_advanced_harness.app import (
    TavilySearchMetrics,
    TavilySearchResourcesServer,
    TavilySearchResourcesServerConfig,
)


_EXCLUDE_DOMAINS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dummy_exclude_domains_file.json")
_EPISODE = EpisodeId(rollout_id="rollout", attempt=0)
_TASK_DATA = {"question": "What is the capital of France?", "ground_truth": "Paris"}


def _server(tmp_path: Path, **overrides) -> TavilySearchResourcesServer:
    config = TavilySearchResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        tavily_api_key="test_api_key",  # pragma: allowlist secret
        exclude_domains_file_path=_EXCLUDE_DOMAINS_FILE,
        use_judge=False,
        workspace_root=str(tmp_path),
        **({"allow_host_execution": True} | overrides),
    )
    return TavilySearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _seed(resources_session_id: str = "resources-session", task_data: dict | None = None) -> dict:
    return ResourcesSeedSessionRequest(
        resources_session_id=resources_session_id,
        episode_id=_EPISODE,
        task_id=TaskId(taskset="browsecomp", task_id="0"),
        task_data=_TASK_DATA if task_data is None else task_data,
    ).model_dump(mode="json")


def _close(resources_session_id: str = "resources-session") -> dict:
    return ResourcesCloseSessionRequest(resources_session_id=resources_session_id, episode_id=_EPISODE).model_dump(
        mode="json"
    )


def _verify_body(answer: str) -> dict:
    response = NeMoGymResponse(
        id="resp",
        created_at=0.0,
        model="policy",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="msg",
                content=[NeMoGymResponseOutputText(annotations=[], text=answer, type="output_text")],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )
    return {**_TASK_DATA, "responses_create_params": {"input": []}, "response": response.model_dump(mode="json")}


@pytest.mark.parametrize("workspace", ["per_session", "none"])
def test_episode_sessions_follow_the_session_contract(tmp_path: Path, workspace: str) -> None:
    server = _server(tmp_path, workspace=workspace)
    check_resources_session_contract(
        server.setup_webserver(),
        ResourcesSeedSessionRequest.model_validate(_seed("contract-session")),
        keeps_state=True,
        holds_state=lambda resources_session_id: (
            resources_session_id in server._episode_sessions or resources_session_id in server._session_workspaces
        ),
        verify_body=_verify_body("Answer: Paris Confidence: 90%"),
    )
    # Both closed sessions' workspaces are gone from disk, not only from the server's map.
    assert list(tmp_path.iterdir()) == []


def test_close_removes_the_workspace_and_metrics_and_refuses_later_tool_calls(tmp_path: Path) -> None:
    server = _server(tmp_path, workspace="per_session")
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)
    assert client.post("/seed_session", json=_seed()).status_code == 200
    workspace = tmp_path / "resources-session"
    assert (workspace / "manifest.tsv").is_file()
    server._session_id_to_metrics["resources-session"] = TavilySearchMetrics()

    assert client.post("/close_session", json=_close()).status_code == 200

    assert not workspace.exists()
    assert "resources-session" not in server._session_id_to_metrics
    # A late tool call names the closed session by its cookie: it is refused and recreates nothing.
    for path, body in (("/bash_command", {"keystrokes": "ls"}), ("/search", {"queries": ["q"]})):
        response = client.post(path, json=body)
        assert response.status_code == 404, response.text
        assert response.json()["detail"]["failure_kind"] == SESSION_LOST
    assert not workspace.exists()
    assert "resources-session" not in server._session_id_to_metrics


def test_a_failed_workspace_removal_fails_the_close_and_a_retry_releases_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _server(tmp_path, workspace="per_session")
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)
    assert client.post("/seed_session", json=_seed()).status_code == 200
    workspace = tmp_path / "resources-session"
    remove_workspace = harness_app._remove_workspace

    def fail(path: Path) -> None:
        raise PermissionError(f"cannot remove {path}")

    monkeypatch.setattr(harness_app, "_remove_workspace", fail)
    assert client.post("/close_session", json=_close()).status_code == 500
    # Not recorded as closed, so the workspace is still held for the caller's retry.
    assert workspace.exists()
    assert "resources-session" in server._session_workspaces
    assert server._fences.get("resources-session") is None

    monkeypatch.setattr(harness_app, "_remove_workspace", remove_workspace)
    assert client.post("/close_session", json=_close()).status_code == 200
    assert not workspace.exists()
    assert server._fences.get("resources-session") == _EPISODE


def test_a_seed_without_the_verify_fields_is_refused_as_final(tmp_path: Path) -> None:
    client = TestClient(_server(tmp_path).setup_webserver(), raise_server_exceptions=False)
    response = client.post("/seed_session", json=_seed(task_data={"question": "q"}))
    assert response.status_code == 422
    assert "ground_truth" in response.json()["detail"]


def test_a_workspace_seed_refuses_an_id_that_is_not_a_directory_name(tmp_path: Path) -> None:
    server = _server(tmp_path / "root", workspace="per_session")
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)
    response = client.post("/seed_session", json=_seed("../escape"))
    assert response.status_code == 422
    assert not (tmp_path / "escape").exists()
    assert server._episode_sessions == {}


def test_the_legacy_run_seed_and_verify_still_own_the_cookie_session(tmp_path: Path) -> None:
    """An agent's /run seeds with its flat row and never closes; its /verify releases the session."""
    server = _server(tmp_path, workspace="per_session")
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)
    row = {**_TASK_DATA, "responses_create_params": {"input": []}}
    seed = client.post("/seed_session", json=row)
    assert seed.status_code == 200
    assert seed.json() == {}
    assert server._episode_sessions == {}
    [sid] = server._session_workspaces
    assert (tmp_path / sid / "pages").is_dir()

    listing = client.post("/bash_command", json={"keystrokes": "ls"})
    assert listing.status_code == 200
    assert "manifest.tsv" in listing.json()["results_string"]
    server._session_id_to_metrics[sid] = TavilySearchMetrics()

    verify = client.post("/verify", json=_verify_body("Answer: Paris Confidence: 90%"))
    assert verify.status_code == 200, verify.text
    assert (verify.json()["reward"], verify.json()["mask_sample"]) == (1.0, False)
    assert server._session_workspaces == {}
    assert not (tmp_path / sid).exists()
    assert sid not in server._session_id_to_metrics


def test_verify_keeps_metrics_for_the_dump_at_shutdown(tmp_path: Path) -> None:
    server = _server(tmp_path, dump_session_id_to_metrics_on_exit=True)
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)
    assert client.post("/seed_session", json=_seed()).status_code == 200
    server._session_id_to_metrics["resources-session"] = TavilySearchMetrics()

    assert client.post("/verify", json=_verify_body("Answer: London Confidence: 90%")).json()["reward"] == 0.0
    assert client.post("/close_session", json=_close()).status_code == 200

    assert "resources-session" in server._session_id_to_metrics


def test_verify_after_close_is_masked(tmp_path: Path) -> None:
    """Masked like every server under the contract, although the answer alone would score 1."""
    server = _server(tmp_path, workspace="per_session")
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)
    assert client.post("/seed_session", json=_seed()).status_code == 200
    assert client.post("/close_session", json=_close()).status_code == 200

    verify = client.post("/verify", json=_verify_body("Answer: Paris Confidence: 90%"))

    assert verify.status_code == 200, verify.text
    verified = verify.json()
    assert (verified["reward"], verified["mask_sample"], verified["failure_kind"]) == (0.0, True, SESSION_LOST)
    assert verified["metrics"] == {"async_tavily_calls": []}
    # The masked verify recreates neither the workspace nor a metrics entry.
    assert not (tmp_path / "resources-session").exists()
    assert "resources-session" not in server._session_id_to_metrics


def test_a_masked_verify_still_reports_a_timed_out_rollout(tmp_path: Path) -> None:
    client = TestClient(_server(tmp_path).setup_webserver(), raise_server_exceptions=False)
    assert client.post("/seed_session", json=_seed()).status_code == 200
    assert client.post("/close_session", json=_close()).status_code == 200
    body = _verify_body("Answer: Paris Confidence: 90%")
    body["response"]["timed_out"] = True

    verified = client.post("/verify", json=body).json()

    assert (verified["mask_sample"], verified["timed_out"]) == (True, 1)


def test_a_caller_that_never_seeds_is_still_graded(tmp_path: Path) -> None:
    """A verify for a session nobody closed, here one nobody seeded, is graded from the response."""
    client = TestClient(_server(tmp_path, workspace="per_session").setup_webserver(), raise_server_exceptions=False)

    verify = client.post("/verify", json=_verify_body("Answer: Paris Confidence: 90%"))

    assert verify.status_code == 200, verify.text
    assert (verify.json()["reward"], verify.json()["mask_sample"]) == (1.0, False)
