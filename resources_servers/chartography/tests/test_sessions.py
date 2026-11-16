# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sessions: the state hooks in both settings, and the lent sandbox of the OpenCode setting, run on the local
sandbox provider in a temporary directory."""

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.core import EpisodeId, NeMoGymChatCompletionCreateParamsNonStreaming, TaskId
from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.server_runtime import (
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
    check_resources_session_contract,
)
from resources_servers.chartography.app import ChartographyConfig, ChartographyResourcesServer
from resources_servers.chartography.tests.test_app import (
    GOLDEN,
    JUDGE_PROMPT_FPATH,
    QUESTION,
    _assistant_msg,
    _judge_content,
    _mock_judge,
    _request,
)


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


def _config(**fields: Any) -> ChartographyConfig:
    return ChartographyConfig(
        name="chartography",
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        judge_chat_create_params=NeMoGymChatCompletionCreateParamsNonStreaming(messages=[]),
        judge_prompt_template_fpath=str(JUDGE_PROMPT_FPATH),
        **fields,
    )


def _server(root: Path, **fields: Any) -> ChartographyResourcesServer:
    config = _config(
        sandbox_provider="sandbox",
        sandbox_config={"image": "unused-by-the-local-provider"},
        sandbox_workdir=str(root),
        **fields,
    )
    return ChartographyResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _plain_server() -> ChartographyResourcesServer:
    return ChartographyResourcesServer(config=_config(), server_client=MagicMock(spec=ServerClient))


def _seed(session_id: str = "session") -> ResourcesSeedSessionRequest:
    return ResourcesSeedSessionRequest(
        resources_session_id=session_id,
        episode_id=EpisodeId(rollout_id=f"rollout-{session_id}"),
        task_id=TaskId(taskset="chartography_opencode:benchmark", task_id="t-1"),
        task_data={"verifier_metadata": {"task_id": "t-1", "prompt": QUESTION, "golden_answer": GOLDEN}},
    )


def _close(session_id: str = "session") -> dict:
    return ResourcesCloseSessionRequest(
        resources_session_id=session_id, episode_id=_seed(session_id).episode_id
    ).model_dump(mode="json")


def _verify_body() -> dict:
    return _request([_assistant_msg("Head: 34.8 m")]).model_dump(mode="json")


# ---------------------------------------------------------------------------
# The OpenCode setting: a sandbox lent per episode
# ---------------------------------------------------------------------------


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

    assert client.post("/close_session", json=_close()).status_code == 200
    assert len(stopped) == 1
    assert server._lent_sandboxes.get("session") is None
    # A repeated close succeeds and stops nothing more.
    assert client.post("/close_session", json=_close()).status_code == 200
    assert len(stopped) == 1


def test_lent_sandbox_follows_the_session_contract(root, stopped) -> None:
    server = _server(root)
    check_resources_session_contract(
        server.setup_webserver(),
        _seed("contract"),
        keeps_state=True,
        holds_state=lambda session_id: server._lent_sandboxes.get(session_id) is not None,
        verify_body=_verify_body(),
    )


def test_verify_stops_the_sandbox_before_the_judge_runs(root, stopped) -> None:
    """Grading never reads the sandbox, so it is not kept alive while the verify waits on the judge."""
    server = _server(root)
    client = TestClient(server.setup_webserver())
    assert client.post("/seed_session", json=_seed().model_dump(mode="json")).status_code == 200
    mock = _mock_judge(server, _judge_content(1))
    stops_at_judge: list[int] = []
    judge_post = mock.side_effect

    async def post(*args: Any, **kwargs: Any) -> Any:
        stops_at_judge.append(len(stopped))
        return await judge_post(*args, **kwargs)

    mock.side_effect = post

    verified = client.post("/verify", json=_verify_body())

    assert verified.status_code == 200, verified.text
    assert verified.json()["reward"] == 1.0
    assert stops_at_judge == [1]
    assert server._lent_sandboxes.get("session") is None
    # The episode's close then finds nothing left to stop.
    assert client.post("/close_session", json=_close()).status_code == 200
    assert len(stopped) == 1


def test_a_verify_retried_after_a_judge_failure_is_graded(root, stopped) -> None:
    """Gym verifies again after a judge failure; the grade reads only the request, so the retry is scored."""
    server = _server(root)
    client = TestClient(server.setup_webserver())
    assert client.post("/seed_session", json=_seed().model_dump(mode="json")).status_code == 200

    _mock_judge(server, '{"scores": []}')  # never parses: a judge failure, masked by judge_failsafe
    failed = client.post("/verify", json=_verify_body())
    assert failed.status_code == 200, failed.text
    assert failed.json()["mask_sample"] is True

    _mock_judge(server, _judge_content(1))
    retried = client.post("/verify", json=_verify_body())
    assert retried.status_code == 200, retried.text
    assert retried.json()["reward"] == 1.0 and not retried.json().get("mask_sample")
    assert len(stopped) == 1


@pytest.mark.parametrize("failed_stops", [1, 2])
def test_a_failed_stop_before_the_judge_keeps_the_grade_and_is_retried(
    root, stopped, monkeypatch, failed_stops
) -> None:
    """verifying() stops the sandbox again on exit, and a stop that fails there too is left for the close."""
    server = _server(root)
    client = TestClient(server.setup_webserver())
    assert client.post("/seed_session", json=_seed().model_dump(mode="json")).status_code == 200
    _mock_judge(server, _judge_content(1))
    attempts: list[AsyncSandbox] = []
    recording_stop = AsyncSandbox.stop

    async def stop(self: AsyncSandbox) -> None:
        attempts.append(self)
        if len(attempts) <= failed_stops:
            raise RuntimeError("the provider refused the stop")
        await recording_stop(self)

    monkeypatch.setattr(AsyncSandbox, "stop", stop)

    verified = client.post("/verify", json=_verify_body())

    assert verified.status_code == 200, verified.text
    assert verified.json()["reward"] == 1.0 and not verified.json().get("mask_sample")
    assert client.post("/close_session", json=_close()).status_code == 200
    assert len(attempts) == failed_stops + 1 and len(stopped) == 1
    assert server._lent_sandboxes.get("session") is None


def test_a_verify_after_the_close_is_masked_as_lost(root, stopped) -> None:
    server = _server(root)
    client = TestClient(server.setup_webserver())
    assert client.post("/seed_session", json=_seed().model_dump(mode="json")).status_code == 200
    assert client.post("/close_session", json=_close()).status_code == 200
    mock = _mock_judge(server, _judge_content(1))

    verified = client.post("/verify", json=_verify_body())

    assert verified.status_code == 200, verified.text
    assert verified.json()["mask_sample"] is True
    assert verified.json()["failure_kind"] == "session_lost"
    assert mock.await_count == 0


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
    # Only the policy model resolves to the address the egress rule should allow.
    monkeypatch.setattr(
        "nemo_gym.server_runtime.sandbox_lending.get_server_url",
        lambda name: {"policy_model": "http://10.1.2.3:9000", "judge": "http://10.9.9.9:9001"}[name],
    )
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


# A malformed typed seed: it names a resources session but has no episode, task or task_data.
_ID_ONLY_SEED = {"resources_session_id": "s"}


@pytest.mark.parametrize(
    "body",
    [
        _ID_ONLY_SEED,
        {**_seed().model_dump(mode="json"), "task_data": None},
        # A cookie seed, as an agent's own /run posts: a dataset row with a scalar task_id.
        {"task_id": "t-1", "verifier_metadata": {"prompt": QUESTION, "golden_answer": GOLDEN}},
    ],
)
def test_a_lending_server_lends_only_to_a_typed_episode_seed(root, stopped, body) -> None:
    server = _server(root)
    client = TestClient(server.setup_webserver())

    refused = client.post("/seed_session", json=body)

    assert refused.status_code == 422, refused.text
    assert server._lent_sandboxes.get("s") is None and server._lent_sandboxes.get("session") is None
    assert stopped == []


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
        ChartographyResourcesServer(config=_config(**fields), server_client=MagicMock(spec=ServerClient))


# ---------------------------------------------------------------------------
# The simple_agent setting: sessions hold nothing
# ---------------------------------------------------------------------------


def test_without_a_sandbox_provider_sessions_hold_no_sandbox() -> None:
    server = _plain_server()
    client = TestClient(server.setup_webserver())

    seeded = client.post("/seed_session", json=_seed().model_dump(mode="json"))

    assert seeded.status_code == 200, seeded.text
    assert seeded.json()["sandbox_access"] is None
    check_resources_session_contract(
        server.setup_webserver(),
        _seed("plain"),
        keeps_state=True,
        holds_state=lambda session_id: server.has_session_state(session_id),
        verify_body=_verify_body(),
    )


def test_without_a_sandbox_provider_a_verify_without_a_seed_is_graded() -> None:
    """An agent's own /run, or a re-verify of stored rollouts, verifies without an episode seed."""
    server = _plain_server()
    client = TestClient(server.setup_webserver())
    _mock_judge(server, _judge_content(1))

    verified = client.post("/verify", json=_verify_body())

    assert verified.status_code == 200, verified.text
    assert verified.json()["reward"] == 1.0


def test_without_a_sandbox_provider_a_cookie_seed_is_accepted() -> None:
    server = _plain_server()
    client = TestClient(server.setup_webserver())

    assert client.post("/seed_session", json=_ID_ONLY_SEED).status_code == 422
    assert client.post("/seed_session", json={"task_id": "t-1"}).status_code == 200


def test_only_the_no_tools_setting_runs_with_several_workers(root) -> None:
    assert _plain_server().multiple_workers_unsupported_reason() is None
    assert _server(root).multiple_workers_unsupported_reason() == "lent sandboxes live in one worker's memory"


@pytest.mark.parametrize(
    ("fields", "status", "reward", "judge_calls"),
    [
        ({"score_unfinished_as_zero": True}, "incomplete", 0.0, 0),
        ({"score_unfinished_as_zero": True}, "failed", 0.0, 0),
        ({"score_unfinished_as_zero": True}, "completed", 1.0, 1),
        # The default (the field unset) keeps upstream's behavior.
        ({}, "incomplete", 1.0, 1),
        ({}, "failed", 1.0, 1),
    ],
    ids=["opencode_incomplete", "opencode_failed", "opencode_completed", "default_incomplete", "default_failed"],
)
def test_an_unfinished_run_scores_zero_only_when_configured(fields, status, reward, judge_calls):
    """A run whose last message is the right answer. One that did not complete (a timeout, an output cap, a context
    limit, or a failed run that the outcome policy sends to verify) scores 0 only when configured."""
    config = _config(**fields)
    server = ChartographyResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    client = TestClient(server.setup_webserver())
    judge = _mock_judge(server, _judge_content(1))
    body = _verify_body()
    body["response"]["status"] = status

    verified = client.post("/verify", json=body)

    assert verified.status_code == 200, verified.text
    assert verified.json()["reward"] == reward
    assert not verified.json().get("mask_sample")
    assert judge.await_count == judge_calls
