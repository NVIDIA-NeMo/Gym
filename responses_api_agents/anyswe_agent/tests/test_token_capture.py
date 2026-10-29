# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Training-token capture wiring for anyswe_agent and its in-sandbox runner."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.anyswe_agent import agent_runner
from responses_api_agents.anyswe_agent.agent_runner import _runner_global_config
from responses_api_agents.anyswe_agent.app import (
    AnySweAgent,
    AnySweRunRequest,
    AnySweServerConfig,
    _model_url_for_rollout,
    _outer_response_id,
)
from responses_api_agents.anyswe_agent.tests.test_app import _config


def test_model_url_carries_the_training_capture_prefix() -> None:
    assert (
        _model_url_for_rollout("http://model-host:8000", "3-1", token_capture=True)
        == "http://model-host:8000/ng-rollout/3-1/training-token-capture"
    )
    assert _model_url_for_rollout("http://model-host:8000", "3-1") == "http://model-host:8000/ng-rollout/3-1"
    assert _model_url_for_rollout("", "3-1", token_capture=True) == ""
    assert _model_url_for_rollout("http://model-host:8000", None, token_capture=True) == "http://model-host:8000"


@pytest.mark.parametrize("rollout_id", [None, "7-2"])
def test_model_url_keeps_trace_context_with_and_without_capture(
    monkeypatch: pytest.MonkeyPatch, rollout_id: str | None
) -> None:
    trace_segment = "/ng-trace/00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"
    monkeypatch.setattr("nemo_gym.server_utils.trace_context_path_segment", lambda: trace_segment)
    capture_prefix = f"/ng-rollout/{rollout_id}/training-token-capture" if rollout_id else ""

    assert (
        _model_url_for_rollout("http://policy:8000", rollout_id, token_capture=True)
        == f"http://policy:8000{capture_prefix}{trace_segment}"
    )


def _agent(tmp_path: Path, *, capture: bool, custom_url: str | None = None) -> AnySweAgent:
    server = AnySweServerConfig(
        run_session_id="s",
        base_results_dir=tmp_path,
        model_server_url=custom_url or "http://policy:8000",
        resolved_sandbox_provider={"opensandbox": {}},
        sandbox_default_metadata={},
    )
    agent = AnySweAgent.__new__(AnySweAgent)
    object.__setattr__(
        agent,
        "__dict__",
        {
            "config": _config(token_id_capture=True, sandbox_model_base_url=custom_url),
            "server_client": ServerClient.model_construct(
                global_config_dict={"token_id_capture": {"enabled": capture}}
            ),
        },
    )
    object.__setattr__(agent, "__pydantic_fields_set__", set())
    object.__setattr__(agent, "__pydantic_extra__", None)
    object.__setattr__(
        agent,
        "__pydantic_private__",
        {"_server": server, "_sem": asyncio.Semaphore(1), "_checkpoint_participant": None, "_restart_only": None},
    )
    return agent


@pytest.mark.parametrize("capture", [False, True])
def test_setup_params_requests_the_capture_prefix_when_capture_is_on(tmp_path: Path, capture: bool) -> None:
    agent = _agent(tmp_path, capture=capture)
    body = NeMoGymResponseCreateParamsNonStreaming(input="fix it", metadata={"instance_id": "astropy__astropy-12907"})

    params = agent._setup_params(body, rollout_id="7-2")

    suffix = "/training-token-capture" if capture else ""
    assert params.model_server_url == f"http://policy:8000/ng-rollout/7-2{suffix}"


def test_inner_agent_learns_that_token_capture_is_on_from_the_model_url() -> None:
    assert _runner_global_config("", "policy_model") == {}
    eval_config = _runner_global_config("http://policy:8000/ng-rollout/7-2", "policy_model")
    assert "token_id_capture" not in eval_config
    assert eval_config["policy_model"]["responses_api_models"]["model"]["port"] == 0
    capture_config = _runner_global_config("http://policy:8000/ng-rollout/7-2/training-token-capture", "policy_model")
    assert capture_config["token_id_capture"] == {"enabled": True, "all_agents": True}


def test_outer_response_keeps_the_inner_served_id_only_under_capture() -> None:
    inner = NeMoGymResponse.model_construct(id="resp_served123")
    instance_id = "astropy__astropy-12907"
    assert _outer_response_id(instance_id, inner, token_capture=True) == "resp_served123"
    assert _outer_response_id(instance_id, inner, token_capture=False) == "anyswe-astropy__astropy-12907"
    assert _outer_response_id(instance_id, None, token_capture=True) == "anyswe-astropy__astropy-12907"


def _saved_response() -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp_served123",
        created_at=0,
        model="policy",
        object="response",
        output=[],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


@pytest.mark.parametrize(
    ("capture", "use_run", "with_indices", "custom_url", "expected_url", "expected_id"),
    [
        (False, True, True, None, "http://policy:8000", "anyswe-repo__repo-1"),
        (True, True, True, None, "http://policy:8000/ng-rollout/7-2/training-token-capture", "resp_served123"),
        (True, False, False, None, "http://policy:8000", "anyswe-repo__repo-1"),
        (True, True, False, None, "http://policy:8000", "anyswe-repo__repo-1"),
        (True, True, True, "http://sandbox:8000", "http://sandbox:8000", "anyswe-repo__repo-1"),
        (
            True,
            True,
            True,
            "http://sandbox/training-token-capture-other",
            "http://sandbox/training-token-capture-other",
            "anyswe-repo__repo-1",
        ),
    ],
)
def test_run_wires_model_url_and_response_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capture: bool,
    use_run: bool,
    with_indices: bool,
    custom_url: str | None,
    expected_url: str,
    expected_id: str,
) -> None:
    environments = []

    class Sandbox:
        def __init__(self, provider, spec) -> None:
            pass

        async def start(self) -> None:
            pass

        async def stop(self) -> None:
            pass

        async def download(self, remote: str, local: Path) -> None:
            assert remote == "/trajectories_mount/response.json"
            local.write_text(_saved_response().model_dump_json())

        async def exec(self, command: str, **kwargs):
            if kwargs.get("env"):
                environments.append(kwargs["env"])
            return SimpleNamespace(
                return_code=int(command == "test -f /trajectories_mount/patch.diff"),
                stdout="",
                stderr="",
                error_type=None,
            )

    monkeypatch.setattr("responses_api_agents.anyswe_agent.app.AsyncSandbox", Sandbox)
    agent = _agent(tmp_path, capture=capture, custom_url=custom_url)
    body = NeMoGymResponseCreateParamsNonStreaming(
        input="fix it", metadata={"instance_id": "repo__repo-1", "problem_statement": "fix it"}
    )
    if use_run:
        indices = {"_ng_task_index": 7, "_ng_rollout_index": 2} if with_indices else {}
        result = asyncio.run(agent.run(AnySweRunRequest(responses_create_params=body, **indices)))
        response = result.response
        assert result.mask_sample is False
    else:
        response = asyncio.run(agent.responses(body))

    assert [env["NGSWE_MODEL_URL"] for env in environments] == [expected_url]
    assert response.id == expected_id


@pytest.mark.parametrize(
    ("model_url", "expected_capture"),
    [
        ("http://policy:8000/ng-rollout/7-2", False),
        ("http://policy:8000/ng-rollout/7-2/training-token-capture", True),
        ("http://policy:8000/training-token-capture-other", False),
        ("http://training-token-capture.example:8000", False),
    ],
)
def test_runner_main_builds_an_agent_with_the_capture_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_url: str, expected_capture: bool
) -> None:
    observed_capture = []

    class Agent(SimpleResponsesAPIAgent):
        async def run(self, body):
            raise AssertionError("the runner should call responses")

        async def responses(self, request, body):
            observed_capture.append(self._token_id_capture_enabled())
            assert self.config.token_id_capture is False
            return _saved_response()

    module = SimpleNamespace(Agent=Agent, Config=BaseResponsesAPIAgentConfig)
    monkeypatch.setattr(agent_runner, "importlib", SimpleNamespace(import_module=lambda name: module))
    monkeypatch.setattr(agent_runner, "_find_repo", lambda: None)
    monkeypatch.setattr(agent_runner, "Path", lambda path: tmp_path / str(path).lstrip("/"))
    for key, value in {
        "NGSWE_MODEL_URL": model_url,
        "NGSWE_MODEL_NAME": "policy",
        "NGSWE_AGENT_MODULE": "test_agent",
        "NGSWE_AGENT_CLASS": "Agent",
        "NGSWE_AGENT_CONFIG_CLASS": "Config",
        "NGSWE_AGENT_KWARGS": "{}",
        "NGSWE_SAMPLING": "{}",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("NGSWE_AGENT_KWARGS_B64", raising=False)
    monkeypatch.delenv("NGSWE_SAMPLING_B64", raising=False)
    trajectory = tmp_path / "trajectories_mount"
    trajectory.mkdir()
    (trajectory / "instruction.txt").write_text("fix it")

    agent_runner.main()

    assert observed_capture == [expected_capture]
    assert NeMoGymResponse.model_validate_json((trajectory / "response.json").read_text()).id == "resp_served123"
