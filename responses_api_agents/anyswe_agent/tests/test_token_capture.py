# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Training-token capture wiring for anyswe_agent and its in-sandbox runner."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.anyswe_agent.agent_runner import _runner_global_config
from responses_api_agents.anyswe_agent.app import (
    AnySweAgent,
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


@pytest.mark.parametrize("capture", [False, True])
def test_setup_params_requests_the_capture_prefix_when_capture_is_on(tmp_path: Path, capture: bool) -> None:
    server = AnySweServerConfig(
        run_session_id="s",
        base_results_dir=tmp_path,
        model_server_url="http://policy:8000",
        resolved_sandbox_provider={"opensandbox": {}},
        sandbox_default_metadata={},
    )
    agent = AnySweAgent.__new__(AnySweAgent)
    object.__setattr__(
        agent,
        "__dict__",
        {
            "config": _config(token_id_capture=True),
            "server_client": SimpleNamespace(global_config_dict={"token_id_capture": {"enabled": capture}}),
        },
    )
    object.__setattr__(agent, "__pydantic_fields_set__", set())
    object.__setattr__(agent, "__pydantic_extra__", None)
    object.__setattr__(agent, "__pydantic_private__", {"_server": server, "_sem": None})
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
