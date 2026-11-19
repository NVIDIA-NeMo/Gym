# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from omegaconf import OmegaConf
from pytest import mark

from nemo_gym.config.config_types import maybe_get_server_instance_config
from nemo_gym.server_runtime.supervisor.server_supervisor import _configured_servers


@mark.parametrize("entrypoint", [{}, {"entrypoint": None}])
def test_external_agent_is_resolved_without_local_launch(entrypoint: dict) -> None:
    agent = OmegaConf.create({"responses_api_agents": {"worker": {"host": "127.0.0.1", "port": 61370, **entrypoint}}})

    config, error = maybe_get_server_instance_config("worker", agent)

    assert error is None
    assert config.get_server_ref().name == "worker"
    assert config.get_inner_run_server_config().port == 61370
    assert config.get_inner_run_server_config().entrypoint is None
    assert not list(_configured_servers(OmegaConf.create({"worker": agent})))


@mark.parametrize("endpoint", [{"host": "localhost"}, {"port": 61370}, {"host": "localhost", "port": 0}])
def test_external_agent_requires_explicit_valid_endpoint(endpoint: dict) -> None:
    agent = OmegaConf.create({"responses_api_agents": {"worker": endpoint}})

    config, error = maybe_get_server_instance_config("worker", agent)

    assert config is None
    assert "An external server requires an explicit host and port" in str(error)
