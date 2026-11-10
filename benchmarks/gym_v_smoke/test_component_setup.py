# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise current Gym startup against the actual component manifests."""

import shlex
from pathlib import Path

import pytest

from nemo_gym.server_runtime.supervisor.server_venv import setup_env_command


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("component", ["resources_servers/gym_v", "responses_api_agents/gymv_agent"])
def test_fresh_component_uses_locked_installer(tmp_path, component):
    config = {
        "uv_venv_dir": str(tmp_path),
        "skip_venv_if_present": False,
        "python_version": "3.13.14",
        "head_server_deps": [],
        "head_server_constraints": [],
        "component_extras": {},
    }
    command = setup_env_command(ROOT / component, config, component)
    args = shlex.split(command)
    installer = shlex.split(args[args.index("--command") + 1].split(" > >(")[0])
    assert installer[1].endswith("/_locked_env.py")
    assert installer[installer.index("--project") + 1] == str(ROOT / component)
    assert not (ROOT / component / "requirements.txt").exists()


def test_benchmark_has_one_dispatch_relay(monkeypatch):
    from omegaconf import OmegaConf

    from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig
    from nemo_gym.rollouts.rollout_collection import _environment_server_for_agent, _environment_servers_by_agent

    monkeypatch.chdir(ROOT)
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": [
                        "benchmarks/gym_v_smoke/config.yaml",
                        "responses_api_models/vllm_model/configs/vllm_model.yaml",
                    ],
                    "policy_base_url": "http://127.0.0.1:8000/v1",
                    "policy_api_key": "EMPTY",
                    "policy_model_name": "super35-step120",
                }
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
            skip_exporters=True,
        )
    )
    routes = _environment_servers_by_agent(config)
    assert _environment_server_for_agent("gym_v_smoke_agent", routes) == "gym_v_environment_server"
    assert routes["gym_v_smoke_agent"] == ["gym_v_environment_server"]
