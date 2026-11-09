# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).parents[3]
SUITE_CONFIG = REPO_ROOT / "benchmarks/spatialclaw/config.yaml"


def test_all_twenty_presets_use_typed_resources_owned_tasks_and_agent_sessions():
    presets = yaml.safe_load(SUITE_CONFIG.read_text())["config_paths"]
    assert len(presets) == 20
    for path in presets:
        preset = yaml.safe_load((REPO_ROOT / path).read_text())
        resource_name = next(k for k in preset if k.endswith("_benchmark_resources_server"))
        agent_name = next(k for k in preset if k.endswith("_benchmark_agent"))
        environment_name = next(k for k in preset if k.endswith("_benchmark_environment_server"))
        resources = preset[resource_name]["resources_servers"]["spatialclaw"]
        agent = preset[agent_name]["responses_api_agents"]["spatialclaw_agent"]
        environment = preset[environment_name]["environment_servers"]
        assert set(environment) == {"single_agent_turn"}
        assert environment["single_agent_turn"]["resources_server"]["name"] == resource_name
        assert environment["single_agent_turn"]["agent_server"]["name"] == agent_name
        assert agent["resources_server"] is None and "datasets" not in agent
        [dataset] = resources["datasets"]
        assert dataset["num_repeats"] == 1
        assert (REPO_ROOT / dataset["prepare_script"]).is_file()
        assert preset["environment_server_routes"] == {dataset["taskset"]: environment_name}
        opencode = yaml.safe_load((REPO_ROOT / "benchmarks/spatialclaw/opencode" / Path(path).name).read_text())
        execution = opencode[resource_name]["resources_servers"]["spatialclaw"]
        assert set(execution) <= {"sandbox_provider", "sandbox_config", "sandbox_network_access", "sandbox_workdir"}
        assert execution["sandbox_provider"] == "spatialclaw_harness_sandbox"
        assert opencode[environment_name]["environment_servers"]["single_agent_turn"]["resources_tool_transports"] == [
            "mcp"
        ]


def test_both_harnesses_resolve_the_same_resource_contract_for_every_benchmark(monkeypatch):
    from omegaconf import OmegaConf

    from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig

    monkeypatch.setenv("SPATIALCLAW_DATA_ROOT", "/data")
    monkeypatch.setenv("SPATIALCLAW_OPENCODE_BIN", "/runtime/opencode")
    presets = yaml.safe_load(SUITE_CONFIG.read_text())["config_paths"]
    for preset_path in presets:
        resolved = []
        for path in [preset_path, str(Path("benchmarks/spatialclaw/opencode") / Path(preset_path).name)]:
            initial = OmegaConf.create(
                {
                    "config_paths": [path],
                    "policy_model_name": "test",
                    "policy_base_url": "http://127.0.0.1:9999/v1",
                    "policy_api_key": "local",
                    "policy_model": {"responses_api_models": {"dummy_model": {"entrypoint": "app.py"}}},
                }
            )
            cfg = GlobalConfigDictParser().parse(
                GlobalConfigDictParserConfig(
                    initial_global_config_dict=initial,
                    skip_load_from_cli=True,
                    skip_load_from_dotenv=True,
                    offline=True,
                )
            )
            resource_name = next(k for k in cfg if k.endswith("_benchmark_resources_server"))
            agent_name = next(k for k in cfg if k.endswith("_benchmark_agent"))
            environment_name = next(k for k in cfg if k.endswith("_benchmark_environment_server"))
            resources = OmegaConf.to_container(cfg[resource_name].resources_servers.spatialclaw, resolve=True)
            resources.pop("host", None)
            resources.pop("port", None)
            # Execution placement may differ; dataset preparation/scoring/media must not.
            for key in [
                "sandbox_provider",
                "sandbox_config",
                "sandbox_image",
                "sandbox_network_access",
                "sandbox_workdir",
            ]:
                resources.pop(key, None)
            resolved.append(resources)
            assert list(cfg[environment_name].environment_servers) == ["single_agent_turn"]
            agent_type = next(iter(cfg[agent_name].responses_api_agents))
            assert agent_type == ("opencode_agent" if "/opencode/" in path else "spatialclaw_agent")
            assert cfg[agent_name].responses_api_agents[agent_type].num_workers == 1
        assert resolved[0] == resolved[1]
