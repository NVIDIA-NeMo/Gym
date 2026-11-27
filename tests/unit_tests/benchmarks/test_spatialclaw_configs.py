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

    from nemo_gym.config.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig

    monkeypatch.setenv("SPATIALCLAW_DATA_ROOT", "/data")
    monkeypatch.setenv("SPATIALCLAW_ROOT", "/authorized/spatialclaw")
    monkeypatch.setenv("SPATIALCLAW_COMMIT", "a" * 40)
    monkeypatch.delenv("SPATIALCLAW_SOURCE_URL", raising=False)
    monkeypatch.delenv("SPATIALCLAW_OPENCODE_BIN", raising=False)
    presets = yaml.safe_load(SUITE_CONFIG.read_text())["config_paths"]
    for preset_path in presets:
        resolved = []
        for path in [preset_path, str(Path("benchmarks/spatialclaw/opencode") / Path(preset_path).name)]:
            initial = OmegaConf.create(
                {
                    "config_paths": [
                        "responses_api_agents/opencode_agent/configs/opencode_agent.yaml",
                        path,
                    ],
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
            assert resources["spatialclaw_root"] == "/authorized/spatialclaw"
            assert resources["spatialclaw_commit"] == "a" * 40
            assert resources["spatialclaw_url"] == ""
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
            agent = cfg[agent_name].responses_api_agents[agent_type]
            assert agent.num_workers == 1
            assert resources["allowed_agents"] == ["spatialclaw_agent", "opencode_agent"]
            if agent_type == "opencode_agent":
                assert agent.image_input is True
                assert str(agent.opencode_version) == "1.17.11"
                assert cfg.opencode_agent.responses_api_agents.opencode_agent.context_window != agent.context_window
        assert resolved[0] == resolved[1]


def test_committed_examples_use_typed_media_contract():
    import json

    path = REPO_ROOT / "resources_servers/spatialclaw/data/example.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 5
    for row in rows:
        task = row["task_input"]
        assert row["task_id"]["task_id"] == task["task_data"]["sample_id"]
        assert row["task_id"]["taskset"] == "spatialclaw:" + task["task_data"]["dataset_config"]
        assert task["task_data"]["media"]["video_path"]
        assert "metadata" not in task["responses_create_params"]
