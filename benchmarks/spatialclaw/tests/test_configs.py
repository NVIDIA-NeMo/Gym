# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).parents[3]
SUITE_CONFIG = REPO_ROOT / "benchmarks" / "spatialclaw" / "config.yaml"


def test_all_twenty_benchmarks_use_the_unified_spatialclaw_path() -> None:
    suite = yaml.safe_load(SUITE_CONFIG.read_text(encoding="utf-8"))
    preset_paths = suite["config_paths"]
    assert len(preset_paths) == 20

    for preset_path in preset_paths:
        preset = yaml.safe_load((REPO_ROOT / preset_path).read_text(encoding="utf-8"))
        assert preset["config_paths"] == [
            "resources_servers/spatialclaw/configs/spatialclaw_resources.yaml",
            "responses_api_agents/spatialclaw_agent/configs/spatialclaw_agent.yaml",
        ]

        resources = {name: value for name, value in preset.items() if name.endswith("_benchmark_resources_server")}
        agents = {name: value for name, value in preset.items() if name.endswith("_benchmark_agent")}
        assert len(resources) == len(agents) == 1

        resource_name, resource = next(iter(resources.items()))
        assert resource["_inherit_from"] == "spatialclaw_resources_server"
        assert set(resource["resources_servers"]) == {"spatialclaw"}

        agent = next(iter(agents.values()))
        spatialclaw_agent = agent["responses_api_agents"]["spatialclaw_agent"]
        assert spatialclaw_agent["resources_server"]["name"] == resource_name
        assert spatialclaw_agent["dataset_config"] == resource["resources_servers"]["spatialclaw"]["dataset_config"]

        datasets = spatialclaw_agent["datasets"]
        assert len(datasets) == 1
        prepare_script = REPO_ROOT / datasets[0]["prepare_script"]
        assert prepare_script.parent == REPO_ROOT / "benchmarks" / "spatialclaw"
        assert prepare_script.is_file()
