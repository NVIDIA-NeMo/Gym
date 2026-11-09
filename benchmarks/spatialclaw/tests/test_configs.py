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
        assert resource_name not in opencode  # Harness swap cannot redefine preparation or verification.
        assert opencode[environment_name]["environment_servers"]["single_agent_turn"]["resources_tool_transports"] == ["mcp"]
