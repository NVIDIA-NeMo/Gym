# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import yaml
from app import NSToolsConfig


def test_coolprop_mcp_config_loads() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "ns_tools_coolprop.yaml"
    data = yaml.safe_load(config_path.read_text())

    server = data["ns_tools_coolprop"]["resources_servers"]["ns_tools_coolprop"]
    assert server["entrypoint"] == "app.py"
    assert server["nemo_skills_tools"] == ["nemo_skills.mcp.servers.coolprop_tool.CoolPropTool"]

    config = NSToolsConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint=server["entrypoint"],
        name="ns_tools_coolprop",
        nemo_skills_tools=server["nemo_skills_tools"],
    )
    assert config.nemo_skills_tools == server["nemo_skills_tools"]
