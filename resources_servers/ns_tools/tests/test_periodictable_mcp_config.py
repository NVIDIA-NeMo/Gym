# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import yaml
from app import NSToolsConfig


def test_periodictable_mcp_config_loads() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "ns_tools_periodictable.yaml"
    data = yaml.safe_load(config_path.read_text())

    server = data["ns_tools_periodictable"]["resources_servers"]["ns_tools_periodictable"]
    assert server["entrypoint"] == "app.py"
    assert server["nemo_skills_tools"] == ["nemo_skills.mcp.servers.periodictable_tool.PeriodictableTool"]

    config = NSToolsConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint=server["entrypoint"],
        name="ns_tools_periodictable",
        nemo_skills_tools=server["nemo_skills_tools"],
    )
    assert config.nemo_skills_tools == server["nemo_skills_tools"]
