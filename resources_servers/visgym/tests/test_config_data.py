# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf

from resources_servers.visgym.schemas import VisGymResourcesServerConfig, VisGymTaskRow


GYM_ROOT = Path(__file__).resolve().parents[3]
VISGYM_ROOT = GYM_ROOT / "resources_servers" / "visgym"


def test_visgym_yaml_points_at_existing_valid_jsonls() -> None:
    cfg = OmegaConf.to_container(
        OmegaConf.load(VISGYM_ROOT / "configs" / "visgym_agent.yaml"),
        resolve=True,
    )

    server_cfg = cfg["visgym_resources_server"]["resources_servers"]["visgym"]
    VisGymResourcesServerConfig(
        **server_cfg,
        name="visgym_resources_server",
        host="0.0.0.0",
        port=8080,
    )

    agent_block = cfg["visgym_agent"]["responses_api_agents"]["visgym_agent"]
    for dataset in agent_block["datasets"]:
        jsonl_path = VISGYM_ROOT / dataset["jsonl_fpath"]
        assert jsonl_path.is_file()
        with jsonl_path.open() as f:
            for line in f:
                if line.strip():
                    VisGymTaskRow.model_validate(json.loads(line))
