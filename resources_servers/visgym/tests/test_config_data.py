# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from resources_servers.visgym.schemas import VisGymResourcesServerConfig, VisGymTaskRow


GYM_ROOT = Path(__file__).resolve().parents[3]
VISGYM_ROOT = GYM_ROOT / "resources_servers" / "visgym"
ENVIRONMENT_CONFIG = GYM_ROOT / "environments" / "visgym" / "config.yaml"


def _config_paths() -> list[Path]:
    return sorted((VISGYM_ROOT / "configs").glob("*.yaml"))


@pytest.mark.parametrize("config_path", _config_paths(), ids=lambda p: p.stem)
def test_visgym_yaml_points_at_existing_valid_jsonls(config_path: Path) -> None:
    """Every shipped config must name datasets that exist and validate."""
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)

    server_cfg = cfg["visgym_resources_server"]["resources_servers"]["visgym"]
    VisGymResourcesServerConfig(
        **server_cfg,
        name="visgym_resources_server",
        host="0.0.0.0",
        port=8080,
    )

    agent_block = cfg["visgym_agent"]["responses_api_agents"]["visgym_agent"]
    for dataset in agent_block.get("datasets") or []:
        jsonl_path = VISGYM_ROOT / dataset["jsonl_fpath"]
        assert jsonl_path.is_file()
        with jsonl_path.open() as f:
            for line in f:
                if line.strip():
                    VisGymTaskRow.model_validate(json.loads(line))


def test_environment_bundle_uses_committed_example_data() -> None:
    cfg = OmegaConf.to_container(OmegaConf.load(ENVIRONMENT_CONFIG), resolve=True)

    server_cfg = cfg["visgym_resources_server"]["resources_servers"]["visgym"]
    VisGymResourcesServerConfig(
        **server_cfg,
        name="visgym_resources_server",
        host="0.0.0.0",
        port=8080,
    )

    agent = cfg["visgym_agent"]["responses_api_agents"]["visgym_agent"]
    dataset_path = GYM_ROOT / agent["datasets"][0]["jsonl_fpath"]
    assert dataset_path == VISGYM_ROOT / "data" / "maze_2d_easy_example.jsonl"
    with dataset_path.open() as handle:
        rows = [VisGymTaskRow.model_validate(json.loads(line)) for line in handle if line.strip()]
    assert rows
