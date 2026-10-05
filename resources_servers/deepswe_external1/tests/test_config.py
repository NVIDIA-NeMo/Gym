# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
import shutil
from pathlib import Path

from omegaconf import OmegaConf

from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.rollout_collection import RolloutCollectionHelper
from nemo_gym.train_data_utils import TrainDataProcessor


def test_readme_collation_produces_rows_routable_by_the_documented_runtime(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[3]
    readme = (repo / "resources_servers/deepswe_external1/README.md").read_text()

    def config_for(command: str):
        config_path = re.search(rf"{command}\s+\\\s+--config\s+(\S+)", readme).group(1)
        return GlobalConfigDictParser().parse(
            GlobalConfigDictParserConfig(
                initial_global_config_dict=OmegaConf.merge(
                    GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                    OmegaConf.load(repo / config_path),
                ),
                skip_load_from_cli=True,
                skip_load_from_dotenv=True,
                offline=True,
            )
        )

    configs = [
        c
        for c in GlobalConfigDictParser().filter_for_server_instance_configs(config_for("gym dataset collate"))
        if c.datasets
    ]
    for instance in configs:
        for dataset in instance.datasets:
            original = repo / dataset.jsonl_fpath
            target = tmp_path / original.name
            shutil.copyfile(original, target)
            dataset.jsonl_fpath = target
    paths = TrainDataProcessor()._collate_samples_single_type("example", configs, task_data_validation="error")
    rows = [json.loads(line) for path in paths for line in path.read_text().splitlines()]
    assert len(rows) == 5
    runtime = config_for("gym env start")
    RolloutCollectionHelper.resolve_task_sources(rows, runtime)
    RolloutCollectionHelper._validate_agent_names(rows, runtime)
    assert {row["agent_ref"]["name"] for row in rows} == {"deepswe_external1_opencode_sandboxed_agent"}


def test_opencode_config_has_environment_server() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs/deepswe_external1_opencode.yaml"
    resolved = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.merge(
                GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                OmegaConf.load(config_path),
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )

    environment = resolved["deepswe_external1_environment_server"]["environment_servers"]["legacy_agent"]
    assert environment["entrypoint"] == "app.py"
    assert environment["agent_server"] == {
        "type": "responses_api_agents",
        "name": "deepswe_external1_opencode_sandboxed_agent",
    }
    agent = resolved[environment["agent_server"]["name"]]["responses_api_agents"]["opencode_sandboxed_agent"]
    assert agent["resources_server"] == {
        "type": "resources_servers",
        "name": "deepswe_external1_opencode_resources_server",
    }
