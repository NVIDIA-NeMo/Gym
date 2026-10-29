# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""All historical SWE sources admit materialized rows for each shared harness."""

import importlib
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig, get_first_server_config_dict
from nemo_gym.rollout_collection import NG_ENVIRONMENT_SERVER_KEY, RolloutCollectionHelper
from nemo_gym.single_agent_turn_types import SingleAgentTurnTaskInput
from nemo_gym.task_materialization import materialize_task


SOURCES = ("scale_swe", "swe_rebench", "swemer_v1", "swemer_v2", "swe_next")
HARNESSES = ("opencode", "claude_code", "pi", "hermes", "nooa")
ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("harness", HARNESSES)
def test_all_swe_sources_admit_materialized_rows_without_changing_task_content(harness: str) -> None:
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.merge(
                GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                {
                    "config_paths": [str(ROOT / f"environment_servers/single_agent_turn/configs/swe_{harness}.yaml")],
                    "sandbox": {"docker": {}},
                },
            ),
            default_config_paths=[],
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    assert set(config.environment_server_routes) == set(SOURCES)
    rows = []
    for source in SOURCES:
        original = json.loads((ROOT / f"resources_servers/{source}/data/example.jsonl").read_text().splitlines()[0])
        # Historical training rows have no agent routing; strip only example-file routing metadata.
        original = {
            key: value
            for key, value in original.items()
            if key not in {"task_source", "agent_ref", "skills_ref"} and not key.startswith("_ng_")
        }
        original["source"] = source
        row = materialize_task(original, taskset=source)
        assert row["task_id"] == {"taskset": source, "task_id": original["instance_id"]}
        assert row["task_input"] == original
        schema = importlib.import_module(f"resources_servers.{source}.task_data").TaskData
        typed_input = SingleAgentTurnTaskInput.model_validate(row["task_input"])
        schema.model_validate(typed_input.task_data)
        assert typed_input.task_data["source"] == source
        environment_name = config.environment_server_routes[source]
        row[NG_ENVIRONMENT_SERVER_KEY] = environment_name
        rows.append(row)
        environment = get_first_server_config_dict(config, environment_name)
        assert environment.agent_server.name == f"{harness}_agent"
        assert environment.resources_server.name == f"{source}_resources_server"
        resources = get_first_server_config_dict(config, environment.resources_server.name)
        expected_agents = (
            ["opencode_agent", "opencode_sandboxed_agent"] if harness == "opencode" else [f"{harness}_agent"]
        )
        assert resources.allowed_agents == expected_agents
        if source in ("swemer_v2", "swe_next"):
            assert resources.num_workers == 1
    RolloutCollectionHelper._validate_environment_servers(rows, config)


@pytest.mark.parametrize("source", ["swemer_v2", "swe_next"])
def test_resource_base_allowlist_keeps_flat_agent_and_all_shared_harnesses(source: str) -> None:
    config = OmegaConf.load(ROOT / f"resources_servers/{source}/configs/{source}.yaml")
    allowed = config[f"{source}_resources_server"].resources_servers[source].allowed_agents
    assert set(allowed) == {"opencode_sandboxed_agent", *(f"{harness}_agent" for harness in HARNESSES)}
