# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The component configs and example data, validated as Gym validates them."""

import json
from unittest.mock import MagicMock

import pytest
import yaml
from omegaconf import OmegaConf

from nemo_gym import PARENT_DIR
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig, get_first_server_config_dict
from nemo_gym.server_utils import ServerClient


def test_frozen_history_cannot_dispatch_live_tools():
    from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServerConfig

    config = yaml.safe_load((PARENT_DIR / "resources_servers/partner_if/configs/partner_if.yaml").read_text())
    agent = config["partner_if_simple_agent"]["responses_api_agents"]["simple_agent"]
    assert agent["execute_tools"] is False
    env = config["partner_if_environment_server"]["environment_servers"]["single_agent_turn_legacy"]
    assert env["resources_tool_transports"] == []
    assert env["resources_server"]["name"] == "partner_if"
    assert env["agent_server"]["name"] == "partner_if_simple_agent"
    parsed = SingleAgentTurnEnvironmentServerConfig.model_validate(
        {**env, "name": "partner_if_environment_server", "host": "127.0.0.1", "port": 8080}
    )
    assert parsed.cleanup_timeout_seconds == 180


# The top-level fields that native rows add, with synthetic values: the Harbor task and, on SWE rows, a Scale-SWE
# instance.
HARBOR_FIELDS = {"task_name": "synthetic-tools-1"}
SWE_FIELDS = HARBOR_FIELDS | {
    "instance_id": "synthetic__notes-1",
    "repo": "synthetic/notes",
    "language": "python",
    "workdir": "/testbed",
    "image_url": "registry.invalid/synthetic-notes:1",
    "patch": "diff --git a/notes.py b/notes.py\n+FIXED = True\n",
    "pre_commands": "pip install -e .",
    "problem_statement": "Log the greeting note.",
    "f2p_patch": "diff --git a/tests/test_notes.py b/tests/test_notes.py\n+def test_log(): pass\n",
    "f2p_script": "pytest tests/test_notes.py",
    "FAIL_TO_PASS": ["tests/test_notes.py::test_log"],
    "PASS_TO_PASS": "[]",
}


@pytest.mark.parametrize(
    ("dataset", "native_fields"),
    [
        ("example.jsonl", {}),
        ("mock_tools_example.jsonl", {}),
        # The live-agent examples as SWE and Harbor rows.
        ("mock_tools_example.jsonl", SWE_FIELDS),
        ("mock_tools_example.jsonl", HARBOR_FIELDS),
    ],
    ids=["text", "mock_tools", "swe", "harbor"],
)
def test_gym_task_data_tooling_validates_the_same_contract(dataset, native_fields):
    from nemo_gym.task_data import TaskDataValidator, load_task_data_schema

    root = PARENT_DIR / "resources_servers/partner_if"
    adapter = load_task_data_schema(root)
    validator = TaskDataValidator("partner_if", adapter, dataset)
    for i, line in enumerate((root / "data" / dataset).read_text().splitlines()):
        validator.validate_row(i, json.loads(line) | native_fields)
    assert validator.report.clean, validator.report.summary()


def test_the_swe_row_carries_every_task_field_of_its_wire():
    # So the swe case above fails when the SWE wire gains a field that the schema does not declare.
    from nemo_gym.base_resources_server import BaseVerifyRequest
    from resources_servers.partner_if.swe_app import SWEIFVerifyRequest

    wire = SWEIFVerifyRequest.model_fields.keys() - BaseVerifyRequest.model_fields.keys() - {"verifier_metadata"}
    assert wire == SWE_FIELDS.keys()


def server_config(config_name, server_name):
    """The block a server validates at startup, resolved offline by Gym's own parser."""
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.merge(
                GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                {"config_paths": [f"resources_servers/partner_if/configs/{config_name}"]},
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    return OmegaConf.to_container(get_first_server_config_dict(config, server_name), resolve=True) | {
        "name": server_name
    }


def test_the_mock_tool_example_starts_with_its_backend_and_rows():
    from resources_servers.partner_if.mock_tools_app import MockToolsResourcesServer, MockToolsServerConfig

    config = MockToolsServerConfig.model_validate(server_config("mock_tools.yaml", "partner_if_mock_tools"))
    resource = MockToolsResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    assert resource.mcp_allowed_tools_for_session(
        {"verifier_metadata": {"partner_if": {"task_id": "synthetic-tools-1"}}}
    ) == ["append", "lookup"]
    assert config.judge_responses_create_params.max_output_tokens == 8192


def test_the_swe_config_validates_and_grades_the_prompt_its_agent_runs_with():
    from resources_servers.partner_if.swe_app import SWEIFConfig
    from responses_api_agents.opencode_sandboxed_agent.app import OpenCodeSandboxedAgentConfig

    scorer = SWEIFConfig.model_validate(server_config("swe.yaml", "partner_if_swe"))
    agent = OpenCodeSandboxedAgentConfig.model_validate(server_config("swe.yaml", "partner_if_swe_agent"))
    assert agent.resources_server.name == "partner_if_swe"
    assert agent.opencode_config["agent"]["build"]["prompt"] == scorer.actor_system_prompt
