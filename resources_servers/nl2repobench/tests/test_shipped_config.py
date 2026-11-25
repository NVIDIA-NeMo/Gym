# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import yaml


OPENCODE_CONFIG = Path(__file__).resolve().parents[3] / "benchmarks/nl2repobench/opencode_sandboxed_agent.yaml"
OPENHANDS_CONFIG = Path(__file__).resolve().parents[3] / "benchmarks/nl2repobench/openhands.yaml"


def test_the_shipped_preset_references_no_private_runtime() -> None:
    text = OPENCODE_CONFIG.read_text()
    agent = yaml.safe_load(text)["nl2repobench_opencode_sandboxed_agent"]["responses_api_agents"][
        "opencode_sandboxed_agent"
    ]

    assert "/mnt/s3-data" not in text
    # with the fields unset the agent downloads the installer, which is what open-egress modes need
    assert not agent.get("remote_opencode_install_script_path")
    assert not agent.get("remote_opencode_binary_path")
    assert not agent.get("preinstalled_opencode")
    assert not agent.get("require_opencode_runtime")


def test_the_openhands_preset_gives_no_clean_score_to_the_task_whose_package_its_runtime_installs() -> None:
    """`pyperclip` is a dependency of the OpenHands runtime and a task of the set: its implementation is readable."""
    from resources_servers.nl2repobench import app as app_module

    server = yaml.safe_load(OPENHANDS_CONFIG.read_text())["nl2repobench_opencode_resources_server"][
        "resources_servers"
    ]["nl2repobench"]
    default = app_module.NL2RepoBenchResourcesServerConfig.model_fields["clean_exclude_tasks"].default_factory()

    assert server["clean_exclude_tasks"] == [*default, "pyperclip"]
    # and the OpenCode preset keeps the server's default
    opencode = yaml.safe_load(OPENCODE_CONFIG.read_text())["nl2repobench_opencode_resources_server"][
        "resources_servers"
    ]
    assert "clean_exclude_tasks" not in opencode["nl2repobench"]


def test_the_openhands_preset_runs_openhands_agent_and_the_server_allows_it() -> None:
    config = yaml.safe_load(OPENHANDS_CONFIG.read_text())
    assert "openhands_agent" in config["nl2repobench_openhands_agent"]["responses_api_agents"]
    assert (
        config["nl2repobench_environment_server"]["environment_servers"]["single_agent_turn_legacy"]["agent_server"][
            "name"
        ]
        == "nl2repobench_openhands_agent"
    )
    shared = yaml.safe_load(
        (OPENHANDS_CONFIG.parents[2] / "resources_servers/nl2repobench/configs/nl2repobench.yaml").read_text()
    )
    assert (
        "openhands_agent"
        in shared["nl2repobench_resources_server"]["resources_servers"]["nl2repobench"]["allowed_agents"]
    )
