# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Built-in tool contracts describe the pinned binary, not hand-maintained guesses."""

from pathlib import Path

import yaml

from resources_servers.agentic_if.task_variants.catalog import (
    BUILTIN_ARGUMENTS,
    BUILTIN_SCHEMA_SNAPSHOT,
    BUILTIN_TOOLS,
)


# bin/opencode of the npm package opencode-linux-x64@1.17.11, the binary the benchmark runs.
PINNED = ("1.17.11", "0254a429cd0e6cf0ba53fc01672cf98e4a8dc728f7fa94be88f1e4b3645e6ded")  # pragma: allowlist secret
CONFIG = Path(__file__).parents[2] / "configs" / "agentic_if.yaml"


def test_all_pinned_opencode_tools_have_complete_schemas_and_derived_arguments():
    snapshot = BUILTIN_SCHEMA_SNAPSHOT
    assert (snapshot["version"], snapshot["binary_sha256"]) == PINNED
    schemas = snapshot["schemas"]
    assert set(schemas) == BUILTIN_TOOLS == set(BUILTIN_ARGUMENTS)
    assert all(schema["type"] == "object" for schema in schemas.values())
    assert schemas["apply_patch"]["properties"]["patchText"]["type"] == "string"
    assert "background" not in schemas["task"]["properties"]
    assert schemas["read"]["properties"]["offset"]["minimum"] == 0


def test_the_example_stack_runs_the_opencode_version_the_snapshot_describes():
    from responses_api_agents.opencode_agent.runtime import OPENCODE_VERSION

    agent = yaml.safe_load(CONFIG.read_text())["agentic_if_opencode_agent"]["responses_api_agents"]["opencode_agent"]
    # Unset, the agent runs its own pinned default.
    assert str(agent.get("opencode_version", OPENCODE_VERSION)) == PINNED[0]
    if agent.get("sandbox_executable_sha256") is not None:
        assert agent["sandbox_executable_sha256"] == PINNED[1]
