# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Environment migration: agent environment repository configs."""

from pathlib import Path

import pytest

from tests.helpers.agent_environment_migration import (
    load_migration,
)


migration = load_migration()


def test_repository_configs_declare_every_environment_server(capsys: pytest.CaptureFixture[str]) -> None:
    """Every agent in the repository's configs has an environment server, so no run needs a generated relay."""
    assert migration.main(["--check"]) == 0, capsys.readouterr().out


def test_repository_non_executing_simple_agents_run_in_sessions_without_tool_access() -> None:
    # simple_agent refuses /run when execute_tools is false, so no relay may front it. A tool grant would let an
    # agent swapped in by composition, which does not keep execute_tools, call tools against a single-call verifier.
    loaded: dict[Path, dict | None] = {}
    fronting: dict[tuple[str, str], tuple[str, list | None]] = {}
    for path in migration.repo_config_files():
        instances = migration.resolved_instances([document for _, document in migration.run_documents(path, loaded)])
        for name, instance in instances.items():
            for server_type, server in (instance.get("environment_servers") or {}).items():
                agent = ((server or {}).get("agent_server") or {}).get("name")
                body = ((instances.get(agent) or {}).get("responses_api_agents") or {}).get("simple_agent") or {}
                if body.get("execute_tools") is False:
                    fronting[(name, agent)] = (server_type, server.get("resources_tool_transports"))

    assert ("swe_pivot_environment_server", "swe_pivot_tool_simulation_agent") in fronting
    assert {key: value for key, value in fronting.items() if value[0] == "legacy_agent" or value[1]} == {}


def test_repository_step_environments_declare_their_resources() -> None:
    loaded: dict[Path, dict | None] = {}
    missing = []
    for path in migration.repo_config_files():
        instances = migration.resolved_instances([document for _, document in migration.run_documents(path, loaded)])
        for name, instance in instances.items():
            for server_type, server in (instance.get("environment_servers") or {}).items():
                if server_type in {"single_agent_turn", "single_agent_turn_legacy"}:
                    if not (server.get("resources_server") or {}).get("name"):
                        missing.append((migration.display(path), name))

    assert missing == []
