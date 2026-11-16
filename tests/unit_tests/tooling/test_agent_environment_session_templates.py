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

"""Environment migration: agent environment session templates."""

from pathlib import Path

import pytest
import yaml

from tests.helpers.agent_environment_migration import (
    _environment_servers,
    load_migration,
)


migration = load_migration()


@pytest.mark.parametrize("resources", [None, {"type": "resources_servers", "name": "my_resources"}])
@pytest.mark.parametrize(
    "agent_type",
    [
        "langgraph_agent",
        "hermes_agent",
        "pi_agent",
        "pool_agent",
        "simple_strands_agent",
        "goose_agent",
        "prime_agent",
        "kimi_code_agent",
        "codex_agent",
        "openclaw_agent",
        "opencode_agent",
        "kilocode_agent",
        "claude_code_agent",
        "cline_agent",
        "osworld_agent",
        "harbor_oracle_agent",
        "terminus_2_sandboxed_agent",
        "mini_swe_agent",
        "miniswe_sandboxed_agent",
    ],
)
def test_migration_respects_agent_session_templates(
    tmp_path: Path, resources: dict[str, str] | None, agent_type: str
) -> None:
    config = tmp_path / "agent.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "my_agent": {
                    "responses_api_agents": {agent_type: {"entrypoint": "app.py", "resources_server": resources}}
                }
            }
        )
    )
    before = config.read_text()
    assert migration.main([str(config)]) == 0
    if agent_type != "osworld_agent" and resources is None:
        assert config.read_text() == before
    else:
        assert _environment_servers(yaml.safe_load(config.read_text())) == {"my_agent": ["my_environment_server"]}
