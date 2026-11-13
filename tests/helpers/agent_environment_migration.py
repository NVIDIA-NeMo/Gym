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

"""Configuration helpers for independently scheduled environment migration tests."""

import importlib.util
from pathlib import Path
from types import ModuleType

from omegaconf import DictConfig, OmegaConf

from nemo_gym.config.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "add_legacy_agent_environment_servers.py"


def load_migration() -> ModuleType:
    """Give each test module its own script instance, as before the module split."""
    spec = importlib.util.spec_from_file_location("add_legacy_agent_environment_servers", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AGENT_CONFIG = """\
my_resources:
  resources_servers:
    mcqa:
      entrypoint: app.py
      domain: other
my_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: my_resources
"""


def _environment_servers(document: dict) -> dict[str, str]:
    """Map each agent to the environment servers that name it."""
    fronting: dict[str, list[str]] = {}
    for name, instance in document.items():
        for server in (instance.get("environment_servers") or {}).values() if isinstance(instance, dict) else ():
            fronting.setdefault(server["agent_server"]["name"], []).append(name)
    return fronting


MULTI_AGENT_CONFIG = (
    AGENT_CONFIG
    + """\
my_user_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: my_resources
my_conversation:
  environment_servers:
    conversation:
      entrypoint: app.py
      user_agent:
        type: responses_api_agents
        name: my_user_agent
      assistant_agent:
        type: responses_api_agents
        name: my_simple_agent
"""
)


def _server_fronting(agent: str, *, name: str, server_type: str) -> str:
    return f"""\
{name}:
  environment_servers:
    {server_type}:
      entrypoint: app.py
      agent_server:
        type: responses_api_agents
        name: {agent}
"""


def _parse(*configs: Path, strict: bool) -> DictConfig:
    return GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.merge(
                GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                *(OmegaConf.load(config) for config in configs),
                {"error_on_agent_without_environment_server": strict},
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )


RELAY = _server_fronting("my_simple_agent", name="my_environment_server", server_type="legacy_agent")


FOLLOWING = """\
# A comment that belongs to the next instance.
my_other_resources:
  resources_servers:
    mcqa:
      entrypoint: app.py
      domain: other
"""


def _switched_server(document: dict, name: str) -> dict:
    servers = document[name]["environment_servers"]
    assert list(servers) == ["single_agent_turn_legacy"]
    return servers["single_agent_turn_legacy"]


CHILD = """\
config_paths:
  - {base}
child_resources:
  _inherit_from: my_resources
child_agent:
  _inherit_from: my_simple_agent
  responses_api_agents:
    simple_agent:
      resources_server:
        name: child_resources
child_environment_server:
  _inherit_from: my_environment_server
  environment_servers:
    legacy_agent:
      agent_server:
        name: child_agent
"""


def _parent_and_child(tmp_path: Path) -> tuple[Path, Path]:
    base = tmp_path / "base.yaml"
    base.write_text(AGENT_CONFIG + RELAY)
    child = tmp_path / "child.yaml"
    child.write_text(CHILD.format(base=base))
    return base, child


HARNESS_OVERRIDE = """\
config_paths:
  - {base}
my_harness_agent:
  responses_api_agents:
    tool_simulation_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: my_resources
my_simple_agent: null
my_environment_server:
  environment_servers:
    legacy_agent:
      agent_server:
        name: my_harness_agent
"""
