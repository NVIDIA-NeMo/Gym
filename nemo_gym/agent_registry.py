# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Registry of agent harnesses under ``responses_api_agents/<name>/``.

An agent harness is one *component* of an environment (an environment = dataset + agent harness +
resources server [verifier and state] + model). This module maps an agent's short ``<name>`` (the
directory name) to its config variant(s) so it can be enumerated by name (``gym list agents``) and
classified by how it composes. Resolving an agent name to a config for *running* belongs to the
config composer (via the CLI's generic asset selectors), so this module is intentionally
discovery-only. The ``self_contained`` flag records only what the agent's *own* config reveals about
its resources-server wiring:

- **Wires a resources server itself (Pattern A, ``self_contained=False``):** the config sets
  ``responses_api_agents.<type>.resources_server``, so the registry can see it pairs with a separate
  resources server + dataset (e.g. the ``simple_agent`` tool-use pattern).
- **Does not wire one in its own config (Pattern B, ``self_contained=True``):** the agent declares
  its own ``agent_framework`` (e.g. ``swe_agents``), drives an external LLM loop (e.g.
  ``claude_code_agent`` via its own model key), or ships only an entrypoint whose resources-server
  pairing is supplied by a separate paired/benchmark config (e.g. ``gymnasium_agent`` + the
  ``blackjack`` resources server). These agents are still composable — but *within* a type
  (gymnasium↔gymnasium-style, simple_agent↔simple_agent-style), not across types. Which pairings are
  actually compatible is the config composer's call, not inferable from the agent config alone.

Discovery only reads config files; it never resolves interpolations or missing values and never
starts servers, so it is safe to call when secrets/API keys referenced by a config are unset.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from omegaconf import OmegaConf

from nemo_gym import PARENT_DIR
from nemo_gym.discovery import discover_components


AGENTS_SUBDIR = "responses_api_agents"
AGENTS_DIR = PARENT_DIR / AGENTS_SUBDIR
AGENT_CONFIGS_SUBDIR = "configs"


@dataclass(frozen=True)
class AgentEntry:
    """A discovered agent: its name, where it lives, its config variants, and how it's wired."""

    name: str
    path: Path
    config_paths: Tuple[Path, ...]  # variant config files, sorted; empty for "zero-config" agents
    # True  = bundles its own environment/framework (Pattern B; run standalone).
    # False = references a separate resources server (Pattern A; wire to a compatible environment).
    self_contained: bool
    description: Optional[str] = None

    @property
    def variants(self) -> Dict[str, Path]:
        """Map variant name (config filename stem) -> config path."""
        return {path.stem: path for path in self.config_paths}


def _iter_agent_blocks(config_path: Path):
    """Yield each ``responses_api_agents.<type>`` mapping in a config (resolution-safe, best effort)."""
    try:
        container = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False, throw_on_missing=False)
    except Exception:
        return
    if not isinstance(container, dict):
        return
    for top_level_value in container.values():
        if not isinstance(top_level_value, dict):
            continue
        agents = top_level_value.get("responses_api_agents")
        if not isinstance(agents, dict):
            continue
        for agent_block in agents.values():
            if isinstance(agent_block, dict):
                yield agent_block


def _is_agent_config(config_path: Path) -> bool:
    """True if the file is a NeMo Gym agent config (a top-level block with ``responses_api_agents``).

    Filters out non-agent YAML that happens to live in an agent's ``configs/`` dir (e.g. a raw
    harness config or an empty stub).
    """
    return next(_iter_agent_blocks(config_path), None) is not None


def _classify(config_paths: Tuple[Path, ...]) -> Tuple[bool, Optional[str]]:
    """Return ``(self_contained, description)`` for an agent from its config variants.

    A harness is NOT self-contained (Pattern A) when some variant references a separate resources
    server, none declares an ``agent_framework``, and none drives an external LLM loop (e.g. its own
    Anthropic key) — it must be wired to a compatible resources server. Otherwise it is
    self-contained (Pattern B). Agents with no parseable config are treated as Pattern A (their
    wiring lives in a paired benchmark/resources-server config).
    """
    has_resources_server = False
    has_agent_framework = False
    drives_external_harness = False
    description: Optional[str] = None

    saw_block = False
    for config_path in config_paths:
        for block in _iter_agent_blocks(config_path):
            saw_block = True
            if "resources_server" in block:
                has_resources_server = True
            if "agent_framework" in block:
                has_agent_framework = True
            if "anthropic_api_key" in block:
                drives_external_harness = True
            if description is None and isinstance(block.get("description"), str):
                description = block["description"]

    if not saw_block:
        return False, description
    references_resources_server = has_resources_server and not has_agent_framework and not drives_external_harness
    return not references_resources_server, description


def _discover_agents_in_dir(agents_dir: Path) -> Dict[str, AgentEntry]:
    """Map agent name -> :class:`AgentEntry` for every agent dir under one ``responses_api_agents/`` dir.

    The name is the directory name. A directory is an agent if it has an ``app.py`` or at least one
    agent config. Returns an empty dict if the directory is missing.
    """
    agents: Dict[str, AgentEntry] = {}
    if not agents_dir.is_dir():
        return agents

    for child in sorted(agents_dir.iterdir()):
        if not child.is_dir():
            continue
        configs_dir = child / AGENT_CONFIGS_SUBDIR
        config_files = sorted(configs_dir.glob("*.yaml")) if configs_dir.is_dir() else []
        agent_configs = tuple(path for path in config_files if _is_agent_config(path))
        if not (child / "app.py").is_file() and not agent_configs:
            continue

        self_contained, description = _classify(agent_configs)
        agents[child.name] = AgentEntry(
            name=child.name,
            path=child,
            config_paths=agent_configs,
            self_contained=self_contained,
            description=description,
        )

    return agents


def discover_agents() -> Dict[str, AgentEntry]:
    """Map agent name -> :class:`AgentEntry` for every discoverable agent dir.

    Scans the ``responses_api_agents/`` subdir of every :func:`~nemo_gym.discovery.component_search_roots`
    root (``NEMO_GYM_EXTRA_ROOTS`` + cwd + built-ins), merged so user agents shadow same-named built-ins.
    """
    return discover_components(AGENTS_SUBDIR, _discover_agents_in_dir)
