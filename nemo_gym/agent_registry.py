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

An *agent* is a directory ``responses_api_agents/<name>/`` providing an agent harness, with zero or
more ``configs/*.yaml`` variants. This module maps an agent's short ``<name>`` (the directory name)
to its config variant(s) so it can be referenced by name — the foundation for ``gym run --agent
<name>`` (run-by-name) — and classifies whether the agent is freely *composable* with an arbitrary
environment.

- **Composable (Pattern A):** the agent references a *separate* resources server
  (``responses_api_agents.<type>.resources_server``), so it can be paired with any environment.
- **Not composable (Pattern B):** the agent is self-contained — it declares an ``agent_framework``
  or bakes in its own environment/external LLM harness (e.g. ``swe_agents``, ``harbor_agent``,
  ``verifiers_agent``, ``claude_code_agent``) — and cannot be dropped onto an arbitrary environment.

Discovery only reads config files; it never resolves interpolations or missing values and never
starts servers, so it is safe to call when secrets/API keys referenced by a config are unset.
"""

from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from omegaconf import OmegaConf

from nemo_gym import PARENT_DIR


AGENTS_DIR = PARENT_DIR / "responses_api_agents"
AGENT_CONFIGS_SUBDIR = "configs"


class AgentNotFoundError(ValueError):
    """An agent was referenced by a name that is not registered under ``responses_api_agents/``."""


class AgentVariantError(ValueError):
    """An agent has no standalone config, or has several and no variant was given to disambiguate."""


class AgentNotComposableError(ValueError):
    """A self-contained (Pattern B) agent was requested for free composition with an environment."""


@dataclass(frozen=True)
class AgentEntry:
    """A discovered agent: its name, where it lives, its config variants, and composability."""

    name: str
    path: Path
    config_paths: Tuple[Path, ...]  # variant config files, sorted; empty for "zero-config" agents
    composable: bool
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
    """Return ``(composable, description)`` for an agent from its config variants.

    Composable iff some variant references a separate resources server, none declares an
    ``agent_framework``, and none drives an external LLM harness (e.g. its own Anthropic key).
    Agents with no parseable config default to composable (their wiring lives in a paired
    benchmark/resources-server config).
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
        return True, description
    composable = has_resources_server and not has_agent_framework and not drives_external_harness
    return composable, description


def discover_agents(agents_dir: Path = AGENTS_DIR) -> Dict[str, AgentEntry]:
    """Map agent name -> :class:`AgentEntry` for every agent dir under ``responses_api_agents/``.

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

        composable, description = _classify(agent_configs)
        agents[child.name] = AgentEntry(
            name=child.name,
            path=child,
            config_paths=agent_configs,
            composable=composable,
            description=description,
        )

    return agents


def _did_you_mean(name: str, available: List[str], noun: str) -> str:
    suggestions = get_close_matches(name, available, n=3, cutoff=0.6)
    if suggestions:
        return "Did you mean: " + ", ".join(repr(s) for s in suggestions) + "?"
    return f"Available {noun}: " + (", ".join(repr(n) for n in available) or "(none)")


def resolve_agent_config_path(
    name: str,
    variant: Optional[str] = None,
    agents_dir: Path = AGENTS_DIR,
    require_composable: bool = False,
) -> str:
    """Return the config path to load to run agent ``name`` — the run-by-name primitive.

    Selection: an explicit ``variant`` wins; otherwise a single config is used directly, and a
    variant whose name equals ``name`` is the default when several exist. Raises
    :class:`AgentNotFoundError` (with a "did you mean?" hint) for an unknown agent,
    :class:`AgentVariantError` for a zero-config or ambiguous-variant agent, and — when
    ``require_composable`` is set — :class:`AgentNotComposableError` for a Pattern B agent.
    """
    agents = discover_agents(agents_dir)
    entry = agents.get(name)
    if entry is None:
        raise AgentNotFoundError(
            f"No agent named '{name}' under {agents_dir}.\n{_did_you_mean(name, sorted(agents), 'agents')}"
        )

    if require_composable and not entry.composable:
        raise AgentNotComposableError(
            f"Agent '{name}' is self-contained (it bakes in its own environment/framework) and cannot "
            "be freely composed with an arbitrary environment; run it with its own config instead."
        )

    variants = entry.variants
    if not variants:
        raise AgentVariantError(
            f"Agent '{name}' ships no standalone config; it is composed via its paired "
            "benchmark/resources-server config."
        )

    if variant is not None:
        if variant not in variants:
            raise AgentVariantError(
                f"Agent '{name}' has no variant '{variant}'.\n{_did_you_mean(variant, sorted(variants), 'variants')}"
            )
        return str(variants[variant])

    if len(variants) == 1:
        return str(next(iter(variants.values())))
    if name in variants:
        return str(variants[name])
    raise AgentVariantError(
        f"Agent '{name}' has multiple config variants: {sorted(variants)}; pass a variant to select one."
    )
