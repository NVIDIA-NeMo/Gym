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
"""Compose an already-merged NeMo Gym config along the agent / dataset axes.

This module takes a *merged* OmegaConf :class:`~omegaconf.DictConfig` — the kind produced by Gym's
Hydra config stack from a benchmark's ``config.yaml`` — and rewrites it in place along two axes:

- **Agent axis:** swap the composable agent harness (``--agent``) carrying the benchmark dataset for
  a different one, re-using the environment's own wiring (resources server, model server, datasets).
- **Dataset axis:** edit the benchmark dataset entry's run parameters (``--num-repeats``,
  ``--prompt-config``).

The *model axis* is intentionally out of scope: model selection is handled by the CLI's
``--model*`` flags long before composition runs, so :func:`substitute_model` is a documented no-op.

Like :mod:`nemo_gym.agent_registry` and :mod:`nemo_gym.benchmarks`, composition is *pure and
resolution-safe*: it never resolves interpolations (``resolve=False``), never starts servers, and
never reads secrets, so it is safe to call when API keys referenced by a config are unset. It also
never emits Hydra override tokens — it returns a new :class:`~omegaconf.DictConfig`.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional

from omegaconf import DictConfig, ListConfig, OmegaConf


# Keys carried over from the environment's existing agent block onto a freshly substituted agent, so
# the agent stays wired to the same resources server, policy model, and benchmark dataset.
_AGENT_TYPE_KEY = "responses_api_agents"
_WIRING_KEYS = ("resources_server", "model_server", "datasets")
_BENCHMARK_DATASET_TYPE = "benchmark"


class ConfigComposerError(ValueError):
    """Base error for failures while composing a merged config."""


class NoComposableAgentBlockError(ConfigComposerError):
    """The merged config has zero (or several ambiguous) agent blocks to compose against."""


class MandatoryPlaceholderError(ConfigComposerError):
    """A mandatory ``???`` (OmegaConf MISSING) field remained after composition."""


class AgentNotComposableError(ConfigComposerError):
    """The requested agent is self-contained (Pattern B) and cannot be wired into an environment.

    Raised by the agent-resolution callable passed to :func:`substitute_agent` / :func:`compose`
    when ``require_composable=True`` and the agent ships its own environment/framework. Composability
    is classified by :func:`nemo_gym.agent_registry.discover_agents` (``AgentEntry.self_contained``).
    """


@dataclass(frozen=True)
class ComposeRequest:
    """A composition request along the agent and dataset axes (model axis handled by the CLI)."""

    agent: Optional[str] = None
    num_repeats: Optional[int] = None
    prompt_config: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        return self.agent is None and self.num_repeats is None and self.prompt_config is None


def _deepcopy(merged: DictConfig) -> DictConfig:
    """Return an independent, resolution-safe copy of ``merged`` (MISSING fields preserved)."""
    return OmegaConf.create(OmegaConf.to_container(merged, resolve=False, throw_on_missing=False))


def _agent_block_type(block: DictConfig) -> Optional[str]:
    """Return the single ``responses_api_agents.<type>`` inner key of a top-level block, if any."""
    agents = block.get(_AGENT_TYPE_KEY) if isinstance(block, DictConfig) else None
    if not isinstance(agents, DictConfig):
        return None
    keys = list(agents.keys())
    if len(keys) != 1:
        return None
    return keys[0]


def _dataset_is_benchmark(dataset) -> bool:
    return isinstance(dataset, DictConfig) and dataset.get("type") == _BENCHMARK_DATASET_TYPE


def find_agent_block_key(merged: DictConfig) -> str:
    """Return the top-level key whose ``responses_api_agents`` block is the composition target.

    Mirroring :mod:`nemo_gym.benchmarks`, the target is the agent block whose ``datasets`` contain a
    ``type: benchmark`` entry. If exactly one agent block carries such a dataset, it wins even when
    other (auxiliary) agent blocks exist. Otherwise the choice must be unambiguous: a single agent
    block overall is accepted. Raises :class:`NoComposableAgentBlockError` for zero candidates or an
    ambiguous selection, listing the candidate keys.
    """
    agent_block_keys: List[str] = []
    benchmark_block_keys: List[str] = []
    for top_level_key in merged:
        block = merged[top_level_key]
        agent_type = _agent_block_type(block)
        if agent_type is None:
            continue
        agent_block_keys.append(top_level_key)
        inner = block[_AGENT_TYPE_KEY][agent_type]
        datasets = inner.get("datasets") if isinstance(inner, DictConfig) else None
        if isinstance(datasets, (list, ListConfig)) and any(_dataset_is_benchmark(d) for d in datasets):
            benchmark_block_keys.append(top_level_key)

    if len(benchmark_block_keys) == 1:
        return benchmark_block_keys[0]
    if len(benchmark_block_keys) > 1:
        raise NoComposableAgentBlockError(
            "Multiple agent blocks carry a `type: benchmark` dataset; cannot pick a composition "
            f"target unambiguously. Candidates: {sorted(benchmark_block_keys)}."
        )
    if len(agent_block_keys) == 1:
        return agent_block_keys[0]
    if len(agent_block_keys) == 0:
        raise NoComposableAgentBlockError(
            "No `responses_api_agents` block found in the merged config to compose against."
        )
    raise NoComposableAgentBlockError(
        "Ambiguous composition target: several agent blocks exist and none carries a "
        f"`type: benchmark` dataset. Candidates: {sorted(agent_block_keys)}."
    )


def substitute_agent(
    merged: DictConfig,
    agent_block_key: str,
    new_agent: str,
    *,
    resolve_agent_config_path: Callable[..., str],
) -> DictConfig:
    """Replace the agent harness in ``agent_block_key`` with ``new_agent``, keeping env wiring.

    ``new_agent`` is resolved via ``resolve_agent_config_path(new_agent, require_composable=True)``;
    a Pattern B (self-contained) agent raises :class:`AgentNotComposableError`, which is left to
    propagate. The new agent config's inner block (shape
    ``<k>.responses_api_agents.<new_agent>``) replaces the existing inner agent block, but the
    environment's ``resources_server`` / ``model_server`` / ``datasets`` win over the agent config's
    own values, and the inner key is re-keyed to ``<new_agent>``.
    """
    merged = _deepcopy(merged)

    new_agent_path = resolve_agent_config_path(new_agent, require_composable=True)
    new_agent_config = OmegaConf.load(new_agent_path)

    new_inner_block: Optional[DictConfig] = None
    for top_level_value in new_agent_config.values():
        agent_type = _agent_block_type(top_level_value)
        if agent_type is not None:
            new_inner_block = top_level_value[_AGENT_TYPE_KEY][agent_type]
            break
    if new_inner_block is None:
        raise NoComposableAgentBlockError(
            f"Agent config for '{new_agent}' at {new_agent_path} has no `responses_api_agents` block."
        )

    old_agents = merged[agent_block_key][_AGENT_TYPE_KEY]
    old_agent_type = next(iter(old_agents.keys()))
    old_inner_block = old_agents[old_agent_type]

    new_inner_block = _deepcopy(new_inner_block)
    for wiring_key in _WIRING_KEYS:
        if wiring_key in old_inner_block:
            new_inner_block[wiring_key] = old_inner_block[wiring_key]

    merged[agent_block_key][_AGENT_TYPE_KEY] = OmegaConf.create({new_agent: new_inner_block})
    return merged


def substitute_dataset_params(
    merged: DictConfig,
    agent_block_key: str,
    *,
    num_repeats: Optional[int] = None,
    prompt_config: Optional[str] = None,
) -> DictConfig:
    """Edit the single ``type: benchmark`` dataset in the agent block; no-op for ``None`` fields."""
    if num_repeats is None and prompt_config is None:
        return _deepcopy(merged)

    merged = _deepcopy(merged)
    agents = merged[agent_block_key][_AGENT_TYPE_KEY]
    inner = agents[next(iter(agents.keys()))]
    datasets = inner.get("datasets") or []
    benchmark_datasets = [d for d in datasets if _dataset_is_benchmark(d)]
    if len(benchmark_datasets) != 1:
        raise NoComposableAgentBlockError(
            f"Expected exactly one `type: benchmark` dataset in agent block '{agent_block_key}', "
            f"found {len(benchmark_datasets)}."
        )

    dataset = benchmark_datasets[0]
    if num_repeats is not None:
        dataset["num_repeats"] = num_repeats
    if prompt_config is not None:
        dataset["prompt_config"] = prompt_config
    return merged


def substitute_model(merged: DictConfig) -> DictConfig:
    """No-op shim: the model axis is owned by the CLI's ``--model*`` flags, not the composer."""
    return merged


def _validate_no_mandatory_placeholders(merged: DictConfig, agent_block_key: str) -> None:
    """Raise :class:`MandatoryPlaceholderError` if any ``???`` remains in the agent or its resources.

    Walks the composed agent block and, if it references one, the resources-server block it points
    at (``resources_server.name`` -> top-level key). Interpolations are *not* resolved.
    """
    missing: List[str] = []

    def _walk(node, path: str) -> None:
        if isinstance(node, DictConfig):
            for key in node:
                if OmegaConf.is_missing(node, key):
                    missing.append(f"{path}.{key}" if path else str(key))
                else:
                    # Use `_get_node` so interpolations (e.g. a composable agent's `${nvidia_api_key}`)
                    # are NOT resolved while we only scan for remaining `???`; `node[key]` would
                    # resolve and raise InterpolationKeyError for keys NO_MODEL doesn't inject.
                    _walk(node._get_node(key), f"{path}.{key}" if path else str(key))
        elif isinstance(node, (list, ListConfig)):
            for index in range(len(node)):
                item = node._get_node(index) if isinstance(node, ListConfig) else node[index]
                _walk(item, f"{path}[{index}]")

    agent_block = merged[agent_block_key]
    _walk(agent_block, agent_block_key)

    agents = agent_block[_AGENT_TYPE_KEY]
    inner = agents[next(iter(agents.keys()))]
    resources_server = inner.get("resources_server") if isinstance(inner, DictConfig) else None
    if isinstance(resources_server, DictConfig) and not OmegaConf.is_missing(resources_server, "name"):
        resources_key = resources_server.get("name")
        if isinstance(resources_key, str) and resources_key in merged:
            _walk(merged[resources_key], resources_key)

    if missing:
        raise MandatoryPlaceholderError(
            "Mandatory `???` field(s) remain after composition; provide them via CLI/config overrides: "
            + ", ".join(sorted(missing))
        )


def compose(
    merged: DictConfig,
    request: ComposeRequest,
    *,
    resolve_agent_config_path: Callable[..., str],
) -> DictConfig:
    """Compose ``merged`` per ``request`` and return a new validated config.

    Pipeline: locate the agent block, optionally swap the agent (carrying env wiring), edit the
    benchmark dataset params, then assert no mandatory ``???`` field remains. An empty request is a
    pure passthrough (still deep-copied and validated).
    """
    result = _deepcopy(merged)
    agent_block_key = find_agent_block_key(result)

    if request.agent is not None:
        result = substitute_agent(
            result,
            agent_block_key,
            request.agent,
            resolve_agent_config_path=resolve_agent_config_path,
        )

    result = substitute_dataset_params(
        result,
        agent_block_key,
        num_repeats=request.num_repeats,
        prompt_config=request.prompt_config,
    )

    result = substitute_model(result)

    _validate_no_mandatory_placeholders(result, agent_block_key)
    return result
