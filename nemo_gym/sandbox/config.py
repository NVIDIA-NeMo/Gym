# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Resolve a sandbox provider reference into a provider config.

An agent selects a sandbox by name (``sandbox_provider: sandbox``). The named
block lives in its own provider config file, so swapping providers is swapping a
``config_paths`` entry, not editing the agent config::

    # nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml
    sandbox:
      opensandbox:
        connection: { ... }

    # agent config
    sandbox_provider: sandbox

An inline single-key mapping (``{provider_name: {...}}``) is also accepted for
keeping everything in one file.
"""

from collections.abc import Mapping
from typing import Any


def _to_plain_dict(value: Any) -> Any:
    """Return a plain ``dict`` for mappings, including OmegaConf ``DictConfig``."""
    try:
        from omegaconf import DictConfig, OmegaConf
    except ImportError:  # pragma: no cover - omegaconf is a core dependency
        DictConfig = ()  # type: ignore[assignment]
        OmegaConf = None  # type: ignore[assignment]

    if OmegaConf is not None and isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, Mapping):
        return dict(value)
    return value


def _candidate_sandbox_names(named_configs: Mapping[str, Any] | None) -> list[str]:
    """List top-level config keys that look like named sandbox provider blocks."""
    if not named_configs:
        return []
    candidates: list[str] = []
    for key, value in named_configs.items():
        plain = _to_plain_dict(value)
        if isinstance(plain, Mapping) and len(plain) == 1:
            candidates.append(str(key))
    return sorted(candidates)


def resolve_provider_config(
    sandbox_provider: str | Mapping[str, Any],
    named_configs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve a ``sandbox_provider`` field into a single-key provider config dict.

    Args:
        sandbox_provider: Either the name of a top-level sandbox config block
            (resolved from ``named_configs``) or an inline single-key provider
            mapping of the form ``{provider_name: {...}}``.
        named_configs: Mapping of top-level config name to config block, typically
            the merged global config dict. Required when ``sandbox_provider`` is a
            name reference.

    Returns:
        A plain ``{provider_name: provider_kwargs}`` dict suitable for
        :func:`nemo_gym.sandbox.create_provider`.

    Raises:
        TypeError: If ``sandbox_provider`` is neither a string nor a mapping.
        ValueError: If a named reference cannot be found, or if the resolved block
            is not a single-key provider mapping.
    """
    if isinstance(sandbox_provider, str):
        name = sandbox_provider
        if not name:
            raise ValueError("Sandbox provider reference must be a non-empty string")
        block = named_configs.get(name) if named_configs is not None else None
        if block is None:
            available = ", ".join(repr(n) for n in _candidate_sandbox_names(named_configs)) or "(none)"
            raise ValueError(
                f"Sandbox provider reference {name!r} is not defined in the merged config. "
                f"Define a top-level '{name}:' block (e.g. via "
                f"nemo_gym/sandbox/providers/<provider>/configs/<provider>.yaml) and include it in "
                f"your config_paths. Available sandbox configs: {available}"
            )
        block = _to_plain_dict(block)
        source = f"reference {name!r}"
    elif isinstance(sandbox_provider, Mapping):
        block = _to_plain_dict(sandbox_provider)
        source = "inline sandbox_provider config"
    else:
        raise TypeError(
            "sandbox_provider must be a name reference (str) or a single-key provider mapping, "
            f"got {type(sandbox_provider).__name__}"
        )

    if not isinstance(block, Mapping) or len(block) != 1:
        raise ValueError(
            f"Sandbox provider config from {source} must be a single-key mapping "
            f"{{provider_name: config}}, got: {block!r}"
        )

    return dict(block)
