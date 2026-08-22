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

"""Shared helpers for Skills-sidecar vs Gym Sandbox API dual-path backends."""

from __future__ import annotations

from typing import Any, Mapping, Union

from nemo_gym.sandbox import (
    SandboxResources,
    SandboxSpec,
    resolve_provider_config,
    resolve_provider_metadata,
)


SANDBOX_BACKEND_SKILLS = "skills_sidecar"
SANDBOX_BACKEND_GYM = "gym_sandbox"
VALID_SANDBOX_BACKENDS = frozenset({SANDBOX_BACKEND_SKILLS, SANDBOX_BACKEND_GYM})

SandboxProviderRef = Union[str, Mapping[str, Any]]


def normalize_sandbox_backend(value: str | None) -> str:
    """Return a validated backend name; unset/empty defaults to Skills sidecar."""
    if value is None or str(value).strip() == "":
        return SANDBOX_BACKEND_SKILLS
    backend = str(value).strip()
    if backend not in VALID_SANDBOX_BACKENDS:
        raise ValueError(
            f"Unknown sandbox_backend={backend!r}. Expected one of: {sorted(VALID_SANDBOX_BACKENDS)}"
        )
    return backend


def resolve_gym_provider_config(
    sandbox_provider: SandboxProviderRef | None,
    named_configs: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Resolve provider config for gym_sandbox; fail fast if misconfigured."""
    if sandbox_provider is None:
        raise ValueError(
            "sandbox_backend=gym_sandbox requires sandbox_provider "
            "(name reference like 'sandbox' or an inline {provider: {...}} mapping)"
        )
    return resolve_provider_config(sandbox_provider, named_configs)


def resolve_gym_provider_metadata(
    sandbox_provider: SandboxProviderRef | None,
    named_configs: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if sandbox_provider is None:
        return {}
    return resolve_provider_metadata(sandbox_provider, named_configs)


def build_sandbox_spec_from_mapping(
    sandbox_spec: Mapping[str, Any] | None,
    *,
    default_metadata: Mapping[str, Any] | None = None,
    extra_files: Mapping[str, str] | None = None,
) -> SandboxSpec:
    """Translate a YAML ``sandbox_spec`` mapping into a SandboxSpec."""
    spec = dict(sandbox_spec or {})
    files = dict(spec.pop("files", {}))
    if extra_files:
        files.update(extra_files)

    metadata = dict(default_metadata or {})
    metadata.update(dict(spec.pop("metadata", {})))

    known = SandboxSpec(
        image=spec.pop("image", None),
        ttl_s=spec.pop("ttl_s", None),
        ready_timeout_s=spec.pop("ready_timeout_s", None),
        workdir=spec.pop("workdir", None),
        env=dict(spec.pop("env", {})),
        files=files,
        metadata=metadata,
        resources=SandboxResources.from_mapping(spec.pop("resources", {})),
        entrypoint=spec.pop("entrypoint", None),
        provider_options=dict(spec.pop("provider_options", {})),
    )
    if spec:
        raise ValueError(f"Unknown sandbox_spec keys: {', '.join(sorted(spec))}")
    return known


def named_configs_from_server_client(server_client: Any) -> Mapping[str, Any] | None:
    """Best-effort global config dict for resolving named sandbox_provider refs."""
    global_config = getattr(server_client, "global_config_dict", None)
    if global_config is None:
        return None
    try:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(global_config, resolve=True)  # type: ignore[return-value]
    except Exception:
        if isinstance(global_config, Mapping):
            return global_config
        return None
