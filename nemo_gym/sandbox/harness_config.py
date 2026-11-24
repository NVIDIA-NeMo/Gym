# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The typed ``sandbox_config`` of a sandbox that runs an agent harness, independent of its owner."""

import dataclasses
from collections.abc import Mapping
from typing import Any

from pydantic import Field

from nemo_gym.sandbox.config import SandboxConfig
from nemo_gym.sandbox.providers.base import SandboxSpec
from nemo_gym.sandbox.utils import cpu_cap_env


class HarnessSandboxConfig(SandboxConfig):
    """``SandboxConfig`` for a harness sandbox owned by an environment server or an agent.

    The owning server creates it through ``OwnedSandboxes`` and lends it through ``borrow()``. These settings also
    support an environment server that supplies the harness sandbox when Resources lends none.

    An adopting agent must refuse setting both its legacy ``network_access`` and this config's ``network_access``;
    a recipe must have one source for its egress rule.

    Attributes:
        image: The sandbox image.
        workdir: The working directory the harness runs in.
        files: Files to place in the sandbox at start, by absolute path.
        entrypoint: The command the sandbox's container runs, when the image's own does not keep it alive.
        user: The user commands run as, when not the image's default.
        ports: The ports the sandbox exposes.
        derive_cpu_env: Cap tool parallelism to ``resources.cpu`` in the sandbox's environment. Off by default, as
            in most agents today. The OpenCode and Pi sandboxed agents turn the caps on in their code rather than in
            a recipe, so a subclass of this config for either defaults it to ``True``.

    A recipe's other ``SandboxSpec`` keys, such as ``sidecars``, are refused here; an agent that moves onto this
    config with recipes that set them warns about them for one release first.
    """

    image: str | None = None
    workdir: str | None = None
    files: dict[str, str] = Field(default_factory=dict)
    entrypoint: list[str] | None = None
    user: str | int | None = None
    ports: list[int] = Field(default_factory=list)
    derive_cpu_env: bool = False

    def spec(
        self,
        sandbox_provider: str | Mapping[str, Any],
        *,
        image: str | None = None,
        workdir: str | None = None,
        metadata: Mapping[str, str] | None = None,
        files: Mapping[str, str | bytes] | None = None,
        named_configs: Mapping[str, Any] | None = None,
        _network_options: Mapping[str, Any] | None = None,
    ) -> SandboxSpec:
        """The spec of the harness sandbox, with this config's image, workdir, files, entrypoint, user and ports.

        ``image`` and ``workdir`` override the config's, and ``files`` add to its files. The CPU caps go under the
        config's ``env``, so a recipe's own value wins.
        """
        spec = super().spec(
            sandbox_provider,
            image=image or self.image,
            workdir=workdir or self.workdir,
            metadata=metadata,
            files={**self.files, **(files or {})},
            named_configs=named_configs,
            _network_options=_network_options,
        )
        cpu_env = cpu_cap_env(spec.resources.cpu) if self.derive_cpu_env else {}
        return dataclasses.replace(
            spec, entrypoint=self.entrypoint, user=self.user, ports=list(self.ports), env={**cpu_env, **spec.env}
        )
