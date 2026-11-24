# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The typed ``sandbox_config`` of a sandbox an agent server starts for its own harness."""

import dataclasses
from collections.abc import Mapping
from typing import Any

from pydantic import Field

from nemo_gym.sandbox.config import SandboxConfig
from nemo_gym.sandbox.providers.base import SandboxSpec
from nemo_gym.sandbox.utils import cpu_cap_env


class AgentSandboxConfig(SandboxConfig):
    """``SandboxConfig`` for the sandbox an agent starts when its environment lends none.

    A lent sandbox is borrowed instead, with ``borrow()`` from ``nemo_gym.server_runtime``; an agent starts its own only
    when nothing is lent, and owns and stops it then.

    A server takes the image, working directory and files of a task sandbox from its task. An agent's own sandbox has
    no task to take them from, so they are configured here, with the keys agent recipes set today.

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
        network_options: Mapping[str, Any] | None = None,
    ) -> SandboxSpec:
        """The spec of the agent's own sandbox, with this config's image, workdir, files, entrypoint, user and ports.

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
            network_options=network_options,
        )
        cpu_env = cpu_cap_env(spec.resources.cpu) if self.derive_cpu_env else {}
        return dataclasses.replace(
            spec, entrypoint=self.entrypoint, user=self.user, ports=list(self.ports), env={**cpu_env, **spec.env}
        )
