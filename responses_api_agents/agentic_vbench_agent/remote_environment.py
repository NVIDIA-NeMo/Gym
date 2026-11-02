# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Upstream Harbor Docker environment backed by the cluster's rootless API."""

import hashlib
import json
import shutil
from pathlib import Path

from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.environments.docker.docker import DockerEnvironment


class RemoteEnvironment(DockerEnvironment):
    _DOCKER_COMPOSE_BASE_PATH = Path(__file__).with_name("remote-compose.yaml")

    def __init__(self, *, environment_dir: Path, runtime_root: str, **kwargs):
        if (environment_dir / "docker-compose.yaml").exists():
            raise ValueError("Refusing to replace a task's existing compose definition")
        staged = Path(runtime_root) / "build-context"
        shutil.copytree(environment_dir, staged)
        dockerfile = staged / "Dockerfile"
        original = dockerfile.read_bytes()
        adapted = original.replace(b"apt-get ", b"apt-get -o APT::Sandbox::User=root ")
        dockerfile.write_bytes(adapted)
        (staged.parent / "rootless-build.json").write_text(
            json.dumps(
                {
                    "source": str(environment_dir),
                    "original_dockerfile_sha256": hashlib.sha256(original).hexdigest(),
                    "build_dockerfile_sha256": hashlib.sha256(adapted).hexdigest(),
                    "adaptation": "APT sandbox uses mapped container root; task inputs and verifiers unchanged",
                },
                indent=2,
            )
            + "\n"
        )
        super().__init__(environment_dir=staged, **kwargs)

    async def start(self, force_build: bool) -> None:
        await super().start(force_build)
        # Agent installers also run apt-get in the live container, where the
        # same single-UID namespace prevents apt's _apt privilege drop.
        result = await self.exec(
            "if [ -d /etc/apt/apt.conf.d ]; then "
            "printf 'APT::Sandbox::User \"root\";\\n' > /etc/apt/apt.conf.d/99-agentic-vbench-rootless; fi",
            user="root",
        )
        if result.return_code != 0:
            raise RuntimeError(f"Failed to configure rootless APT sandbox: {result.stdout} {result.stderr}")

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return super().capabilities.model_copy(update={"mounted": False})
