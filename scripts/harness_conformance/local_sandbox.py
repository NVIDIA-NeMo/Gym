# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Borrowable local execution for the controlled conformance fixture.

Like the other probes, commands execute on the Linux runner. The descriptor only
hands the fixture's existing workspace to an adapter that requires a resources-
owned sandbox; this does not qualify a remote sandbox provider.
"""

import os
import pwd
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from nemo_gym.sandbox.providers.base import SandboxHandle
from nemo_gym.sandbox.providers.local.provider import LocalProvider
from nemo_gym.sandbox.providers.registry import get_provider_class, register_provider


class ProbeLocalProvider(LocalProvider):
    """Execute real commands through LocalProvider with an explicit borrow contract."""

    async def exec(self, handle: SandboxHandle, command: str, *, user=None, **kwargs):
        # Terminus explicitly requests root for setup. Accept an explicit current
        # user, never pretend that a local subprocess changed its identity.
        if user is not None:
            uid = int(user) if str(user).isdigit() else pwd.getpwnam(user).pw_uid
            if uid != os.geteuid():
                raise ValueError("conformance local execution requires the requested user to be the current user")
        return await super().exec(handle, command, **kwargs)

    async def serialize_handle(self, handle: SandboxHandle, *, scope: str | None = None) -> dict[str, Any]:
        return {
            "sandbox_id": handle.sandbox_id,
            "workspace": str(handle.raw["workspace"]),
            "env": dict(handle.raw["env"]),
        }

    async def connect(self, descriptor: Mapping[str, Any]) -> SandboxHandle:
        workspace = Path(descriptor["workspace"])
        if not workspace.is_absolute() or not workspace.is_dir():
            raise ValueError("conformance sandbox workspace must already exist")
        return SandboxHandle(
            sandbox_id=descriptor["sandbox_id"],
            provider_name=self.name,
            raw={"workspace": workspace, "env": dict(descriptor["env"]), "owned": False},
        )


@contextmanager
def borrowable_local_provider() -> Iterator[None]:
    """Add borrowing only inside this fixture process, retaining local URL routing."""
    previous = get_provider_class("local")
    register_provider("local", ProbeLocalProvider, override=True)
    try:
        yield
    finally:
        register_provider("local", previous, override=True)
