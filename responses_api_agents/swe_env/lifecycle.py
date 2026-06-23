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

"""Sandbox lifecycle: a thin acquire context manager.

``acquire_sandbox`` starts a fresh sandbox and always tears it down on exit —
normal return, exception, or ``asyncio.CancelledError``.

Per-task teardown relies on the ``finally`` block below, which covers graceful
exit and cancellation. Backends that honor ``SandboxSpec.ttl_s`` (e.g.
opensandbox) self-expire any sandbox orphaned by a hard crash (SIGKILL/OOM);
``docker`` ignores ``ttl_s`` so its orphans (rare: only on un-catchable death)
need a manual sweep.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Mapping

from nemo_gym.sandbox import SandboxProvider, SandboxSpec
from responses_api_agents.swe_env.environment import AsyncSweEnvironment


@asynccontextmanager
async def acquire_sandbox(
    provider: Mapping[str, Any] | SandboxProvider,
    spec: SandboxSpec,
    *,
    instance_id: str = "",
) -> AsyncIterator[AsyncSweEnvironment]:
    """Start a fresh sandbox, yield it, and always stop it on exit.

    Args:
        provider: Either a ``SandboxProvider`` instance or a mapping describing
            the provider configuration used to create the sandbox.
        spec: The ``SandboxSpec`` describing how to provision the sandbox.
        instance_id: Identifier accepted for logging/telemetry; it does not
            affect behavior.

    Yields:
        AsyncSweEnvironment: The started environment wrapping the sandbox,
        which is cleaned up when the context manager exits.
    """
    env: AsyncSweEnvironment | None = None
    try:
        env = await AsyncSweEnvironment.start(provider, spec)
        yield env
    finally:
        if env is not None:
            try:
                await env.cleanup()
            except Exception:
                pass
