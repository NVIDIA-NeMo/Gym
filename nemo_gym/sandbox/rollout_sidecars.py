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
"""Sidecars from configuration for the sandbox an agent's harness runs in, never for verification sandboxes.

A sidecar listed in a sandbox spec's ``provider_options["sidecars"]`` (a list of sidecar specs, each a mapping with
at least ``name`` and ``image``) is moved into ``SandboxSpec.sidecars`` for the rollout sandbox and dropped for any
other sandbox, with no resources server, agent, or sandbox provider code. :func:`apply_rollout_sidecars` does this as
a spec transform in ``AsyncSandbox.start``; the server base classes register it when they build their app:

- Resources servers (scoped): ``SimpleResourcesServer.setup_webserver`` registers it with ``scoped=True``. Only
  sandboxes started while the server is serving ``/seed_session`` (with or without the rollout prefix) get the
  sidecars. Any other sandbox, such as one created in ``/verify``, has the option dropped. A server that creates the
  rollout sandbox later, for example from a tool call, does not get the sidecars.
- Agent servers (unscoped): ``SimpleResponsesAPIAgent.setup_webserver`` registers it with ``scoped=False``. Every
  sandbox the agent process starts gets the sidecars, because an agent only creates sandboxes its harness runs in.
  An agent that starts helper containers through ``AsyncSandbox`` would give each of them the sidecars too; such an
  agent declares sidecars in ``SandboxSpec.sidecars`` directly instead of in this option.
- Any other process (environment servers, scripts, agents that build their app without the base class) registers
  nothing, so the option reaches the provider unchanged: a provider that validates its options rejects it, and no
  sidecar is started either way.

If both modes are registered in one process, scoped wins, so verification sandboxes never get the sidecars.

Tasks and threads spawned during ``/seed_session`` copy the request's context and so stay in scope for their whole
lifetime, also after the response. Spawn one with a fresh ``contextvars.Context()`` to leave the scope.

A sandbox that gets the sidecars on a provider without sidecar support fails to start with ``NotImplementedError``.
A malformed option raises ``ValueError`` on every start, in or out of scope, so a misconfiguration fails at the first
sandbox.
"""

import logging
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from typing import Any, Literal

from nemo_gym.sandbox.api import register_spec_transform
from nemo_gym.sandbox.providers import SandboxSidecarSpec, SandboxSpec


LOGGER = logging.getLogger(__name__)

#: ``provider_options`` key for sidecars that only the rollout sandbox gets.
ROLLOUT_SIDECARS_OPTION = "sidecars"

_SEEDING_SESSION: ContextVar[bool] = ContextVar("nemo_gym_seeding_session", default=False)

# How this process applies the option; set by register_rollout_sidecars. None means not registered.
_mode: Literal["scoped", "unscoped"] | None = None


@contextmanager
def seeding_session() -> Iterator[None]:
    """Mark the current request as a resources server's ``/seed_session``."""
    token = _SEEDING_SESSION.set(True)
    try:
        yield
    finally:
        _SEEDING_SESSION.reset(token)


def is_seeding_session() -> bool:
    """Whether the current request is a resources server's ``/seed_session``."""
    return _SEEDING_SESSION.get()


class SeedSessionMiddleware:
    """Run a resources server's ``/seed_session`` requests inside :func:`seeding_session`.

    Add it before ``RolloutContextMiddleware`` (so it runs inside it) to see the path without the rollout prefix.
    """

    def __init__(self, app: Any) -> None:
        self._app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") == "http" and scope.get("path") == "/seed_session":
            with seeding_session():
                await self._app(scope, receive, send)
            return
        await self._app(scope, receive, send)


def register_rollout_sidecars(*, scoped: bool) -> None:
    """Apply ``provider_options["sidecars"]`` to the sandboxes this process starts.

    ``scoped=True`` (resources servers) applies it only while serving ``/seed_session``; ``scoped=False`` (agent
    servers) applies it to every sandbox. Registering again has no effect, except that scoped wins over unscoped.
    """
    global _mode
    if _mode is not None and _mode != ("scoped" if scoped else "unscoped"):
        LOGGER.warning(
            "Both a resources server and an agent server registered rollout sidecars in this process; only sandboxes "
            "started while serving /seed_session get provider_options[%r].",
            ROLLOUT_SIDECARS_OPTION,
        )
    if scoped or _mode is None:
        _mode = "scoped" if scoped else "unscoped"
    register_spec_transform(apply_rollout_sidecars)


def _option_sidecars(value: object) -> tuple[SandboxSidecarSpec | Mapping[str, Any], ...]:
    """The sidecars listed in the option; raise on a value that is not a list of sidecar specs."""
    if not value:  # None (a null in YAML) or empty
        return ()
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and all(isinstance(sidecar, (SandboxSidecarSpec, Mapping)) for sidecar in value)
    ):
        return tuple(value)
    raise ValueError(
        f"provider_options[{ROLLOUT_SIDECARS_OPTION!r}] must be a list of sidecar specs, each a mapping with at least "
        f"'name' and 'image', e.g. [{{name: recorder, image: ...}}]; got {value!r}"
    )


def apply_rollout_sidecars(spec: SandboxSpec) -> SandboxSpec:
    """Move ``provider_options["sidecars"]`` into ``spec.sidecars`` for a rollout sandbox; drop it otherwise.

    The option never reaches the provider and the given spec and its options are not modified. Unless this process
    registered the unscoped mode, only a sandbox started while serving ``/seed_session`` is a rollout sandbox.
    """
    if not spec.provider_options or ROLLOUT_SIDECARS_OPTION not in spec.provider_options:
        return spec
    options = dict(spec.provider_options)
    sidecars = _option_sidecars(options.pop(ROLLOUT_SIDECARS_OPTION))
    if _mode != "unscoped" and not is_seeding_session():
        sidecars = ()
    return replace(spec, provider_options=options, sidecars=(*spec.sidecars, *sidecars))
