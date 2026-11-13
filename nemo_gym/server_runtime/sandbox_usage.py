# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind the measured usage of the sandboxes a request stopped to the sandbox observations it returns.

``AsyncSandbox.stop()`` reports each measured sandbox it stopped to a listener registered here. The shared agent base
collects them over an agent ``/run`` or session close, and the environment base over an episode; each then fills the
``SandboxObservation`` records with a matching ``sandbox_id`` in the observations it returns. A sandbox stopped inside
an observed verification belongs to that verification instead (see ``verifier_observability``).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from pydantic import BaseModel

from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.observability import add_stopped_sandbox_listener
from nemo_gym.server_runtime.verifier_observability import verification_active
from nemo_gym.trajectory.trajectory_record import (
    NG_AGENT_OBSERVATIONS_KEY,
    AgentObservationBundle,
    SandboxObservation,
    resource_usage_fields,
)


# Usage fields of the measured sandboxes stopped during the current request, by sandbox id.
_stopped_sandbox_usage: ContextVar[dict[str, dict[str, Any]] | None] = ContextVar(
    "stopped_sandbox_usage", default=None
)


def _record_stopped_sandbox(sandbox: AsyncSandbox) -> None:
    """Keep a sandbox stopped during a collecting request; one stopped during a verification belongs to it."""
    registry = _stopped_sandbox_usage.get()
    if registry is None or sandbox.sandbox_id is None or verification_active():
        return
    usage = resource_usage_fields(sandbox.resource_usage)
    if usage:
        registry[sandbox.sandbox_id] = usage


add_stopped_sandbox_listener(_record_stopped_sandbox)


@contextmanager
def stopped_sandbox_usage() -> Iterator[dict[str, dict[str, Any]]]:
    """Collect the usage of the measured sandboxes stopped while the block runs, by sandbox id."""
    registry: dict[str, dict[str, Any]] = {}
    token = _stopped_sandbox_usage.set(registry)
    try:
        yield registry
    finally:
        _stopped_sandbox_usage.reset(token)


def fill_sandbox_usage(bundle: AgentObservationBundle | Mapping[str, Any] | None, usage: Mapping[str, Any]) -> None:
    """Add collected usage to each sandbox observation in ``bundle`` whose ``sandbox_id`` matches, in place.

    ``bundle`` is an observation bundle or its JSON form. An observation that already carries usage keeps it,
    so a sandbox's usage is recorded once whether its observation was built before or after the stop.
    """
    if not usage or bundle is None:
        return
    records = bundle.records if isinstance(bundle, AgentObservationBundle) else bundle.get("records")
    for index, record in enumerate(records if isinstance(records, list) else []):
        if isinstance(record, SandboxObservation):
            if record.resource_usage is None and record.sandbox_id in usage:
                for name, value in usage[record.sandbox_id].items():
                    setattr(record, name, value)
        elif isinstance(record, dict) and record.get("kind") == "sandbox" and record.get("sandbox_id") in usage:
            if record.get("resource_usage") is None:
                filled = SandboxObservation.model_validate(record).model_copy(update=usage[record["sandbox_id"]])
                records[index] = filled.model_dump(mode="json")


# The fields that carry an agent observation bundle: a run or episode reply, and an agent-session close reply.
_BUNDLE_FIELDS = (NG_AGENT_OBSERVATIONS_KEY, "agent_observations")


def fill_reply_sandbox_usage(reply: Any, usage: Mapping[str, Any], *, depth: int = 4) -> None:
    """Fill collected usage into every agent observation bundle a reply carries, in place.

    Bundles are found under ``ng_agent_observations`` or ``agent_observations`` in the reply, a model or its JSON
    form, including nested results and turns (an Environment Server's episode result), down to ``depth`` levels.
    """
    if not usage or depth < 0:
        return
    if isinstance(reply, BaseModel):
        items = [(name, getattr(reply, name, None)) for name in type(reply).model_fields]
        items += list((reply.model_extra or {}).items())
    elif isinstance(reply, Mapping):
        items = list(reply.items())
    elif isinstance(reply, list):
        for item in reply:
            fill_reply_sandbox_usage(item, usage, depth=depth - 1)
        return
    else:
        return
    for name, value in items:
        if name in _BUNDLE_FIELDS and isinstance(value, (AgentObservationBundle, Mapping)):
            fill_sandbox_usage(value, usage)
        elif isinstance(value, (BaseModel, Mapping, list)):
            fill_reply_sandbox_usage(value, usage, depth=depth - 1)
