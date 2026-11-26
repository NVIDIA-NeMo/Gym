# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind the measured usage of the sandboxes a request stopped to the sandbox observations it returns.

``AsyncSandbox.stop()`` reports each measured sandbox it stopped to a listener registered here. The shared agent base
collects them over an agent ``/run`` or session close, and the environment base over an episode; each then fills the
``SandboxObservation`` records with a matching ``sandbox_id`` in the observations it returns. A sandbox stopped inside
an observed verification belongs to that verification instead (see ``verifier_observability``).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from pydantic import BaseModel, ValidationError

from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.observability import add_stopped_sandbox_listener
from nemo_gym.server_runtime.verifier_observability import verification_active
from nemo_gym.trajectory.trajectory_record import (
    NG_AGENT_OBSERVATIONS_KEY,
    AgentObservationBundle,
    SandboxObservation,
    resource_usage_fields,
)


LOGGER = logging.getLogger(__name__)

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
def stopped_sandbox_usage(*, registry: dict[str, dict[str, Any]] | None = None) -> Iterator[dict[str, dict[str, Any]]]:
    """Collect the usage of the measured sandboxes stopped while the block runs, by sandbox id.

    Pass a ``registry`` that outlives the block when the stop may finish in a shielded task after the block's
    waiter was cancelled, so a retry that joins that task still finds the usage.
    """
    registry = {} if registry is None else registry
    token = _stopped_sandbox_usage.set(registry)
    try:
        yield registry
    finally:
        _stopped_sandbox_usage.reset(token)


def fill_sandbox_usage(bundle: AgentObservationBundle | Mapping[str, Any] | None, usage: Mapping[str, Any]) -> None:
    """Add collected usage to each sandbox observation in ``bundle`` whose ``sandbox_id`` matches, in place.

    ``bundle`` is an observation bundle or its JSON form. An observation that already carries usage keeps it,
    so a sandbox's usage is recorded once whether its observation was built before or after the stop; a usage field
    a producer already set is never overwritten. A JSON record that is not a valid observation is left as it is
    (trajectory building reports it), and only the filled keys are added to it.
    """
    if not usage or bundle is None:
        return
    records = bundle.records if isinstance(bundle, AgentObservationBundle) else bundle.get("records")
    for record in records if isinstance(records, list) else []:
        if isinstance(record, SandboxObservation):
            if record.resource_usage is None and record.sandbox_id in usage:
                for name, value in usage[record.sandbox_id].items():
                    if getattr(record, name, None) is None:
                        setattr(record, name, value)
        elif isinstance(record, dict) and record.get("kind") == "sandbox" and record.get("sandbox_id") in usage:
            if record.get("resource_usage") is not None:
                continue
            updates = {name: value for name, value in usage[record["sandbox_id"]].items() if record.get(name) is None}
            try:
                filled = SandboxObservation.model_validate({**record, **updates})
            except ValidationError:
                LOGGER.debug("Sandbox observation left without usage: not a valid observation", exc_info=True)
                continue
            record.update(filled.model_dump(mode="json", include=set(updates)))


# The fields that carry an agent observation bundle: a run or episode reply, and an agent-session close reply.
_BUNDLE_FIELDS = (NG_AGENT_OBSERVATIONS_KEY, "agent_observations")
# Conversation and trajectory payloads, which never hold a bundle and can be large: never walked.
_PAYLOAD_FIELDS = frozenset(
    ("responses_create_params", "response", "ng_trajectory", "trajectory", "ng_model_call_capture")
)


def fill_reply_sandbox_usage(reply: Any, usage: Mapping[str, Any], *, depth: int = 4) -> None:
    """Fill collected usage into every agent observation bundle a reply carries, in place.

    Bundles are found under ``ng_agent_observations`` or ``agent_observations`` in the reply, a model or its JSON
    form, including nested results and turns (an Environment Server's episode result), down to ``depth`` levels.
    Conversation and trajectory payloads are skipped.
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
        elif name not in _PAYLOAD_FIELDS and isinstance(value, (BaseModel, Mapping, list)):
            fill_reply_sandbox_usage(value, usage, depth=depth - 1)
