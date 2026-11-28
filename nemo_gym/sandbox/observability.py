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
"""Hooks the sandbox API offers to whoever runs the sandbox: span attributes and stopped-sandbox listeners.

The sandbox package knows nothing about rollouts. Gym's telemetry setup registers a provider
here that returns the current rollout id; the sandbox spans in :mod:`nemo_gym.sandbox.api` add
whatever the provider returns when they open. Gym's observation layers register listeners that
bind a stopped sandbox's measured usage to the verification or agent request that stopped it.
"""

import logging
from collections.abc import Callable, Mapping
from typing import Any


LOGGER = logging.getLogger(__name__)


SpanAttributeProvider = Callable[[], Mapping[str, Any]]
_span_attribute_provider: SpanAttributeProvider | None = None


def set_span_attribute_provider(provider: SpanAttributeProvider | None) -> None:
    """Register a callable whose attributes are added to every ``gym.sandbox.*`` span.

    Called once by the telemetry setup of the embedding application; None removes it. The
    provider runs when a span opens, so it may read request-scoped state such as a ContextVar.
    An explicit ``span_attributes`` argument to ``AsyncSandbox.start`` or ``AsyncSandbox.exec``
    wins over it key by key.
    """
    global _span_attribute_provider
    _span_attribute_provider = provider


def span_attributes(explicit: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The provider's attributes, overridden key by key by *explicit*."""
    attributes: dict[str, Any] = dict(_span_attribute_provider()) if _span_attribute_provider is not None else {}
    if explicit:
        attributes.update(explicit)
    return attributes


# Called with each ``AsyncSandbox`` whose ``stop()`` confirmed teardown after measuring its usage.
StoppedSandboxListener = Callable[[Any], None]
_stopped_sandbox_listeners: list[StoppedSandboxListener] = []


def add_stopped_sandbox_listener(listener: StoppedSandboxListener) -> None:
    """Register a callable told about every measured sandbox this process created and stopped.

    Registered once by an upper layer that binds the usage to whatever it is recording, such as the
    active verification or agent request; registering the same callable again does nothing. The listener
    runs inside ``stop()``, so it may read request-scoped state such as a ContextVar.
    """
    if listener not in _stopped_sandbox_listeners:
        _stopped_sandbox_listeners.append(listener)


def notify_stopped_sandbox(sandbox: Any) -> None:
    """Tell every listener about a stopped sandbox; a failing listener never fails the stop."""
    for listener in list(_stopped_sandbox_listeners):
        try:
            listener(sandbox)
        except Exception:
            LOGGER.warning("Stopped-sandbox listener failed", exc_info=True)
