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
"""NeMo Gym span groups, declared into nemo-lens's ``SpanRegistry``.

A span group is checked at every instrumentation site before any work happens, so a
disabled group costs one frozenset membership test. nemo-lens ships no group names of its
own: a consuming library registers the groups it emits under its own namespace, and users
select from them with the ``span_groups`` spec. Importing this module registers Gym's groups
and presets, so it must be imported before ``setup_telemetry``; ``init_telemetry`` does that.

``GymSpanGroup`` is a bag of ``str`` constants rather than a lens subclass, because
``managed_span`` and ``is_span_group_enabled`` take the group as a plain string. Call sites
keep reading ``GymSpanGroup.SANDBOX`` so the spelling lives in one place, and the constants
stay importable without nemo-lens: the *gate* is conditional, not the name.

Presets
-------
``default``
    ``job`` plus the cross-process spine (``server``, ``http_client``, ``rollout``). This
    is deliberately enough on its own to produce **one trace per rollout spanning the
    agent, model, and resources server processes** — the whole point of the integration
    works without tuning.
``per_rollout``
    The spine plus per-request detail (``verify``, ``agent``, ``model_call``). Omits
    ``job`` so each rollout is its own bounded root trace rather than nesting every
    rollout under one run-long span — the same reasoning behind NeMo-RL's ``per_step``.
``all``
    Reserved by nemo-lens: every group registered in the process, including ``sandbox``.

Registration is process-global, and presets **union** across namespaces. When Gym runs
inside a NeMo-RL process, ``default`` selects NeMo-RL's default groups *and* Gym's, and the
``job`` and ``rollout`` names are shared with NeMo-RL's groups of the same name.

There is deliberately no ``tool_call`` or ``dataset`` group. A resources-server tool call
is already a SERVER span named after its route (``POST /get_weather``), which answers the
same questions without a second layer; and Gym's dataset code is CLI upload/download
helpers, not a runtime path worth tracing. A span group with no call site is a knob that
silently does nothing, so neither is declared until something emits under it.

Disabling ``server`` or ``http_client`` breaks cross-process trace joining: ``server``
is the FastAPI ingress side that adopts an inbound ``traceparent`` as its parent, and
``http_client`` is the egress side that emits one. They are in every preset for that
reason.
"""

from typing import ClassVar, Final


#: Registry namespace Gym owns. Also the key for ``SpanRegistry.unregister``.
NAMESPACE = "nemo_gym"


class GymSpanGroup:
    """Span group names for NeMo Gym instrumentation."""

    JOB = "job"
    """The whole ``gym eval`` / rollout-collection run, driver side. Shares its name with
    NeMo-RL's run-level group, so one spec entry selects both."""

    SERVER = "server"
    """Inbound FastAPI request spans on every Gym server process. The ingress half of
    cross-process propagation: adopts an inbound ``traceparent`` as the span's parent."""

    HTTP_CLIENT = "http_client"
    """Outbound spans around ``nemo_gym.server_utils.request`` — Gym's single aiohttp
    egress point. The egress half of cross-process propagation: injects ``traceparent``
    into the outgoing headers."""

    ROLLOUT = "rollout"
    """Rollout collection spans (one per task attempt, driver side)."""

    VERIFY = "verify"
    """Resources-server ``/verify`` spans."""

    AGENT = "agent"
    """Agent-server ``/run`` and ``/v1/responses`` spans."""

    MODEL_CALL = "model_call"
    """Model-server ``/v1/chat/completions``, ``/v1/responses`` and ``/v1/messages`` spans."""

    SANDBOX = "sandbox"
    """Sandbox provider create/exec/delete spans."""

    CHECKPOINT = "checkpoint"
    """Partial-rollout checkpoint spans: each participant's control operations (prepare, commit, restore,
    resume, retire) with their export, write, read, and install steps, and the controller's coordination
    calls. Enable it with ``default,checkpoint``; the checkpoint metrics are recorded either way."""

    ALL_GROUPS: Final[frozenset] = frozenset(
        [JOB, SERVER, HTTP_CLIENT, ROLLOUT, VERIFY, AGENT, MODEL_CALL, SANDBOX, CHECKPOINT]
    )

    #: The groups that make one rollout appear as one trace across Gym's server
    #: processes. Every preset is a superset of this.
    CROSS_PROCESS_SPINE: Final[frozenset] = frozenset([SERVER, HTTP_CLIENT, ROLLOUT])

    #: ``all`` is not here: nemo-lens reserves it as a wildcard over every registered group.
    _PRESETS: ClassVar[dict] = {
        "default": frozenset([JOB]) | CROSS_PROCESS_SPINE,
        # NOTE: ``per_rollout`` deliberately omits ``job`` so each rollout is its own root
        # trace with a bounded span count. ``job`` wraps a whole eval run and lives in
        # ``default`` and ``all``.
        "per_rollout": frozenset([VERIFY, AGENT, MODEL_CALL]) | CROSS_PROCESS_SPINE,
    }

    @classmethod
    def resolve(cls, spec: str) -> frozenset:
        """Resolve a ``span_groups`` spec against every group registered in this process.

        Entries that name nothing registered are dropped rather than raised: the library
        that owns them may not be imported in this process. nemo-lens logs a warning naming
        them when the spec is applied in ``setup_telemetry``.

        Raises:
            RuntimeError: nemo-lens is not installed, so there is nothing to resolve.
        """
        try:
            from nemo.lens.groups import SpanRegistry
        except ImportError as e:
            raise RuntimeError(
                "GymSpanGroup.resolve() requires nemo-lens. Install it with: uv sync --extra telemetry"
            ) from e
        return SpanRegistry.resolve(spec)[0]


def register_span_groups() -> None:
    """Declare Gym's groups and presets to nemo-lens. A no-op without nemo-lens.

    Called at import. ``allow_override`` makes it idempotent, so a re-import or a test that
    cleared the registry can call it again, and silences the shared-name warning for the
    ``job`` and ``rollout`` groups NeMo-RL also registers.
    """
    try:
        from nemo.lens.groups import SpanRegistry
    except ImportError:
        return
    SpanRegistry.register(
        NAMESPACE, groups=GymSpanGroup.ALL_GROUPS, presets=GymSpanGroup._PRESETS, allow_override=True
    )


register_span_groups()
