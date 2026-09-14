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
"""NeMo Gym span groups — the knob that decides which spans exist at all.

A span group is checked at every instrumentation site before any work happens, so a
disabled group costs one frozenset membership test. ``GymSpanGroup`` extends the shared
``nemo.lens.groups.SpanGroup`` with Gym-shaped groups and replaces its training-oriented
presets with rollout-oriented ones.

Span groups live downstream, in the consumer, not in nemo-lens — the same call the
NeMo-RL integration made (``RLSpanGroup``), and the direction lens itself is heading.

Presets
-------
``default``
    ``job`` plus the cross-process spine (``server``, ``http_client``, ``rollout``). This
    is deliberately enough on its own to produce **one trace per rollout spanning the
    agent, model, and resources server processes** — the whole point of the integration
    works without tuning.
``per_rollout``
    The spine plus per-request detail (``verify``, ``agent``, ``model_call``,
    ``tool_call``). Omits ``job`` so each rollout is its own bounded root trace rather
    than nesting every rollout under one run-long span — the same reasoning behind
    NeMo-RL's ``per_step``.
``all``
    Every group, including ``sandbox`` and the groups inherited from nemo-lens.

Only groups Gym actually emits under appear in ``default`` and ``per_rollout``. The
training-oriented groups inherited from ``nemo.lens.groups.SpanGroup`` (``checkpoint``,
``step``, ``optimizer``, ``evaluate``, ...) remain resolvable, and are reachable through
``all``, but no Gym call site emits under them.

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


try:
    from nemo.lens.groups import SpanGroup
except ImportError:
    # TODO(ahmadki): SpanGroups are moving from nemo-lens to downstream consumers, at
    # which point this stub and the try/except disappear and GymSpanGroup stands alone.
    class SpanGroup:  # type: ignore[no-redef]
        """Minimal stub used when nemo-lens is not installed.

        Mirrors ``nemo.lens.groups.SpanGroup`` at commit ``b85578fc``. ``resolve()``
        raises rather than returning a wrong answer: without lens there is nothing to
        enable, and silently returning an empty set would make a typo in
        ``telemetry.span_groups`` indistinguishable from a working config.
        """

        JOB = "job"
        CHECKPOINT = "checkpoint"
        EVALUATE = "evaluate"
        MODEL_INIT = "model_init"
        LOAD_CHECKPOINT = "load_checkpoint"
        STEP = "step"
        FORWARD_BACKWARD = "forward_backward"
        OPTIMIZER = "optimizer"

        ALL_GROUPS: Final[frozenset] = frozenset(
            [
                JOB,
                CHECKPOINT,
                EVALUATE,
                MODEL_INIT,
                LOAD_CHECKPOINT,
                STEP,
                FORWARD_BACKWARD,
                OPTIMIZER,
            ]
        )

        _PRESETS: ClassVar[dict] = {
            "default": frozenset([JOB, CHECKPOINT, EVALUATE]),
            "per_step": frozenset(
                [JOB, CHECKPOINT, EVALUATE, MODEL_INIT, LOAD_CHECKPOINT, STEP, FORWARD_BACKWARD, OPTIMIZER]
            ),
            "all": ALL_GROUPS,
        }

        @classmethod
        def resolve(cls, spec: str) -> frozenset:
            raise RuntimeError("SpanGroup.resolve() requires nemo-lens. Install it with: uv sync --extra telemetry")


class GymSpanGroup(SpanGroup):
    """Span groups for NeMo Gym instrumentation."""

    # ------------------------------------------------------------------ #
    # Gym-specific groups
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    # All groups and presets
    # ------------------------------------------------------------------ #

    ALL_GROUPS: Final[frozenset] = SpanGroup.ALL_GROUPS | frozenset(
        [
            SERVER,
            HTTP_CLIENT,
            ROLLOUT,
            VERIFY,
            AGENT,
            MODEL_CALL,
            SANDBOX,
        ]
    )

    #: The groups that make one rollout appear as one trace across Gym's server
    #: processes. Every preset is a superset of this.
    CROSS_PROCESS_SPINE: Final[frozenset] = frozenset([SERVER, HTTP_CLIENT, ROLLOUT])

    _PRESETS: ClassVar[dict] = {
        "default": frozenset([SpanGroup.JOB]) | CROSS_PROCESS_SPINE,
        # NOTE: ``per_rollout`` deliberately omits ``job`` so each rollout is its own root
        # trace with a bounded span count. ``job`` wraps a whole eval run and lives in
        # ``default`` and ``all``.
        "per_rollout": frozenset([VERIFY, AGENT, MODEL_CALL]) | CROSS_PROCESS_SPINE,
        "all": ALL_GROUPS,
    }
