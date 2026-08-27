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
"""Span wrappers for Gym's endpoint handlers.

Applied in the three `SimpleServer` subclasses where routes are registered, so every one
of Gym's ~150 servers is instrumented without touching any of them individually.

Why a wrapper and not the FastAPI auto-instrumentation alone: the instrumentor gives one
SERVER span per HTTP request, named after the route. That is the right thing for the
transport, but it cannot know that `/run` is a rollout and `/verify` is a verification, it
cannot be switched on per span group, and it cannot attach Gym's rollout id. These
wrappers add the semantic layer on top.

The rollout id comes from `nemo_gym.rollout_correlation.current_rollout_id`, the
ContextVar Gym already sets from `RolloutContextMiddleware` and the agent's `/run`
wrapper. It is bridged onto the span rather than replaced: one correlation scheme, now
visible from traces, Gym's own logs, and captured trajectories alike.
"""

from functools import wraps
from typing import Any, Callable, Optional

from nemo_gym.rollout_correlation import current_rollout_id
from nemo_gym.telemetry._fallbacks import is_span_group_enabled, managed_span, safe_set_span_attributes


#: Span attribute carrying Gym's existing rollout correlation id.
ROLLOUT_ID_ATTRIBUTE = "nemo.gym.rollout.id"


def traced_endpoint(
    group: str,
    span_name: str,
    handler: Callable,
    static_attributes: Optional[dict] = None,
) -> Callable:
    """Wrap an async FastAPI handler in a span-group-gated span.

    `functools.wraps` sets `__wrapped__`, which is what FastAPI's `inspect.signature`
    follows to build the request model — so the route keeps its body type, its validation
    and its OpenAPI schema. Gym already relies on this for
    `SimpleResponsesAPIAgent.run_with_rollout_context`.

    Args:
        group: Span group gating this site. Checked at **call** time, not decoration time:
            span groups are configured during `init_telemetry`, long after import.
        span_name: Span name, e.g. `gym.verify`.
        handler: The async handler to wrap.
        static_attributes: Attributes constant for this route, e.g. the server name.
            Evaluated once at wrap time, not per request.

    Returns:
        The wrapped handler.
    """
    base_attributes = dict(static_attributes) if static_attributes else {}

    @wraps(handler)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Gate first, with nothing above it — not even building the attribute dict.
        # kb/knowledge/conventions/hot-path-overhead.md.
        if not is_span_group_enabled(group):
            return await handler(*args, **kwargs)

        with managed_span(group, span_name) as span:
            if span is not None:
                attributes = dict(base_attributes)
                rollout_id = current_rollout_id()
                if rollout_id:
                    attributes[ROLLOUT_ID_ATTRIBUTE] = rollout_id
                safe_set_span_attributes(span, attributes)
            return await handler(*args, **kwargs)

    return wrapper


def traced_verify_endpoint(handler: Callable, static_attributes: Optional[dict] = None) -> Callable:
    """`traced_endpoint` for `/verify`, plus the `gym.verify.*` metrics.

    `succeeded` records whether the **verification call completed**, not whether the task
    passed. Reward and accuracy are experiment telemetry and belong in W&B, not in an
    application-telemetry metric — see
    `kb/knowledge/concepts/application-vs-experiment-telemetry.md`. A verifier that
    correctly scores an answer as wrong is a success here; a verifier that raises is not.
    """
    import time

    from nemo_gym.telemetry.metrics import record_verify
    from nemo_gym.telemetry.span_groups import GymSpanGroup

    traced = traced_endpoint(GymSpanGroup.VERIFY, "gym.verify", handler, static_attributes)

    @wraps(handler)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not is_span_group_enabled(GymSpanGroup.VERIFY):
            return await handler(*args, **kwargs)

        started = time.perf_counter()
        succeeded = False
        try:
            result = await traced(*args, **kwargs)
            succeeded = True
            return result
        finally:
            record_verify((time.perf_counter() - started) * 1000.0, succeeded=succeeded)

    return wrapper


def traced_rollout_endpoint(handler: Callable, static_attributes: Optional[dict] = None) -> Callable:
    """`traced_endpoint` for the agent's `/run`, plus `gym.rollout.duration_ms`.

    One `/run` is one rollout, which makes this the span everything else in a rollout
    hangs off — the model calls and verifications it triggers become its descendants
    through W3C context propagation.
    """
    import time

    from nemo_gym.telemetry.metrics import record_rollout_duration
    from nemo_gym.telemetry.span_groups import GymSpanGroup

    traced = traced_endpoint(GymSpanGroup.ROLLOUT, "gym.rollout", handler, static_attributes)

    @wraps(handler)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not is_span_group_enabled(GymSpanGroup.ROLLOUT):
            return await handler(*args, **kwargs)

        started = time.perf_counter()
        try:
            return await traced(*args, **kwargs)
        finally:
            record_rollout_duration((time.perf_counter() - started) * 1000.0)

    return wrapper
