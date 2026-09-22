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

from collections.abc import Mapping
from functools import wraps
from typing import Any, Callable, Optional

from nemo_gym.rollout_correlation import current_rollout_id, decode_rollout_id
from nemo_gym.telemetry._fallbacks import is_span_group_enabled, managed_span, safe_set_span_attributes
from nemo_gym.telemetry.cpu import sample_cpu_percent, sample_process_tree_cpu_percent
from nemo_gym.telemetry.gpu import last_gpu_utilization_percent
from nemo_gym.telemetry.gym_metrics import (
    record_host_memory_total_mib,
    record_host_memory_used_mib,
    record_process_cpu_percent,
    record_process_tree_cpu_percent,
    record_process_tree_memory_used_mib,
)
from nemo_gym.telemetry.memory import sample_host_memory_mib, sample_process_tree_memory_mib
from nemo_gym.telemetry.setup import (
    cpu_min_resample_interval_s,
    current_run_id,
    is_cpu_sampling_enabled,
    is_gpu_sampling_enabled,
    is_memory_sampling_enabled,
    memory_min_resample_interval_s,
)


#: Span attribute carrying Gym's existing rollout correlation id.
ROLLOUT_ID_ATTRIBUTE = "nemo.gym.rollout.id"

#: Span attributes carrying the rest of the Gap-C correlation chain. `RUN_ID_ATTRIBUTE` is
#: this run's fleet-wide id (`telemetry.setup.current_run_id`); `TASK_ID_ATTRIBUTE` /
#: `REPEAT_INDEX_ATTRIBUTE` are decoded from the rollout id itself
#: (`rollout_correlation.decode_rollout_id`) rather than threaded as extra ContextVar state,
#: since the encoding already crosses every process boundary as part of `rollout_id`.
RUN_ID_ATTRIBUTE = "nemo.gym.run.id"
TASK_ID_ATTRIBUTE = "nemo.gym.task.id"
REPEAT_INDEX_ATTRIBUTE = "nemo.gym.repeat.index"

#: Span attribute carrying a CPU-utilization-at-span-end reading. See
#: `nemo_gym.telemetry.cpu` for why this is sampled inline here rather than by a
#: decoupled background sampler (exemplar linkage needs the active span context).
CPU_PERCENT_ATTRIBUTE = "nemo.gym.cpu.percent"

#: Span attribute carrying the process-tree CPU reading (this process plus every child
#: process) alongside `CPU_PERCENT_ATTRIBUTE` (this process alone) -- see
#: `nemo_gym.telemetry.cpu.sample_process_tree_cpu_percent`.
PROCESS_TREE_CPU_PERCENT_ATTRIBUTE = "nemo.gym.process_tree.cpu.percent"

#: Span attributes carrying a host-memory-at-span-end reading. Same inline-sampling
#: reasoning as CPU (see `nemo_gym.telemetry.memory`) -- host-wide, not process-scoped.
MEMORY_USED_MIB_ATTRIBUTE = "nemo.gym.host.memory_used_mib"
MEMORY_TOTAL_MIB_ATTRIBUTE = "nemo.gym.host.memory_total_mib"

#: Span attribute carrying the process-tree RSS reading, alongside the host-wide pair
#: above -- see `nemo_gym.telemetry.memory.sample_process_tree_memory_mib`.
PROCESS_TREE_MEMORY_USED_MIB_ATTRIBUTE = "nemo.gym.process_tree.memory_used_mib"

#: Span attribute carrying the background GPU sampler's most recent reading (summed
#: across every visible GPU) -- see `nemo_gym.telemetry.gpu.last_gpu_utilization_percent`
#: for why this reads a cache rather than forcing a fresh `nvidia-smi` call per span, and
#: for the staleness/shared-tenant caveats this figure inherits. Existing purely so
#: `nemo_gym.pareto_analysis` has a per-rollout GPU figure to sum, the GPU half of the
#: CPU-seconds join it already does.
GPU_UTILIZATION_PERCENT_ATTRIBUTE = "nemo.gym.gpu.utilization_percent"


def traced_endpoint(
    group: str,
    span_name: str,
    handler: Callable,
    static_attributes: Optional[dict] = None,
    response_attributes: Optional[Callable[[Any], Optional[dict]]] = None,
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
        response_attributes: Optional callback given the handler's return value, producing
            extra span attributes only knowable from the response (e.g. a model-generated
            request id). Called after the handler returns, inside the same `finally` block
            as every other attribute here; a raising or non-dict-returning callback is
            swallowed rather than failing the request.

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
            result = None
            try:
                result = await handler(*args, **kwargs)
                return result
            finally:
                # Attributes are set here, after the handler returns, rather than
                # before — so a CPU reading (added below) reflects span-end, not
                # span-start. Moved intentionally; this used to run before the handler
                # call, which is why `attributes` construction lives in a `finally`
                # around it now instead of ahead of it.
                if span is not None:
                    attributes = dict(base_attributes)
                    run_id = current_run_id()
                    if run_id:
                        attributes[RUN_ID_ATTRIBUTE] = run_id
                    rollout_id = current_rollout_id()
                    if rollout_id:
                        attributes[ROLLOUT_ID_ATTRIBUTE] = rollout_id
                        task_index, _, attempt_index = decode_rollout_id(rollout_id)
                        if task_index is not None:
                            attributes[TASK_ID_ATTRIBUTE] = task_index
                            attributes[REPEAT_INDEX_ATTRIBUTE] = attempt_index
                    if is_cpu_sampling_enabled():
                        cpu_percent = sample_cpu_percent(cpu_min_resample_interval_s())
                        if cpu_percent is not None:
                            attributes[CPU_PERCENT_ATTRIBUTE] = cpu_percent
                            record_process_cpu_percent(cpu_percent)  # still inside `with managed_span`
                        tree_cpu_percent = sample_process_tree_cpu_percent(cpu_min_resample_interval_s())
                        if tree_cpu_percent is not None:
                            attributes[PROCESS_TREE_CPU_PERCENT_ATTRIBUTE] = tree_cpu_percent
                            record_process_tree_cpu_percent(tree_cpu_percent)
                    if is_memory_sampling_enabled():
                        memory_reading = sample_host_memory_mib(memory_min_resample_interval_s())
                        if memory_reading is not None:
                            used_mib, total_mib = memory_reading
                            attributes[MEMORY_USED_MIB_ATTRIBUTE] = used_mib
                            attributes[MEMORY_TOTAL_MIB_ATTRIBUTE] = total_mib
                            record_host_memory_used_mib(used_mib)  # still inside `with managed_span`
                            record_host_memory_total_mib(total_mib)
                        tree_rss_mib = sample_process_tree_memory_mib(memory_min_resample_interval_s())
                        if tree_rss_mib is not None:
                            attributes[PROCESS_TREE_MEMORY_USED_MIB_ATTRIBUTE] = tree_rss_mib
                            record_process_tree_memory_used_mib(tree_rss_mib)
                    if is_gpu_sampling_enabled():
                        gpu_percent = last_gpu_utilization_percent()
                        if gpu_percent is not None:
                            attributes[GPU_UTILIZATION_PERCENT_ATTRIBUTE] = gpu_percent
                    if response_attributes is not None and result is not None:
                        try:
                            extra = response_attributes(result)
                        except Exception:
                            extra = None
                        if extra:
                            attributes.update(extra)
                    safe_set_span_attributes(span, attributes)

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


#: Span attribute carrying the model-generated request/response id (join key into
#: `TrajectoryModelCall.response_metadata.response_id`).
MODEL_REQUEST_ID_ATTRIBUTE = "nemo.gym.model_request.id"


def _model_request_id_attribute(result: Any) -> Optional[dict]:
    """Best-effort extraction of a model response's own id, for `MODEL_REQUEST_ID_ATTRIBUTE`.

    Handles both a dict-shaped response body and a Pydantic response object (chat
    completions, Responses API and Messages dialects all expose an `id` field, just as
    either an attribute or a mapping key depending on how the handler serializes its
    return value).
    """
    response_id = result.get("id") if isinstance(result, Mapping) else getattr(result, "id", None)
    return {MODEL_REQUEST_ID_ATTRIBUTE: response_id} if response_id else None


def traced_model_call_endpoint(
    handler: Callable, span_name: str, dialect: str, static_attributes: Optional[dict] = None
) -> Callable:
    """`traced_endpoint` for one model-server dialect route, plus `gym.model.call_duration_ms`.

    One model server registers three of these (`chat_completions`/`responses`/`messages`).
    `dialect` becomes an attribute on the duration histogram so the three are comparable
    against each other in a dashboard rather than collapsed into one undimensioned number
    — the same reasoning `gym.rollout.duration_ms` gets away with skipping, because a
    rollout is one comparable unit of work and a `/v1/messages` call is not a
    `/v1/responses` call.

    For a streaming response this measures "handler returned", not "stream fully
    drained" — a pre-existing limit of the underlying `gym.model.*` spans, not introduced
    here.
    """
    import time

    from nemo_gym.telemetry.gym_metrics import record_model_call_duration
    from nemo_gym.telemetry.span_groups import GymSpanGroup

    traced = traced_endpoint(
        GymSpanGroup.MODEL_CALL, span_name, handler, static_attributes, response_attributes=_model_request_id_attribute
    )
    server_name = (static_attributes or {}).get("nemo.gym.server.name")

    @wraps(handler)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not is_span_group_enabled(GymSpanGroup.MODEL_CALL):
            return await handler(*args, **kwargs)

        started = time.perf_counter()
        try:
            return await traced(*args, **kwargs)
        finally:
            record_model_call_duration(
                (time.perf_counter() - started) * 1000.0, dialect=dialect, server_name=server_name
            )

    return wrapper


#: Routes a resources server registers itself (see `SimpleResourcesServer.setup_webserver`)
#: plus the handful every `SimpleServer` exposes -- anything else on a resources server is
#: a tool route. A prefix set, not exact matches: `/mcp` covers `/mcp/...` sub-paths too.
_NON_TOOL_ROUTE_PREFIXES = (
    "/seed_session",
    "/verify",
    "/aggregate_metrics",
    "/reverify_mode",
    "/health",
    "/mcp",
    "/server_instances",
    "/global_config_dict_yaml",
    "/docs",
    "/openapi.json",
    "/redoc",
)


def _is_tool_route(path: str) -> bool:
    if not path or path == "/":
        return False
    return not any(path == prefix or path.startswith(f"{prefix}/") for prefix in _NON_TOOL_ROUTE_PREFIXES)


class ToolCallTelemetryMiddleware:
    """Per-tool-name call count/duration for a resources server's tool routes.

    Raw ASGI (like `RolloutContextMiddleware`), not `traced_endpoint`: tool routes are
    registered dynamically by each resources server's own `app.py`, so there is no single
    handler-wrapping call site the way `/verify`/`/run`/the model dialects have one each
    in their respective base classes. A middleware is the one place that sees every route
    a resources server exposes without touching any of their ~150 individual `app.py`
    files.

    Measures "the ASGI app call returned", the same "handler returned, not response fully
    flushed" approximation `traced_model_call_endpoint` already accepts for streaming --
    consistent with the rest of this module rather than a new precision standard.
    """

    def __init__(self, app: Any, *, server_name: Optional[str] = None) -> None:
        self._app = app
        self._server_name = server_name

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        from nemo_gym.telemetry.span_groups import GymSpanGroup

        path = scope.get("path", "") if scope.get("type") == "http" else ""
        if scope.get("type") != "http" or not _is_tool_route(path) or not is_span_group_enabled(GymSpanGroup.TOOL_CALL):
            await self._app(scope, receive, send)
            return

        import time

        from nemo_gym.telemetry.gym_metrics import (
            record_tool_call_cpu_percent,
            record_tool_call_duration,
            record_tool_call_memory_used_mib,
        )

        tool_name = path.lstrip("/")
        started = time.perf_counter()
        with managed_span(
            GymSpanGroup.TOOL_CALL,
            "gym.tool_call",
            **{"nemo.gym.tool.name": tool_name, "nemo.gym.server.name": self._server_name or ""},
        ) as span:
            # Read after RolloutContextMiddleware has already run (it is added after this
            # middleware in `SimpleResourcesServer.setup_webserver`, which makes it
            # outermost -- see that call site's comment), so the ContextVar is populated
            # by the time this middleware's body executes. Attached the same way
            # `traced_endpoint` attaches it, to close the "Tool and sandbox activity" link
            # in the run/rollout/task/.../tool/sandbox correlation chain.
            if span is not None:
                rollout_id = current_rollout_id()
                if rollout_id:
                    safe_set_span_attributes(span, {ROLLOUT_ID_ATTRIBUTE: rollout_id})
            try:
                await self._app(scope, receive, send)
            finally:
                record_tool_call_duration(
                    (time.perf_counter() - started) * 1000.0, tool_name=tool_name, server_name=self._server_name
                )
                # Sampled at span-close, same approximation scope as everywhere else
                # process-tree CPU/memory is read (see `record_tool_call_cpu_percent`'s
                # docstring) -- "how busy was the process during this tool call", not an
                # isolated per-call measurement.
                if is_cpu_sampling_enabled():
                    cpu_percent = sample_process_tree_cpu_percent(cpu_min_resample_interval_s())
                    if cpu_percent is not None:
                        if span is not None:
                            safe_set_span_attributes(span, {PROCESS_TREE_CPU_PERCENT_ATTRIBUTE: cpu_percent})
                        record_tool_call_cpu_percent(cpu_percent, tool_name=tool_name)
                if is_memory_sampling_enabled():
                    tree_rss_mib = sample_process_tree_memory_mib(memory_min_resample_interval_s())
                    if tree_rss_mib is not None:
                        if span is not None:
                            safe_set_span_attributes(span, {PROCESS_TREE_MEMORY_USED_MIB_ATTRIBUTE: tree_rss_mib})
                        record_tool_call_memory_used_mib(tree_rss_mib, tool_name=tool_name)


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
