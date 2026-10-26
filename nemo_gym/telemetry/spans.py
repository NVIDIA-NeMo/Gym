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
"""Spans nemo-lens cannot produce: a CLIENT-kind span, a linked root for a new trace, and a child of an ended span.

``client_span``: ``nemo.lens.helpers.managed_span`` calls ``tracer.start_span(name)`` with no
``kind`` argument and exposes no way to pass one, so every span it creates is
``SpanKind.INTERNAL``. For spans *inside* a service that is right. For Gym's outbound calls it
is not: an INTERNAL span on a cross-service hop means a backend cannot tell that the agent
server called the model server, so Jaeger/Tempo/Honeycomb service maps lose the edge and
client-vs-server latency attribution stops working. Gym is pinned to nemo-lens ``4d045c03``,
so rather than change lens this module creates that one span directly against the OTel API.
Adding a ``kind`` parameter to ``managed_span`` is the proper fix; when it lands, delete
``client_span`` and pass ``kind=`` instead.

``linked_root_span``: starts a span as the root of a **new trace**, linked to the span it
splits from, when the current span is that span; otherwise an ordinary child span. nemo-lens
has ``create_linked_span`` for cross-rank correlation but nothing that also decides whether
to split and attaches the result as current. The eval driver uses it so that a run is one
small job trace plus one bounded trace per rollout.

``record_child_span``: emits a finished span under a parent given as a ``SpanContext``, which may
have ended already. The eval driver evaluates a rollout's health after ``gym.rollout.dispatch``
closed and puts the verdict on a child of it.

``client_span`` and ``linked_root_span`` are otherwise identical to ``managed_span``: context
attach/detach, exception recording, and an unconditional ``end()`` in ``finally``.

**Callers must gate on the span group themselves.** Nothing here checks it, so that a
disabled site does not pay for entering a context manager at all
(``kb/knowledge/conventions/hot-path-overhead.md``).
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from opentelemetry.trace import Span, SpanContext


@contextmanager
def client_span(name: str, tracer: Any = None, **attributes: Any):
    """Start a ``SpanKind.CLIENT`` span, attach it as current, and always end it.

    Args:
        name: Span name.
        tracer: OTel tracer; defaults to the globally registered one.
        **attributes: Set on the span via ``safe_set_span_attributes``, so non-scalar
            values are dropped and sensitive keys are redacted rather than exported.

    Yields:
        The active span. Only call this behind a span-group gate.
    """
    from opentelemetry import context as otel_ctx
    from opentelemetry import trace as otel_trace
    from opentelemetry.trace import SpanKind, StatusCode, set_span_in_context

    from nemo_gym.telemetry._fallbacks import safe_set_span_attributes

    if tracer is None:
        tracer = otel_trace.get_tracer("nemo.gym")

    span = tracer.start_span(name, kind=SpanKind.CLIENT)
    if attributes:
        safe_set_span_attributes(span, attributes)
    token = otel_ctx.attach(set_span_in_context(span))
    try:
        yield span
    except Exception as exc:
        span.record_exception(exc)
        span.set_status(StatusCode.ERROR, str(exc))
        raise
    finally:
        otel_ctx.detach(token)
        span.end()


@contextmanager
def linked_root_span(
    name: str, tracer: Any = None, *, split_from: "SpanContext | None" = None, **attributes: Any
) -> "Iterator[Span]":
    """Start *name* as the root of a new trace, linked to *split_from*, when that span is current.

    Under the run-long ``gym.job`` span, parent/child nesting would put every rollout of a run
    into one trace whose size grows with the run and which stays open until the job ends;
    backends cap and buffer traces as units, so long runs lose spans. Splitting gives each
    rollout a trace bounded by its own work, and the span link keeps the job ↔ rollout relation.

    The split happens only when the current span **is** *split_from*, the ``SpanContext`` of a
    ``gym.job`` span the driver opened as a standalone root. In every other case, an embedder's
    span, no owned job, a remote non-recording context, or a child of the job being current,
    the span nests as an ordinary child, so the caller's trace keeps its shape.

    Args:
        name: Span name.
        tracer: OTel tracer; defaults to the globally registered one.
        split_from: ``SpanContext`` of the driver-owned job span, or None when no such span exists.
        **attributes: Set via ``safe_set_span_attributes``.

    Yields:
        The active span. Only call this behind a span-group gate.
    """
    from opentelemetry import context as otel_ctx
    from opentelemetry import trace as otel_trace
    from opentelemetry.trace import Link, StatusCode, set_span_in_context

    from nemo_gym.telemetry._fallbacks import safe_set_span_attributes

    if tracer is None:
        tracer = otel_trace.get_tracer("nemo.gym")

    current_ctx = otel_trace.get_current_span().get_span_context()
    start_new_trace = split_from is not None and current_ctx.is_valid and current_ctx == split_from
    if start_new_trace:
        span = tracer.start_span(name, context=otel_ctx.Context(), links=[Link(current_ctx)])
    else:
        span = tracer.start_span(name)
    if attributes:
        safe_set_span_attributes(span, attributes)
    token = otel_ctx.attach(set_span_in_context(span))
    try:
        yield span
    except Exception as exc:
        span.record_exception(exc)
        span.set_status(StatusCode.ERROR, str(exc))
        raise
    finally:
        otel_ctx.detach(token)
        span.end()


def record_child_span(name: str, parent: "SpanContext", tracer: Any = None, **attributes: Any) -> None:
    """Emit *name* as a finished child of *parent*, which may have ended. Gate on the span group first."""
    from opentelemetry import trace as otel_trace
    from opentelemetry.trace import NonRecordingSpan, set_span_in_context

    from nemo_gym.telemetry._fallbacks import safe_set_span_attributes

    if tracer is None:
        tracer = otel_trace.get_tracer("nemo.gym")

    span = tracer.start_span(name, context=set_span_in_context(NonRecordingSpan(parent)))
    safe_set_span_attributes(span, attributes)
    span.end()
