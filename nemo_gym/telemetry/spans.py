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
"""A CLIENT-kind span, which nemo-lens cannot currently produce.

``nemo.lens.helpers.managed_span`` calls ``tracer.start_span(name)`` with no ``kind``
argument and exposes no way to pass one, so every span it creates is ``SpanKind.INTERNAL``.
For spans *inside* a service that is right. For Gym's outbound calls it is not: an
INTERNAL span on a cross-service hop means a backend cannot tell that the agent server
called the model server, so Jaeger/Tempo/Honeycomb service maps lose the edge and
client-vs-server latency attribution stops working.

Gym is pinned to nemo-lens ``b85578fc``, so rather than change lens this module creates
that one span directly against the OTel API. Adding a ``kind`` parameter to
``managed_span`` is the proper fix and is raised as a decision in the PR body; when it
lands, delete this module and pass ``kind=`` instead.

Everything else is deliberately identical to ``managed_span``: context attach/detach,
exception recording, and an unconditional ``end()`` in ``finally``.

**Callers must gate on the span group themselves.** Nothing here checks it, so that a
disabled site does not pay for entering a context manager at all
(``kb/knowledge/conventions/hot-path-overhead.md``).
"""

from contextlib import contextmanager
from typing import Any


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
