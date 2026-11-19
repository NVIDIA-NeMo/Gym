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
"""Gym's boundary onto nemo-lens's propagation and framework-instrumentation helpers.

`nemo_gym.telemetry` is the only package that imports `nemo.lens`. Everything outside it
reaches telemetry through this package, so a reader never has to work out whether a given
`nemo.lens` import is reachable when the extra is not installed. The invariant is
enforced by `tests/unit_tests/telemetry/test_call_sites.py`.

`_fallbacks` covers the primitives used at instrumentation sites. This module covers the
two helpers that are not primitives: W3C context injection at Gym's HTTP egress, and
FastAPI auto-instrumentation at each server's startup. Both are called only on paths that
already established telemetry is active, so neither needs a no-op twin — but both are
defensive anyway, because telemetry must never be the reason a server fails to start or a
request fails to send.
"""

import logging
from typing import Any


logger = logging.getLogger(__name__)


def inject_trace_context(headers: dict) -> None:
    """Write the current span's W3C ``traceparent`` into *headers*.

    The egress half of cross-process propagation. A no-op when there is no recording span,
    because the propagator has no context to write.

    Call only behind a span-group gate: the import below is function-local so a disabled
    site never reaches it.
    """
    try:
        from nemo.lens.propagation import inject_context

        inject_context(headers)
    except Exception:
        # A request that cannot carry trace context is still a valid request.
        logger.debug("nemo-lens: could not inject trace context", exc_info=True)


def instrument_fastapi_app(app: Any) -> bool:
    """Apply OpenTelemetry FastAPI auto-instrumentation to *app*.

    The ingress half of cross-process propagation: it extracts an inbound ``traceparent``
    and parents this server's SERVER span to the caller's span.

    ``service_name`` is deliberately not passed. ``nemo.lens.contrib.fastapi``'s helper
    accepts it and never uses it at the pinned commit; the name comes from
    ``OTEL_SERVICE_NAME``, which ``init_telemetry`` resolves.

    Returns:
        True when instrumentation was applied. False when the optional instrumentation
        package is missing or the instrumentor refused the app — the server then runs
        without inbound context extraction rather than failing to start.
    """
    try:
        from nemo.lens.contrib.fastapi import instrument_fastapi

        instrument_fastapi(app)
        return True
    except Exception:
        logger.warning("nemo-lens: FastAPI instrumentation unavailable; continuing without it", exc_info=True)
        return False
