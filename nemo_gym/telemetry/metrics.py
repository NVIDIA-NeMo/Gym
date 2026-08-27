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
"""Gym-side wrapper over nemo-lens's ``gym.*`` metric instruments.

``nemo.lens.instruments.gym.record_gym_metrics`` records **without attributes** at the
pinned commit — every instrument is undimensioned. That is not a detail that can be
papered over, so this module takes an explicit position on each of the five:

``gym.rollout.duration_ms`` (histogram)
    Used. One rollout is one comparable unit of work, so an undimensioned distribution is
    still meaningful. :func:`record_rollout_duration`.

``gym.verify.duration_ms`` (histogram)
    Used. Same reasoning, per verification. :func:`record_verify`.

``gym.verify.success_rate`` (gauge)
    Used, with a stated window. A gauge is last-value, so it cannot express "rate" on its
    own; this module keeps a process-local running tally and sets the gauge to the
    **cumulative** success fraction since process start. That is a well-defined number,
    but it is not a windowed rate, and it flattens as a process ages — see the module
    docs. Counters would be the right instrument; that needs a lens change.

``gym.servers.active`` (gauge)
    Used from **exactly one process**. A gauge is last-value semantics, so if every
    server process set it the exported value would be whichever process happened to write
    last — a meaningless number that looks like a real one. :func:`record_active_servers`
    is orchestrator-only and refuses to run anywhere else.

``gym.server.request_duration_ms`` (histogram)
    **Deliberately unused.** With no attributes it would collapse every endpoint of every
    server type into one histogram: ``/verify`` on a resources server, ``/v1/responses``
    on a model server and a liveness probe would land in the same bucket set, and the
    result answers no question anyone has. The FastAPI auto-instrumentation Gym enables
    already emits ``http.server.request.duration`` dimensioned by ``http.route``,
    ``http.request.method`` and ``http.response.status_code``, which is strictly better.
    Use that instead; see ``fern/versions/latest/pages/observability/metrics.mdx``.

Every function here is a no-op unless telemetry is initialised *and* exporting, so call
sites do not need their own guards for correctness — though they should still sit under a
span-group gate to stay free when disabled.
"""

import logging
import threading


logger = logging.getLogger(__name__)

#: Cumulative verification tally backing ``gym.verify.success_rate``. Process-local: each
#: server process reports its own fraction, which is the correct scope given the gauge
#: carries no attribute to distinguish them.
_VERIFY_LOCK = threading.Lock()
_VERIFY_TOTAL = 0
_VERIFY_SUCCEEDED = 0


def _record(**kwargs) -> None:
    """Forward to ``record_gym_metrics`` when a meter is available; never raise."""
    from nemo_gym.telemetry.setup import get_telemetry

    telemetry = get_telemetry()
    if telemetry is None or not telemetry.is_exporting:
        return
    try:
        from nemo.lens.instruments.gym import record_gym_metrics
    except ImportError:  # pragma: no cover - unreachable while a handle exists
        return
    try:
        record_gym_metrics(telemetry.meter, **kwargs)
    except Exception:
        logger.debug("nemo-lens: failed to record Gym metrics", exc_info=True)


def record_rollout_duration(duration_ms: float) -> None:
    """Record one rollout's wall-clock duration into ``gym.rollout.duration_ms``."""
    _record(rollout_duration_ms=duration_ms)


def record_verify(duration_ms: float, succeeded: bool) -> None:
    """Record one verification's duration and fold it into the success rate.

    Sets ``gym.verify.success_rate`` to the cumulative fraction of successful
    verifications in this process since start — not a windowed rate. A long-lived server
    therefore shows a figure that moves more slowly over time; read it as "this process's
    success fraction so far", and use the trace data for anything finer.
    """
    global _VERIFY_TOTAL, _VERIFY_SUCCEEDED
    with _VERIFY_LOCK:
        _VERIFY_TOTAL += 1
        if succeeded:
            _VERIFY_SUCCEEDED += 1
        rate = _VERIFY_SUCCEEDED / _VERIFY_TOTAL
    _record(verify_duration_ms=duration_ms, verify_success_rate=rate)


def record_active_servers(count: int) -> None:
    """Set ``gym.servers.active`` — orchestrator only.

    ``gym.servers.active`` is a gauge, so the exported value is whatever was written last.
    Calling this from more than one process produces a number that is not the fleet size,
    not any process's view of it, and impossible to interpret after the fact. The
    orchestrator is the only process that knows the fleet size, so it is the only caller.
    """
    _record(active_servers=count)


def _reset_verify_tally_for_testing() -> None:
    """Reset the cumulative verify tally. Test-only."""
    global _VERIFY_TOTAL, _VERIFY_SUCCEEDED
    with _VERIFY_LOCK:
        _VERIFY_TOTAL = 0
        _VERIFY_SUCCEEDED = 0
