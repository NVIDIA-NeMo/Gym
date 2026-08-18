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
"""Schema for the ``telemetry:`` block of a NeMo Gym config.

``nemo_gym.telemetry.setup`` translates this block into ``NEMO_GYM_OTEL_*`` environment
variables in the orchestrating CLI process *before* it spawns the server processes, so
every server, and every Ray worker under them, inherits one consistent configuration.
Raw ``NEMO_GYM_OTEL_*`` / ``NEMO_LENS_*`` / ``OTEL_*`` env vars always win over these
YAML values: the translation uses ``os.environ.setdefault``.

This module imports only pydantic. It never requires nemo-lens, so it is safe to import
unconditionally from Gym's config machinery.
"""

from typing import Optional

from pydantic import BaseModel


class TelemetryConfig(BaseModel, extra="allow"):
    """OpenTelemetry / nemo-lens configuration for NeMo Gym.

    Telemetry is optional twice over: it activates only when ``enabled`` is true *and*
    nemo-lens is installed (``uv sync --extra telemetry``). When either is missing every
    instrumentation site in ``nemo_gym`` degrades to a ~0-cost no-op and Gym behaves
    exactly as it does today.
    """

    enabled: bool = False
    """Master switch. When false, all instrumentation is a ~0-cost no-op."""

    service_name: str = "nemo-gym"
    """``service.name`` reported to the backend.

    Passed through the standard ``OTEL_SERVICE_NAME`` env var, not through
    ``instrument_fastapi(service_name=...)`` — that parameter is accepted and ignored by
    nemo-lens at the pinned commit (``contrib/fastapi.py``).

    Each server process appends its own Gym server name, so a run yields
    ``nemo-gym/example_single_tool_call``, ``nemo-gym/policy_model``, and so on rather
    than three indistinguishable ``nemo-gym`` services. Set ``service_name_per_server``
    to false to opt out.
    """

    service_name_per_server: bool = True
    """Suffix ``service_name`` with each server's Gym config name.

    On by default because Gym runs several processes that would otherwise share one
    ``service.name``, which makes a backend's service map useless. The unsuffixed name
    stays available as the ``nemo.gym.service_group`` resource attribute."""

    span_groups: str = "default"
    """Span-group spec: a preset (``default`` | ``per_rollout`` | ``all``) or a
    comma-separated list of group names (e.g. ``"default,sandbox"``). See
    :class:`~nemo_gym.telemetry.span_groups.GymSpanGroup`."""

    export_strategy: str = "all_ranks"
    """Which processes export: ``all_ranks`` | ``single_rank`` | ``sampled`` |
    ``first_rank_per_node``.

    Defaults to ``all_ranks``, unlike NeMo-RL and Megatron-LM which default to
    ``single_rank``. Those run one process tree per job where rank 0 sees a
    representative slice; Gym runs N independent server processes, each rank 0 of its own
    world. Silencing any of them puts a hole in the middle of every distributed trace."""

    export_rank: int = -1
    """For ``single_rank``: which rank exports (``-1`` = last rank)."""

    traces_enabled: bool = True
    """Emit trace spans."""

    metrics_enabled: bool = True
    """Emit metric instruments (the ``gym.*`` histograms/gauges plus the FastAPI
    instrumentor's ``http.server.*``)."""

    logs_enabled: bool = False
    """Bridge Python logging to OTel logs, exported with trace correlation."""

    exporter: str = "otlp"
    """Exporter backend: ``otlp`` | ``console``. The OTLP endpoint, headers and protocol
    come from the standard ``OTEL_EXPORTER_OTLP_*`` env vars, so any OTLP-compatible
    backend or an OpenTelemetry Collector works. ``console`` writes one JSON object per
    line to stdout, which is what the local validation flow greps."""

    instrument_aiohttp: bool = True
    """Use nemo-lens's aiohttp auto-instrumentation for outbound calls, which produces a
    CLIENT span per request in addition to injecting ``traceparent``.

    When false, ``nemo_gym.server_utils.request`` still injects ``traceparent`` manually,
    so cross-process traces stay joined — you lose the client-side span, not the trace.
    Turn it off if the per-request patching cost matters at very high concurrency."""

    run_id: Optional[str] = None
    """Correlates every process of one ``gym env start`` / ``gym env test`` invocation.
    Generated in the orchestrator and inherited by the servers when unset."""
