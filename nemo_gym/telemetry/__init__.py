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
"""NeMo Gym telemetry: optional OpenTelemetry instrumentation via nemo-lens.

Importing this package never requires nemo-lens. Install it with
``uv sync --extra telemetry`` and switch it on with a ``telemetry:`` block or
``NEMO_LENS_ENABLED=1``; with either missing, every instrumentation site in ``nemo_gym``
is a ~0-cost no-op.

Public surface
--------------
* :class:`~nemo_gym.telemetry.config.TelemetryConfig` — the ``telemetry:`` config block.
* :class:`~nemo_gym.telemetry.span_groups.GymSpanGroup` — Gym span groups and presets.
* :func:`~nemo_gym.telemetry.setup.configure_telemetry_env` — orchestrator side; hands
  the settings to spawned server processes through the environment.
* :func:`~nemo_gym.telemetry.setup.init_telemetry` /
  :func:`~nemo_gym.telemetry.setup.get_telemetry` /
  :func:`~nemo_gym.telemetry.setup.shutdown_telemetry` — per-process lifecycle.

Instrumentation primitives (``managed_span`` / ``span_cm`` / ``trace_fn`` /
``is_span_group_enabled`` / ``safe_set_span_attributes``) come from
:mod:`nemo_gym.telemetry._fallbacks`, which resolves to the real nemo-lens
implementations when it is installed and to no-op stubs when it is not.

See ``nemo_gym/telemetry/README.md`` for the design, and
``fern/versions/latest/pages/observability/`` for user documentation.
"""

from nemo_gym.telemetry._fallbacks import (
    is_span_group_enabled,
    managed_span,
    safe_set_span_attributes,
    span_cm,
    trace_fn,
)
from nemo_gym.telemetry.config import TelemetryConfig
from nemo_gym.telemetry.setup import (
    TELEMETRY_KEY_NAME,
    configure_telemetry_env,
    get_telemetry,
    init_telemetry,
    is_telemetry_env_enabled,
    shutdown_telemetry,
    telemetry_config_from_global_config,
)
from nemo_gym.telemetry.span_groups import GymSpanGroup


__all__ = [
    "TELEMETRY_KEY_NAME",
    "GymSpanGroup",
    "TelemetryConfig",
    "configure_telemetry_env",
    "get_telemetry",
    "init_telemetry",
    "is_span_group_enabled",
    "is_telemetry_env_enabled",
    "managed_span",
    "safe_set_span_attributes",
    "shutdown_telemetry",
    "span_cm",
    "telemetry_config_from_global_config",
    "trace_fn",
]
