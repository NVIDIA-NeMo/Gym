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

"""The in-memory meter fixture shared by tests that assert on Gym metrics.

A plain module, not a test module: importing it skips nothing. Without the OTel SDK, only the tests that request the
fixture skip.
"""

import pytest

from nemo_gym.telemetry import gym_metrics as telemetry_metrics
from nemo_gym.telemetry import setup as telemetry_setup


@pytest.fixture
def collected_metrics(monkeypatch):
    """A live meter on an in-memory reader, installed as the process handle."""
    pytest.importorskip("opentelemetry.sdk.metrics")
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])

    class _Handle:
        is_exporting = True
        meter = provider.get_meter("test")

    monkeypatch.setattr(telemetry_setup, "_TELEMETRY_HANDLE", _Handle())
    telemetry_metrics._reset_for_testing()

    def collect():
        data = reader.get_metrics_data()
        out = {}
        # The reader hands back None when no instrument has recorded yet.
        for resource_metric in data.resource_metrics if data is not None else ():
            for scope_metric in resource_metric.scope_metrics:
                for metric in scope_metric.metrics:
                    out[metric.name] = list(metric.data.data_points)
        return out

    return collect
