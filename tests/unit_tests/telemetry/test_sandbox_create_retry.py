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

"""The create-retry counter, recorded from each retrying provider's own retry callback."""

from types import SimpleNamespace

import pytest

from nemo_gym.telemetry import gym_metrics
from tests.unit_tests.telemetry.test_sandbox_active import collected_metrics  # noqa: F401 - fixture


pytest.importorskip("opentelemetry.sdk.metrics")

PROVIDER = gym_metrics.SANDBOX_PROVIDER_ATTRIBUTE
RETRY = gym_metrics.SANDBOX_CREATE_RETRY_INSTRUMENT


def _points(collected, name):
    return collected().get(name, [])


def _retry_state(attempt: int = 1):
    return SimpleNamespace(
        attempt_number=attempt,
        outcome=SimpleNamespace(exception=lambda: RuntimeError("boom")),
        next_action=SimpleNamespace(sleep=1.5),
    )


@pytest.mark.parametrize(
    ("module", "provider"),
    [
        ("nemo_gym.sandbox.providers.opensandbox.provider", "opensandbox"),
        ("nemo_gym.sandbox.providers.daytona.provider", "daytona"),
    ],
)
def test_provider_create_retry_callbacks_count_by_provider(collected_metrics, monkeypatch, module, provider):  # noqa: F811
    mod = pytest.importorskip(module)
    monkeypatch.setattr(mod, "is_span_group_enabled", lambda group: True)
    mod._log_create_retry(_retry_state(1))
    mod._log_create_retry(_retry_state(2))

    (point,) = _points(collected_metrics, RETRY)
    assert point.attributes == {PROVIDER: provider}
    assert point.value == 2


def test_provider_create_retry_is_silent_with_the_group_off(collected_metrics, monkeypatch):  # noqa: F811
    mod = pytest.importorskip("nemo_gym.sandbox.providers.opensandbox.provider")
    monkeypatch.setattr(mod, "is_span_group_enabled", lambda group: False)
    mod._log_create_retry(_retry_state())
    assert RETRY not in collected_metrics()
