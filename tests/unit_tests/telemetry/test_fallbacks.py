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
"""The nemo-lens-absent path: Gym's no-op stand-ins, and binding the real helpers otherwise.

Installs without the ``telemetry`` extra have no nemo-lens, so every instrumentation site
runs against the stand-ins in ``nemo_gym/telemetry/_fallbacks.py``. That configuration is
the one nobody runs by accident; these tests make it routine.
"""

import pytest

from tests.unit_tests.telemetry.conftest import import_without_lens


@pytest.fixture(scope="module")
def gym_shim_without_lens():
    """Gym's ``_fallbacks`` module imported with nemo-lens unavailable (the no-op branch)."""
    return import_without_lens("nemo_gym.telemetry._fallbacks")


def test_telemetry_package_imports_without_lens():
    """Importing nemo_gym.telemetry must not require nemo-lens."""
    module = import_without_lens("nemo_gym.telemetry._fallbacks")
    assert module.NEMO_LENS_AVAILABLE is False


def test_no_op_primitives_are_inert(gym_shim_without_lens):
    """Every shim behaves as a no-op, not merely as an importable name."""
    shim = gym_shim_without_lens

    assert shim.is_span_group_enabled("server") is False
    assert shim.is_span_group_enabled("anything-at-all") is False

    # Called the way Gym's call sites call them: dotted attribute names as keywords.
    attributes = {"nemo.gym.sandbox.provider": "test"}
    with shim.managed_span("server", "gym.test", **attributes) as span:
        assert span is None
    with shim.span_cm("gym.test", **attributes) as span:
        assert span is None

    calls = []

    @shim.trace_fn("server", "gym.test")
    def instrumented(value):
        calls.append(value)
        return value * 2

    assert instrumented(21) == 42, "trace_fn must return the function's own result"
    assert calls == [21]

    # Must tolerate a None span rather than raising, since that is what managed_span yields.
    assert shim.safe_set_span_attributes(None, {"a": 1}) is None


def test_managed_span_propagates_exceptions(gym_shim_without_lens):
    """The no-op context manager must not swallow the body's exception."""
    with pytest.raises(ValueError, match="boom"):
        with gym_shim_without_lens.managed_span("server", "gym.test"):
            raise ValueError("boom")


def test_real_lens_is_used_when_installed():
    """With lens installed, Gym binds the real helpers — not the no-ops.

    This is the failure NeMo-RL's ``_fallbacks.py`` shape invites: re-exporting
    ``nemo.lens.fallbacks`` means a call site that imports from ``_fallbacks`` gets a
    permanent no-op even with lens present.
    """
    pytest.importorskip("nemo.lens")
    from nemo_gym.telemetry import _fallbacks

    assert _fallbacks.NEMO_LENS_AVAILABLE is True
    assert _fallbacks.managed_span.__module__ == "nemo.lens.helpers"
    assert _fallbacks.is_span_group_enabled.__module__ == "nemo.lens.state"
    assert _fallbacks.safe_set_span_attributes.__module__ == "nemo.lens.helpers"
