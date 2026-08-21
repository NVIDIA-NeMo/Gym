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
"""Cost of a disabled instrumentation site.

`kb/knowledge/conventions/hot-path-overhead.md`: the span-group gate comes first and
nothing — including imports — runs above it. Gym serves at 16k+ concurrency, so
`server_utils.request` and every per-request handler is a hot path, and a user who turned
telemetry off must not pay for it.

Absolute nanosecond figures are machine-specific and are NOT asserted on. Every assertion
here is a **ratio between two timings taken in the same process on the same run**, which
is the only reproducible form (see the convention's "compare deltas, never absolutes").
"""

import timeit
from contextlib import nullcontext

import pytest

from nemo_gym.telemetry._fallbacks import is_span_group_enabled, managed_span
from nemo_gym.telemetry.span_groups import GymSpanGroup


#: Wide enough that ordinary scheduler noise on a loaded machine does not flake the test,
#: tight enough to catch the regression this convention exists for: PR lens-23 moved a
#: disabled site from 166ns to 378ns (2.3x) by putting an import above the gate.
MAX_RATIO_VS_NULLCONTEXT = 4.0

_NUMBER = 50_000
_REPEAT = 5


def _min_ns(stmt, globals_dict):
    """Minimum per-call nanoseconds over `_REPEAT` runs.

    Minimum rather than mean: it is the run least disturbed by other processes.
    """
    timings = timeit.repeat(stmt, number=_NUMBER, repeat=_REPEAT, globals=globals_dict)
    return min(timings) / _NUMBER * 1e9


@pytest.fixture(autouse=True)
def telemetry_is_off():
    """Every timing below is of the *disabled* path, which is the one that must be free."""
    assert is_span_group_enabled(GymSpanGroup.SANDBOX) is False


def test_disabled_gate_is_not_measurably_worse_than_a_bare_call():
    """The gate itself is a frozenset membership test and nothing else."""
    ns_gate = _min_ns("is_span_group_enabled('sandbox')", {"is_span_group_enabled": is_span_group_enabled})
    ns_baseline = _min_ns("noop()", {"noop": lambda: None})

    print(f"\ndisabled gate: {ns_gate:.1f} ns   bare call: {ns_baseline:.1f} ns")
    assert ns_gate < ns_baseline * MAX_RATIO_VS_NULLCONTEXT, (
        f"the span-group gate costs {ns_gate / ns_baseline:.1f}x a bare call "
        f"({ns_gate:.1f} ns vs {ns_baseline:.1f} ns) — something is running above it"
    )


def test_the_gym_call_site_shape_is_free_when_disabled():
    """What a disabled Gym instrumentation site actually costs.

    Gym's sites gate with `is_span_group_enabled` *before* entering `managed_span`, so
    the disabled path is the gate and nothing else. This is the number that matters for a
    user who turned telemetry off; `test_managed_span_alone_is_not_free` below is why the
    site is written that way rather than leaning on managed_span's internal gate.
    """
    ns_site = _min_ns(
        "if is_span_group_enabled('sandbox'):\n    with managed_span('sandbox', 'gym.test'): pass",
        {"is_span_group_enabled": is_span_group_enabled, "managed_span": managed_span},
    )
    ns_null = _min_ns("with nullcontext(): pass", {"nullcontext": nullcontext})

    print(f"\ngym site (disabled): {ns_site:.1f} ns   nullcontext: {ns_null:.1f} ns")
    assert ns_site < ns_null, (
        f"a disabled Gym instrumentation site ({ns_site:.1f} ns) should cost less than entering "
        f"an empty context manager ({ns_null:.1f} ns) — it should not be entering one at all"
    )


def test_managed_span_alone_is_not_free_which_is_why_sites_gate_first():
    """nemo-lens's `managed_span` is not free when disabled, which is why sites gate first.

    Entering a disabled `managed_span` still costs more than an empty context manager, for
    two reasons:

    * The `@contextmanager` generator has to be built and driven to its `yield` on enter
      and to `StopIteration` on exit. This is the larger share.
    * `managed_span` resolves its own gate through a function-local
      `from nemo.lens.state import is_span_group_enabled`, which sits above that gate and
      so is paid on the disabled path too.

    Both are nemo-lens concerns on a commit Gym pins, and gating at the call site avoids
    them. This test pins the reasoning so nobody "simplifies" the sites by deleting the
    outer gate.

    Asserted as a ratio, and only that the cost is non-trivial — the point is that it is
    not free, not how unfree it is on any particular machine.
    """
    ns_span = _min_ns("with managed_span('sandbox', 'gym.test'): pass", {"managed_span": managed_span})
    ns_null = _min_ns("with nullcontext(): pass", {"nullcontext": nullcontext})

    print(f"\nmanaged_span (disabled): {ns_span:.1f} ns   nullcontext: {ns_null:.1f} ns")
    assert ns_span > ns_null, (
        "managed_span now appears free when disabled — nemo-lens may have moved the import "
        f"below its gate ({ns_span:.1f} ns vs {ns_null:.1f} ns). If so, delete this test and "
        "consider dropping the outer gate at Gym's call sites."
    )


def test_attribute_building_stays_below_the_gate():
    """The pattern Gym's call sites use must not build attributes when disabled.

    Written as an executable example of the required shape: the `if` comes first, so the
    dict literal and the f-string never run on the disabled path. A site that instead
    passes attributes as keyword arguments to `managed_span` pays for building them on
    every call whether telemetry is on or not.
    """
    calls = []

    def expensive_attribute():
        calls.append(1)
        return "value"

    # The shape every Gym instrumentation site uses.
    if is_span_group_enabled(GymSpanGroup.SANDBOX):  # pragma: no cover - disabled in this test
        with managed_span(GymSpanGroup.SANDBOX, "gym.test", attr=expensive_attribute()):
            pass

    assert calls == [], "attribute construction ran on the disabled path"
