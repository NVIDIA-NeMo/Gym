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
"""GymSpanGroup preset resolution and membership."""

import pytest

from nemo_gym.telemetry.span_groups import GymSpanGroup
from tests.unit_tests.telemetry.conftest import import_without_lens, requires_lens


#: These exercise the telemetry-enabled path, which needs nemo-lens. The absent-lens path
#: is covered by test_fallbacks.py, which runs either way.
pytestmark = requires_lens


GYM_SPECIFIC = {
    "server",
    "http_client",
    "rollout",
    "verify",
    "agent",
    "model_call",
    "sandbox",
}


def test_gym_groups_extend_the_shared_base():
    """Gym adds its own groups without dropping the shared ones."""
    from nemo.lens.groups import SpanGroup

    assert GYM_SPECIFIC <= GymSpanGroup.ALL_GROUPS
    assert SpanGroup.ALL_GROUPS <= GymSpanGroup.ALL_GROUPS
    assert GymSpanGroup.ALL_GROUPS == SpanGroup.ALL_GROUPS | GYM_SPECIFIC


@pytest.mark.parametrize("preset", ["default", "per_rollout", "all"])
def test_every_preset_carries_the_cross_process_spine(preset):
    """Losing `server`, `http_client` or `rollout` would silently break trace joining.

    These three are what make one rollout appear as a single trace across the agent,
    model and resources server processes. A preset that omits one still 'works' — it just
    produces disconnected traces, which is the failure this integration exists to
    prevent. Pin them into every preset.
    """
    resolved = GymSpanGroup.resolve(preset)
    assert GymSpanGroup.CROSS_PROCESS_SPINE <= resolved, (
        f"preset {preset!r} is missing {sorted(GymSpanGroup.CROSS_PROCESS_SPINE - resolved)}"
    )


def test_default_preset_is_coarse():
    """`default` is the run-level view: the spine plus job/evaluate, and nothing per-request."""
    resolved = GymSpanGroup.resolve("default")
    assert resolved == {"job", "server", "http_client", "rollout"}
    for fine_grained in ("verify", "agent", "model_call", "sandbox"):
        assert fine_grained not in resolved


def test_per_rollout_adds_request_detail_and_drops_job():
    """`per_rollout` bounds each trace at one rollout instead of one run."""
    resolved = GymSpanGroup.resolve("per_rollout")
    assert {"verify", "agent", "model_call"} <= resolved
    assert "job" not in resolved, "per_rollout must not nest every rollout under one run-long span"


def test_all_preset_is_every_group():
    assert GymSpanGroup.resolve("all") == GymSpanGroup.ALL_GROUPS


def test_individual_group_names_resolve():
    assert GymSpanGroup.resolve("sandbox") == {"sandbox"}
    assert GymSpanGroup.resolve("verify,agent") == {"verify", "agent"}


def test_every_preset_group_has_a_call_site():
    """A preset must not advertise a group nothing emits under.

    `default` and `per_rollout` are what users actually select, so a group listed there
    with no instrumentation is a knob that silently does nothing. The inherited
    training-oriented groups stay resolvable through `all` but are kept out of the
    curated presets.
    """
    emitting_groups = {"job", "server", "http_client", "rollout", "verify", "agent", "model_call", "sandbox"}
    for preset in ("default", "per_rollout"):
        assert GymSpanGroup.resolve(preset) <= emitting_groups, (
            f"preset {preset!r} advertises groups with no call site: "
            f"{sorted(GymSpanGroup.resolve(preset) - emitting_groups)}"
        )


def test_preset_and_group_names_can_be_mixed():
    resolved = GymSpanGroup.resolve("default,sandbox")
    assert resolved == GymSpanGroup.resolve("default") | {"sandbox"}


def test_resolution_is_case_and_whitespace_insensitive():
    assert GymSpanGroup.resolve("  DEFAULT , Sandbox ") == GymSpanGroup.resolve("default,sandbox")


def test_unknown_group_is_rejected_with_the_valid_options():
    """A typo must fail loudly — silently resolving to nothing looks identical to
    'telemetry is on but nothing is instrumented', which is very hard to debug."""
    with pytest.raises(ValueError, match="Unknown span group or preset"):
        GymSpanGroup.resolve("rollouts")  # note the plural


def test_unknown_group_error_lists_gym_groups_not_just_lens_ones():
    with pytest.raises(ValueError) as excinfo:
        GymSpanGroup.resolve("nope")
    message = str(excinfo.value)
    for name in ("model_call", "per_rollout", "sandbox"):
        assert name in message, f"the error message should offer {name!r} as a valid option"


def test_empty_spec_resolves_to_nothing():
    assert GymSpanGroup.resolve("") == frozenset()
    assert GymSpanGroup.resolve(" , ") == frozenset()


# --------------------------------------------------------------------------- #
# The nemo-lens-absent path
# --------------------------------------------------------------------------- #


def test_gym_groups_are_usable_without_lens():
    """`GymSpanGroup.SERVER` and friends must still be importable constants.

    Instrumentation sites reference these names unconditionally — the *gate* is what is
    conditional, not the constant — so an ImportError here would break every call site on
    a checkout without the telemetry extra.
    """
    module = import_without_lens("nemo_gym.telemetry.span_groups")

    assert module.GymSpanGroup.SERVER == "server"
    assert module.GymSpanGroup.HTTP_CLIENT == "http_client"
    assert module.GymSpanGroup.ROLLOUT == "rollout"
    assert "server" in module.GymSpanGroup.ALL_GROUPS


def test_resolving_a_preset_without_lens_fails_loudly():
    """Without lens there is nothing to enable, so resolution must raise rather than
    return an empty set — a silent empty result is indistinguishable from a working
    config that happens to trace nothing."""
    module = import_without_lens("nemo_gym.telemetry.span_groups")

    with pytest.raises(RuntimeError, match="requires nemo-lens"):
        module.GymSpanGroup.resolve("default")


def test_the_stub_mirrors_the_lens_span_group_surface():
    """The stub stands in for `nemo.lens.groups.SpanGroup`; if lens grows a group the
    stub does not have, `GymSpanGroup.ALL_GROUPS` silently differs depending on whether
    the extra is installed."""
    from nemo.lens.groups import SpanGroup as RealSpanGroup

    module = import_without_lens("nemo_gym.telemetry.span_groups")
    stub = module.GymSpanGroup.__mro__[1]

    assert stub.ALL_GROUPS == RealSpanGroup.ALL_GROUPS, (
        "the no-lens stub in span_groups.py has drifted from nemo.lens.groups.SpanGroup"
    )
