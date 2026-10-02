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
"""GymSpanGroup registration, preset resolution, and membership."""

import pytest

from nemo_gym.telemetry.span_groups import GymSpanGroup
from tests.unit_tests.telemetry.conftest import import_without_lens, no_lens, requires_lens


#: These exercise the telemetry-enabled path, which needs nemo-lens. The absent-lens path
#: is covered by test_fallbacks.py, which runs either way.
pytestmark = requires_lens


GYM_GROUPS = {
    "job",
    "server",
    "http_client",
    "rollout",
    "verify",
    "agent",
    "model_call",
    "sandbox",
}


def test_importing_span_groups_registers_them_with_lens():
    """nemo-lens ships no group names, so an unregistered Gym group can never be enabled."""
    from nemo.lens.groups import SpanRegistry

    from nemo_gym.telemetry.span_groups import NAMESPACE

    assert GymSpanGroup.ALL_GROUPS == GYM_GROUPS
    assert NAMESPACE in SpanRegistry.namespaces()
    assert GYM_GROUPS <= SpanRegistry.groups()


def test_registration_is_idempotent():
    """A re-import, or a test that re-registers, must not raise on the existing namespace."""
    from nemo_gym.telemetry.span_groups import register_span_groups

    register_span_groups()
    assert GymSpanGroup.resolve("default") == {"job", "server", "http_client", "rollout"}


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
    """`default` is the run-level view: the spine plus job, and nothing per-request."""
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


def test_unknown_group_is_reported_as_pending_not_enabled():
    """nemo-lens no longer raises on an unknown name, because the library that owns it may
    not be imported in this process. It must still enable nothing and be reported back."""
    from nemo.lens.groups import SpanRegistry

    assert GymSpanGroup.resolve("rollouts") == frozenset()  # note the plural
    assert SpanRegistry.resolve("default,rollouts")[1] == {"rollouts"}


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
    with no_lens():
        with pytest.raises(RuntimeError, match="requires nemo-lens"):
            GymSpanGroup.resolve("default")
