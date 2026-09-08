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
"""The nemo-lens-absent path, and the shim-parity rule that guards it.

``kb/knowledge/conventions/fallback-sync.md``: Lens's no-op surface exists in four places
— ``nemo/lens/fallbacks.py`` plus one ``_fallbacks.py`` per consumer. A signature that
drifts breaks consumers only in the configuration where lens is *absent*, which is the
configuration nobody runs by accident. These tests make that configuration routine.
"""

import inspect

import pytest

from tests.unit_tests.telemetry.conftest import import_without_lens


#: Every name Gym's shim is required to provide. Adding a name to
#: ``nemo/lens/fallbacks.py`` without adding it here fails ``test_shim_covers_every_lens_fallback``.
EXPECTED_NAMES = frozenset(
    {"trace_fn", "managed_span", "span_cm", "is_span_group_enabled", "safe_set_span_attributes"}
)


def signature_map(module, names):
    """Map name -> parameter list for *names* on *module*."""
    return {name: list(inspect.signature(getattr(module, name)).parameters.values()) for name in names}


def compare_signatures(reference, candidate, names):
    """Return a list of human-readable drift descriptions, empty when the two agree.

    Factored out so ``test_comparison_detects_drift`` can prove this helper actually
    catches a mismatch — without that, a parity test that never sees drift asserts
    nothing.
    """
    problems = []
    ref_sigs = signature_map(reference, names)
    cand_sigs = signature_map(candidate, names)
    for name in sorted(names):
        ref, cand = ref_sigs[name], cand_sigs[name]
        if [(p.name, p.kind, p.default) for p in ref] != [(p.name, p.kind, p.default) for p in cand]:
            problems.append(f"{name}: lens has {ref}, Gym shim has {cand}")
    return problems


@pytest.fixture(scope="module")
def gym_shim_without_lens():
    """Gym's ``_fallbacks`` module imported with nemo-lens unavailable (the no-op branch)."""
    return import_without_lens("nemo_gym.telemetry._fallbacks")


def test_telemetry_package_imports_without_lens():
    """Importing nemo_gym.telemetry must not require nemo-lens."""
    module = import_without_lens("nemo_gym.telemetry._fallbacks")
    assert module.NEMO_LENS_AVAILABLE is False


def test_shim_provides_every_expected_name(gym_shim_without_lens):
    for name in sorted(EXPECTED_NAMES):
        assert hasattr(gym_shim_without_lens, name), f"Gym's no-lens shim is missing {name}"


def test_shim_covers_every_lens_fallback(gym_shim_without_lens):
    """Gym's shim must cover every public no-op nemo-lens declares — no more, no less.

    Fails when lens grows a fallback Gym has not mirrored, which is the drift direction a
    per-name check would miss.
    """
    lens_fallbacks = pytest.importorskip("nemo.lens.fallbacks")
    lens_names = {
        name
        for name, obj in vars(lens_fallbacks).items()
        if not name.startswith("_") and inspect.isfunction(obj) and obj.__module__ == lens_fallbacks.__name__
    }
    # contextmanager-wrapped functions report the decorated module; pick them up by name.
    lens_names |= {
        name for name in vars(lens_fallbacks) if not name.startswith("_") and callable(getattr(lens_fallbacks, name))
    } - {"contextmanager"}

    assert lens_names == EXPECTED_NAMES, (
        "nemo.lens.fallbacks changed its public surface. Update nemo_gym/telemetry/_fallbacks.py "
        f"and EXPECTED_NAMES in the same PR. lens={sorted(lens_names)} expected={sorted(EXPECTED_NAMES)}"
    )


def test_shim_signatures_match_lens(gym_shim_without_lens):
    """Gym's no-op shim must match nemo.lens.fallbacks parameter-for-parameter."""
    lens_fallbacks = pytest.importorskip("nemo.lens.fallbacks")
    problems = compare_signatures(lens_fallbacks, gym_shim_without_lens, EXPECTED_NAMES)
    assert not problems, "Gym's _fallbacks.py has drifted from nemo.lens.fallbacks:\n" + "\n".join(problems)


def test_comparison_detects_drift(gym_shim_without_lens):
    """The parity check must actually fail on a mismatch.

    Without this, ``test_shim_signatures_match_lens`` would pass just as happily against a
    comparison that never reports anything — the failure mode
    ``kb/knowledge/conventions/test-strength.md`` exists for.
    """
    lens_fallbacks = pytest.importorskip("nemo.lens.fallbacks")

    class DriftedShim:
        # Renamed parameter (group -> span_group) and a dropped keyword.
        @staticmethod
        def is_span_group_enabled(span_group):
            return False

    problems = compare_signatures(lens_fallbacks, DriftedShim, ["is_span_group_enabled"])
    assert problems, "compare_signatures failed to notice a renamed parameter"
    assert "is_span_group_enabled" in problems[0]


def test_no_op_primitives_are_inert(gym_shim_without_lens):
    """Every shim behaves as a no-op, not merely as an importable name."""
    shim = gym_shim_without_lens

    assert shim.is_span_group_enabled("server") is False
    assert shim.is_span_group_enabled("anything-at-all") is False

    with shim.managed_span("server", "gym.test") as span:
        assert span is None
    with shim.span_cm("gym.test") as span:
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
