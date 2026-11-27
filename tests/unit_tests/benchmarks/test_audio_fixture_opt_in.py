# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The audio stand-in fixtures of this folder's ``conftest.py`` apply to a module only when it opts in.

The folder holds many non-audio tests, so a module name such as ``test_x_benchmark`` must not silently give a test the
``soundfile``/``soxr`` stand-ins and pinned ports. This module does not opt in; ``test_audio_fixture_opted_in.py`` does.
"""

import sys
from types import SimpleNamespace

from tests.unit_tests.benchmarks import stdlib_audio
from tests.unit_tests.benchmarks.conftest import _opted_in


def test_only_an_explicit_true_opts_in() -> None:
    assert not _opted_in(SimpleNamespace(module=SimpleNamespace()))
    assert not _opted_in(SimpleNamespace(module=SimpleNamespace(USES_AUDIO_LIBRARIES="yes")))
    assert not _opted_in(SimpleNamespace(module=SimpleNamespace(USES_AUDIO_LIBRARIES=False)))
    assert _opted_in(SimpleNamespace(module=SimpleNamespace(USES_AUDIO_LIBRARIES=True)))


def test_a_module_that_did_not_opt_in_gets_no_stand_ins() -> None:
    assert getattr(sys.modules.get("soundfile"), "read", None) is not stdlib_audio.read
