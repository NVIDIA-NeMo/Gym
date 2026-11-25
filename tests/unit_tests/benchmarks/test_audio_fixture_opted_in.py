# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A module that sets ``USES_AUDIO_LIBRARIES = True`` gets the audio stand-ins and pinned ports (see the opt-in test)."""

import importlib.util
import sys

import pytest

import nemo_gym.config.global_config as global_config_module
from tests.unit_tests.benchmarks import stdlib_audio


USES_AUDIO_LIBRARIES = True
REAL_SOUNDFILE = importlib.util.find_spec("soundfile") is not None


@pytest.mark.skipif(REAL_SOUNDFILE, reason="the real soundfile is installed, so no stand-in is needed")
def test_an_opted_in_module_has_the_stand_ins() -> None:
    assert sys.modules["soundfile"].read is stdlib_audio.read


def test_an_opted_in_module_has_pinned_ports() -> None:
    assert global_config_module._find_open_port_using_range() == 12345
