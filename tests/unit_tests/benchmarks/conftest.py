# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for the audio benchmark tests."""

import pytest

import nemo_gym.config.global_config as global_config_module
from tests.unit_tests.benchmarks import stdlib_audio


@pytest.fixture
def mock_port_allocation(monkeypatch):
    """Parsing a benchmark declaration allocates server ports; pin the allocation so the tests need no free ports."""
    monkeypatch.setattr(global_config_module, "_find_open_port_using_range", lambda **_: 12345)


def _opted_in(request) -> bool:
    return getattr(request.module, "USES_AUDIO_LIBRARIES", False) is True


@pytest.fixture(autouse=True)
def _audio_libraries(request, monkeypatch):
    """The ``prepare()`` tests of the audio benchmarks need ``soundfile`` and ``soxr``, which the core environment lacks:
    use the stdlib stand-ins. This folder also holds many non-audio tests, so a module opts in explicitly with
    ``USES_AUDIO_LIBRARIES = True`` (a module name is not a reliable signal); no other module is affected."""
    if _opted_in(request):
        stdlib_audio.install(monkeypatch)


@pytest.fixture(autouse=True)
def _mock_port_allocation_for_audio_benchmark_tests(request):
    """An audio benchmark test module (``USES_AUDIO_LIBRARIES = True``) parses benchmark declarations: pin the ports."""
    if _opted_in(request):
        request.getfixturevalue("mock_port_allocation")
