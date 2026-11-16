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


@pytest.fixture(autouse=True)
def _audio_libraries(request, monkeypatch):
    """The ``prepare()`` tests of the audio benchmarks need ``soundfile`` and ``soxr``, which the core environment lacks:
    use the stdlib stand-ins. Only ``test_<name>_benchmark.py`` modules and modules that set
    ``USES_AUDIO_LIBRARIES = True`` are affected."""
    if request.module.__name__.endswith("_benchmark") or getattr(request.module, "USES_AUDIO_LIBRARIES", False):
        stdlib_audio.install(monkeypatch)


@pytest.fixture(autouse=True)
def _mock_port_allocation_for_benchmark_tests(request):
    """Every ``test_<name>_benchmark.py`` module parses benchmark declarations; no other test is affected."""
    if request.module.__name__.endswith("_benchmark"):
        request.getfixturevalue("mock_port_allocation")
