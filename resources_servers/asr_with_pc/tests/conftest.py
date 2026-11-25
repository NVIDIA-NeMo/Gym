# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the benchmark tests that run with this server's scoring libraries."""

import pytest

import nemo_gym.config.global_config as global_config_module
from tests.unit_tests.benchmarks import stdlib_audio


@pytest.fixture(autouse=True)
def _audio_libraries(monkeypatch):
    """The benchmark prepare() code needs ``soundfile`` and ``soxr``: use the stdlib stand-ins when they are absent."""
    stdlib_audio.install(monkeypatch)


@pytest.fixture(autouse=True)
def _mock_port_allocation(monkeypatch):
    """Parsing a benchmark declaration allocates server ports; pin the allocation so the tests need no free ports."""
    monkeypatch.setattr(global_config_module, "_find_open_port_using_range", lambda **_: 12345)
