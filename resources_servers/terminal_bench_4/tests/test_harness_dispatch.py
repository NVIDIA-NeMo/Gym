# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harness selection: default mini-SWE, explicit `name: hermes`, and unknown-name rejection."""

import pytest
from pydantic import ValidationError

from resources_servers.terminal_bench_4.app import TerminalBench4Config
from resources_servers.terminal_bench_4.tests.test_environment import environment_config
from responses_api_agents.hermes_sandboxed_agent.harness import HermesConfig
from responses_api_agents.miniswe_sandboxed_agent.harness import MiniSWEConfig


def _config(harness=None):
    kwargs = {
        "host": "localhost",
        "port": 1,
        "name": "tb4",
        "entrypoint": "app.py",
        "environment": environment_config(sandbox_provider={"local": {}}),
        "model_server": {"type": "responses_api_models", "name": "model"},
    }
    if harness is not None:
        kwargs["harness"] = harness
    return TerminalBench4Config(**kwargs)


def test_default_harness_remains_miniswe():
    """Existing configurations with no harness name keep selecting mini-SWE unchanged."""
    assert isinstance(_config().harness, MiniSWEConfig)


def test_name_hermes_selects_hermes():
    """`harness.name: hermes` selects the Hermes harness configuration."""
    assert isinstance(_config({"name": "hermes"}).harness, HermesConfig)


def test_unknown_harness_name_is_rejected():
    """An unrecognized harness name fails validation instead of silently becoming Hermes."""
    with pytest.raises(ValidationError, match="Unknown harness name"):
        _config({"name": "not-a-real-harness"})


def test_invalid_hermes_option_is_rejected():
    """An unsupported Hermes option fails validation instead of being silently ignored."""
    with pytest.raises(ValidationError):
        _config({"name": "hermes", "not_a_real_option": True})
