# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A legacy_agent relay gets a six-hour episode deadline unless its config opts out.

Some agents bound their own episode for longer than that, or wrap environments that set no deadline, so their relays
say `default_episode_timeout_seconds: null`. Without it the relay would end a valid episode as `Episode timed out`.
"""

from pathlib import Path

import pytest
import yaml

from environment_servers.legacy_agent.app import LegacyAgentEnvironmentServerConfig


ROOT = Path(__file__).resolve().parents[3]

# (config file, environment server instance) pairs whose relay must not end an episode at the default deadline.
UNBOUNDED_RELAYS = [
    # mini-SWE's own budget is 8 hours (`agent_timeout_sec`).
    (
        "benchmarks/terminal_bench_science/miniswe_legacy_agent.yaml",
        "terminal_bench_science_miniswe_environment_server",
    ),
    ("benchmarks/terminal_bench_4/miniswe_legacy_agent.yaml", "terminal_bench_4_miniswe_environment_server"),
    (
        "resources_servers/terminal_bench_4/configs/terminal_bench_4.yaml",
        "terminal_bench_4_miniswe_environment_server",
    ),
]


def relay_block(path: str, instance: str) -> dict:
    config = yaml.safe_load((ROOT / path).read_text())
    return config[instance]["environment_servers"]["legacy_agent"]


@pytest.mark.parametrize(("path", "instance"), UNBOUNDED_RELAYS)
def test_the_relay_opts_out_of_the_default_episode_deadline(path: str, instance: str) -> None:
    block = relay_block(path, instance)

    assert "default_episode_timeout_seconds" in block, "an omitted deadline would default to six hours"
    config = LegacyAgentEnvironmentServerConfig(host="127.0.0.1", port=1, name=instance, **block)
    assert config.default_episode_timeout_seconds is None


@pytest.mark.parametrize(
    "path", ["benchmarks/partner_if/config.yaml", "environment_servers/partner_if/configs/partner_if.yaml"]
)
def test_partner_if_environments_keep_their_null_deadlines(path: str) -> None:
    # The partner_if server waits for routed episodes whose text and tool turns have no deadline.
    config = yaml.safe_load((ROOT / path).read_text())
    environments = [
        block
        for value in config.values()
        if isinstance(value, dict)
        for implementation, block in (value.get("environment_servers") or {}).items()
        if implementation in ("partner_if", "single_agent_turn_legacy")
    ]

    assert any("task_routes" in block for block in environments)
    for block in environments:
        assert block["default_episode_timeout_seconds"] is None


def test_an_ordinary_relay_keeps_the_six_hour_default() -> None:
    block = relay_block(
        "responses_api_agents/mini_swe_agent/configs/mini_swe_agent.yaml", "mini_swe_simple_environment_server"
    )

    assert "default_episode_timeout_seconds" not in block
    config = LegacyAgentEnvironmentServerConfig(host="127.0.0.1", port=1, name="relay", **block)
    assert config.default_episode_timeout_seconds == 21600
