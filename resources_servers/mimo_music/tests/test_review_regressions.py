# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression cases for config migration and renderer process failures."""

import signal
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.server_utils import entrypoint_ray_enabled
from resources_servers.mimo_music.app import MimoMusicResourcesServer
from resources_servers.mimo_music.tests.test_app import ABC, request, server


def test_music_server_does_not_require_ray() -> None:
    assert MimoMusicResourcesServer.ray_enabled is False
    assert entrypoint_ray_enabled(Path(__file__).resolve().parents[1] / "app.py") is False


def test_config_resolves_with_environment_server() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "mimo_music.yaml"
    initial = OmegaConf.merge(
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        OmegaConf.load(config_path),
    )
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=initial,
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    relay = config.mimo_music_environment_server.environment_servers.legacy_agent
    assert relay.entrypoint == "app.py"
    assert relay.agent_server.type == "responses_api_agents"
    assert relay.agent_server.name == "mimo_music_simple_agent"
    agent = config.mimo_music_simple_agent.responses_api_agents.simple_agent
    assert agent.resources_server.name == "mimo_music"
    assert agent.max_steps == 1


@pytest.mark.parametrize("termination_signal", [signal.SIGKILL, signal.SIGTERM])
@pytest.mark.parametrize("partial_midi", [False, True])
@pytest.mark.asyncio
async def test_renderer_signal_is_masked(tmp_path: Path, termination_signal: int, partial_midi: bool) -> None:
    executable = tmp_path / "crashing-renderer"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        f"if {partial_midi!r}:\n"
        "    open(sys.argv[sys.argv.index('-o') + 1], 'wb').write(b'partial MIDI')\n"
        f"os.kill(os.getpid(), {int(termination_signal)})\n"
    )
    executable.chmod(0o755)
    music_server = server()
    music_server._abc2midi = str(executable)
    result = await music_server.verify(request(ABC))
    assert result.reward == 0
    assert result.mask_sample is True
    assert result.failure_kind == "mimo_music:scorer_error"
    assert f"signal {int(termination_signal)}" in result.failure_reason
    assert f"return code {-int(termination_signal)}" in result.failure_reason


@pytest.mark.parametrize("exit_code", [0, 1])
@pytest.mark.asyncio
async def test_normal_renderer_rejection_stays_valid_zero(tmp_path: Path, exit_code: int) -> None:
    executable = tmp_path / "rejecting-renderer"
    executable.write_text(f"#!{sys.executable}\nimport sys\nprint('Error: invalid ABC')\nsys.exit({exit_code})\n")
    executable.chmod(0o755)
    music_server = server()
    music_server._abc2midi = str(executable)
    result = await music_server.verify(request(ABC))
    assert result.reward == 0
    assert result.mask_sample is False
    assert result.failure_kind is None
    assert result.scorer_details["skip"] == "no_midi"
    assert result.scorer_details["err"] == 1
