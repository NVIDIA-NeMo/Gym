# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from contextvars import ContextVar
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


pytest.importorskip("harbor")

from responses_api_agents.partner_if_harbor_agent.environment import ArtifactEnvironment


async def test_harbor_user_and_scoped_environment_reach_sandbox():
    env = object.__new__(ArtifactEnvironment)
    env._sandbox = SimpleNamespace(exec=AsyncMock(return_value=SimpleNamespace(stdout="ok", stderr="", return_code=0)))
    env._exec_shell = "bash -c"
    env._default_exec_timeout_s = 1800
    env._cpu_pin_enabled = False
    env._persistent_env = {"BASE": "base"}
    env._exec_env_overlays = ContextVar("test_exec_overlays", default=())
    with env.scoped_exec_env({"LLM_API_KEY": "proxy-placeholder", "BASE": "scoped"}):
        result = await env.exec("echo ok", cwd="/root", user="root", env={"BASE": "per-command"}, timeout_sec=15)
    assert result.stdout == "ok"
    assert env._sandbox.exec.call_args.kwargs == {
        "cwd": "/root",
        "user": "root",
        "timeout_s": 15,
        "env": {"BASE": "scoped", "LLM_API_KEY": "proxy-placeholder"},
    }
    await env.exec("true", user="root")
    assert env._sandbox.exec.call_args.kwargs["env"] == {"BASE": "base"}


def test_asset_paths_cannot_escape_prepared_environment(tmp_path):
    (tmp_path / "skills").mkdir()
    from responses_api_agents.partner_if_harbor_agent.environment import actor_directory

    assert actor_directory(tmp_path, "skills") == tmp_path / "skills"
    with pytest.raises(ValueError):
        actor_directory(tmp_path, "../rubrics")
    (tmp_path / "escape").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError):
        actor_directory(tmp_path, "escape")


async def test_bare_ray_worker_initializes_http_without_parsing_worker_arguments(monkeypatch, tmp_path):
    import sys

    import nemo_gym.global_config as global_config
    import nemo_gym.server_utils as server_utils
    from responses_api_agents.partner_if_harbor_agent.environment import NemoGymSandboxEnvironment

    monkeypatch.setattr(global_config, "_GLOBAL_CONFIG_DICT", None)
    monkeypatch.setattr(server_utils, "_GLOBAL_AIOHTTP_CLIENT", None)
    monkeypatch.delenv("NEMO_GYM_CONFIG_DICT", raising=False)
    monkeypatch.setattr(sys, "argv", ["default_worker.py", "--node-ip-address=192.0.2.1"])

    def no_worker_cli(_self):
        raise AssertionError("Ray's arguments are not Gym command-line configuration")

    monkeypatch.setattr(global_config.GlobalConfigDictParser, "parse_global_config_dict_from_cli", no_worker_cli)

    async def sandbox_start(_self, _force_build):
        # The SDK's aiohttp transport asks for this shared client on first I/O.
        assert server_utils.get_global_aiohttp_client() is not None

    monkeypatch.setattr(NemoGymSandboxEnvironment, "start", sandbox_start)
    env = object.__new__(ArtifactEnvironment)
    env.environment_dir = tmp_path
    env._actor_directories = {}
    env._output_directory = "/root/output"
    env._bootstrap_packages = []
    env.exec = AsyncMock(return_value=SimpleNamespace(return_code=0))
    env.stop = AsyncMock()
    try:
        await env.start(False)
        assert sys.argv == ["default_worker.py", "--node-ip-address=192.0.2.1"]
        env.exec.assert_awaited_once_with("mkdir -p /root/output", timeout_sec=600)
        env.stop.assert_not_called()
    finally:
        if server_utils._GLOBAL_AIOHTTP_CLIENT is not None:
            await server_utils._GLOBAL_AIOHTTP_CLIENT.close()
