# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

from nemo_gym.server_utils import ServerClient
from resources_servers.legal_agent_bench.app import (
    LegalAgentBenchResourcesServer,
    LegalAgentBenchResourcesServerConfig,
    _load_judge_env_from_global_config,
)


def _server(tmp_path) -> LegalAgentBenchResourcesServer:
    config = LegalAgentBenchResourcesServerConfig(
        name="legal_agent_bench",
        host="127.0.0.1",
        port=0,
        entrypoint="app.py",
        harbor_tasks_cache_dir=str(tmp_path / "cache"),
        harbor_tasks_dir=str(tmp_path / "runtime"),
        harness_skills_dir=str(tmp_path / "skills"),
    )
    return LegalAgentBenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def test_defaults_use_full_task_and_auto_prepare(tmp_path) -> None:
    server = _server(tmp_path)

    assert server.config.reward_mode == "full_task"
    assert server.config.auto_prepare_assets is True
    assert server.config.judge_parallelism == 6


def test_startup_prepares_assets_then_rebuilds_runtime(tmp_path) -> None:
    server = _server(tmp_path)
    judge_env = {"LAB_JUDGE_API_KEY": "secret"}
    with (
        patch(
            "resources_servers.legal_agent_bench.app.ensure_assets",
            return_value={"tasks": tmp_path / "cache", "skills": tmp_path / "skills"},
        ) as ensure,
        patch("resources_servers.legal_agent_bench.app.hydrate_runtime_tasks") as hydrate,
        patch(
            "resources_servers.legal_agent_bench.app._load_judge_env_from_global_config",
            return_value=judge_env,
        ),
    ):
        app = server.setup_webserver()

    ensure.assert_called_once_with(
        tasks_dir=str(tmp_path / "cache"),
        skills_dir=str(tmp_path / "skills"),
        allow_download=True,
    )
    hydrate.assert_called_once_with(
        tmp_path / "cache",
        str(tmp_path / "runtime"),
        verifier_env={**judge_env, "LAB_JUDGE_PARALLELISM": "6"},
        reward_mode="full_task",
        cache_is_validated=True,
    )
    assert "/verify" in {route.path for route in app.routes}


def test_judge_model_is_preserved_through_openai_compatible_prefix() -> None:
    with patch(
        "nemo_gym.global_config.get_global_config_dict",
        return_value={
            "judge_base_url": "https://judge.example/v1",
            "judge_api_key": "secret",
            "judge_model_name": "provider/model-name",
        },
    ):
        env = _load_judge_env_from_global_config()

    assert env == {
        "LAB_JUDGE_BASE_URL": "https://judge.example/v1",
        "LAB_JUDGE_API_KEY": "secret",
        "LAB_JUDGE_MODEL": "openai-compatible/provider/model-name",
    }
