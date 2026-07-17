# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import yaml
from aviary.core import Message

from nemo_gym.openai_utils import NeMoGymResponse
from resources_servers.aviary.schemas import AviarySeedSessionRequest
from resources_servers.bbh_hypotest.app import (
    SANDBOX_ARTIFACTS_METADATA_KEY,
    BBHHypotestConfig,
    BBHHypotestResourcesServer,
    BBHVerifyRequest,
)


def _response(metadata: dict | None = None) -> NeMoGymResponse:
    return NeMoGymResponse.model_validate(
        {
            "id": "response-1",
            "created_at": 0.0,
            "model": "policy_model",
            "object": "response",
            "output": [
                {
                    "id": "message-1",
                    "content": [
                        {
                            "annotations": [],
                            "text": "The treatment effect is supported.",
                            "type": "output_text",
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": 1,
                "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0},
                "output_tokens": 1,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 2,
            },
            "metadata": metadata,
        }
    )


class _FakeEnvironment:
    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self.state = SimpleNamespace(
            done=False,
            score=0.0,
            raw_score=None,
            total_reward=0.0,
            notebook_runtime_errors=[],
        )
        self.run_cell = AsyncMock()
        self.close = AsyncMock()

    async def reset(self):
        return [Message(role="user", content="Test whether the treatment changed the outcome.")], []

    async def submit_answer(self, answer: str):
        assert answer == "The treatment effect is supported."
        self.state.done = True
        self.state.score = 0.75
        self.state.raw_score = 3
        self.state.total_reward = 0.75


def _server(env: _FakeEnvironment) -> BBHHypotestResourcesServer:
    config = BBHHypotestConfig.model_construct(
        seed_concurrency=1,
        sandbox_workdir="/data_workspace",
        artifact_paths=["analysis.py"],
        max_artifact_bytes=1024,
    )
    server = BBHHypotestResourcesServer.model_construct(
        config=config,
        server_client=MagicMock(),
        dataset=MagicMock(get_new_env_by_idx=MagicMock(return_value=env)),
        env_id_to_env={},
        env_id_to_total_reward={},
    )
    object.__setattr__(server, "_seed_sem", asyncio.Semaphore(1))
    return server


async def test_seed_and_verify_replays_artifact_and_closes_environment(tmp_path: Path, caplog):
    env = _FakeEnvironment(tmp_path)
    server = _server(env)
    caplog.set_level(logging.WARNING, logger="resources_servers.bbh_hypotest.app")

    seed = await server.seed_session(MagicMock(), AviarySeedSessionRequest(task_idx=4))

    assert "Submission requirements" in seed.responses_create_params.input[0].content
    assert seed.sandbox_setup == {
        "workspace_path": str(tmp_path),
        "workdir": "/data_workspace",
        "artifact_paths": ["analysis.py"],
        "max_artifact_bytes": 1024,
    }

    artifact = "# %%\nimport pandas as pd\n# %%\nprint('effect=2.5')\n"
    response = _response({SANDBOX_ARTIFACTS_METADATA_KEY: json.dumps({"analysis.py": artifact})})
    result = await server.verify(
        MagicMock(),
        BBHVerifyRequest(
            responses_create_params=seed.responses_create_params,
            response=response,
            seed_session=seed.verify_context,
        ),
    )

    assert result.reward == 0.75
    assert result.mask_sample is False
    assert result.evidence_path == "analysis.py"
    assert result.metadata["evidence_cell_count"] == 2
    assert [call.args[0] for call in env.run_cell.await_args_list] == [
        "import pandas as pd",
        "print('effect=2.5')",
    ]
    env.close.assert_awaited_once()
    assert seed.env_id not in server.env_id_to_env
    assert (
        f"BBH verify env_id={seed.env_id} reward=0.7500 evidence_path=analysis.py "
        "evidence_error=None mask_sample=False done=True score=0.75 raw_score=3 "
        "notebook_runtime_errors=[]"
    ) in caplog.messages


async def test_unknown_environment_is_masked():
    env = _FakeEnvironment(Path("/tmp/unused"))
    server = _server(env)
    result = await server.verify(
        MagicMock(),
        BBHVerifyRequest(
            responses_create_params={"input": "task"},
            response=_response(),
            seed_session={"env_id": "missing"},
        ),
    )
    assert result.reward == 0.0
    assert result.mask_sample is True
    assert "Unknown or closed" in result.evidence_error


def test_bbh_opencode_config_uses_supported_auto_flag():
    config_path = Path(__file__).parents[2] / "environments" / "bbh_hypotest" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    extra_args = config["bbh_sandbox_opencode"]["responses_api_agents"]["sandbox_agent"]["agent_config"]["extra_args"]

    assert extra_args == ["--auto", "--title", "bbh-hypotest"]


def test_bbh_opencode_config_requires_artifact_before_long_exploration():
    config_path = Path(__file__).parents[2] / "environments" / "bbh_hypotest" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    agent_config = config["bbh_sandbox_opencode"]["responses_api_agents"]["sandbox_agent"]["agent_config"]
    build_agent = agent_config["opencode_config"]["agent"]["build"]
    model_limit = agent_config["opencode_config"]["provider"]["nvinf"]["models"]["policy_model"]["limit"]

    assert build_agent["steps"] == 128
    assert agent_config["repo_dir"] == "/data_workspace"
    assert "Create /data_workspace/analysis.py immediately" in build_agent["prompt"]
    assert model_limit == {"context": "${oc.env:BBH_OPENCODE_CONTEXT,79999}", "output": 8192}


def test_bbh_workspace_mount_matches_hypotest_kernel_path():
    environment_path = Path(__file__).parents[2] / "environments" / "bbh_hypotest" / "config.yaml"
    resource_path = (
        Path(__file__).parents[2]
        / "resources_servers"
        / "bbh_hypotest"
        / "configs"
        / "bbh_hypotest.yaml"
    )
    environment = yaml.safe_load(environment_path.read_text(encoding="utf-8"))
    resource = yaml.safe_load(resource_path.read_text(encoding="utf-8"))
    agent = environment["bbh_sandbox_opencode"]["responses_api_agents"]["sandbox_agent"]

    assert resource["bbh_hypotest"]["resources_servers"]["bbh_hypotest"]["sandbox_workdir"] == "/data_workspace"
    assert agent["agent_config"]["repo_dir"] == "/data_workspace"
    assert agent["sandbox_spec"]["workdir"] == "/data_workspace"


def test_bbh_opencode_runner_deadline_precedes_sandbox_deadline():
    config_path = Path(__file__).parents[2] / "environments" / "bbh_hypotest" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    sandbox = config["bbh_sandbox_opencode"]["responses_api_agents"]["sandbox_agent"]

    assert sandbox["agent_config"]["timeout"] == 1200
    assert sandbox["rollout_timeout"] == 1500


def test_bbh_verifier_installs_sandbox_scientific_dependencies():
    repo_root = Path(__file__).parents[2]
    config = yaml.safe_load((repo_root / "environments" / "bbh_hypotest" / "config.yaml").read_text(encoding="utf-8"))
    setup_commands = " ".join(
        config["bbh_sandbox_opencode"]["responses_api_agents"]["sandbox_agent"]["setup_commands"]
    )
    requirements = {
        line.strip()
        for line in (repo_root / "resources_servers" / "bbh_hypotest" / "requirements.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.startswith("#")
    }

    scientific_dependencies = {"openpyxl", "scikit-learn", "scipy", "statsmodels"}
    assert scientific_dependencies <= requirements
    assert all(dependency in setup_commands.split() for dependency in scientific_dependencies)
