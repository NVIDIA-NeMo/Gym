# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load the real catalog and type-check every checked-in task envelope."""

import json
from pathlib import Path

import pytest

from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServerConfig
from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.environment.validation import validate_environment
from nemo_gym.server_runtime import SingleAgentTurnTaskInput
from resources_servers.mm_coding_training.task_data import TaskData


ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def runtime_artifact_bindings(monkeypatch):
    for key in ("OPENCODE_PATH", "RG_PATH", "PLUGIN_BUNDLE_PATH", "RG_SOURCE", "PLUGIN_BUNDLE_SOURCE"):
        monkeypatch.setenv("MM_CODING_" + key, "/opt/test-runtime/fixture")
    for key in ("OPENCODE_SHA256", "RG_SHA256", "PLUGIN_BUNDLE_SHA256"):
        monkeypatch.setenv("MM_CODING_" + key, "a" * 64)
    monkeypatch.setenv("MM_CODING_SANDBOX_IMAGE", "example/mm-coding-sandbox@sha256:" + "a" * 64)
    monkeypatch.setenv("MM_CODING_JUDGE_MODEL", "quality-judge")


@pytest.mark.parametrize("mode", ["single", "multi", "mixed"])
def test_training_config_resolves_resource_owned_sessions(monkeypatch, mode):
    from omegaconf import OmegaConf

    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "offline-validation-only")
    config_name = "mm_coding_training" if mode == "single" else f"mm_coding_{mode}"
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": [
                        "responses_api_models/vllm_model/configs/vllm_model_for_training.yaml",
                        f"resources_servers/mm_coding_training/configs/{config_name}.yaml",
                        "nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml",
                    ],
                    "policy_model_name": "offline-validation-only",
                    "policy_base_url": "http://localhost:1/v1",
                    "policy_api_key": "offline-validation-only",
                }
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    assert config.mm_coding_training.resources_servers.mm_coding_training.entrypoint == "app.py"
    if mode != "single":
        owners = set()
        for harness in ("opencode",):
            environment_name = f"mm_coding_{harness}_environment"
            env = config[environment_name].environment_servers.single_agent_turn
            # Resolve the actual deployment schema, including required lifecycle limits.
            SingleAgentTurnEnvironmentServerConfig.model_validate(
                OmegaConf.to_container(env, resolve=True) | {"name": environment_name}
            )
            agent = config[env.agent_server.name].responses_api_agents[f"{harness}_agent"]
            assert agent.resources_server is None
            # Exercise the stock adapter's seed allowlist, not only Pydantic schema parsing.
            from responses_api_agents.opencode_agent.app import _session_opencode_config_error

            if harness == "opencode":
                assert _session_opencode_config_error(OmegaConf.to_container(agent.opencode_config)) is None
                assert agent.context_window == 16384 - agent.max_output_tokens
                assert agent.opencode_config.compaction.auto is False
                assert agent.opencode_config.agent.build.steps == 8
            owner = env.resources_server.name
            owners.add(owner)
            resource = config[owner].resources_servers.mm_coding_training
            assert resource.entrypoint == "session_app.py"
            assert resource.sandbox_provider == "sandbox"
            assert resource.judge_model_server.name == "mm_coding_quality_judge"
        assert len(owners) == 1


@pytest.mark.parametrize("harness", ["single", "opencode"])
def test_pairing_manifest_and_task_data_are_consumable(monkeypatch, harness):
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "offline-validation-only")
    report = validate_environment(ROOT / f"environments/mm_coding_{harness}/manifest.yaml")
    assert not report.warnings
    assert len(report.datasets) == 1 and report.datasets[0].rows == 5
    for line in Path(report.datasets[0].path).read_text().splitlines():
        row = json.loads(line)
        if harness == "single":
            task = TaskData(task=row["task"])
            params = row["responses_create_params"]
        else:
            assert row["task_id"]["taskset"] == f"mm-coding:{harness}"
            parsed = SingleAgentTurnTaskInput.model_validate(row["task_input"])
            task = TaskData.model_validate(parsed.task_data)
            params = parsed.responses_create_params.model_dump()
        assert task.task.ground_truth not in json.dumps(params) or not task.task.ground_truth
        assert task.task.source_split == "example"


@pytest.mark.asyncio
@pytest.mark.parametrize("harness", ["single", "opencode"])
async def test_reward_fixture_is_executable_through_real_onboarding_api(monkeypatch, harness):
    from nemo_gym.environment.onboarding import exercise_verifier_run, prepare_verifier_run
    from nemo_gym.environment.registry import EnvironmentCatalogEntry

    monkeypatch.setenv("OPENSANDBOX_API_KEY", "offline-validation-only")
    path = ROOT / f"environments/mm_coding_{harness}"
    entry = EnvironmentCatalogEntry(
        name=f"mm_coding_{harness}", path=path, config_path=path / "config.yaml", manifest_path=path / "manifest.yaml"
    )
    report = await exercise_verifier_run(prepare_verifier_run(entry))
    assert len(report.cases) == 3
    assert {case.name for case in report.cases} == {"execution_failed", "faithful", "missing_judge"}
