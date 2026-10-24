# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Resolve the shipped profiles through Gym's actual parser and Harbor schema."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from environments.skill_trigger.agent import SkillOptions
from environments.skill_trigger.package import HARNESSES
from nemo_gym.environment.manifest import load_manifest
from nemo_gym.environment.validation import _mirror_differences, _resolve_manifest_composition
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from responses_api_agents.harbor_agent_general.app import HarborAgentConfig


ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("harness", HARNESSES)
def test_profiles_resolve_to_exact_harness_and_shared_verifier(harness, tmp_path):
    path = ROOT / "benchmarks/skill_trigger/configs" / harness / "config.yaml"
    options = GlobalConfigDictParserConfig(
        initial_global_config_dict=OmegaConf.merge(
            GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
            {"config_paths": [str(path)], "policy_model_name": "test-model", "policy_base_url": "http://provider/v1"},
        ),
        skip_load_from_cli=True,
        skip_load_from_dotenv=True,
        offline=True,
    )
    resolved = GlobalConfigDictParser().parse(options)
    # Runtime choices are explicit, never inferred from a model name.
    resolved.skill_trigger_settings.provider = "openai" if harness == "pi" else None
    resolved.skill_trigger_settings.api = "openai-responses" if harness == "pi" else None
    if harness == "codex_guidance":
        resolved.skill_trigger_settings.model_catalog_path = str(tmp_path / "model-catalog.json")
        resolved.skill_trigger_settings.model_catalog_sha256 = "catalog-hash"
    config = HarborAgentConfig.model_validate(
        {"name": "skill_trigger", "host": "127.0.0.1", "port": 8080}
        | OmegaConf.to_container(resolved.skill_trigger.responses_api_agents.skill_trigger, resolve=True)
    )
    agent = config.job.agents[0]
    profile = SkillOptions.model_validate(agent.kwargs)
    assert profile.harness == harness
    assert config.job.verifier.import_path == "environments.skill_trigger.verifier:SkillOpeningVerifier"
    assert config.job.verifier.kwargs["harness"] == harness
    assert config.job.verifier.kwargs["manifest_sha256"] == profile.manifest_sha256
    assert agent.import_path == "environments.skill_trigger.agent:SkillAgent"
    assert agent.model_name == "test-model"
    assert config.job.n_attempts == config.job.n_concurrent_trials == 1
    assert config.job.retry.max_retries == 0 and config.job.environment.delete
    assert config.job.environment.cpu_enforcement_policy == "ignore"
    assert config.harbor_ray_max_retries == 0
    assert config.harbor_unique_job_names and config.harbor_reward_key == "reward"
    assert resolved.count_failure_classes_as_zero == [] and not resolved.count_missing_rollouts_as_zero
    assert resolved.skills.path.endswith("/catalogs/bionemo-9984d9c3ec80")
    assert "${" not in OmegaConf.to_container(resolved)["skills"]["path"]
    assert "api_key" not in agent.kwargs and profile.api_key_env == "SKILL_MODEL_API_KEY"
    manifest = load_manifest(path.with_name("manifest.yaml"))
    assert manifest.experimental and manifest.prompt_source == "agent"
    assert not _mirror_differences(manifest, _resolve_manifest_composition(path))
