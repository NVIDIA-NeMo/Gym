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

from pathlib import Path

from omegaconf import OmegaConf


PACKAGE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]


def test_role_models_copy_the_standard_policy_model() -> None:
    config = OmegaConf.load(PACKAGE_DIR / "configs" / "conversational_tool_use_simulation.yaml")

    assert config["simulator_model"]["_copy"] == "policy_model"
    assert config["judge_model"]["_copy"] == "policy_model"


def test_generation_role_models_copy_the_standard_policy_model() -> None:
    config_paths = {
        "domain_generation_model": (
            "responses_api_agents/conversational_tool_use/domain_generation/configs/conversational_tool_use_domain_generation.yaml"
        ),
        "policy_generation_model": (
            "responses_api_agents/conversational_tool_use/policy_tool_generation/configs/conversational_tool_use_policy_tool_generation.yaml"
        ),
        "policy_tool_judge_model": (
            "responses_api_agents/conversational_tool_use/policy_tool_generation/configs/conversational_tool_use_policy_tool_generation.yaml"
        ),
        "scenario_generation_model": (
            "responses_api_agents/conversational_tool_use/scenario_generation/configs/conversational_tool_use_scenario_generation.yaml"
        ),
    }

    for model_name, relative_path in config_paths.items():
        config = OmegaConf.load(REPO_ROOT / relative_path)
        assert config[model_name]["_copy"] == "policy_model"
