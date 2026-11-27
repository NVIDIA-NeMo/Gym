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
import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from nemo_gym.config.global_config import model_call_sampling_config


def test_the_global_block_marks_a_run_whose_rows_carry_values() -> None:
    config = model_call_sampling_config(
        OmegaConf.create(
            {
                "policy_model": {"responses_api_models": {"vllm_model": {}}},
                "model_call_sampling": {"required_model_servers": ["policy_model"]},
            }
        )
    )

    assert config is not None and config.required_model_servers == ["policy_model"]
    assert model_call_sampling_config(OmegaConf.create({"model_call_sampling": {}})).required_model_servers == []
    assert model_call_sampling_config(OmegaConf.create({})) is None
    assert model_call_sampling_config(None) is None
    with pytest.raises(ValidationError):
        model_call_sampling_config({"model_call_sampling": {"required_model_server": ["policy_model"]}})


@pytest.mark.parametrize("name", ["polcy_model", "policy_agent"])
def test_a_required_server_that_is_not_a_model_server_fails(name: str) -> None:
    """A misspelled or wrong name would otherwise leave the trained agent's Model Server unenforced."""
    with pytest.raises(ValueError, match=rf"not Model Servers in this run: \['{name}'\]"):
        model_call_sampling_config(
            OmegaConf.create(
                {
                    "policy_model": {"responses_api_models": {"vllm_model": {}}},
                    "policy_agent": {"responses_api_agents": {"simple_agent": {}}},
                    "model_call_sampling": {"required_model_servers": [name]},
                }
            )
        )
