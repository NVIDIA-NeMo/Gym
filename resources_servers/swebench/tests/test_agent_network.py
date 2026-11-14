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

from nemo_gym.config import ModelServerRef
from resources_servers.swebench.agent_network import check_agent_network_config


MODEL_SERVER = ModelServerRef(type="responses_api_models", name="policy_model")


def test_a_workable_config_passes() -> None:
    check_agent_network_config({"opensandbox": {}}, MODEL_SERVER, {})


def test_a_missing_model_server_fails_at_startup() -> None:
    # Deny-all egress would cut the agent in the sandbox off from the policy model.
    with pytest.raises(ValueError, match="sandbox_model_server"):
        check_agent_network_config({"opensandbox": {}}, None, {})


def test_a_provider_without_network_policies_fails_at_startup() -> None:
    with pytest.raises(ValueError, match="OpenSandbox"):
        check_agent_network_config({"docker": {}}, MODEL_SERVER, {})


def test_a_configured_network_policy_fails_at_startup() -> None:
    with pytest.raises(ValueError, match="not both"):
        check_agent_network_config({"opensandbox": {}}, MODEL_SERVER, {"provider_options": {"network_policy": {}}})
