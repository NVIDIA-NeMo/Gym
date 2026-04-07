# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

try:
    import reasoning_gym  # noqa: F401

    HAS_REASONING_GYM = True
except ImportError:
    HAS_REASONING_GYM = False

from unittest.mock import MagicMock

from nemo_gym.base_resources_server import BaseResourcesServerConfig
from nemo_gym.server_utils import ServerClient


@pytest.mark.skipif(not HAS_REASONING_GYM, reason="reasoning_gym not installed")
class TestApp:
    def test_sanity(self) -> None:
        from resources_servers.reasoning_gym_env.app import ReasoningGymEnv

        config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="")
        ReasoningGymEnv(config=config, server_client=MagicMock(spec=ServerClient))
