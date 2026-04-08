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
from unittest.mock import MagicMock

from nemo_gym.base_resources_server import BaseResourcesServerConfig
from nemo_gym.server_utils import ServerClient
from resources_servers.example_gymnasium import GymnasiumServer


class TestGymnasiumServer:
    def test_sanity(self) -> None:
        class ConcreteEnv(GymnasiumServer):
            async def step(self, action, metadata, session_id=None):
                return None, 1.0, True, False, {}

        config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="")
        env = ConcreteEnv(config=config, server_client=MagicMock(spec=ServerClient))
        app = env.setup_webserver()
        routes = {r.path for r in app.routes}
        assert "/reset" in routes
        assert "/step" in routes
