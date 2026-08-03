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

import json
import sys
from asyncio import run

from omegaconf import OmegaConf

from nemo_gym.global_config import get_first_server_config_dict, get_global_config_dict
from nemo_gym.server_utils import ServerClient
from responses_api_agents.opencode_sandboxed_agent.app import OpenCodeSandboxedAgent, OpenCodeSandboxedAgentConfig


async def main():
    global_config_dict = get_global_config_dict()

    server_config_dict = OmegaConf.to_container(
        get_first_server_config_dict(global_config_dict, "opencode_sandboxed_agent"), resolve=True
    )
    server_config = OpenCodeSandboxedAgentConfig.model_validate(
        server_config_dict | {"name": "opencode_sandboxed_agent"}
    )
    server_client = ServerClient(
        head_server_config=ServerClient.load_head_server_config(),
        global_config_dict=global_config_dict,
    )
    server = OpenCodeSandboxedAgent(config=server_config, server_client=server_client)

    sandbox = await server._start_sandbox()

    command = f"""
    echo "Shell: $SHELL" \
    && curl -fsSL https://opencode.ai/install | VERSION={server.config.opencode_version} bash \
    && export PATH=$HOME/.opencode/bin:$PATH \
    && opencode run 'hello' \
    && opencode export $OPENCODE_SESSION_ID
    """

    result = await sandbox.exec(
        command=command,
        env={"OPENCODE_CONFIG_CONTENT": json.dumps(server._create_opencode_config())},
    )

    print("STDOUT: ", result.stdout, file=sys.stderr)
    print("STDERR: ", result.stderr, file=sys.stderr)
    await sandbox.stop()


if __name__ == "__main__":
    run(main())
