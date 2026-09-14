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
"""Runs another agent server's responses() inside the sandbox."""

import asyncio
import importlib
import json
import sys
from unittest.mock import MagicMock


sys.path.insert(0, "/gym_mount")

import nemo_gym  # noqa: E402


assert nemo_gym.__file__.startswith("/gym_mount"), f"wrong nemo_gym: {nemo_gym.__file__}"

from omegaconf import OmegaConf  # noqa: E402

from nemo_gym.config_types import BaseServerConfig  # noqa: E402
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming  # noqa: E402
from nemo_gym.server_utils import ServerClient  # noqa: E402


def main() -> None:
    rc = json.load(open("/work/runner_config.json"))
    body = json.load(open("/work/request.json"))
    model_url = open("/work/model_url.txt").read().strip()
    cfg_raw = open("/work/agent_config.json").read().replace("__SANDBOX_MODEL_URL__", model_url)

    module = importlib.import_module(rc["agent_module"])
    agent_class = getattr(module, rc["agent_class"])
    config_class = getattr(module, rc["agent_config_class"])

    cfg = config_class(host="", port=0, entrypoint="", name="agent", **json.loads(cfg_raw))
    sc = ServerClient(
        head_server_config=BaseServerConfig(host="127.0.0.1", port=0),
        global_config_dict=OmegaConf.create({}),
    )
    agent = agent_class(config=cfg, server_client=sc)

    params = NeMoGymResponseCreateParamsNonStreaming.model_validate(body)
    resp = asyncio.run(agent.responses(MagicMock(), params))
    open("/work/response.json", "w").write(resp.model_dump_json())
    print("RUNNER_DONE")


if __name__ == "__main__":
    main()
