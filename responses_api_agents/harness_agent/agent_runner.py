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
import asyncio
import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace


WORK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK_DIR / "gym_mount"))

import nemo_gym  # noqa: E402


sys.path.insert(0, str(WORK_DIR / "gym_mount"))
assert nemo_gym.__file__.startswith(str(WORK_DIR / "gym_mount")), f"wrong nemo_gym: {nemo_gym.__file__}"

from omegaconf import OmegaConf  # noqa: E402

from nemo_gym.config_types import BaseServerConfig  # noqa: E402
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming  # noqa: E402
from nemo_gym.server_utils import ServerClient  # noqa: E402


def main() -> None:
    rc = json.loads((WORK_DIR / "runner_config.json").read_text())
    body = json.loads((WORK_DIR / "request.json").read_text())
    model_url = (WORK_DIR / "model_url.txt").read_text().strip()
    cfg_raw = (WORK_DIR / "agent_config.json").read_text().replace("__SANDBOX_MODEL_URL__", model_url)
    cfg_data = json.loads(cfg_raw)

    module = importlib.import_module(rc["agent_module"])
    agent_class = getattr(module, rc["agent_class"])
    config_class = getattr(module, rc["agent_config_class"])

    if cwd := rc.get("cwd"):
        os.chdir(cwd)

    cfg = config_class(host="", port=0, entrypoint="", name="agent", **cfg_data)
    model_ref = cfg_data.get("model_server") or {}
    global_config = {}
    if model_ref.get("name"):
        global_config[model_ref["name"]] = {"responses_api_models": {"sandbox_model": {"host": "", "port": 0}}}
    sc = ServerClient(
        head_server_config=BaseServerConfig(host="127.0.0.1", port=0),
        global_config_dict=OmegaConf.create(global_config),
    )
    sc._build_server_base_url = lambda _: model_url
    agent = agent_class(config=cfg, server_client=sc)

    params = NeMoGymResponseCreateParamsNonStreaming.model_validate(body)
    request = SimpleNamespace(path_params={}, url=SimpleNamespace(path=""))
    resp = asyncio.run(agent.responses(request, params))
    (WORK_DIR / "response.json").write_text(resp.model_dump_json())
    print("RUNNER_DONE")


if __name__ == "__main__":
    main()
