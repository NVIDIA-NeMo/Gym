# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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


sys.path.insert(0, "/nemo_gym_mount")
agent_deps_dir = os.environ.get("NV_AGENT_DEPS_DIR", "/agent_deps_mount")
os.environ["PATH"] = f"{agent_deps_dir}/bin:" + os.environ.get("PATH", "")
agent_home = Path(os.environ.get("NV_AGENT_HOME", "/code/.home"))
os.environ["HOME"] = str(agent_home)
os.environ["XDG_CACHE_HOME"] = str(agent_home / ".cache")
os.environ["XDG_CONFIG_HOME"] = str(agent_home / ".config")
os.environ["XDG_DATA_HOME"] = str(agent_home / ".local" / "share")
agent_home.mkdir(parents=True, exist_ok=True)


def main() -> None:
    model_url = os.environ.get("NV_MODEL_URL", "")
    model_name = os.environ["NV_MODEL_NAME"]
    traj_dir = os.environ.get("NV_TRAJ_DIR", "/trajectories_mount")
    instruction = Path(traj_dir, "instruction.txt").read_text()
    system = os.environ.get("NV_SYSTEM_PROMPT", "") or None
    agent_kwargs = json.loads(os.environ.get("NV_AGENT_KWARGS", "{}"))
    sampling = json.loads(os.environ.get("NV_SAMPLING", "{}"))

    agent_module = os.environ["NV_AGENT_MODULE"]
    agent_class = os.environ["NV_AGENT_CLASS"]
    agent_cfg_class = os.environ["NV_AGENT_CFG_CLASS"]

    from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponseCreateParamsNonStreaming

    module = importlib.import_module(agent_module)
    AgentClass = getattr(module, agent_class)
    AgentConfigClass = getattr(module, agent_cfg_class)

    messages = [NeMoGymEasyInputMessage(role="user", content=instruction)]
    if system:
        messages.insert(0, NeMoGymEasyInputMessage(role="system", content=system))
    body = NeMoGymResponseCreateParamsNonStreaming(input=messages, model=model_name, **sampling)

    if AgentConfigClass.__name__ == "AgentHarnessConfig":
        agent = AgentClass(config=AgentConfigClass(**agent_kwargs))
        base_url = model_url if not model_url or model_url.endswith("/v1") else f"{model_url}/v1"
        response = asyncio.run(agent.run(body, model_base_url=base_url or None))
    else:
        from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
        from nemo_gym.server_utils import ServerClient

        mock_client = ServerClient.model_construct(global_config_dict={})
        mock_client._build_server_base_url = lambda cfg: model_url
        cfg_sampling = {k: v for k, v in sampling.items() if k in AgentConfigClass.model_fields}
        model_server = ModelServerRef(name=model_name, type="responses_api_models") if model_url else None
        config = AgentConfigClass(
            host="0.0.0.0",
            port=0,
            name=agent_class.lower(),
            entrypoint="app.py",
            model_server=model_server,
            resources_server=ResourcesServerRef(name="in_sandbox", type="resources_servers"),
            **{**cfg_sampling, **agent_kwargs},
        )
        agent = AgentClass(config=config, server_client=mock_client)
        request = SimpleNamespace(path_params={})
        response = asyncio.run(agent.responses(request=request, body=body))
    Path(traj_dir, "response.json").write_text(response.model_dump_json())
    print(f"agent finished: {len(response.output)} output items", flush=True)


if __name__ == "__main__":
    main()
