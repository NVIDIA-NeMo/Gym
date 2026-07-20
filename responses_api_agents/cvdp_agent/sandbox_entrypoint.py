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

"""Guest entrypoint copied verbatim into the sandbox and executed as ``python agent_runner.py``.

It imports the configured gym agent (module/class chosen via ``NV_AGENT_*`` env vars),
points it at the model server, calls ``responses()`` so the agent edits files with its own
tools, and writes the trajectory out. Kept as a plain, lintable module rather than a string
template so it is diffable and syntax-checked with the rest of the package;
``app.load_runner_source`` reads its source and drops it into the container unchanged.
"""

import asyncio
import importlib
import json
import os
import sys
from pathlib import Path


# the mounts only exist inside the sandbox; do this before importing nemo_gym / the agent
sys.path.insert(0, "/nemo_gym_mount")
os.environ["PATH"] = "/agent_deps_mount/bin:" + os.environ.get("PATH", "")


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

    from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
    from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponseCreateParamsNonStreaming
    from nemo_gym.server_utils import ServerClient

    module = importlib.import_module(agent_module)
    AgentClass = getattr(module, agent_class)
    AgentConfigClass = getattr(module, agent_cfg_class)

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

    if model_url:
        if hasattr(agent, "_resolve_model_base_url"):
            v1 = model_url if model_url.endswith("/v1") else model_url + "/v1"
            agent._resolve_model_base_url = lambda: v1
        if hasattr(agent, "_resolve_base_url"):
            agent._resolve_base_url = lambda: model_url

    messages = [NeMoGymEasyInputMessage(role="user", content=instruction)]
    if system:
        messages.insert(0, NeMoGymEasyInputMessage(role="system", content=system))
    body = NeMoGymResponseCreateParamsNonStreaming(input=messages, model=model_name, **sampling)

    response = asyncio.run(agent.responses(request=None, body=body))
    Path(traj_dir, "response.json").write_text(response.model_dump_json())
    print(f"agent finished: {len(response.output)} output items", flush=True)


if __name__ == "__main__":
    main()
