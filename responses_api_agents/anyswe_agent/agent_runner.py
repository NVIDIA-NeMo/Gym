# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import base64
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path


sys.path.insert(0, "/nemo_gym_mount")
os.environ["PATH"] = "/agent_deps_mount/bin:" + os.environ.get("PATH", "")


def _json_env(name: str) -> dict:
    encoded = os.environ.get(f"{name}_B64")
    if encoded:
        return json.loads(base64.b64decode(encoded).decode())
    return json.loads(os.environ.get(name, "{}"))


def main() -> None:
    model_url = os.environ.get("NGSWE_MODEL_URL", "")
    model_name = os.environ["NGSWE_MODEL_NAME"]
    instruction = Path("/trajectories_mount/instruction.txt").read_text()
    agent_kwargs = _json_env("NGSWE_AGENT_KWARGS")
    sampling = _json_env("NGSWE_SAMPLING")

    from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
    from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponseCreateParamsNonStreaming
    from nemo_gym.server_utils import ServerClient

    module = importlib.import_module(os.environ["NGSWE_AGENT_MODULE"])
    agent_class = getattr(module, os.environ["NGSWE_AGENT_CLASS"])
    config_class = getattr(module, os.environ["NGSWE_AGENT_CONFIG_CLASS"])

    client = ServerClient.model_construct(global_config_dict={})
    client._build_server_base_url = lambda config: model_url
    config_sampling = {key: value for key, value in sampling.items() if key in config_class.model_fields}
    model_server = ModelServerRef(name=model_name, type="responses_api_models") if model_url else None
    config = config_class(
        host="0.0.0.0",
        port=0,
        name=agent_class.__name__.lower(),
        entrypoint="app.py",
        model_server=model_server,
        resources_server=ResourcesServerRef(name="anyswe", type="resources_servers"),
        **{**agent_kwargs, **config_sampling},
    )
    agent = agent_class(config=config, server_client=client)

    if model_url:
        if hasattr(agent, "_resolve_model_base_url"):
            v1_url = model_url if model_url.endswith("/v1") else f"{model_url}/v1"
            agent._resolve_model_base_url = lambda: v1_url
        if hasattr(agent, "_resolve_base_url"):
            agent._resolve_base_url = lambda: model_url

    body = NeMoGymResponseCreateParamsNonStreaming(
        input=[NeMoGymEasyInputMessage(role="user", content=instruction)],
        model=model_name,
        **sampling,
    )
    response = asyncio.run(agent.responses(request=None, body=body))
    Path("/trajectories_mount/response.json").write_text(response.model_dump_json())
    print(f"agent finished: {len(response.output)} output items", flush=True)

    patch = ""
    for candidate in ["/testbed", "/workspace/repo", "/app", "/root/repo"]:
        repo = Path(candidate)
        if repo.exists() and (repo / ".git").exists():
            subprocess.run(["git", "add", "-A"], check=True, cwd=repo)
            patch = subprocess.run(
                ["git", "diff", "--no-color", "--cached", "HEAD"],
                capture_output=True,
                text=True,
                errors="replace",
                check=True,
                cwd=repo,
            ).stdout
            print(f"patch: {len(patch)} chars from {repo}", flush=True)
            break
    Path("/trajectories_mount/patch.diff").write_text(patch)


if __name__ == "__main__":
    main()
