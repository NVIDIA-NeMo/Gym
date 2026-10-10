# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import importlib
import inspect
import json
import os
import re
import sys
from pathlib import Path

from fastapi import Request, Response


sys.path[:0] = ["/trajectories_mount/agent_src", "/opt/Gym"]
os.environ["PATH"] = "/opt/agent/bin:" + os.environ.get("PATH", "")

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef  # noqa: E402
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming  # noqa: E402
from nemo_gym.server_utils import ServerClient  # noqa: E402


def main() -> None:
    module = importlib.import_module(os.environ["KB_AGENT_MODULE"])
    agent_class = getattr(module, os.environ["KB_AGENT_CLASS"])
    config_class = getattr(module, os.environ["KB_AGENT_CONFIG_CLASS"])
    model_url = os.environ["KB_MODEL_URL"]
    model_server = ModelServerRef(name="policy_model", type="responses_api_models")
    client = ServerClient.model_construct(global_config_dict={"policy_model": {"responses_api_models": {"model": {}}}})
    client._build_server_base_url = lambda config: model_url
    agent_kwargs = json.loads(os.environ["KB_AGENT_KWARGS"])
    model_api_key = agent_kwargs.pop("model_api_key", None)
    if model_api_key:
        post = client.post

        async def authenticated_post(*args, **kwargs):
            kwargs["headers"] = {"Authorization": f"Bearer {model_api_key}"}
            return await post(*args, **kwargs)

        object.__setattr__(client, "post", authenticated_post)
    config_values = {
        "host": "0.0.0.0",
        "port": 0,
        "name": "kernel_gym_harness",
        "entrypoint": "app.py",
        **agent_kwargs,
    }
    if "model_server" in config_class.model_fields:
        config_values["model_server"] = model_server
    if "resources_server" in config_class.model_fields:
        config_values["resources_server"] = ResourcesServerRef(name="unused", type="resources_servers")
    agent = agent_class(config=config_class(**config_values), server_client=client)
    body = NeMoGymResponseCreateParamsNonStreaming.model_validate_json(os.environ["KB_BODY"])
    body.model = body.model or os.environ["KB_MODEL_NAME"]
    solution = Path("/workspace/solution.py")
    original = solution.read_text()
    call = {"request": Request({"type": "http", "path": "/v1/responses", "headers": []}), "body": body}
    if "response" in inspect.signature(agent.responses).parameters:
        call["response"] = Response()
    response = asyncio.run(agent.responses(**call))
    if solution.read_text() == original:
        text = "".join(
            part.text
            for item in response.output
            if item.type == "message"
            for part in item.content
            if part.type == "output_text"
        )
        match = re.search(r"```(?:python|cpp)?\s*\n(.*?)```", text, re.DOTALL)
        solution.write_text(match.group(1) if match else text)
    Path("/trajectories_mount/response.json").write_text(response.model_dump_json())
    print(f"agent finished: {len(response.output)} output items", flush=True)


if __name__ == "__main__":
    main()
