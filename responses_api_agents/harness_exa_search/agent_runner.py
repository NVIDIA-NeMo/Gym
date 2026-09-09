# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import importlib
import inspect
import json
import sys
from pathlib import Path


async def main() -> None:
    from fastapi import Request

    from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
    from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
    from nemo_gym.server_utils import ServerClient

    settings = json.loads(Path(sys.argv[1]).read_text())
    module = importlib.import_module(settings["harness_module"])
    agent_class = getattr(module, settings["harness_class"])
    config_class = getattr(module, settings["harness_config_class"])
    model_url = settings["model_url"]
    model_name = "policy_model"
    global_config = {model_name: {"responses_api_models": {"model": {"host": "0.0.0.0", "port": 0}}}}
    client = ServerClient.model_construct(global_config_dict=global_config)
    client._build_server_base_url = lambda _: model_url
    config_values = {
        "host": "0.0.0.0",
        "port": 0,
        "name": "sandboxed_harness",
        "entrypoint": "app.py",
        **settings["harness_kwargs"],
    }
    if "model_server" in config_class.model_fields:
        config_values["model_server"] = ModelServerRef(name=model_name, type="responses_api_models")
    if "resources_server" in config_class.model_fields:
        config_values["resources_server"] = ResourcesServerRef(name="unused", type="resources_servers")
    if "mcp_config" in config_class.model_fields and settings.get("exa_api_key"):
        mcp_path = Path(settings["input_path"]).with_name("exa_mcp.json")
        mcp_path.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "exa": {
                            "command": "npx",
                            "args": ["-y", "exa-mcp-server"],
                            "env": {"EXA_API_KEY": settings["exa_api_key"]},
                        }
                    }
                }
            )
        )
        config_values["mcp_config"] = str(mcp_path)
    config = config_class(**config_values)
    body = NeMoGymResponseCreateParamsNonStreaming.model_validate_json(Path(settings["input_path"]).read_text())
    agent = agent_class(config=config, server_client=client)
    if "request" in inspect.signature(agent.responses).parameters:
        request = Request({"type": "http", "path": "/v1/responses", "path_params": {}, "headers": []})
        response = await agent.responses(request=request, body=body)
    else:
        response = await agent.responses(body=body)
    Path(settings["output_path"]).write_text(response.model_dump_json())


if __name__ == "__main__":
    asyncio.run(main())
