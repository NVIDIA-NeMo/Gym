# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import base64
import importlib
import inspect
import json
import os
import socket
from pathlib import Path


def decode(name: str) -> dict:
    return json.loads(base64.b64decode(os.environ[name]).decode())


async def start_model_proxy(upstream: str, api_key: str, model: str):
    import aiohttp
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import Response, StreamingResponse

    from nemo_gym.anthropic_converter import AnthropicConverter
    from nemo_gym.openai_utils import NeMoGymChatCompletion
    from nemo_gym.responses_converter import ResponsesConverter

    app = FastAPI()
    anthropic = AnthropicConverter()
    responses = ResponsesConverter(return_token_id_information=False)

    @app.post("/v1/messages")
    async def messages(request: Request):
        body = await request.json()
        response_params = anthropic.anthropic_request_to_responses(body)
        response_params.model = model
        chat_params = responses.responses_to_chat_completion_create_params(response_params)
        headers = {
            "authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{upstream.rstrip('/')}/chat/completions",
                json=chat_params.model_dump(mode="json", exclude_none=True),
                headers=headers,
            ) as upstream_response:
                content = await upstream_response.read()
                if upstream_response.status >= 400:
                    return Response(content, status_code=upstream_response.status, media_type="application/json")

        chat_response = NeMoGymChatCompletion.model_validate_json(content)
        response = responses.chat_completion_to_response(response_params, chat_response)
        anthropic_response = anthropic.responses_to_anthropic_response(response, model=body["model"])
        if body.get("stream"):
            return StreamingResponse(
                anthropic.anthropic_response_to_sse(anthropic_response), media_type="text/event-stream"
            )
        return Response(json.dumps(anthropic_response), media_type="application/json")

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)
    return server, task, f"http://127.0.0.1:{port}"


async def main() -> None:
    from fastapi import Request

    from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
    from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
    from nemo_gym.server_utils import ServerClient

    module = importlib.import_module(os.environ["NG_HARNESS_MODULE"])
    agent_class = getattr(module, os.environ["NG_HARNESS_CLASS"])
    config_class = getattr(module, os.environ["NG_HARNESS_CONFIG_CLASS"])
    model_url = os.environ.get("NG_MODEL_URL", "")
    model_name = "policy_model"
    global_config = {model_name: {"responses_api_models": {"model": {"host": "0.0.0.0", "port": 0}}}}
    client = ServerClient.model_construct(global_config_dict=global_config)
    client._build_server_base_url = lambda _: model_url
    config_values = {
        "host": "0.0.0.0",
        "port": 0,
        "name": "sandboxed_harness",
        "entrypoint": "app.py",
        **decode("NG_HARNESS_KWARGS"),
    }
    proxy = None
    if config_values.get("anthropic_base_url") and os.environ.get("NVIDIA_API_KEY"):
        upstream = config_values["anthropic_base_url"]
        upstream_model = config_values["model"]
        proxy = await start_model_proxy(upstream, os.environ["NVIDIA_API_KEY"], upstream_model)
        config_values["anthropic_base_url"] = proxy[2]
        config_values["model"] = "claude-sonnet-4-6"
    if "model_server" in config_class.model_fields and not config_values.get("anthropic_base_url"):
        config_values["model_server"] = ModelServerRef(name=model_name, type="responses_api_models")
    if "anthropic_api_key" in config_class.model_fields and os.environ.get("NVIDIA_API_KEY"):
        config_values["anthropic_api_key"] = os.environ["NVIDIA_API_KEY"]
    if "resources_server" in config_class.model_fields:
        config_values["resources_server"] = ResourcesServerRef(name="unused", type="resources_servers")
    if "mcp_config" in config_class.model_fields and os.environ.get("EXA_API_KEY"):
        mcp_path = Path(os.environ["NG_INPUT_PATH"]).with_name("exa_mcp.json")
        mcp_path.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "exa": {
                            "command": "npx",
                            "args": ["-y", "exa-mcp-server"],
                            "env": {"EXA_API_KEY": os.environ["EXA_API_KEY"]},
                        }
                    }
                }
            )
        )
        config_values["mcp_config"] = str(mcp_path)
    config = config_class(**config_values)
    body = NeMoGymResponseCreateParamsNonStreaming.model_validate_json(Path(os.environ["NG_INPUT_PATH"]).read_text())
    try:
        agent = agent_class(config=config, server_client=client)
        if "request" in inspect.signature(agent.responses).parameters:
            request = Request({"type": "http", "path": "/v1/responses", "path_params": {}, "headers": []})
            response = await agent.responses(request=request, body=body)
        else:
            response = await agent.responses(body=body)
        Path(os.environ["NG_OUTPUT_PATH"]).write_text(response.model_dump_json())
    finally:
        if proxy:
            proxy[0].should_exit = True
            await proxy[1]


if __name__ == "__main__":
    asyncio.run(main())
