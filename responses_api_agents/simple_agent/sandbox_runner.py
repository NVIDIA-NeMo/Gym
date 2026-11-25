# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run Simple Agent's agent loop once inside the sandbox its environment lends.

The agent server unpacks ``nemo_gym`` and this package beside this file and launches it with the sandbox's Python
runtime: ``sandbox_runner.py <input.json> <output.json>``. It builds the same ``SimpleAgent`` the agent server runs,
with the agent server's own config, and calls its ``_create_episode`` loop with the arguments the agent server
staged. The loop's model calls go to the model server's address as the sandbox reaches it, under the model path the
agent server built, so token capture and the episode's sampling values apply as they do in process.

``sandbox_runner.py --check`` only imports the loop, so a seed fails early when the runtime cannot.
"""

import asyncio
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


async def run_loop(payload: dict[str, Any]) -> dict[str, Any]:
    """Run ``SimpleAgent._create_episode`` with the staged arguments and return its results as JSON."""
    from omegaconf import OmegaConf

    from nemo_gym.client import ServerClient
    from nemo_gym.config import BaseServerConfig
    from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
    from nemo_gym.server_runtime import DirectHTTPToolAccess
    from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig, _cookie_values
    from responses_api_agents.simple_agent.python_tool import PythonTools, PythonToolSpec

    config = SimpleAgentConfig.model_validate(payload["config"])
    model = payload["model_server"]
    # The only server the loop calls by name is its model server, at the address the sandbox reaches.
    global_config = OmegaConf.create(
        {config.model_server.name: {"responses_api_models": {"model": {"host": model["host"], "port": model["port"]}}}}
    )
    client = ServerClient(
        head_server_config=BaseServerConfig(host=model["host"], port=model["port"]), global_config_dict=global_config
    )
    agent = SimpleAgent(config=config, server_client=client)
    arguments = dict(payload["loop"])
    if arguments.get("tool_access") is not None:
        arguments["tool_access"] = DirectHTTPToolAccess.model_validate(arguments["tool_access"])
    body = NeMoGymResponseCreateParamsNonStreaming.model_validate(payload["body"])
    specs = payload.get("python_tools") or {}
    # The task's Python tools run here, beside the loop: their workers are this process's children in the sandbox.
    python_tools = PythonTools({name: PythonToolSpec.model_validate(spec) for name, spec in specs.items()})
    try:
        response, trajectory, model_server_cookies, resources_server_cookies = await agent._create_episode(
            body, **arguments, python_tools=python_tools if specs else None
        )
    finally:
        await python_tools.close()
    return {
        "response": response.model_dump(mode="json"),
        "trajectory": trajectory.model_dump(mode="json") if trajectory is not None else None,
        "model_server_cookies": _cookie_values(model_server_cookies),
        "resources_server_cookies": (
            _cookie_values(resources_server_cookies) if resources_server_cookies is not None else None
        ),
    }


def error_output(error: Exception) -> dict[str, Any]:
    """What the agent server needs to raise ``error`` again: a model server's HTTP error keeps its status, headers and
    body, as the agent boundary reads them; any other error keeps its type and message."""
    from aiohttp import ClientResponseError

    output: dict[str, Any] = {
        "error": f"{type(error).__name__}: {error}",
        "error_type": type(error).__name__,
        "traceback": traceback.format_exc(),
    }
    if isinstance(error, ClientResponseError):
        content = getattr(error, "response_content", None)
        output["http_error"] = {
            "url": str(error.request_info.real_url),
            "method": error.request_info.method,
            "status": error.status,
            "message": error.message,
            "headers": list((error.headers or {}).items()),
            "content": content.decode(errors="replace") if isinstance(content, bytes) else content,
        }
    return output


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    if argv[1:] == ["--check"]:
        import responses_api_agents.simple_agent.app  # noqa: F401

        return 0
    input_path, output_path = Path(argv[1]), Path(argv[2])
    payload = json.loads(input_path.read_text())
    try:
        output = asyncio.run(run_loop(payload))
    except Exception as error:
        # The agent server raises it again; the loop's own failures, such as a rejected model call, are in the response.
        output = error_output(error)
    partial = output_path.with_name(f".{output_path.name}.{os.getpid()}")
    partial.write_text(json.dumps(output))
    partial.replace(output_path)
    return 1 if "error" in output else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
