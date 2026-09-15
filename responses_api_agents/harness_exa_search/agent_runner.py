# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import base64
import importlib
import inspect
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from time import monotonic

from aiohttp import web


_RELAY_PREFIX = "/__nemo_gym_model_relay"


class ModelRelay:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[dict] = asyncio.Queue()
        self._pending: dict[str, asyncio.Future[dict]] = {}
        self._waiters: set[asyncio.Task] = set()
        self.app = web.Application(client_max_size=1024**3)
        self.app.on_shutdown.append(self.shutdown)
        self.app.router.add_get(f"{_RELAY_PREFIX}/next", self.next_request)
        self.app.router.add_post(f"{_RELAY_PREFIX}/result/{{request_id}}", self.submit_result)
        self.app.router.add_route("*", "/{path:.*}", self.forward)

    async def forward(self, request: web.Request) -> web.Response:
        request_id = uuid.uuid4().hex
        result = asyncio.get_running_loop().create_future()
        self._pending[request_id] = result
        await self._queue.put(
            {
                "id": request_id,
                "method": request.method,
                "path": request.raw_path,
                "headers": dict(request.headers),
                "body": base64.b64encode(await request.read()).decode(),
            }
        )
        try:
            payload = await result
        finally:
            self._pending.pop(request_id, None)
        return web.Response(
            status=payload["status"],
            headers=payload.get("headers"),
            body=base64.b64decode(payload.get("body", "")),
        )

    async def next_request(self, request: web.Request) -> web.Response:
        timeout = min(max(float(request.query.get("timeout", "15")), 0.0), 30.0)
        waiter = asyncio.create_task(self._queue.get())
        self._waiters.add(waiter)
        try:
            payload = await asyncio.wait_for(waiter, timeout=timeout)
        except TimeoutError:
            return web.Response(status=204)
        finally:
            self._waiters.discard(waiter)
        return web.json_response(payload)

    async def submit_result(self, request: web.Request) -> web.Response:
        pending = self._pending.get(request.match_info["request_id"])
        if pending is None or pending.done():
            return web.Response(status=404)
        pending.set_result(await request.json())
        return web.Response(status=204)

    async def shutdown(self, _: web.Application) -> None:
        for task in self._waiters:
            task.cancel()
        for future in self._pending.values():
            future.cancel()


async def start_model_relay(port: int) -> web.AppRunner:
    relay = ModelRelay()
    runner = web.AppRunner(relay.app)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", port).start()
    return runner


async def main() -> None:
    from fastapi import Request

    from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
    from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
    from nemo_gym.server_utils import ServerClient

    logging.basicConfig(level=logging.WARNING)
    settings = json.loads(Path(sys.argv[1]).read_text())
    module = importlib.import_module(settings["harness_module"])
    agent_class = getattr(module, settings["harness_class"])
    config_class = getattr(module, settings["harness_config_class"])
    model_url = settings["model_url"]
    relay_runner = None
    if settings.get("model_relay_port"):
        relay_runner = await start_model_relay(settings["model_relay_port"])
        model_url = f"http://127.0.0.1:{settings['model_relay_port']}"
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
    exa_api_key = settings.get("exa_api_key")
    if exa_api_key:
        os.environ["EXA_API_KEY"] = exa_api_key
    if "model_server" in config_class.model_fields:
        config_values["model_server"] = ModelServerRef(name=model_name, type="responses_api_models")
    if "resources_server" in config_class.model_fields:
        config_values["resources_server"] = ResourcesServerRef(name="unused", type="resources_servers")
    if "mcp_config" in config_class.model_fields and exa_api_key:
        mcp_path = Path(settings["input_path"]).with_name("exa_mcp.json")
        mcp_path.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "exa": {
                            "command": "npx",
                            "args": ["-y", "exa-mcp-server"],
                            "env": {"EXA_API_KEY": exa_api_key},
                        }
                    }
                }
            )
        )
        config_values["mcp_config"] = str(mcp_path)
    if "extra_config" in config_class.model_fields and exa_api_key:
        config_values.setdefault("extra_config", {}).setdefault("mcp_servers", {})["exa"] = {
            "command": "npx",
            "args": ["-y", "exa-mcp-server"],
            "env": {"EXA_API_KEY": exa_api_key},
        }
    for config_field in ("opencode_config", "kilo_config"):
        if config_field in config_class.model_fields and exa_api_key:
            config_values.setdefault(config_field, {}).setdefault("mcp", {})["exa"] = {
                "type": "local",
                "command": ["npx", "-y", "exa-mcp-server"],
                "environment": {"EXA_API_KEY": exa_api_key},
                "enabled": True,
            }
    if "fabric_config" in config_class.model_fields and exa_api_key:
        config_values.setdefault("fabric_config", {}).setdefault("mcp", {}).setdefault("servers", {})["exa"] = {
            "transport": "stdio",
            "url": "npx",
            "args": ["-y", "exa-mcp-server"],
            "env": {"EXA_API_KEY": exa_api_key},
            "exposure": "harness_native",
        }
    if "openclaw_config" in config_class.model_fields and exa_api_key:
        config_values.setdefault("openclaw_config", {}).setdefault("mcp", {}).setdefault("servers", {})["exa"] = {
            "command": "npx",
            "args": ["-y", "exa-mcp-server"],
            "env": {"EXA_API_KEY": exa_api_key},
        }
    if settings.get("pi_extension_path") and settings["harness_class"] == "PiAgent":
        config_values.setdefault("extra_args", []).extend(["--extension", settings["pi_extension_path"]])
    try:
        config = config_class(**config_values)
        body = NeMoGymResponseCreateParamsNonStreaming.model_validate_json(Path(settings["input_path"]).read_text())
        agent = agent_class(config=config, server_client=client)
        started_at = monotonic()
        if "request" in inspect.signature(agent.responses).parameters:
            request = Request({"type": "http", "path": "/v1/responses", "path_params": {}, "headers": []})
            response = await agent.responses(request=request, body=body)
        else:
            response = await agent.responses(body=body)
        metadata = dict(response.metadata or {})
        diagnostics = json.loads(metadata.get("agent_run", "{}"))
        diagnostics.update(
            harness_module=settings["harness_module"],
            harness_class=settings["harness_class"],
            runner_status="returned",
            runner_duration_ms=(monotonic() - started_at) * 1000,
        )
        response.metadata = metadata | {"agent_run": json.dumps(diagnostics, sort_keys=True)}
        Path(settings["output_path"]).write_text(response.model_dump_json())
    finally:
        if relay_runner is not None:
            await relay_runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
