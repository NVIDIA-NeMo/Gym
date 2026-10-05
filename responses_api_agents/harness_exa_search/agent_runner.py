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
    module = importlib.import_module(settings["agent_module"])
    agent_class = getattr(module, settings["agent_class"])
    config_class = getattr(module, settings["agent_config_class"])
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
        **settings["agent_kwargs"],
    }
    search_provider = settings.get("search_provider", "exa")
    budget_env = {
        "SEARCH_MAX_RESULTS": str(settings.get("search_max_results", 20)),
        "SEARCH_MAX_CHARS_PER_RESULT": str(settings.get("search_max_chars_per_result", 30000)),
        "SEARCH_MAX_CHARS_TOTAL": str(settings.get("search_max_chars_total", 65000)),
    }
    if search_provider == "exa":
        api_key = settings.get("exa_api_key")
        mcp_command = sys.executable
        mcp_args = [settings["search_mcp_path"]]
        mcp_env = {"EXA_API_KEY": api_key, **budget_env}
    elif search_provider == "parallel":
        api_key = settings.get("parallel_api_key")
        mcp_command = sys.executable
        mcp_args = [settings["search_mcp_path"]]
        mcp_env = {"PARALLEL_API_KEY": api_key, **budget_env}
    elif search_provider == "brave":
        api_key = settings.get("brave_api_key")
        mcp_command = sys.executable
        mcp_args = [settings["search_mcp_path"]]
        mcp_env = {"BRAVE_API_KEY": api_key, **budget_env}
    else:
        raise ValueError(f"unsupported search_provider: {search_provider}")
    if "model_server" in config_class.model_fields:
        config_values["model_server"] = ModelServerRef(name=model_name, type="responses_api_models")
    if "resources_server" in config_class.model_fields:
        config_values["resources_server"] = ResourcesServerRef(name="unused", type="resources_servers")
    if "mcp_config" in config_class.model_fields and api_key:
        mcp_path = Path(settings["input_path"]).with_name("search_mcp.json")
        mcp_path.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        search_provider: {"command": mcp_command, "args": mcp_args, "env": mcp_env},
                    }
                }
            )
        )
        config_values["mcp_config"] = str(mcp_path)
    if "extra_config" in config_class.model_fields and api_key:
        config_values.setdefault("extra_config", {}).setdefault("mcp_servers", {})[search_provider] = {
            "command": mcp_command,
            "args": mcp_args,
            "env": mcp_env,
        }
    for config_field in ("opencode_config", "kilo_config"):
        if config_field in config_class.model_fields and api_key:
            config_values.setdefault(config_field, {}).setdefault("mcp", {})[search_provider] = {
                "type": "local",
                "command": [mcp_command, *mcp_args],
                "environment": mcp_env,
                "enabled": True,
            }
    if "fabric_config" in config_class.model_fields and api_key:
        config_values.setdefault("fabric_config", {}).setdefault("mcp", {}).setdefault("servers", {})[
            search_provider
        ] = {
            "transport": "stdio",
            "url": mcp_command,
            "args": mcp_args,
            "env": mcp_env,
            "exposure": "harness_native",
        }
    if "openclaw_config" in config_class.model_fields and api_key:
        config_values.setdefault("openclaw_config", {}).setdefault("mcp", {}).setdefault("servers", {})[
            search_provider
        ] = {"command": mcp_command, "args": mcp_args, "env": mcp_env}
    if settings.get("pi_extension_path") and settings["agent"] == "pi":
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
            agent_module=settings["agent_module"],
            agent_class=settings["agent_class"],
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
