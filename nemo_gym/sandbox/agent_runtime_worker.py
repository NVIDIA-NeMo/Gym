# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-task agent process. Invoked by the runtime host inside the task sandbox."""

import asyncio
import importlib
import json
import os
import sys
from http.cookies import SimpleCookie
from pathlib import Path


async def invoke(payload: dict) -> dict:
    # Set before importing Gym or the harness: both may consult configuration at import time.
    os.environ["NEMO_GYM_CONFIG_DICT"] = json.dumps(payload["global_config"])
    for key in ("IS_NEMO_GYM_FASTAPI_WORKER", "IS_NEMO_GYM_FASTAPI_ENTRYPOINT", "NEMO_GYM_FASTAPI_NUM_WORKERS"):
        os.environ.pop(key, None)
    from omegaconf import OmegaConf

    from nemo_gym.config_types import BaseServerConfig
    from nemo_gym.server_utils import ServerClient

    class RuntimeServerClient(ServerClient):
        def _build_server_base_url(self, server_config_dict):
            return server_config_dict.get("runtime_base_url") or super()._build_server_base_url(server_config_dict)

    module_name, class_name = payload["harness"].split(":", 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    config = cls.model_fields["config"].annotation.model_validate(payload["config"])
    client = RuntimeServerClient(
        head_server_config=BaseServerConfig(host="127.0.0.1", port=1),
        global_config_dict=OmegaConf.create(payload["global_config"]),
    )
    agent = cls(config=config, server_client=client)
    app = agent.setup_webserver()
    content = json.dumps(payload["body"]).encode()
    cookies = SimpleCookie()
    for key, value in payload["cookies"].items():
        cookies[key] = value
    headers = [(b"content-type", b"application/json"), (b"cookie", cookies.output(header="", sep=";").encode())]
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": payload["path"],
        "raw_path": payload["path"].encode(),
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "server": ("localhost", 1),
        "client": ("127.0.0.1", 1),
    }
    received = False
    completed = asyncio.Event()
    response_body = bytearray()
    response_cookies = SimpleCookie()
    status = None

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": content, "more_body": False}
        await completed.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        nonlocal status
        if message["type"] == "http.response.start":
            status = message["status"]
            for key, value in message.get("headers", []):
                if key.lower() == b"set-cookie":
                    response_cookies.load(value.decode("latin-1"))
        elif message["type"] == "http.response.body":
            response_body.extend(message.get("body", b""))
            if not message.get("more_body", False):
                completed.set()

    async with app.router.lifespan_context(app):
        await app(scope, receive, send)
    if status != 200:
        raise RuntimeError(f"Harness /responses returned HTTP {status}")
    return {
        "response": json.loads(response_body),
        "cookies": {key: value.value for key, value in response_cookies.items()},
    }


if __name__ == "__main__":
    payload = json.loads(Path(sys.argv[1]).read_text())
    result = asyncio.run(invoke(payload))
    Path(sys.argv[2]).write_text(json.dumps(result))
