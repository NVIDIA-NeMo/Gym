# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serve a Resources Server's tool routes in its tool sidecar.

The Resources Server unpacks ``nemo_gym`` and its own package beside this file and runs it with the sidecar's Python
runtime:

- ``tool_sidecar_host.py <spec.json>`` builds the Resources Server class the spec names, with its config, and serves
  its tool routes on ``0.0.0.0:<port>``. Grading and the session's lifecycle stay with the Resources Server: the
  episode routes other than ``/seed_session`` are not served, and ``/seed_session`` and ``/_tool_sidecar/state``
  (``export_tool_state()`` of the caller's session) answer only callers inside the sidecar, so the agent container
  reaches the tools alone. ``/_tool_sidecar/ready`` answers anyone.
- ``tool_sidecar_host.py --ready <port>`` exits 0 once the tool host answers.
- ``tool_sidecar_host.py --call <port> <request.json> <reply.json>`` sends one request to the tool host and writes its
  status, body and cookies, so the Resources Server can seed a session and read its state from outside the group.

The probe and the call use only the standard library, so they start quickly.
"""

import http.client
import http.cookies
import importlib
import json
import sys
from pathlib import Path
from typing import Any


READY_PATH = "/_tool_sidecar/ready"
STATE_PATH = "/_tool_sidecar/state"
# Grading and the session's lifecycle stay with the Resources Server.
_DROPPED_ROUTES = frozenset(
    {
        "/verify",
        "/aggregate_metrics",
        "/close_session",
        "/reverify_mode",
        "/prepare_sandbox",
        "/tool_sidecar/import_state",
    }
)
_SEED_PATH = "/seed_session"
_LOOPBACK = frozenset({"127.0.0.1", "::1"})


def build_app(spec: dict[str, Any]) -> Any:
    """The Resources Server's app as the tool host serves it."""
    from fastapi import Depends, FastAPI, HTTPException, Request
    from fastapi.routing import APIRoute
    from omegaconf import OmegaConf

    from nemo_gym.client.server_client import SESSION_ID_KEY, ServerClient
    from nemo_gym.config.config_types import BaseServerConfig

    server_class: Any = importlib.import_module(spec["module"])
    for name in spec["class"].split("."):
        server_class = getattr(server_class, name)
    config = server_class.model_fields["config"].annotation.model_validate(spec["config"])
    client = ServerClient(
        head_server_config=BaseServerConfig(host="127.0.0.1", port=spec["port"]),
        global_config_dict=OmegaConf.create({}),
    )
    server = server_class(config=config, server_client=client)
    app: FastAPI = server.setup_webserver()
    app.router.routes[:] = [
        route for route in app.router.routes if getattr(route, "path", None) not in _DROPPED_ROUTES
    ]

    async def from_inside_the_sidecar(request: Request) -> None:
        # Checked on the route, after any rollout prefix is stripped. The agent container reaches this host by its
        # address, never as loopback; the Resources Server calls from inside the sidecar (``--call``).
        if request.client is None or request.client.host not in _LOOPBACK:
            raise HTTPException(
                status_code=403, detail="only the Resources Server seeds this tool host or reads its state"
            )

    async def ready() -> dict[str, bool]:
        return {"ready": True}

    async def state(request: Request) -> dict[str, Any]:
        return {"state": server.export_tool_state(request.session[SESSION_ID_KEY])}

    local_only = [Depends(from_inside_the_sidecar)]
    existing = [route for route in app.router.routes if getattr(route, "path", None) != _SEED_PATH]
    seeds = [route for route in app.router.routes if isinstance(route, APIRoute) and route.path == _SEED_PATH]
    app.router.routes[:] = []
    app.get(READY_PATH)(ready)
    app.post(STATE_PATH, dependencies=local_only)(state)
    for seed in seeds:
        app.add_api_route(
            seed.path, seed.endpoint, methods=list(seed.methods), name=seed.name, dependencies=local_only
        )
    # First, so no catch-all tool route of the server answers for them.
    app.router.routes.extend(existing)
    return app


def serve(spec_path: str) -> int:
    import uvicorn

    spec = json.loads(Path(spec_path).read_text())
    # No proxy headers: the seed and state routes trust the client's own address, which a forwarded header would
    # replace where the image allows it (``FORWARDED_ALLOW_IPS``).
    uvicorn.run(
        build_app(spec),
        host="0.0.0.0",
        port=spec["port"],
        log_level="warning",
        access_log=False,
        proxy_headers=False,
    )
    return 0


def ready(port: int) -> int:
    try:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request("GET", READY_PATH)
        return 0 if connection.getresponse().status == 200 else 1
    except OSError:
        return 1


def call(port: int, request_path: str, reply_path: str) -> int:
    """Send the request ``request_path`` describes; write ``{"status", "body", "cookies"}`` to ``reply_path``."""
    request = json.loads(Path(request_path).read_text())
    headers = {"Content-Type": "application/json"}
    if request["cookies"]:
        headers["Cookie"] = "; ".join(f"{name}={value}" for name, value in request["cookies"].items())
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=280)
    connection.request(request["method"], request["path"], body=json.dumps(request["body"]), headers=headers)
    response = connection.getresponse()
    cookies: dict[str, str] = {}
    for header in response.headers.get_all("Set-Cookie") or []:
        parsed = http.cookies.SimpleCookie()
        parsed.load(header)
        cookies.update({name: morsel.value for name, morsel in parsed.items()})
    reply = {"status": response.status, "body": response.read().decode(errors="replace"), "cookies": cookies}
    partial = Path(f"{reply_path}.partial")
    partial.write_text(json.dumps(reply))
    partial.replace(reply_path)
    return 0


def main(argv: list[str]) -> int:
    if argv[1:2] == ["--ready"]:
        return ready(int(argv[2]))
    if argv[1:2] == ["--call"]:
        return call(int(argv[2]), argv[3], argv[4])
    return serve(argv[1])


if __name__ == "__main__":
    # Run as a script in the sidecar: the sources sit two directories above this file, outside any task directory.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.exit(main(sys.argv))
