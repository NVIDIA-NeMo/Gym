# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The live half of the static sandbox policy (tests/unit_tests/test_sandbox_policy.py, R4): what the running
server serves and what its episode response carries. It imports the server package, so it lives here, with the
server's suite, and not in the core unit run whose coverage gate must not count the server's modules."""

import typing

from pydantic import BaseModel

from environment_servers.harbor.app import HarborEpisodeResponse
from environment_servers.harbor.tests.conftest import make_server
from nemo_gym.sandbox import SandboxAccess


# Request fields that would let a caller run something in the sandbox through the server.
COMMAND_FIELDS = {"command", "commands", "argv", "cmd", "script", "stdin", "shell"}


def test_r4_the_server_serves_only_the_base_routes_and_no_command_field(tmp_path, monkeypatch):
    server, _, _, _ = make_server(tmp_path, monkeypatch)
    app = server.setup_webserver()
    served = {route.path for route in app.routes if "POST" in (getattr(route, "methods", None) or set())}
    assert served == {"/run", "/aggregate_metrics"}
    documentation = {route.path for route in app.routes} - served
    assert documentation == {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}
    for route in app.routes:
        if route.path not in served:
            continue
        body = route.endpoint.__annotations__["body"]
        assert issubclass(body, BaseModel)
        for field in body.model_fields:
            assert field not in COMMAND_FIELDS, f"{route.path} takes a {field!r}"


def test_r4_the_episode_response_carries_no_sandbox_descriptor():
    carried: list[str] = []

    def walk(model: type[BaseModel], prefix: str, seen: set[type]) -> None:
        if model in seen:
            return
        seen.add(model)
        for name, field in model.model_fields.items():
            if name in {"sandbox_access", "descriptor", "connection", "provider_config_ref"}:
                carried.append(f"{prefix}.{name}")
            for nested in nested_models(field.annotation):
                if nested is SandboxAccess:
                    carried.append(f"{prefix}.{name} ({nested.__name__})")
                walk(nested, f"{prefix}.{name}", seen)

    walk(HarborEpisodeResponse, "HarborEpisodeResponse", set())
    assert carried == []


def nested_models(annotation) -> list[type[BaseModel]]:
    found: list[type[BaseModel]] = []
    stack = [annotation]
    while stack:
        current = stack.pop()
        if isinstance(current, type) and issubclass(current, BaseModel):
            found.append(current)
        stack.extend(typing.get_args(current))
    return found
