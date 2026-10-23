# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check agent-generated model URLs against the real sampling middleware."""

from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from starlette.testclient import TestClient

from nemo_gym.rollout_sampling import ModelSamplingMiddleware


def assert_model_sampling(
    base_url: str, controls: dict[str, Any], *, routed_base_path: str = "/v1", mount_path: str = ""
) -> None:
    """Send model requests through a staged URL and check its body and retained routing."""

    async def echo(request: Request) -> JSONResponse:
        return JSONResponse({"path": request.url.path, "body": await request.json()})

    model_app = ModelSamplingMiddleware(Starlette(routes=[Route("/{path:path}", echo, methods=["POST"])]))
    app = Starlette(routes=[Mount(mount_path, app=model_app)]) if mount_path else model_app
    with TestClient(app) as client:
        for endpoint, token_field in (
            ("chat/completions", "max_completion_tokens"),
            ("responses", "max_output_tokens"),
        ):
            body = {"model": "test-model", "temperature": 1.2, "top_p": 0.5, token_field: 2048}
            expected = body | {key: value for key, value in controls.items() if key != "max_output_tokens"}
            if "max_output_tokens" in controls:
                expected[token_field] = min(2048, controls["max_output_tokens"])
            response = client.post(f"{base_url.rstrip('/')}/{endpoint}", json=body)
            assert response.status_code == 200, response.text
            assert response.json() == {"path": f"{routed_base_path}/{endpoint}", "body": expected}
