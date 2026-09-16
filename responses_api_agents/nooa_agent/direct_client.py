# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct HTTP transport used by a NOOA runtime executing inside a sandbox."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from nemo_gym.server_utils import request


class DirectServerClient:
    """Minimal ``ServerClient``-compatible transport backed by explicit endpoints."""

    def __init__(self, endpoints: dict[str, str]) -> None:
        self._endpoints = {name: endpoint.rstrip("/") for name, endpoint in endpoints.items()}

    async def post(self, server_name: str, url_path: str, **kwargs: Any) -> Any:
        try:
            endpoint = self._endpoints[server_name]
        except KeyError as error:
            raise ValueError(f"No direct endpoint was configured for server {server_name!r}") from error
        body = kwargs.get("json")
        if isinstance(body, BaseModel):
            kwargs["json"] = body.model_dump(exclude_unset=True)
        return await request("POST", f"{endpoint}{url_path}", _internal=True, **kwargs)
