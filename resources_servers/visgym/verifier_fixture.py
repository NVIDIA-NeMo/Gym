# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise grading of completed server-owned episodes without model inference.

These scorer fixtures do not establish simulator success predicates; native
provider tests and live rollouts cover that boundary separately.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from starlette.requests import Request


if TYPE_CHECKING:
    from resources_servers.visgym.app import VisGymResourcesServer
    from resources_servers.visgym.schemas import (
        VisGymAgentVerifyRequest,
        VisGymAgentVerifyResponse,
        VisGymResourcesServerConfig,
    )


def create_server(
    server_class: type[VisGymResourcesServer], config_class: type[VisGymResourcesServerConfig]
) -> VisGymResourcesServer:
    """Install trusted completed state, independently of the submitted response."""
    # The manifest loader imports the app in isolation. Bind its classes rather
    # than importing a second copy with incompatible Pydantic model identities.
    return server_class.model_construct(
        config=config_class(
            name="visgym_fixture",
            host="127.0.0.1",
            port=9000,
            entrypoint="app.py",
            reward_mode="task_success",
        ),
        # Equal positive partial returns must not turn failure into success.
        env_id_to_total_reward={"solved": 0.25, "unsolved": 0.25},
        env_id_to_task_success={"solved": True, "unsolved": False},
    )


async def invoke(server: VisGymResourcesServer, body: VisGymAgentVerifyRequest) -> VisGymAgentVerifyResponse:
    """Use the public verifier and require exact retry and masking semantics."""
    request = Request({"type": "http", "headers": []})
    result = await server.verify(request, body)
    repeated = await server.verify(request, body)
    if repeated.model_dump() != result.model_dump():
        raise AssertionError("Verification retry changed the grade")
    missing = body.response.env_id not in {"solved", "unsolved"}
    if result.mask_sample != missing:
        raise AssertionError("Lost state and legitimate task failures must have different masks")
    return result
