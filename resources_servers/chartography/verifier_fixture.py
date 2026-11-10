# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Service-free chartography checks using a fixed judge reply (``gym env test chartography``)."""

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

from omegaconf import OmegaConf

from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymChatCompletion, NeMoGymChatCompletionMessage, NeMoGymChoice


if TYPE_CHECKING:
    from resources_servers.chartography.app import (
        ChartographyResourcesServer,
        ChartographyVerifyRequest,
        ChartographyVerifyResponse,
    )

CONFIG_FPATH = Path(__file__).resolve().parent / "configs/chartography.yaml"


def create_chartography_server() -> "ChartographyResourcesServer":
    """Build the verifier from its shipped config without starting the judge model server."""
    from resources_servers.chartography.app import ChartographyConfig, ChartographyResourcesServer

    shipped = OmegaConf.to_container(OmegaConf.load(CONFIG_FPATH), resolve=True)
    values = dict(shipped["chartography"]["resources_servers"]["chartography"])
    config = ChartographyConfig.model_validate({**values, "name": "chartography", "host": "127.0.0.1", "port": 8080})
    return ChartographyResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


async def invoke_chartography(
    server: "ChartographyResourcesServer", request: "ChartographyVerifyRequest"
) -> "ChartographyVerifyResponse":
    """Run verification with the judge returning the fixed reply stored in the fixture case."""
    completion = NeMoGymChatCompletion(
        id="fixture_judge",
        created=0,
        model="fixture",
        object="chat.completion",
        choices=[
            NeMoGymChoice(
                index=0,
                finish_reason="stop",
                message=NeMoGymChatCompletionMessage(role="assistant", content=request.fixture_judge_output),
            )
        ],
    )
    reply = MagicMock(status=200)
    reply.read = AsyncMock(return_value=completion.model_dump_json().encode())
    server.server_client.post = AsyncMock(return_value=reply)

    result = await server.verify(request)

    # One judge call: the fixed reply always parses.
    assert server.server_client.post.await_count == 1
    return result
