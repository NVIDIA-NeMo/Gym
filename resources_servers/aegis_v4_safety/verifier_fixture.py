# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Service-free Aegis v4 verifier checks using fixed judge replies."""

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

from omegaconf import OmegaConf

from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient


if TYPE_CHECKING:
    from resources_servers.aegis_v4_safety.app import (
        AegisV4SafetyResourcesServer,
        AegisV4SafetyVerifyRequest,
        AegisV4SafetyVerifyResponse,
    )


def create_aegis_v4_safety_server() -> "AegisV4SafetyResourcesServer":
    """Build the benchmark verifier without starting model services."""
    from resources_servers.aegis_v4_safety.app import AegisV4SafetyConfig, AegisV4SafetyResourcesServer

    server_dir = Path(__file__).resolve().parent
    config_path = server_dir.parents[1] / "benchmarks/aegis_v4_safety/config.yaml"
    resolved = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.merge(
                GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                {"config_paths": [str(config_path)]},
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    config_values = OmegaConf.to_container(
        resolved.aegis_v4_safety_benchmark_resources_server.resources_servers.aegis_v4_safety,
        resolve=True,
    )
    config_values["name"] = "aegis_v4_safety_benchmark_resources_server"
    config = AegisV4SafetyConfig.model_validate(config_values)
    return AegisV4SafetyResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


async def invoke_aegis_v4_safety(
    server: "AegisV4SafetyResourcesServer",
    request: "AegisV4SafetyVerifyRequest",
) -> "AegisV4SafetyVerifyResponse":
    """Run verification with the fixed Aegis reply stored in a fixture case."""
    response = NeMoGymResponse(
        id="fixture_judge",
        created_at=0,
        model="fixture",
        object="response",
        status="completed",
        output=[
            {
                "id": "fixture_message",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": request.fixture_aegis_output,
                        "annotations": [],
                    }
                ],
            }
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )
    reply = MagicMock(status=200)
    reply.read = AsyncMock(return_value=response.model_dump_json().encode())
    server.server_client.post = AsyncMock(return_value=reply)

    result = await server.verify(request)

    server.server_client.post.assert_awaited_once()
    assert result.judge_evaluation is not None
    return result
