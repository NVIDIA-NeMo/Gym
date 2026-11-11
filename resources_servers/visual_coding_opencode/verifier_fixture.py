# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Service-free visual_coding_opencode verifier checks using fixed judge replies."""

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

from omegaconf import OmegaConf

from nemo_gym.client import ServerClient
from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.core import NeMoGymChatCompletion


if TYPE_CHECKING:
    from resources_servers.visual_coding_opencode.app import (
        VisualCodingOpenCodeServer,
        VisualCodingOpenCodeVerifyRequest,
        VisualCodingOpenCodeVerifyResponse,
    )


def create_visual_coding_opencode_server() -> "VisualCodingOpenCodeServer":
    """Build the verifier from its config without starting model services."""
    from resources_servers.visual_coding_opencode.app import VisualCodingOpenCodeConfig, VisualCodingOpenCodeServer

    config_path = Path(__file__).resolve().parent / "configs" / "visual_coding_opencode.yaml"
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
        resolved.visual_coding_opencode.resources_servers.visual_coding_opencode, resolve=True
    )
    config_values["name"] = "visual_coding_opencode"
    # Grading never reads the lent sandbox, so the verifier runs without one (and without a sandbox image).
    config_values["sandbox_provider"] = None
    config_values["sandbox_network_access"] = "inherit"
    config = VisualCodingOpenCodeConfig.model_validate(config_values)
    return VisualCodingOpenCodeServer(config=config, server_client=MagicMock(spec=ServerClient))


async def invoke_visual_coding_opencode(
    server: "VisualCodingOpenCodeServer",
    request: "VisualCodingOpenCodeVerifyRequest",
) -> "VisualCodingOpenCodeVerifyResponse":
    """Run verification with the judge reply stored in the case's fixture_judge_reply."""
    completion = NeMoGymChatCompletion.model_validate(
        {
            "id": "fixture_judge",
            "created": 0,
            "model": "fixture",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": getattr(request, "fixture_judge_reply", "")},
                }
            ],
        }
    )
    reply = MagicMock(status=200, ok=True)
    reply.read = AsyncMock(return_value=completion.model_dump_json().encode())
    server.server_client.post = AsyncMock(return_value=reply)
    return await server.verify(request)
