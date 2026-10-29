# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Service-free proof jury checks using a fixed judge reply."""

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

from omegaconf import OmegaConf

from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient


if TYPE_CHECKING:
    from resources_servers.proof_jury_judge.app import (
        ProofJuryJudgeServer,
        ProofJuryVerifyRequest,
        ProofJuryVerifyResponse,
    )


def create_proof_jury_judge_server() -> "ProofJuryJudgeServer":
    """Build the jury verifier from its shipped config without starting model services."""
    from resources_servers.proof_jury_judge.app import ProofJuryJudgeConfig, ProofJuryJudgeServer

    config_path = Path(__file__).resolve().parent / "configs/proof_jury_judge.yaml"
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
    config_values = OmegaConf.to_container(resolved.proof_jury_judge.resources_servers.proof_jury_judge, resolve=True)
    config_values["name"] = "proof_jury_judge"
    config = ProofJuryJudgeConfig.model_validate(config_values)
    return ProofJuryJudgeServer(config=config, server_client=MagicMock(spec=ServerClient))


async def invoke_proof_jury_judge(
    server: "ProofJuryJudgeServer", request: "ProofJuryVerifyRequest"
) -> "ProofJuryVerifyResponse":
    """Run verification with every judge returning the fixed reply stored in the fixture case."""
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
                "content": [{"type": "output_text", "text": request.fixture_judge_output, "annotations": []}],
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

    # One call per judge: every judge returns the same score, so no reconciliation round runs.
    assert server.server_client.post.await_count == len(server.config.judges)
    return result
