# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Service-free HLE-Verified checks using fixed judge replies."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from omegaconf import OmegaConf

from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient


def create_hle_verified_server():
    from resources_servers.equivalence_llm_judge.app import LLMJudgeResourcesServer, LLMJudgeResourcesServerConfig

    server_dir = Path(__file__).resolve().parent
    config_path = server_dir.parents[1] / "benchmarks/hle_verified/config.yaml"
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
    config = LLMJudgeResourcesServerConfig.model_validate(
        OmegaConf.to_container(
            resolved.hle_verified_equivalence_llm_judge_resources_server.resources_servers.equivalence_llm_judge,
            resolve=True,
        )
    )
    config.judge_prompt_template_fpath = str((server_dir / config.judge_prompt_template_fpath).resolve())
    return LLMJudgeResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


async def invoke_hle_verified(server, request):
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
                "content": [{"type": "output_text", "text": request.fixture_judge_text, "annotations": []}],
            }
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )
    reply = MagicMock(ok=True)
    reply.read = AsyncMock(return_value=response.model_dump_json())
    server.server_client.post = AsyncMock(return_value=reply)

    result = await server.verify(request)

    server.server_client.post.assert_awaited_once()
    assert result.judge_evaluations[0].response is not None
    return result
