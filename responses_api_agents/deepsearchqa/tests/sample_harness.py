# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import time

from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseOutputMessage, NeMoGymResponseOutputText


class SampleHarnessConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef


class SampleHarness:
    def __init__(self, config, server_client):
        self.config = config
        self.server_client = server_client

    async def responses(self, request, body):
        return NeMoGymResponse(
            id="smoke",
            created_at=int(time.time()),
            model=body.model,
            object="response",
            output=[
                NeMoGymResponseOutputMessage(
                    id="message",
                    content=[
                        NeMoGymResponseOutputText(
                            annotations=[], text=os.environ["MOCK_EXA_SEARCH_RESULT"], type="output_text"
                        )
                    ],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            ],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )
