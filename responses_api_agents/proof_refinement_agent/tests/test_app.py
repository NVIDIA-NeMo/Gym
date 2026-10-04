# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from unittest.mock import AsyncMock, MagicMock

from fastapi import Request

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.proof_refinement_agent.app import (
    ModelServerRef,
    ProofRefinementAgent,
    ProofRefinementAgentConfig,
    ProofRefinementRunRequest,
    ResourcesServerRef,
)


class TestApp:
    def test_sanity(self) -> None:
        config = ProofRefinementAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            resources_server=ResourcesServerRef(
                type="resources_servers",
                name="",
            ),
            model_server=ModelServerRef(
                type="responses_api_models",
                name="",
            ),
        )
        ProofRefinementAgent(config=config, server_client=MagicMock(spec=ServerClient))

    def test_config_defaults(self) -> None:
        """Test that config has correct default values."""
        config = ProofRefinementAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            resources_server=ResourcesServerRef(
                type="resources_servers",
                name="math_formal_lean",
            ),
            model_server=ModelServerRef(
                type="responses_api_models",
                name="policy_model",
            ),
        )
        assert config.max_correction_turns == 0  # Default: single-turn
        assert config.include_all_attempts is True  # Default: include all attempts

    async def test_correction_turn_preserves_model_request_metadata(self) -> None:
        config = ProofRefinementAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="math_formal_lean_refinement_agent",
            resources_server=ResourcesServerRef(
                type="resources_servers",
                name="math_formal_lean",
            ),
            model_server=ModelServerRef(
                type="responses_api_models",
                name="policy_model",
            ),
            max_correction_turns=1,
        )

        model_response = {
            "id": "response-id",
            "created_at": 1,
            "model": "policy-model",
            "object": "response",
            "output": [],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        }
        request_params = NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "user", "content": "initial proof prompt"}],
            metadata={"extra_body": ('{"_ng_task_index": 7, "_ng_rollout_index": 3, "_ng_target_weight_version": 0}')},
            max_output_tokens=4096,
            temperature=0.7,
            top_p=0.9,
        )
        first_verify_response = {
            "responses_create_params": request_params.model_dump(),
            "response": model_response,
            "reward": 0.0,
            "proof_status": "failed",
            "needs_correction": True,
            "correction_prompt": "repair this proof",
        }
        second_verify_response = first_verify_response | {
            "reward": 1.0,
            "proof_status": "completed",
            "needs_correction": False,
            "correction_prompt": None,
        }

        def response(payload: dict) -> AsyncMock:
            mock = AsyncMock()
            mock.ok = True
            mock.cookies = {}
            mock.json.return_value = payload
            return mock

        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock(
            side_effect=[
                response({}),
                response(model_response),
                response(first_verify_response),
                response(model_response),
                response(second_verify_response),
            ]
        )
        agent = ProofRefinementAgent(config=config, server_client=server_client)
        run_request = ProofRefinementRunRequest.model_validate(
            {
                "responses_create_params": request_params.model_dump(),
                "_ng_task_index": 7,
                "_ng_rollout_index": 3,
                "_ng_target_weight_version": 0,
            }
        )
        http_request = MagicMock(spec=Request)
        http_request.cookies = {}

        result = await agent.run(http_request, run_request)

        assert result.total_turns == 2
        correction_request = server_client.post.call_args_list[3].kwargs["json"]
        assert correction_request.input[0].content == "repair this proof"
        assert correction_request.metadata == request_params.metadata
        assert correction_request.max_output_tokens == 4096
        assert correction_request.temperature == 0.7
        assert correction_request.top_p == 0.9
