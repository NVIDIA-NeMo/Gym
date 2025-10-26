# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from unittest.mock import MagicMock

import pytest
from fastapi import Request

from nemo_gym.integrations.atropos import (
    AtroposCloseRequest,
    AtroposResourcesServerConfig,
    AtroposSeedSessionRequest,
    AtroposStepRequest,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.atropos.gsm8k_app import GSM8kAtroposServer


class TestGSM8kAtroposApp:
    """Test the GSM8k Atropos integration."""

    @pytest.mark.asyncio
    async def test_server_lifecycle(self) -> None:
        """Test the full lifecycle: seed, step, verify, close."""
        config = AtroposResourcesServerConfig(
            name="",
            host="0.0.0.0",
            port=8080,
            entrypoint="",
        )
        server = GSM8kAtroposServer(config=config, server_client=MagicMock(spec=ServerClient))

        mock_request = MagicMock(spec=Request)
        seed_resp = await server.seed_session(mock_request, AtroposSeedSessionRequest(task_idx=0))

        assert seed_resp.env_id, "Expected env_id"
        assert seed_resp.obs, "Expected non-empty observations"
        assert len(seed_resp.obs) >= 1, "Expected at least system/user message"
        assert seed_resp.system_prompt, "Expected system prompt"

        user_message = [msg for msg in seed_resp.obs if msg.role == "user"][0]
        assert user_message.content, "Expected question content"

        model_response = "<think>Let me solve this step by step...</think>\n\\boxed{42}"

        step_resp = await server.step(
            mock_request,
            AtroposStepRequest(env_id=seed_resp.env_id, action=model_response),
        )

        assert step_resp.done, "Expected done=True for single-turn task"
        assert "reward" in step_resp.model_dump() or step_resp.reward is not None, "Expected reward"
        assert step_resp.info, "Expected info dict"

        close_resp = await server.close(mock_request, AtroposCloseRequest(env_id=seed_resp.env_id))
        assert close_resp.success, "Expected success"
        assert seed_resp.env_id not in server.env_id_to_state, "Expected environment to be removed"

    @pytest.mark.asyncio
    async def test_correct_answer_scoring(self) -> None:
        """Test that correct answers get reward=1.0."""
        config = AtroposResourcesServerConfig(
            name="",
            host="0.0.0.0",
            port=8080,
            entrypoint="",
        )
        server = GSM8kAtroposServer(config=config, server_client=MagicMock(spec=ServerClient))

        mock_request = MagicMock(spec=Request)
        seed_resp = await server.seed_session(mock_request, AtroposSeedSessionRequest(task_idx=0))

        state = server.env_id_to_state[seed_resp.env_id]
        gold_answer = state.item["answer"].split("#")[-1].strip().replace(",", "")

        model_response = f"<think>Calculating...</think>\n\\boxed{{{gold_answer}}}"

        step_resp = await server.step(
            mock_request,
            AtroposStepRequest(env_id=seed_resp.env_id, action=model_response),
        )

        assert step_resp.reward == 1.0, f"Expected reward=1.0 for correct answer, got {step_resp.reward}"
        assert step_resp.info["correct"] is True, "Expected correct=True"

    @pytest.mark.asyncio
    async def test_incorrect_answer_scoring(self) -> None:
        """Test that incorrect answers get reward=0.0."""
        config = AtroposResourcesServerConfig(
            name="",
            host="0.0.0.0",
            port=8080,
            entrypoint="",
        )
        server = GSM8kAtroposServer(config=config, server_client=MagicMock(spec=ServerClient))

        mock_request = MagicMock(spec=Request)
        seed_resp = await server.seed_session(mock_request, AtroposSeedSessionRequest(task_idx=0))

        model_response = "<think>Let me guess...</think>\n\\boxed{999999}"

        step_resp = await server.step(
            mock_request,
            AtroposStepRequest(env_id=seed_resp.env_id, action=model_response),
        )

        assert step_resp.reward == 0.0, f"Expected reward=0.0 for incorrect answer, got {step_resp.reward}"
        assert step_resp.info["correct"] is False, "Expected correct=False"
