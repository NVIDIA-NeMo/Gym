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
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from nemo_gym.server_utils import ServerClient
from responses_api_agents.proof_refinement_agent.app import (
    ModelServerRef,
    ProofRefinementAgent,
    ProofRefinementAgentConfig,
    ProofRefinementRunRequest,
    ResourcesServerRef,
)


class _FakeResp:
    ok = True

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.cookies: dict[str, str] = {}

    async def json(self) -> dict[str, Any]:
        return self._payload


def _usage(input_tokens: int, output_tokens: int, *, cached: int = 0, reasoning: int = 0) -> dict[str, Any]:
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": cached},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": reasoning},
        "total_tokens": input_tokens + output_tokens,
    }


def _model_response(response_id: str, text: str, usage: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": response_id,
        "created_at": 1,
        "model": "policy-model",
        "object": "response",
        "output": [
            {
                "id": f"msg-{response_id}",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "usage": usage,
    }


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

    async def test_run_sums_usage_across_correction_turns(self) -> None:
        config = ProofRefinementAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="proof_refinement_agent",
            resources_server=ResourcesServerRef(type="resources_servers", name="math_formal_lean"),
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
            max_correction_turns=2,
        )
        body = ProofRefinementRunRequest(responses_create_params={"input": "Prove it."})
        first = _model_response("resp-0", "proof attempt", _usage(300, 1400, cached=100, reasoning=1200))
        second = _model_response("resp-1", "fixed proof", _usage(1818, 24, cached=300))

        def verify_result(response: dict[str, Any], *, needs_correction: bool) -> dict[str, Any]:
            # math_formal_lean's /verify echoes the request body, including the model response.
            return body.model_dump() | {
                "response": response,
                "reward": 0.0 if needs_correction else 1.0,
                "proof_status": "error" if needs_correction else "success",
                "needs_correction": needs_correction,
                "correction_prompt": "Fix the proof." if needs_correction else None,
            }

        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock(
            side_effect=[
                _FakeResp({}),  # /seed_session
                _FakeResp(first),
                _FakeResp(verify_result(first, needs_correction=True)),
                _FakeResp(second),
                _FakeResp(verify_result(second, needs_correction=False)),
            ]
        )
        agent = ProofRefinementAgent(config=config, server_client=server_client)

        result = await agent.run(MagicMock(cookies={}), body)

        assert result.total_turns == 2
        assert [attempt["response"]["id"] for attempt in result.all_attempts] == ["resp-0", "resp-1"]
        assert result.response.id == "resp-1"
        # Usage covers every model call in the episode, not just the last one.
        assert result.response.usage.model_dump() == _usage(2118, 1424, cached=400, reasoning=1200)

    async def test_correction_turns_keep_the_callers_sampling_params(self) -> None:
        config = ProofRefinementAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="proof_refinement_agent",
            resources_server=ResourcesServerRef(type="resources_servers", name="math_formal_lean"),
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
            max_correction_turns=1,
        )
        body = ProofRefinementRunRequest(
            responses_create_params={"input": "Prove it.", "max_output_tokens": 1024, "temperature": 0.6, "top_p": 0.9}
        )
        first = _model_response("resp-0", "proof attempt", _usage(10, 20))
        second = _model_response("resp-1", "fixed proof", _usage(30, 40))
        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock(
            side_effect=[
                _FakeResp({}),  # /seed_session
                _FakeResp(first),
                _FakeResp(
                    body.model_dump()
                    | {"response": first, "reward": 0.0, "needs_correction": True, "correction_prompt": "Fix it."}
                ),
                _FakeResp(second),
                _FakeResp(body.model_dump() | {"response": second, "reward": 1.0, "needs_correction": False}),
            ]
        )
        agent = ProofRefinementAgent(config=config, server_client=server_client)

        await agent.run(MagicMock(cookies={}), body)

        model_calls = [
            c.kwargs["json"]
            for c in server_client.post.await_args_list
            if c.kwargs["url_path"].endswith("/v1/responses")
        ]
        correction_request = model_calls[1]
        assert correction_request["input"] == [{"role": "user", "content": "Fix it."}]
        # The output cap must survive the correction turn (Responses calls it max_output_tokens, not max_tokens).
        assert correction_request["max_output_tokens"] == 1024
        assert correction_request["temperature"] == 0.6
        assert correction_request["top_p"] == 0.9
