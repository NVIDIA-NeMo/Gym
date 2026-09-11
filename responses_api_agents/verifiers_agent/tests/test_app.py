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
import asyncio
import json
from unittest.mock import MagicMock, patch

import httpx
from openai import AsyncOpenAI

from nemo_gym.config_types import ModelServerRef
from nemo_gym.server_utils import ServerClient
from responses_api_agents.verifiers_agent.app import (
    VerifiersAgent,
    VerifiersAgentConfig,
    _NoStoreCookieJar,
)


POLICY_URL = "http://policy.test/v1"
SESSION_COOKIE = "VLLMModel___policy_model=abc123; Path=/"


def _policy_server_that_sets_a_session_cookie(seen_cookie_headers: list) -> httpx.MockTransport:
    """Stand-in for Gym's vllm_model server.

    Like the real one (SessionMiddleware + add_session_id in
    nemo_gym/server_utils.py) it puts a session cookie on EVERY response, and
    it records the Cookie header each request arrived with so a test can see
    whether the client replayed it.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        seen_cookie_headers.append(request.headers.get("cookie"))
        body = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"},
            ],
        }
        return httpx.Response(200, json=body, headers={"set-cookie": SESSION_COOKIE})

    return httpx.MockTransport(handler)


async def _three_chat_completions(openai_client: AsyncOpenAI) -> None:
    for _ in range(3):
        await openai_client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])


class TestApp:
    def test_sanity(self) -> None:
        config = VerifiersAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            model_server=ModelServerRef(type="responses_api_models", name=""),
        )
        VerifiersAgent(config=config, server_client=MagicMock(spec=ServerClient))

    def test_convert_completion_keeps_tool_outputs_as_response_items(self) -> None:
        config = VerifiersAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            model_server=ModelServerRef(type="responses_api_models", name=""),
        )
        agent = VerifiersAgent(config=config, server_client=MagicMock(spec=ServerClient))

        rollout_output = {
            "prompt": [{"role": "user", "content": "q"}],
            "completion": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        json.dumps(
                            {
                                "id": "call_1",
                                "name": "python",
                                "arguments": json.dumps({"expr": "2+2"}),
                            }
                        )
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "4"},
                {"role": "assistant", "content": "answer"},
            ],
            "trajectory": [
                {
                    "completion": [
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "name": "python",
                                    "arguments": json.dumps({"expr": "2+2"}),
                                }
                            ],
                        }
                    ],
                    "tokens": {
                        "prompt_ids": [1],
                        "completion_ids": [2],
                        "completion_logprobs": [0.0],
                        "routed_experts": [[[0, 1]], [[2, 3]]],
                    },
                },
                {
                    "completion": [{"role": "assistant", "content": "answer"}],
                    "tokens": {
                        "prompt_ids": [3],
                        "completion_ids": [4],
                        "completion_logprobs": [-0.1],
                    },
                },
            ],
        }

        output = agent._convert_trajectory_to_output(rollout_output)

        assert [item["type"] for item in output] == ["function_call", "function_call_output", "message"]
        assert output[0]["call_id"] == "call_1"
        assert output[0]["name"] == "python"
        assert output[0]["arguments"] == json.dumps({"expr": "2+2"})
        assert output[0]["prompt_token_ids"] == [1]
        assert output[0]["routed_experts"] == [[[0, 1]], [[2, 3]]]
        assert output[1]["call_id"] == "call_1"
        assert output[1]["output"] == "4"
        assert output[2]["content"][0]["text"] == "answer"
        assert output[2]["prompt_token_ids"] == [3]


class TestPolicyClient:
    """The policy client decides which vLLM engine serves a rollout.

    vllm_model routes a request to ``sha256(session_id) % n_engines`` and mints
    the session id per cookie jar, so a client that replays Set-Cookie pins
    every rollout in this process to one engine (CMH 3670120, 2026-09-10: 6 of
    48 engines busy). These tests hold the two properties the fix relies on:
    the client is shared (a hot connection pool; per-rollout clients aborted
    468/512 rollouts on CMH 3670792 racing the router's keep-alive close) and
    it never sends a cookie back.
    """

    @staticmethod
    def _agent() -> VerifiersAgent:
        config = VerifiersAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        )
        return VerifiersAgent(config=config, server_client=MagicMock(spec=ServerClient))

    def test_policy_client_is_shared_across_rollouts(self) -> None:
        agent = self._agent()
        with patch.object(VerifiersAgent, "resolve_model_base_url", return_value=POLICY_URL):
            first = agent._get_client()
            second = agent._get_client()
        assert first is second
        assert str(first.client.base_url).rstrip("/") == POLICY_URL

    def test_client_base_url_carries_the_rollout_capture_prefix(self) -> None:
        agent = self._agent()

        def fake_resolve(model_server_name: str, rollout_id: str | None = None) -> str:
            prefix = f"/ng-rollout/{rollout_id}" if rollout_id else ""
            return f"http://policy{prefix}/v1"

        with (
            patch.object(VerifiersAgent, "resolve_model_base_url", side_effect=fake_resolve),
            patch.object(VerifiersAgent, "rollout_id_from_run", lambda _self, body: body.rollout_id),
        ):
            first = agent._get_client(MagicMock(rollout_id="7-2"))
            second = agent._get_client(MagicMock(rollout_id="7-3"))

        assert str(first.client.base_url).rstrip("/") == "http://policy/ng-rollout/7-2/v1"
        assert str(second.client.base_url).rstrip("/") == "http://policy/ng-rollout/7-3/v1"
        assert second is not first
        assert second.client._client is first.client._client

    def test_client_base_url_is_unprefixed_when_capture_is_disabled(self) -> None:
        agent = self._agent()
        with (
            patch.object(VerifiersAgent, "resolve_model_base_url", return_value=POLICY_URL),
            patch.object(VerifiersAgent, "rollout_id_from_run", lambda _self, body: None),
        ):
            first = agent._get_client(MagicMock())
            second = agent._get_client(MagicMock())

        assert first is second
        assert str(first.client.base_url).rstrip("/") == POLICY_URL

    def test_policy_client_never_replays_the_session_cookie(self) -> None:
        agent = self._agent()
        with patch.object(VerifiersAgent, "resolve_model_base_url", return_value=POLICY_URL):
            openai_client = agent._get_client().client

        # The jar must be ours. httpx.Cookies adopts a CookieJar instance but
        # copies an httpx.Cookies into a fresh stdlib jar, which silently
        # discards the subclass -- exactly the mistake this line catches.
        assert isinstance(openai_client._client.cookies.jar, _NoStoreCookieJar)

        seen = []
        openai_client._client._transport = _policy_server_that_sets_a_session_cookie(seen)
        asyncio.run(_three_chat_completions(openai_client))

        assert seen == [None, None, None], seen

    def test_a_default_openai_client_would_replay_the_session_cookie(self) -> None:
        """Positive control: proves the previous test can fail.

        The stock AsyncOpenAI sits on an httpx client that persists cookies, so
        from the second request on it carries the session cookie back -- the
        behaviour that pinned every rollout to one engine.
        """
        openai_client = AsyncOpenAI(base_url=POLICY_URL, api_key="EMPTY")  # pragma: allowlist secret
        seen = []
        openai_client._client._transport = _policy_server_that_sets_a_session_cookie(seen)
        asyncio.run(_three_chat_completions(openai_client))

        assert seen[0] is None
        assert seen[1] == seen[2] == SESSION_COOKIE.split(";")[0]
