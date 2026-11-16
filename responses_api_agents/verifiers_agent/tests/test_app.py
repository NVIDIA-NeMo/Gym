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
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest
from fastapi.testclient import TestClient
from openai import DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT, AsyncOpenAI, Timeout
from verifiers.clients import NeMoRLChatCompletionsClient

from nemo_gym.client import ServerClient, apply_rollout_prefix
from nemo_gym.config import OBSERVABILITY_ENABLED_KEY_NAME, ModelServerRef
from nemo_gym.core import FailureReportRow, StageOutcomePolicyConfig, model_sampling_context
from nemo_gym.core.outcome_policy import is_policy_counted_zero
from nemo_gym.rollouts.rollout_recovery import masked_infrastructure_kind
from responses_api_agents.verifiers_agent.app import (
    VerifiersAgent,
    VerifiersAgentConfig,
    _as_response_usage,
    _NoStoreCookieJar,
    _rollout_error_fact,
)
from responses_api_agents.verifiers_agent.app import (
    logger as agent_app_logger,
)


POLICY_URL = "http://policy.test/v1"
SESSION_COOKIE = "VLLMModel___policy_model=abc123; Path=/"


def _policy_server_that_sets_a_session_cookie(seen_cookie_headers: list) -> httpx.MockTransport:
    """Stand-in for Gym's vllm_model server.

    Like the real one (SessionMiddleware + SessionIDMiddleware, set up in
    nemo_gym/server_runtime/base_server.py) it puts a session cookie on EVERY response, and
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

    def test_eval_rollout_without_token_ids_is_not_reported_as_empty(self, caplog) -> None:
        """Eval mode (`return_token_id_information: false`) returns no token ids, but the trajectory is complete."""
        tool_call = {"id": "call_1", "name": "python", "arguments": json.dumps({"expr": "2+2"})}
        completion = [
            {"role": "assistant", "content": "", "tool_calls": [json.dumps(tool_call)]},
            {"role": "tool", "tool_call_id": "call_1", "content": "4"},
            {"role": "assistant", "content": "answer"},
        ]
        rollout_output = {
            "completion": completion,
            "trajectory": [
                {"completion": [{"role": "assistant", "content": "", "tool_calls": [tool_call]}]},
                {"completion": [{"role": "assistant", "content": "answer"}]},
            ],
        }

        with caplog.at_level(logging.DEBUG, logger=agent_app_logger.name):
            output = self._agent()._convert_trajectory_to_output(rollout_output)

        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert "token ids were likely not requested" in caplog.text
        assert [item["type"] for item in output] == ["function_call", "function_call_output", "message", "message"]
        assert output[-1]["id"] == "msg_empty"
        assert output[-1]["generation_token_ids"] == [0]

    @pytest.mark.parametrize(
        ("completion", "expect_warning"),
        [
            # Stopped by max_turns right after a tool call: only function_call items, no assistant message.
            (
                [
                    {"role": "assistant", "content": "", "tool_calls": [json.dumps({"id": "c1", "name": "f"})]},
                    {"role": "tool", "tool_call_id": "c1", "content": "ok"},
                ],
                False,
            ),
            # Only non-assistant messages: a user or tool message must not count as an assistant turn.
            ([{"role": "user", "content": "q"}, {"role": "tool", "tool_call_id": "c1", "content": "ok"}], True),
        ],
    )
    def test_each_kind_of_assistant_turn_decides_the_warning(self, caplog, completion, expect_warning) -> None:
        with caplog.at_level(logging.WARNING, logger=agent_app_logger.name):
            self._agent()._convert_trajectory_to_output({"completion": completion, "trajectory": []})

        assert ("Returning empty trajectory" in caplog.text) is expect_warning

    def test_rollout_without_completion_items_still_warns_empty_trajectory(self, caplog) -> None:
        rollout_output = {"completion": [], "trajectory": [], "error": {"error": "sandbox quota exceeded"}}

        with caplog.at_level(logging.WARNING, logger=agent_app_logger.name):
            output = self._agent()._convert_trajectory_to_output(rollout_output)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "Returning empty trajectory" in warnings[0].getMessage()
        assert "sandbox quota exceeded" in warnings[0].getMessage()
        assert [item["id"] for item in output] == ["msg_empty"]

    def test_rollout_error_after_assistant_turns_is_still_logged(self, caplog) -> None:
        rollout_output = {
            "completion": [{"role": "assistant", "content": "partial answer"}],
            "trajectory": [{"completion": [{"role": "assistant", "content": "partial answer"}]}],
            "error": {"error": "ToolError", "error_chain_repr": "ToolError('sandbox died')"},
        }

        with caplog.at_level(logging.WARNING, logger=agent_app_logger.name):
            output = self._agent()._convert_trajectory_to_output(rollout_output)

        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "ToolError" in warnings[0]
        assert "Returning empty trajectory" not in warnings[0]
        assert output[-1]["id"] == "msg_empty"

    @staticmethod
    def _agent() -> VerifiersAgent:
        config = VerifiersAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            model_server=ModelServerRef(type="responses_api_models", name=""),
        )
        return VerifiersAgent(config=config, server_client=MagicMock(spec=ServerClient))


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
    def _agent(**config_overrides) -> VerifiersAgent:
        config = VerifiersAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
            **config_overrides,
        )
        return VerifiersAgent(config=config, server_client=MagicMock(spec=ServerClient))

    def test_policy_client_keeps_the_sdk_deadlines_by_default(self) -> None:
        agent = self._agent()
        with patch.object(VerifiersAgent, "resolve_model_base_url", return_value=POLICY_URL):
            client = agent._get_client().client
        assert client.timeout == DEFAULT_TIMEOUT
        assert client.max_retries == DEFAULT_MAX_RETRIES

    def test_policy_client_applies_configured_deadlines_and_retries(self) -> None:
        agent = self._agent(client_timeout_s=3600, client_connect_timeout_s=60, client_max_retries=0)
        with patch.object(VerifiersAgent, "resolve_model_base_url", return_value=POLICY_URL):
            client = agent._get_client().client
        assert client.timeout == Timeout(connect=60, read=3600, write=3600, pool=3600)
        assert client.max_retries == 0

    def test_connect_timeout_alone_keeps_the_sdk_read_deadline_and_retries(self) -> None:
        agent = self._agent(client_connect_timeout_s=60)
        with patch.object(VerifiersAgent, "resolve_model_base_url", return_value=POLICY_URL):
            client = agent._get_client().client
        assert client.timeout == Timeout(
            connect=60, read=DEFAULT_TIMEOUT.read, write=DEFAULT_TIMEOUT.write, pool=DEFAULT_TIMEOUT.pool
        )
        assert client.max_retries == DEFAULT_MAX_RETRIES

    def test_rollout_prefixed_client_inherits_deadlines_and_retries(self) -> None:
        agent = self._agent(client_timeout_s=3600, client_connect_timeout_s=60, client_max_retries=0)

        def fake_resolve(model_server_name: str, rollout_id: str | None = None, *, trace_context: bool = True) -> str:
            prefix = f"/ng-rollout/{rollout_id}" if rollout_id else ""
            return f"http://policy{prefix}/v1"

        with (
            patch.object(VerifiersAgent, "resolve_model_base_url", side_effect=fake_resolve),
            patch.object(VerifiersAgent, "rollout_id_from_run", lambda _self, body: body.rollout_id),
        ):
            prefixed = agent._get_client(MagicMock(rollout_id="7-2")).client

        assert str(prefixed.base_url).rstrip("/") == "http://policy/ng-rollout/7-2/v1"
        assert prefixed.timeout == Timeout(connect=60, read=3600, write=3600, pool=3600)
        assert prefixed.max_retries == 0

    def test_policy_client_is_shared_across_rollouts(self) -> None:
        agent = self._agent()
        with patch.object(VerifiersAgent, "resolve_model_base_url", return_value=POLICY_URL):
            first = agent._get_client()
            second = agent._get_client()
        assert first is second
        assert str(first.client.base_url).rstrip("/") == POLICY_URL

    def test_client_base_url_carries_the_rollout_capture_prefix(self) -> None:
        agent = self._agent()

        def fake_resolve(model_server_name: str, rollout_id: str | None = None, *, trace_context: bool = True) -> str:
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

    def test_a_rollout_with_sampling_values_gets_its_own_url_with_capture_off(self) -> None:
        agent = self._agent()

        def fake_resolve(model_server_name: str, rollout_id: str | None = None, *, trace_context: bool = True) -> str:
            return f"{apply_rollout_prefix('http://policy', rollout_id, trace_context=trace_context)}/v1"

        with (
            patch.object(VerifiersAgent, "resolve_model_base_url", side_effect=fake_resolve),
            patch.object(VerifiersAgent, "rollout_id_from_run", lambda _self, body: None),
        ):
            with model_sampling_context("SEG"):
                sampled = agent._get_client(MagicMock())
            unsampled = agent._get_client(MagicMock())

        # The shared client is built inside the first rollout, yet its URL carries no rollout's values.
        assert str(unsampled.client.base_url).rstrip("/") == "http://policy/v1"
        assert str(sampled.client.base_url).rstrip("/") == "http://policy/ng-sampling/SEG/v1"
        assert sampled.client._client is unsampled.client._client

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


class TestPrefixedResponsesRoute:
    """The `/ng-rollout/<id>/v1/responses` route must correlate too.

    `rollout_id_from_run` only sees `/run`, where rollout collection injects
    `_ng_rollout_id` into the body. The prefixed Responses route carries the id
    in the PATH, and agents get no `RolloutContextMiddleware` (that is installed
    on resources servers), so nothing recovered it and the route silently built
    an unprefixed client -- capture lost for every call made through it.

    These drive the real FastAPI routes rather than patching
    `rollout_id_from_run`, so they exercise the actual protocol.
    """

    @staticmethod
    def _agent(*, observability: bool) -> VerifiersAgent:
        config = VerifiersAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
            vf_env_id="stub_env",
        )
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {OBSERVABILITY_ENABLED_KEY_NAME: observability}
        return VerifiersAgent(config=config, server_client=server_client)

    @staticmethod
    def _resolve(model_server_name: str, rollout_id: str | None = None, *, trace_context: bool = True) -> str:
        return "http://policy%s/v1" % (f"/ng-rollout/{rollout_id}" if rollout_id else "")

    def _base_url_seen_by(self, agent: VerifiersAgent, url: str) -> str:
        """POST `url` through the real app and report the client's base_url."""
        seen: list[str] = []

        async def fake_run_group(*, group_inputs, client, model, sampling_args, state_columns):
            seen.append(str(client.client.base_url).rstrip("/"))
            return [{"reward": 0.0, "metrics": {}, "completion": [], "trajectory": []}]

        env = MagicMock()
        env.run_group = fake_run_group

        with (
            patch.object(VerifiersAgent, "resolve_model_base_url", side_effect=self._resolve),
            patch.object(VerifiersAgent, "_get_env", return_value=env),
        ):
            client = TestClient(agent.setup_webserver())
            response = client.post(
                url,
                json={
                    "task_idx": 0,
                    "responses_create_params": {"input": [{"role": "user", "content": "hi"}]},
                },
            )
        assert response.status_code == 200, response.text
        assert seen, "run_group was never reached; the route did not execute"
        return seen[0]

    def test_prefixed_route_correlates_the_model_client(self) -> None:
        agent = self._agent(observability=True)
        assert self._base_url_seen_by(agent, "/ng-rollout/7-2/v1/responses") == ("http://policy/ng-rollout/7-2/v1")

    def test_unprefixed_route_stays_unprefixed(self) -> None:
        agent = self._agent(observability=True)
        assert self._base_url_seen_by(agent, "/v1/responses") == "http://policy/v1"

    def test_prefixed_route_is_inert_when_capture_is_disabled(self) -> None:
        """Capture off means the shared unprefixed client, as on the body path."""
        agent = self._agent(observability=False)
        assert self._base_url_seen_by(agent, "/ng-rollout/7-2/v1/responses") == "http://policy/v1"


def _policy_server_returning(responses: list[dict]) -> httpx.MockTransport:
    """Stand-in policy server that replays `responses` in order, then repeats the last."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        index = min(calls["n"], len(responses) - 1)
        calls["n"] += 1
        return httpx.Response(200, json=responses[index])

    return httpx.MockTransport(handler)


def _chat_completion(response_id: str, *, prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


class TestRolloutObservability:
    """The evidence rollout health reads has to come from somewhere.

    `verifiers` owns the call loop, so the agent never sees a model response and
    for a long time emitted no `ng_trajectory` at all. Every check that reads
    turns or bound model calls then reported `unobserved`: a run of this agent
    scored `0 healthy, 0 unhealthy, N unobserved`, which is indistinguishable
    from a healthy run at a glance. These hold the two things the checks need --
    one turn per model call, each claiming the call it came from.
    """

    @staticmethod
    def _agent() -> VerifiersAgent:
        config = VerifiersAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
            vf_env_id="stub_env",
        )
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {OBSERVABILITY_ENABLED_KEY_NAME: True}
        return VerifiersAgent(config=config, server_client=server_client)

    def test_recording_client_records_every_response_id_and_sums_usage(self) -> None:
        agent = self._agent()
        with (
            patch.object(VerifiersAgent, "resolve_model_base_url", return_value="http://policy/ng-rollout/7-2/v1"),
            patch.object(VerifiersAgent, "rollout_id_from_run", lambda _self, body: "7-2"),
        ):
            client = agent._get_client(MagicMock(), invocation_id="inv-1")

        client.client._client._transport = _policy_server_returning(
            [
                _chat_completion("chatcmpl-a", prompt_tokens=10, completion_tokens=3),
                _chat_completion("chatcmpl-b", prompt_tokens=20, completion_tokens=5),
            ]
        )
        for _ in range(2):
            asyncio.run(
                client.get_native_response(
                    prompt=[{"role": "user", "content": "hi"}], model="m", sampling_args={}, tools=None
                )
            )

        assert client.recorded_response_ids == ["chatcmpl-a", "chatcmpl-b"]
        assert client.recorded_usage.input_tokens == 30
        assert client.recorded_usage.output_tokens == 8
        assert client.recorded_usage.total_tokens == 38

    def _trajectory_for(self, output: list[dict], response_ids: list[str]):
        agent = self._agent()
        return agent._build_trajectory(
            invocation_id="inv-1",
            task_id="0",
            rollout_id="0-0",
            output=output,
            response_ids=response_ids,
            conversation=[],
            status="completed" if response_ids else "incomplete",
        )

    @staticmethod
    def _two_turn_output() -> list[dict]:
        return [
            {"type": "message", "id": "msg_1", "content": [{"type": "output_text", "text": "first"}]},
            {"type": "function_call", "id": "fc_1", "call_id": "c1", "name": "t", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "done"},
            {"type": "message", "id": "msg_2", "content": [{"type": "output_text", "text": "second"}]},
        ]

    def test_each_turn_claims_only_the_call_it_came_from(self) -> None:
        """`_canonical_model_call_references` reads `turns[*].model_calls`.

        Handing every turn the whole call list still binds, but then no finding
        can be attributed to a turn and a per-call locator points anywhere.
        """
        trajectory = self._trajectory_for(self._two_turn_output(), ["chatcmpl-a", "chatcmpl-b"])

        assert [turn.turn_no for turn in trajectory.turns] == [1, 2]
        assert [[ref.response_id for ref in turn.model_calls] for turn in trajectory.turns] == [
            ["chatcmpl-a"],
            ["chatcmpl-b"],
        ]
        assert trajectory.gaps == []

    def test_turn_answers_are_sliced_per_turn(self) -> None:
        """`agent_turn_hollow` asks whether THIS turn produced anything.

        Giving each turn the whole rollout's items makes every turn look
        non-hollow as long as any one turn spoke, so the check can never fire.
        """
        trajectory = self._trajectory_for(self._two_turn_output(), ["chatcmpl-a", "chatcmpl-b"])

        assert [[item["id"] for item in turn.answer] for turn in trajectory.turns] == [["msg_1", "fc_1"], ["msg_2"]]
        # cumulative within the invocation: turn 2 adds no tool call of its own
        assert [turn.step_count for turn in trajectory.turns] == [1, 1]

    def test_a_turn_with_no_items_behind_it_stays_empty(self) -> None:
        """A claimed call that produced nothing is what `agent_turn_hollow` is for."""
        trajectory = self._trajectory_for(self._two_turn_output(), ["chatcmpl-a", "chatcmpl-b", "chatcmpl-c"])

        assert len(trajectory.turns) == 3
        assert trajectory.turns[2].answer == []
        # the running count carries through a turn that did nothing
        assert [turn.step_count for turn in trajectory.turns] == [1, 1, 1]

    def test_the_empty_token_placeholder_does_not_stand_in_for_a_turn(self) -> None:
        """`_convert_trajectory_to_output` appends a `msg_empty` placeholder when
        a rollout came back with no token ids. It is a message item, so counting
        it would dress the one case worth catching -- a call that produced
        nothing at all -- as a turn that spoke, and `agent_turn_hollow` would
        pass on it."""
        output = [{"type": "message", "id": "msg_empty", "content": [{"type": "output_text", "text": ""}]}]
        trajectory = self._trajectory_for(output, ["chatcmpl-a"])

        assert len(trajectory.turns) == 1
        assert trajectory.turns[0].answer == []

    def test_a_rollout_with_no_model_calls_reports_a_gap(self) -> None:
        """Every call failing must not look like an ordinary empty trajectory."""
        trajectory = self._trajectory_for([], [])

        assert trajectory.turns == []
        assert [gap.code for gap in trajectory.gaps] == ["model_call_reference_unavailable"]
        assert trajectory.invocations[0].status == "incomplete"


class TestExportStateColumns:
    """``export_state_columns`` copies extra verifiers State keys into the response, and is off by default."""

    RECORDS = [{"index": 0, "type": "gmail_message_sent_to", "app": "gmail", "role": "objective", "passed": True}]

    def _run(self, **config_overrides) -> tuple[list[str], dict]:
        config = VerifiersAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
            vf_env_id="stub_env",
            **config_overrides,
        )
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {OBSERVABILITY_ENABLED_KEY_NAME: False}
        agent = VerifiersAgent(config=config, server_client=server_client)
        seen_columns = []

        async def fake_run_group(*, group_inputs, client, model, sampling_args, state_columns):
            seen_columns.extend(state_columns)
            state = {
                "reward": 1.0,
                "metrics": {},
                "completion": [],
                "trajectory": [],
                "assertion_results": self.RECORDS,
            }
            # Like verifiers' state_to_output: a requested column missing from the State comes back as None.
            return [{**state, **{column: state.get(column) for column in state_columns}}]

        env = MagicMock()
        env.run_group = fake_run_group
        with (
            patch.object(VerifiersAgent, "resolve_model_base_url", return_value=POLICY_URL),
            patch.object(VerifiersAgent, "_get_env", return_value=env),
        ):
            response = TestClient(agent.setup_webserver()).post(
                "/run",
                json={"task_idx": 0, "responses_create_params": {"input": [{"role": "user", "content": "hi"}]}},
            )
        assert response.status_code == 200, response.text
        return seen_columns, response.json()

    def test_default_requests_only_the_trajectory_and_omits_the_field(self) -> None:
        columns, body = self._run()
        assert columns == ["trajectory"]
        assert "exported_state" not in body["response"]

    def test_requested_columns_reach_the_response(self) -> None:
        columns, body = self._run(export_state_columns=["assertion_results", "trajectory"])
        assert columns == ["trajectory", "assertion_results"]
        assert body["response"]["exported_state"] == {"assertion_results": self.RECORDS}

    def test_columns_missing_from_the_state_are_omitted_with_a_warning(self, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger=agent_app_logger.name):
            _, body = self._run(export_state_columns=["assertion_result"])
        assert "exported_state" not in body["response"]
        assert "['assertion_result'] not on the verifiers State" in caplog.text


class TestSharedClientTraceContext:
    """The cached policy client is shared across rollouts, so its base URL must not name a span.

    Per-run copies may: they live for one rollout. See Gym #4836.
    """

    SEGMENT = "/ng-trace/00-" + "a" * 32 + "-" + "b" * 16 + "-01"

    def _agent(self, *, observability: bool) -> VerifiersAgent:
        config = VerifiersAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
            vf_env_id="stub_env",
        )
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {
            OBSERVABILITY_ENABLED_KEY_NAME: observability,
            "policy_model": {"responses_api_models": {"vllm_model": {"host": "policy", "port": 8000}}},
        }
        server_client._build_server_base_url = lambda cfg: "http://policy:8000"
        return VerifiersAgent(config=config, server_client=server_client)

    def test_cached_client_has_no_trace_context_while_per_run_copies_do(self, monkeypatch) -> None:
        # Stand in for an exporting tracer with a current span at every call.
        monkeypatch.setattr("nemo_gym.client.server_client.trace_context_path_segment", lambda: self.SEGMENT)

        agent = self._agent(observability=False)
        shared = agent._get_client({"_ng_task_index": 0, "_ng_rollout_index": 0})
        assert str(shared.client.base_url).rstrip("/") == "http://policy:8000/v1"
        assert agent._get_client({"_ng_task_index": 1, "_ng_rollout_index": 0}) is shared

        agent = self._agent(observability=True)
        per_run = agent._get_client({"_ng_task_index": 2, "_ng_rollout_index": 0})
        assert str(per_run.client.base_url).rstrip("/") == f"http://policy:8000/ng-rollout/2-0{self.SEGMENT}/v1"
        cached = agent.client_cache["policy_model"]
        assert str(cached.client.base_url).rstrip("/") == "http://policy:8000/v1"


class TestOwnGradingRecord:
    """The agent runs its rubric itself, so with observability on it returns its own verifier invocation."""

    @staticmethod
    def _run(funcs, *, observability: bool = True) -> dict:
        import types

        import verifiers as vf

        rubric = vf.Rubric(funcs=funcs)

        async def fake_run_group(*, group_inputs, client, model, sampling_args, state_columns):
            # Score as verifiers does: each reward function through the rubric, an error scored 0.0.
            state = {"prompt": [], "completion": [], "answer": "", "info": {}, "input": {}}
            rewards = [await rubric._call_individual_reward_func(func, state) for func in rubric.funcs]
            return [{"reward": sum(rewards), "metrics": {}, "completion": [], "trajectory": []}]

        config = VerifiersAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="automationbench",
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
            vf_env_id="stub_env",
        )
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {OBSERVABILITY_ENABLED_KEY_NAME: observability}
        agent = VerifiersAgent(config=config, server_client=server_client)
        env = types.SimpleNamespace(rubric=rubric, run_group=fake_run_group)
        with (
            patch.object(VerifiersAgent, "resolve_model_base_url", return_value=POLICY_URL),
            patch("responses_api_agents.verifiers_agent.app.vf.load_environment", return_value=env),
        ):
            response = TestClient(agent.setup_webserver()).post(
                "/run",
                json={
                    "task_idx": 0,
                    "_ng_task_index": 0,
                    "_ng_rollout_index": 0,
                    "responses_create_params": {"input": [{"role": "user", "content": "hi"}]},
                },
            )
        assert response.status_code == 200, response.text
        return response.json()

    def test_a_synchronous_rubric_returns_the_agents_own_scored_grading_record(self) -> None:
        def aa_headline(state, **kwargs) -> float:
            return 1.0

        body = self._run([aa_headline])
        [invocation] = body["ng_verifier_invocations"]
        assert invocation["status"] == "completed" and invocation["failure_kind"] is None
        assert invocation["model_calls"] == [] and invocation.get("grading_operations") is None
        record = invocation["grading_record"]
        assert (record["verifier"], record["score"], record["verdict"]) == ("automationbench", 1.0, "scored")
        # Bound to the row as collection saves it: the response and reward of this /run reply.
        from nemo_gym.trajectory import grading_result_binding

        assert record["result_binding_sha256"] == grading_result_binding(body)
        assert body["reward"] == 1.0 and body.get("mask_sample") is False

    def test_a_reward_function_that_raises_is_a_grading_fault_not_a_zero_score(self) -> None:
        def partial_credit(state, **kwargs) -> float:
            raise KeyError("world_state")

        body = self._run([partial_credit])
        assert body["mask_sample"] is True and body["failure_kind"] == "verifier_error"
        assert "partial_credit" in body["failure_reason"]
        [invocation] = body["ng_verifier_invocations"]
        assert invocation["failure_kind"] == "verifier_error"
        assert invocation["grading_record"]["verdict"] == "masked"

    def test_a_rubric_that_may_ask_a_model_is_not_observed(self) -> None:
        """No invocation at all, as before: one without a record would read as ungraded and be re-verified."""

        async def judge(state, **kwargs) -> float:
            return 1.0

        body = self._run([judge])
        assert not body.get("ng_verifier_invocations")
        assert body["reward"] == 1.0 and body.get("mask_sample") is False

    def test_observability_off_returns_no_verifier_invocation(self) -> None:
        def aa_headline(state, **kwargs) -> float:
            return 1.0

        body = self._run([aa_headline], observability=False)
        assert not body.get("ng_verifier_invocations")
        assert body["reward"] == 1.0


class TestProgrammaticRubricOfRealEnvironments:
    """verifiers' own environments wrap the rubric in a RubricGroup with their weight-0 monitor rubrics."""

    @staticmethod
    def _tool_env(*funcs):
        import verifiers as vf
        from datasets import Dataset

        return vf.ToolEnv(
            dataset=Dataset.from_list([{"question": "q", "answer": "a"}]),
            rubric=vf.Rubric(funcs=list(funcs)),
            tools=[],
            max_turns=2,
        )

    def test_a_tool_env_with_synchronous_reward_functions_is_programmatic_and_watched(self) -> None:
        from responses_api_agents.verifiers_agent.app import _programmatic_rubric

        def correct(completion, answer, **kwargs) -> float:
            return 1.0

        env = self._tool_env(correct)
        assert [type(m).__name__ for m in env.rubric.rubrics] == [
            "Rubric",
            "MultiTurnMonitorRubric",
            "ToolMonitorRubric",
        ]
        assert _programmatic_rubric(env) is True
        [watched] = env.rubric.rubrics[0].funcs
        assert watched._ng_watched_reward_func is True and watched.__wrapped__ is correct
        # The monitors' own metrics are left as they are.
        assert all(not getattr(f, "_ng_watched_reward_func", False) for m in env.rubric.rubrics[1:] for f in m.funcs)

    def test_a_tool_env_whose_reward_function_is_async_is_not_programmatic(self) -> None:
        from responses_api_agents.verifiers_agent.app import _programmatic_rubric

        async def judged(completion, answer, **kwargs) -> float:
            return 1.0

        assert _programmatic_rubric(self._tool_env(judged)) is False

    def test_a_monitor_that_carries_reward_weight_is_not_left_out(self) -> None:
        from responses_api_agents.verifiers_agent.app import _programmatic_rubric

        def correct(completion, answer, **kwargs) -> float:
            return 1.0

        env = self._tool_env(correct)
        env.rubric.rubrics[1].weights[0] = 1.0
        assert _programmatic_rubric(env) is False


class TestRolloutErrorFact:
    """The error a rollout ended on is recorded on the row as a typed fact; the reward and the mask do not change."""

    @pytest.mark.parametrize(
        "error, expected",
        [
            (None, None),
            ({}, None),
            (
                {
                    "error": "ModelError",
                    "message": "Error code: 502 - {'detail': 'Bad Gateway'}",
                    "error_chain_repr": "ModelError(...) -> InternalServerError('Error code: 502')",
                    "error_chain_str": "ModelError -> InternalServerError",
                },
                {"type": "ModelError", "status_code": 502, "kind_candidate": "http_5xx"},
            ),
            (
                {
                    "error": "ModelError",
                    "message": "Request timed out.",
                    "error_chain_repr": "ModelError('Request timed out.') -> APITimeoutError('Request timed out.')",
                    "error_chain_str": "ModelError -> APITimeoutError",
                },
                {"type": "ModelError", "status_code": None, "kind_candidate": "timeout"},
            ),
            (
                {
                    "error": "ModelError",
                    "message": "Connection error.",
                    "error_chain_repr": "ModelError('Connection error.') -> APIConnectionError('Connection error.')",
                    "error_chain_str": "ModelError -> APIConnectionError",
                },
                {"type": "ModelError", "status_code": None, "kind_candidate": "connection"},
            ),
            (
                {
                    "error": "RateLimitError",
                    "message": "Error code: 429",
                    "error_chain_repr": "",
                    "error_chain_str": "",
                },
                {"type": "RateLimitError", "status_code": 429, "kind_candidate": "http_429"},
            ),
            (
                {
                    "error": "SandboxError",
                    "message": "sandbox died",
                    "error_chain_repr": "SandboxError('sandbox died')",
                    "error_chain_str": "SandboxError",
                },
                {"type": "SandboxError", "status_code": None, "kind_candidate": "infra"},
            ),
            (
                {
                    "error": "EmptyModelResponseError",
                    "message": "empty",
                    "error_chain_repr": "EmptyModelResponseError('empty')",
                    "error_chain_str": "EmptyModelResponseError",
                },
                {"type": "EmptyModelResponseError", "status_code": None, "kind_candidate": "model_response"},
            ),
            ("sandbox quota exceeded", {"type": None, "status_code": None, "kind_candidate": "other"}),
        ],
    )
    def test_the_fact_types_the_error(self, error, expected) -> None:
        fact = _rollout_error_fact(error)
        if expected is None:
            assert fact is None
            return
        assert {key: fact[key] for key in expected} == expected
        assert set(fact) == {"type", "chain", "status_code", "kind_candidate", "message_head"}

    def test_the_message_head_is_bounded(self) -> None:
        fact = _rollout_error_fact({"error": "ValueError", "message": "x" * 5000})
        assert len(fact["message_head"]) == 500

    def _run(self, error, *, call_errors=(), rubric_reward=0.0, **config_options) -> dict:
        """One rollout whose policy client sees each of ``call_errors`` (an exception raises, anything else returns)
        before verifiers reports ``error`` as the error the rollout ended on."""
        config = VerifiersAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
            vf_env_id="stub_env",
            **config_options,
        )
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {OBSERVABILITY_ENABLED_KEY_NAME: False}
        agent = VerifiersAgent(config=config, server_client=server_client)

        async def fake_run_group(*, group_inputs, client, model, sampling_args, state_columns):
            with patch.object(NeMoRLChatCompletionsClient, "get_native_response", AsyncMock(side_effect=call_errors)):
                for _ in call_errors:
                    try:
                        await client.get_native_response([], "policy", {})
                    except Exception:
                        pass
            return [{"reward": rubric_reward, "metrics": {}, "completion": [], "trajectory": [], "error": error}]

        env = MagicMock()
        env.run_group = fake_run_group
        with (
            patch.object(VerifiersAgent, "resolve_model_base_url", return_value=POLICY_URL),
            patch.object(VerifiersAgent, "_get_env", return_value=env),
        ):
            response = TestClient(agent.setup_webserver()).post(
                "/run",
                json={"task_idx": 0, "responses_create_params": {"input": [{"role": "user", "content": "hi"}]}},
            )
        assert response.status_code == 200, response.text
        return response.json()

    @staticmethod
    def _http_error(status: int) -> openai.APIStatusError:
        request = httpx.Request("POST", "http://policy:8000/v1/chat/completions")
        return openai.APIStatusError(
            f"Error code: {status} - private-body-marker",
            response=httpx.Response(status, request=request),
            body=None,
        )

    @staticmethod
    def _ended_on_model_error(status: int | None = None) -> dict:
        message = f"Error code: {status}" if status else "Request timed out."
        return {"error": "ModelError", "message": message, "error_chain_str": "ModelError"}

    def test_a_rollout_that_ended_on_an_error_records_it_on_the_row(self) -> None:
        body = self._run(self._ended_on_model_error(503))
        assert body["rollout_error"]["kind_candidate"] == "http_5xx" and body["rollout_error"]["status_code"] == 503
        assert body["response"]["rollout_error"] == body["rollout_error"]
        # The error text names a status, but no model call reported it: the diagnostic candidate decides nothing.
        assert body["reward"] == 0.0 and not body.get("mask_sample") and body.get("failure_kind") is None

    def test_a_first_turn_503_without_an_answer_is_a_provider_event_the_rerun_path_acts_on(self) -> None:
        body = self._run(self._ended_on_model_error(503), call_errors=[self._http_error(503)])

        assert body["mask_sample"] is True and body["reward"] == 0.0
        assert (
            body["failure_kind"] == "model_request_error"
            and body["failure_reason"] == "HTTP 503 from the model server"
        )
        assert (body["failure_outcome_action"], body["failure_outcome_attribution"]) == ("retry", "evaluation")
        # The saved row names its selected event; it is a model-boundary event with its status, stage and cause.
        report = FailureReportRow.report(body)
        event = report.events[report.failure_id]
        assert (event.boundary, event.failure_kind, event.http_status, event.stage) == (
            "model",
            "model_request_error",
            503,
            "agent",
        )
        assert event.details.request_upload_complete is True
        assert (event.foreign_error.exception_type, event.foreign_error.http_status) == ("openai.APIStatusError", 503)
        # The SDK's text, which holds the reply body, is not copied.
        assert "private-body-marker" not in json.dumps(body)
        # Recovery acts on the saved row alone: it is a masked, evaluation-side result with no answer to keep.
        assert masked_infrastructure_kind(body) == "model_request_error"

    def test_the_same_503_is_not_rerun_when_the_policy_ends_it_ungraded(self) -> None:
        policy = StageOutcomePolicyConfig(by_stage={"agent": {"by_status_code": {503: "fail"}}})
        body = self._run(self._ended_on_model_error(503), call_errors=[self._http_error(503)], outcome_policy=policy)

        assert body["mask_sample"] is True and body["failure_outcome_action"] == "fail"
        assert masked_infrastructure_kind(body) is None

    def test_a_model_side_rejection_is_counted_as_a_zero(self) -> None:
        body = self._run(self._ended_on_model_error(400), call_errors=[self._http_error(400)])

        assert (body["reward"], body["mask_sample"], body["failure_kind"]) == (0.0, False, "model_request_error")
        assert is_policy_counted_zero(body)
        assert masked_infrastructure_kind(body) is None

    def test_a_counted_zero_is_a_zero_even_when_the_rubric_scored_the_errored_rollout_positive(self) -> None:
        body = self._run(self._ended_on_model_error(400), call_errors=[self._http_error(400)], rubric_reward=1.0)

        assert (body["reward"], body["mask_sample"], body["failure_outcome_action"]) == (0.0, False, "count_as_zero")
        assert is_policy_counted_zero(body)

    def test_a_masked_retry_keeps_the_rubric_reward_it_was_given(self) -> None:
        body = self._run(self._ended_on_model_error(503), call_errors=[self._http_error(503)], rubric_reward=1.0)

        assert body["mask_sample"] is True and body["failure_outcome_action"] == "retry"
        assert body["reward"] == 1.0

    @pytest.mark.parametrize(
        "override",
        [
            StageOutcomePolicyConfig(by_stage={"agent": {"by_status_code": {503: "fail"}}}),
            {"by_stage": {"agent": {"by_status_code": {503: "fail"}}}},
        ],
    )
    def test_a_partial_policy_override_keeps_the_untouched_default_rules(self, override) -> None:
        agent_policy = VerifiersAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
            outcome_policy=override,
        ).outcome_policy.by_stage["agent"]

        assert agent_policy.by_status_code[503] == "fail"
        assert agent_policy.by_status_code[429] == "retry" and agent_policy.by_status_code[400] == "count_as_zero"
        assert agent_policy.by_failure_kind["model_transport_error"] == "retry"

        body = self._run(self._ended_on_model_error(429), call_errors=[self._http_error(429)], outcome_policy=override)
        assert body["failure_outcome_action"] == "retry"
        body = self._run(self._ended_on_model_error(400), call_errors=[self._http_error(400)], outcome_policy=override)
        assert is_policy_counted_zero(body)

    def test_a_configured_rule_counts_a_server_error_as_a_zero(self) -> None:
        policy = StageOutcomePolicyConfig(by_stage={"agent": {"by_status_code": {503: "count_as_zero"}}})
        body = self._run(self._ended_on_model_error(503), call_errors=[self._http_error(503)], outcome_policy=policy)

        assert body["mask_sample"] is False and is_policy_counted_zero(body)

    def test_a_429_is_neither_a_zero_nor_classified_as_a_transport_failure(self) -> None:
        body = self._run(self._ended_on_model_error(429), call_errors=[self._http_error(429)])

        assert (body["mask_sample"], body["failure_kind"], body["failure_outcome_action"]) == (
            True,
            "model_request_error",
            "retry",
        )
        event = FailureReportRow.report(body).events[body["_ng_failure_id"]]
        assert event.http_status == 429 and event.boundary == "model"

    @pytest.mark.parametrize("status", [429, 408, 425])
    def test_a_retryable_status_row_is_rerun_by_recovery_on_its_recorded_retry_action(self, status: int) -> None:
        body = self._run(self._ended_on_model_error(status), call_errors=[self._http_error(status)])

        assert body["mask_sample"] is True and body["failure_outcome_action"] == "retry"
        # The recorded decision is honored; the attribution is not changed to satisfy the classifier.
        assert body["failure_kind"] == "model_request_error"
        assert masked_infrastructure_kind(body) == "model_request_error"

    def test_a_call_that_got_no_reply_is_a_transport_failure_without_a_status(self) -> None:
        timeout = openai.APITimeoutError(request=httpx.Request("POST", "http://policy:8000/v1/chat/completions"))
        body = self._run(self._ended_on_model_error(), call_errors=[timeout])

        assert (body["mask_sample"], body["failure_kind"]) == (True, "model_transport_error")
        event = FailureReportRow.report(body).events[body["_ng_failure_id"]]
        assert event.http_status is None and event.details.request_upload_complete is None
        assert event.foreign_error.exception_type == "openai.APITimeoutError"

    def test_a_failed_call_that_a_later_call_recovered_from_does_not_mask_another_error(self) -> None:
        body = self._run(
            {"error": "ToolError", "message": "bad tool", "error_chain_str": "ToolError"},
            call_errors=[self._http_error(503), SimpleNamespace(id="chatcmpl-a", usage=None)],
        )

        assert body["reward"] == 0.0 and not body.get("mask_sample") and body.get("failure_kind") is None

    @pytest.mark.parametrize(
        "error",
        [
            {"error": "ModelError", "message": "Error code: 400", "error_chain_str": "ModelError"},
            {"error": "ToolError", "message": "bad tool", "error_chain_str": "ToolError"},
            {"error": "EmptyModelResponseError", "message": "empty", "error_chain_str": "EmptyModelResponseError"},
            # Text that mentions a status, or a transport name in the chain, is not evidence of either.
            {"error": "ToolError", "message": "Error code: 503", "error_chain_str": "ToolError -> TimeoutError"},
            {"error": "SandboxError", "message": "sandbox died", "error_chain_str": "SandboxError"},
        ],
    )
    def test_an_error_no_model_call_reported_is_still_scored(self, error) -> None:
        body = self._run(error)
        assert body["reward"] == 0.0 and not body.get("mask_sample") and body.get("failure_kind") is None

    def test_a_clean_rollout_has_no_field(self) -> None:
        body = self._run(None)
        assert "rollout_error" not in body and "rollout_error" not in body["response"]


@pytest.mark.parametrize(
    ("prompt_tokens_details", "expected"),
    [(None, None), ({"cached_tokens": None}, None), ({"cached_tokens": 0}, 0), ({"cached_tokens": 12}, 12)],
)
def test_usage_reports_cache_reads_only_when_the_model_does(prompt_tokens_details, expected) -> None:
    usage = openai.types.CompletionUsage.model_validate(
        {
            "prompt_tokens": 20,
            "completion_tokens": 5,
            "total_tokens": 25,
            "prompt_tokens_details": prompt_tokens_details,
        }
    )
    assert _as_response_usage(usage).input_tokens_details.cached_tokens == expected
