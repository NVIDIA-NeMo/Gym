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
import json
from unittest.mock import MagicMock, patch

from nemo_gym.config_types import ModelServerRef
from nemo_gym.server_utils import ServerClient
from responses_api_agents.verifiers_agent.app import (
    VerifiersAgent,
    VerifiersAgentConfig,
)


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

    def test_client_base_url_carries_the_rollout_capture_prefix(self) -> None:
        """Model-call capture is keyed by the ``/ng-rollout/<id>`` URL prefix.

        Without it the capture store stays empty for every rollout, the projected
        trajectory reports ``model_call_capture_no_records``, and every
        rollout-health check degrades to ``unobserved`` -- a silently broken run
        then looks identical to a healthy low-scoring one.
        """
        config = VerifiersAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        )
        agent = VerifiersAgent(config=config, server_client=MagicMock(spec=ServerClient))

        def fake_resolve(model_server_name: str, rollout_id: str | None = None) -> str:
            prefix = f"/ng-rollout/{rollout_id}" if rollout_id else ""
            return f"http://policy{prefix}/v1"

        with (
            patch.object(VerifiersAgent, "resolve_model_base_url", side_effect=fake_resolve) as resolve,
            patch.object(VerifiersAgent, "rollout_id_from_run", lambda _self, body: body.rollout_id),
        ):
            first = agent._get_client(MagicMock(rollout_id="7-2"))
            assert str(first.client.base_url).rstrip("/") == "http://policy/ng-rollout/7-2/v1"

            # A second rollout must not reuse the first rollout's prefixed client.
            second = agent._get_client(MagicMock(rollout_id="7-3"))
            assert str(second.client.base_url).rstrip("/") == "http://policy/ng-rollout/7-3/v1"
            assert second is not first

            # Same rollout is still served from cache, not re-resolved.
            assert agent._get_client(MagicMock(rollout_id="7-2")) is first
            assert resolve.call_count == 2

    def test_client_base_url_is_unprefixed_when_capture_is_disabled(self) -> None:
        """``rollout_id_from_run`` returns ``None`` with capture off; URL is unchanged."""
        config = VerifiersAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        )
        agent = VerifiersAgent(config=config, server_client=MagicMock(spec=ServerClient))

        def fake_resolve(model_server_name: str, rollout_id: str | None = None) -> str:
            prefix = f"/ng-rollout/{rollout_id}" if rollout_id else ""
            return f"http://policy{prefix}/v1"

        with (
            patch.object(VerifiersAgent, "resolve_model_base_url", side_effect=fake_resolve),
            patch.object(VerifiersAgent, "rollout_id_from_run", lambda _self, body: None),
        ):
            client = agent._get_client(MagicMock())
            assert str(client.client.base_url).rstrip("/") == "http://policy/v1"
            assert agent._get_client(MagicMock()) is client
