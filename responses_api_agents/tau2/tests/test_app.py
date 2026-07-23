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
import json
from pathlib import Path
from typing import Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from tau2.data_model.simulation import RewardInfo, SimulationRun, TerminationReason

from nemo_gym.base_responses_api_agent import AggregateMetricsRequest
from nemo_gym.server_utils import ServerClient
from responses_api_agents.tau2.app import (
    ModelServerRef,
    Tau2Agent,
    Tau2Config,
)


class TestApp:
    def _dummy_server(self) -> Tuple[Tau2Config, Tau2Agent]:
        config = Tau2Config(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            model_server=ModelServerRef(
                type="responses_api_models",
                name="",
            ),
            user_model_server=ModelServerRef(
                type="responses_api_models",
                name="",
            ),
            max_steps=4,
        )
        server = Tau2Agent(config=config, server_client=MagicMock(spec=ServerClient))

        return config, server

    def test_sanity(self) -> None:
        self._dummy_server()

    def test_sanity_query_input(self) -> None:
        example_jsonl = Path(__file__).parent.parent / "data" / "example.jsonl"
        with example_jsonl.open() as f:
            data = list(map(json.loads, f))

        _, server = self._dummy_server()

        app = server.setup_webserver()
        client = TestClient(app)

        async_openai_mock = MagicMock()
        async_openai_mock.create_chat_completion = AsyncMock(
            return_value={
                "id": "chtcmpl-123",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "message": {"content": "<think>thinking</think>hello", "role": "assistant", "tool_calls": []},
                    }
                ],
                "created": 0,
                "model": "dummy_model",
                "object": "chat.completion",
            },
        )

        with (
            patch("responses_api_agents.tau2.app.get_server_url", return_value="dummy base url"),
            patch("tau2.utils.llm_utils.NeMoGymAsyncOpenAI", return_value=async_openai_mock),
        ):
            response = client.post("/run", json=data[0])

        actual_response_dict = response.json()
        expected_response_dict = json.loads((Path(__file__).parent / "test_data.json").read_text())
        # with open("temp.json", "w") as f:
        #     json.dump(actual_response_dict, f, indent=4)

        def _clean(d):
            d["result"].pop("duration")
            d["result"].pop("end_time")
            d["result"].pop("id")
            d["result"].pop("start_time")
            d["result"].pop("timestamp")
            for m in d["result"]["messages"]:
                m.pop("timestamp")
                m.pop("generation_time_seconds", None)

            d["response"].pop("created_at")

            for o in d["response"]["output"]:
                o.pop("id", None)

            for o in d["responses_create_params"]["input"]:
                o.pop("id", None)

            d["duration"] = 0.0

            return d

        assert _clean(expected_response_dict) == _clean(actual_response_dict)

    @pytest.mark.parametrize(
        ("observability_enabled", "url_suffix"),
        [(True, "/ng-rollout/7-2/v1"), (False, "/v1")],
    )
    def test_policy_and_user_model_calls_share_rollout_correlation(
        self, observability_enabled: bool, url_suffix: str
    ) -> None:
        example_jsonl = Path(__file__).parent.parent / "data" / "example.jsonl"
        request_body = json.loads(example_jsonl.read_text().splitlines()[0])
        request_body |= {"_ng_task_index": 7, "_ng_rollout_index": 2}

        config, server = self._dummy_server()
        config.model_server.name = "policy"
        config.user_model_server.name = "user"
        server.server_client.global_config_dict = {"observability_enabled": observability_enabled}
        with patch("responses_api_agents.tau2.app.ensure_tau2_data_dir"):
            client = TestClient(server.setup_webserver())

        result = SimulationRun(
            id="run-1",
            task_id="task-1",
            start_time="2026-07-22T00:00:00Z",
            end_time="2026-07-22T00:00:00Z",
            duration=0,
            termination_reason=TerminationReason.AGENT_STOP,
            reward_info=RewardInfo(reward=1),
            messages=[],
        )
        model_urls = {"policy": "http://policy:8000", "user": "http://user:8001"}
        with (
            patch("responses_api_agents.tau2.app.get_server_url", side_effect=model_urls.__getitem__),
            patch("responses_api_agents.tau2.app.run_single_task", AsyncMock(return_value=result)),
        ):
            response = client.post("/run", json=request_body)

        assert response.status_code == 200
        response_config = response.json()["config"]
        assert response_config["llm_args_agent"]["api_base"] == model_urls["policy"] + url_suffix
        assert response_config["llm_args_user"]["api_base"] == model_urls["user"] + url_suffix

    async def test_compute_metrics(self) -> None:
        example_rollouts_fpath = Path(__file__).parent.parent / "data" / "example_rollouts.jsonl"
        with example_rollouts_fpath.open() as f:
            rollouts = list(map(json.loads, f))

        _, server = self._dummy_server()

        actual_metrics = server.compute_metrics([rollouts])
        expected_metrics = {
            "macro_average": 1.0,
            "telecom/num_samples_unique": 1,
            "retail/num_samples_total": 1,
            "telecom/num_samples_total": 3,
            "airline/num_samples_total": 1,
            "retail/reward": 1.0,
            "telecom/reward": 1.0,
            "airline/reward": 1.0,
            "telecom/service_issue/reward": 1.0,
            "retail/trajectory_termination_reason/user_stop/count": 1,
            "telecom/trajectory_termination_reason/user_stop/count": 3,
            "airline/trajectory_termination_reason/user_stop/count": 1,
            "trajectory_termination_reason/user_stop/count": 5,
            "trajectory_termination_reason/user_stop/pct": 1.0,
            "message_finish_reason/tool_calls/count": 20,
            "message_finish_reason/stop/count": 5,
            "message_finish_reason/tool_calls/pct": 0.8,
            "message_finish_reason/stop/pct": 0.2,
            "trajectory_transfer_to_human_agents/count": 4,
            "trajectory_transfer_to_human_agents/pct": 0.8,
            "tool_call_hallucination/count/total": 0,
            "trajectory_missing_tool_call/count": 0,
            "trajectory_missing_tool_call/pct": 0.0,
            "messages_with_incomplete_reasoning/count": 0,
            "messages_with_incomplete_reasoning/pct": 0.0,
        }
        assert expected_metrics == actual_metrics

        actual_aggregate_metrics = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=rollouts))
        expected_key_aggregate_metrics = {
            "mean/reward": 1.0,
            "macro_average": 1.0,
            "airline/num_samples_unique": 1,
            "telecom/num_samples_unique": 3,
            "retail/num_samples_unique": 1,
            "airline/num_samples_total": 1,
            "telecom/num_samples_total": 3,
            "retail/num_samples_total": 1,
            "airline/reward": 1.0,
            "telecom/reward": 1.0,
            "retail/reward": 1.0,
            "telecom/service_issue/reward": 1.0,
            "airline/trajectory_termination_reason/user_stop/count": 1,
            "telecom/trajectory_termination_reason/user_stop/count": 3,
            "retail/trajectory_termination_reason/user_stop/count": 1,
            "trajectory_termination_reason/user_stop/count": 5,
            "trajectory_termination_reason/user_stop/pct": 1.0,
            "message_finish_reason/tool_calls/count": 20,
            "message_finish_reason/stop/count": 5,
            "message_finish_reason/tool_calls/pct": 0.8,
            "message_finish_reason/stop/pct": 0.2,
            "trajectory_transfer_to_human_agents/count": 4,
            "trajectory_transfer_to_human_agents/pct": 0.8,
            "tool_call_hallucination/count/total": 0,
            "trajectory_missing_tool_call/count": 0,
            "trajectory_missing_tool_call/pct": 0.0,
            "messages_with_incomplete_reasoning/count": 0,
            "messages_with_incomplete_reasoning/pct": 0.0,
        }
        assert expected_key_aggregate_metrics == actual_aggregate_metrics.key_metrics
