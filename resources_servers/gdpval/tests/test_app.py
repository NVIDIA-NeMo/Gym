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
from unittest.mock import MagicMock, patch

import pytest

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.gdpval.app import (
    GDPValResourcesServer,
    GDPValResourcesServerConfig,
    GDPValVerifyRequest,
)


def _server(reward_mode: str = "rubric", **extra) -> GDPValResourcesServer:
    kwargs = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        reward_mode=reward_mode,
        judge_model_server={"type": "responses_api_models", "name": "judge"},
    )
    kwargs.update(extra)
    config = GDPValResourcesServerConfig(**kwargs)
    return GDPValResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _verify_request(**fields) -> GDPValVerifyRequest:
    deliverable_text = fields.pop("deliverable_text", "A text deliverable.")
    return GDPValVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=NeMoGymResponse(
            id="resp-1",
            created_at=0.0,
            model="model",
            object="response",
            output=[
                NeMoGymResponseOutputMessage(
                    id="msg-1",
                    type="message",
                    role="assistant",
                    status="completed",
                    content=[NeMoGymResponseOutputText(type="output_text", text=deliverable_text, annotations=[])],
                )
            ],
            status="completed",
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        ),
        task_id="task-1",
        prompt="Write a report on X.",
        rubric_json=fields.pop("rubric_json", None),
        rubric_pretty=fields.pop("rubric_pretty", ""),
        **fields,
    )


class TestApp:
    def test_sanity_rubric(self) -> None:
        _server(reward_mode="rubric")

    def test_sanity_comparison(self) -> None:
        _server(reward_mode="comparison", reference_deliverables_dir="/tmp/fork-deliverables")

    def test_comparison_requires_reference_dir(self) -> None:
        import pytest as _pytest

        with _pytest.raises(ValueError, match="reference_deliverables_dir"):
            _server(reward_mode="comparison")

    @pytest.mark.asyncio
    async def test_verify_rubric_no_rubric_returns_zero(self) -> None:
        server = _server(reward_mode="rubric")
        body = _verify_request(rubric_json=None, rubric_pretty="")
        resp = await server.verify(body)
        assert resp.reward == 0.0
        assert resp.verify_mode == "rubric"
        assert resp.invalid_judge_response is True

    @pytest.mark.asyncio
    async def test_verify_rubric_with_canned_judge(self) -> None:
        server = _server(reward_mode="rubric")

        canned_result = {"overall_score": 0.7, "criteria_scores": [{"score": 0.7}]}

        async def fake_score_with_rubric(**_kwargs):
            return 0.7, canned_result

        body = _verify_request(
            rubric_json=[{"criterion": "clarity", "score": 1}],
            deliverable_text="Deliverable body text.",
        )

        with (
            patch("resources_servers.gdpval.scoring.score_with_rubric", side_effect=fake_score_with_rubric),
            patch("resources_servers.gdpval.app.get_server_url", return_value="http://localhost:9999"),
        ):
            resp = await server.verify(body)

        assert resp.reward == 0.7
        assert resp.verify_mode == "rubric"
        assert resp.invalid_judge_response is False
        assert resp.judge_response == canned_result

    @pytest.mark.asyncio
    async def test_verify_rubric_passes_create_overrides_through(self) -> None:
        """``judge_responses_create_params_overrides`` must reach the scoring fn.

        ``model`` and ``api_key`` are pulled out as their own kwargs; everything
        else (e.g. ``max_tokens``, ``temperature``) flows through as
        ``create_overrides`` and gets merged into ``client.chat.completions.create``.
        """
        server = _server(
            reward_mode="rubric",
            judge_responses_create_params_overrides={
                "model": "custom-judge",
                "api_key": "sk-custom",  # pragma: allowlist secret
                "max_tokens": 16384,
                "temperature": 0.0,
            },
        )

        captured: dict = {}

        async def fake_score_with_rubric(**kwargs):
            captured.update(kwargs)
            return 0.5, {"overall_score": 0.5}

        body = _verify_request(rubric_json=[{"criterion": "clarity", "score": 1}])

        with (
            patch("resources_servers.gdpval.scoring.score_with_rubric", side_effect=fake_score_with_rubric),
            patch("resources_servers.gdpval.app.get_server_url", return_value="http://localhost:9999"),
        ):
            await server.verify(body)

        assert captured["model_name"] == "custom-judge"
        assert captured["api_key"] == "sk-custom"  # pragma: allowlist secret
        assert captured["create_overrides"] == {"max_tokens": 16384, "temperature": 0.0}

    @pytest.mark.asyncio
    async def test_verify_comparison_missing_reference(self, tmp_path) -> None:
        server = _server(
            reward_mode="comparison",
            reference_deliverables_dir=str(tmp_path / "no-such-dir"),
        )
        body = _verify_request(rubric_json=[{"criterion": "clarity", "score": 1}])
        resp = await server.verify(body)
        assert resp.reward == 0.0
        assert resp.verify_mode == "comparison"
        assert resp.judge_response == {"error": "reference_missing"}

    def test_aggregate_metrics_comparison_elo(self) -> None:
        from nemo_gym.config_types import AggregateMetricsRequest

        server = _server(
            reward_mode="comparison",
            reference_deliverables_dir="/tmp/fork-deliverables",
            reference_elo=1000.0,
        )

        def _row(task_idx, reward, win, loss, tie):
            return {
                "_ng_task_index": task_idx,
                "_ng_rollout_index": 0,
                "reward": reward,
                "win": win,
                "loss": loss,
                "tie": tie,
                "response": {},
            }

        responses = (
            [_row(i, 1.0, True, False, False) for i in range(7)]
            + [_row(7 + i, 0.0, False, True, False) for i in range(2)]
            + [_row(9, 0.5, False, False, True)]
        )
        import asyncio as _asyncio

        body = AggregateMetricsRequest(verify_responses=responses)
        result = _asyncio.run(server.aggregate_metrics(body))
        assert result.agent_metrics["comparison/wins"] == 7
        assert result.agent_metrics["comparison/losses"] == 2
        assert result.agent_metrics["comparison/ties"] == 1
        assert result.agent_metrics["comparison/judged"] == 10
        assert abs(result.agent_metrics["comparison/win_rate"] - 0.75) < 1e-6
        # win_rate=0.75 → ELO = 1000 - 400 * (log10(0.25) - log10(0.75)) ≈ 1190.85
        assert 1180 < result.agent_metrics["comparison/eval_elo"] < 1200
