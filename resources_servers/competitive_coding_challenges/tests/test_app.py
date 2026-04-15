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

import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


GYM_ROOT = Path(__file__).resolve().parents[3]
RESOURCE_SERVER_DIR = Path(__file__).resolve().parents[1]
for import_path in (GYM_ROOT, RESOURCE_SERVER_DIR):
    import_path_str = str(import_path)
    if import_path_str not in sys.path:
        sys.path.insert(0, import_path_str)

ccc_eval_stub = types.ModuleType("ccc_eval")


class _StubCCCEvaluator:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


ccc_eval_stub.CCCEvaluator = _StubCCCEvaluator
sys.modules.setdefault("ccc_eval", ccc_eval_stub)

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from resources_servers.competitive_coding_challenges.app import (
    CompetitiveCodingChallengesResourcesServer,
    CompetitiveCodingChallengesResourcesServerConfig,
    CompetitiveCodingChallengesVerifyRequest,
)


def _make_server() -> CompetitiveCodingChallengesResourcesServer:
    config = CompetitiveCodingChallengesResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="competitive_coding_challenges",
    )
    return CompetitiveCodingChallengesResourcesServer(
        config=config,
        server_client=MagicMock(spec=ServerClient),
    )


def _make_verify_request(
    *,
    text: str = "```cpp\nint main() { return 0; }\n```",
    competition_id: str = "comp-1",
    problem_id: str = "prob-1",
    subtask: str | None = "samples",
    subtask_score: float | None = None,
) -> CompetitiveCodingChallengesVerifyRequest:
    return CompetitiveCodingChallengesVerifyRequest(
        competition_id=competition_id,
        problem_id=problem_id,
        subtask=subtask,
        subtask_score=subtask_score,
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "user", "content": "Solve the problem."}],
        ),
        response=NeMoGymResponse(
            id="resp-1",
            object="response",
            created_at=0.0,
            model="dummy",
            output=[
                {
                    "id": "msg-1",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "annotations": [],
                            "text": text,
                        }
                    ],
                }
            ],
            tool_choice="auto",
            tools=[],
            parallel_tool_calls=False,
        ),
    )


@pytest.fixture
def server() -> CompetitiveCodingChallengesResourcesServer:
    return _make_server()


def test_sanity(server: CompetitiveCodingChallengesResourcesServer) -> None:
    assert server.config.name == "competitive_coding_challenges"


def test_setup_webserver_initializes_evaluator(server: CompetitiveCodingChallengesResourcesServer) -> None:
    with patch("resources_servers.competitive_coding_challenges.app.CCCEvaluator") as evaluator_cls:
        app = server.setup_webserver()

    evaluator_cls.assert_called_once_with(
        config={
            "test_file": server.config.test_file,
            "test_batch_size": server.config.test_batch_size,
            "time_scale": server.config.time_scale,
            "shared_dir": server.config.shared_dir,
        },
        num_parallel_requests=server.config.num_parallel_requests,
    )
    assert app is not None
    assert server._evaluator is evaluator_cls.return_value


@pytest.mark.asyncio
async def test_verify_passes_competition_context_and_defaults_name(
    server: CompetitiveCodingChallengesResourcesServer,
) -> None:
    request = _make_verify_request()
    evaluator = MagicMock()
    evaluator.eval_single = AsyncMock(
        return_value={
            "test_case_results": {
                "samples": {
                    "score": 10.0,
                    "outputs": [
                        {"score": 1.0, "test_group": "sample"},
                        {"score": 1.0, "test_group": "secret"},
                    ],
                }
            }
        }
    )
    evaluator.get_problem_metadata.return_value = {
        "subtasks": {
            "samples": {
                "score": 10.0,
                "aggregation": "min",
                "test_names": ["sample-1", "secret-1"],
            }
        }
    }
    server._evaluator = evaluator

    response = await server.verify(request)

    assert response.reward == 1.0
    assert response.name == "prob-1"
    evaluator.eval_single.assert_awaited_once_with(
        {
            "competition_id": "comp-1",
            "name": "prob-1",
            "problem_id": "prob-1",
            "subtask": "samples",
            "generation": "```cpp\nint main() { return 0; }\n```",
        }
    )
    evaluator.get_problem_metadata.assert_called_once_with("prob-1", "comp-1")


@pytest.mark.asyncio
async def test_verify_partial_subtask_score_returns_zero_reward() -> None:
    server = _make_server()
    request = _make_verify_request(subtask="sub1")
    evaluator = MagicMock()
    evaluator.eval_single = AsyncMock(
        return_value={
            "test_case_results": {
                "sub1": {
                    "score": 3.0,
                    "outputs": [{"score": 0.0, "test_group": "secret"}],
                }
            }
        }
    )
    evaluator.get_problem_metadata.return_value = {
        "subtasks": {
            "sub1": {
                "score": 12.0,
                "aggregation": "min",
                "test_names": ["t1", "t2"],
            }
        }
    }
    server._evaluator = evaluator

    response = await server.verify(request)

    assert response.reward == 0.0
    evaluator.get_problem_metadata.assert_called_once_with("prob-1", "comp-1")


@pytest.mark.asyncio
async def test_verify_full_problem_reward_requires_all_subtasks() -> None:
    server = _make_server()
    request = _make_verify_request(subtask=None)
    evaluator = MagicMock()
    evaluator.eval_single = AsyncMock(
        return_value={
            "test_case_results": {
                "sub1": {
                    "score": 5.0,
                    "outputs": [{"score": 1.0, "test_group": "sample"}],
                },
                "sub2": {
                    "score": 0.0,
                    "outputs": [{"score": 0.0, "test_group": "secret"}],
                },
            }
        }
    )
    evaluator.get_problem_metadata.return_value = {
        "subtasks": {
            "sub1": {"score": 5.0, "aggregation": "min", "test_names": ["t1"]},
            "sub2": {"score": 7.0, "aggregation": "min", "test_names": ["t2"]},
        }
    }
    server._evaluator = evaluator

    response = await server.verify(request)

    assert response.reward == 0.0


@pytest.mark.asyncio
async def test_verify_returns_error_details_when_evaluator_fails(
    server: CompetitiveCodingChallengesResourcesServer,
) -> None:
    request = _make_verify_request()
    evaluator = MagicMock()
    evaluator.eval_single = AsyncMock(side_effect=RuntimeError("boom"))
    server._evaluator = evaluator

    response = await server.verify(request)

    assert response.reward == 0.0
    assert response.details == {"error": "boom"}
