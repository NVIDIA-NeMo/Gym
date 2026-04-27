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


# ---------------------------------------------------------------------------
# Aggregate-metrics path: _score_fn / _lookup_subtask_cap / compute_metrics /
# get_key_metrics. Bare-instance pattern (model_construct + stub _evaluator)
# avoids spinning up FastAPI for these pure-function checks.
# ---------------------------------------------------------------------------


from types import SimpleNamespace  # noqa: E402


def _make_bare_server() -> CompetitiveCodingChallengesResourcesServer:
    return CompetitiveCodingChallengesResourcesServer.model_construct()


def _make_server_with_evaluator(metadata: dict) -> CompetitiveCodingChallengesResourcesServer:
    server = _make_bare_server()
    server._evaluator = SimpleNamespace(
        get_problem_metadata=lambda problem_id, competition_id: metadata[problem_id],
    )
    return server


def _make_rollout(problem_id: str, subtask: str, tcr: dict, competition_id: str = "ioi24") -> dict:
    return {
        "problem_id": problem_id,
        "competition_id": competition_id,
        "subtask": subtask,
        "extracted_code": "<stub code>",
        "details": {"test_case_results": tcr},
    }


@pytest.fixture
def server_with_two_problems() -> CompetitiveCodingChallengesResourcesServer:
    return _make_server_with_evaluator(
        {
            "nile": {
                "subtasks": {
                    "01-equal": {"score": 6.0},
                    "02-permutation": {"score": 13.0},
                }
            },
            "message": {
                "subtasks": {
                    "01-len64": {"score": 10.0},
                    "02-full": {"score": 90.0},
                }
            },
        }
    )


class TestScoreFn:
    def test_empty_tcr_scores_zero(self) -> None:
        assert CompetitiveCodingChallengesResourcesServer._score_fn({"details": {"test_case_results": {}}}) == {
            "accuracy": 0.0
        }

    def test_missing_details_scores_zero(self) -> None:
        assert CompetitiveCodingChallengesResourcesServer._score_fn({}) == {"accuracy": 0.0}

    def test_any_zero_subtask_blocks_accuracy(self) -> None:
        result = {"details": {"test_case_results": {"a": {"score": 6.0}, "b": {"score": 0.0}}}}
        assert CompetitiveCodingChallengesResourcesServer._score_fn(result) == {"accuracy": 0.0}

    def test_all_positive_subtasks_score_one(self) -> None:
        result = {"details": {"test_case_results": {"a": {"score": 6.0}, "b": {"score": 13.0}}}}
        assert CompetitiveCodingChallengesResourcesServer._score_fn(result) == {"accuracy": 1.0}

    def test_none_score_treated_as_zero(self) -> None:
        result = {"details": {"test_case_results": {"a": {"score": None}}}}
        assert CompetitiveCodingChallengesResourcesServer._score_fn(result) == {"accuracy": 0.0}


class TestLookupSubtaskCap:
    def test_returns_zero_when_no_evaluator(self) -> None:
        server = _make_bare_server()
        server._evaluator = None
        assert server._lookup_subtask_cap("ioi24", "nile", "01-equal") == 0.0

    def test_returns_subtask_score(self) -> None:
        server = _make_server_with_evaluator(
            {"nile": {"subtasks": {"01-equal": {"score": 6.0}, "02-permutation": {"score": 13.0}}}}
        )
        assert server._lookup_subtask_cap("ioi24", "nile", "01-equal") == 6.0
        assert server._lookup_subtask_cap("ioi24", "nile", "02-permutation") == 13.0

    def test_returns_zero_for_unknown_subtask(self) -> None:
        server = _make_server_with_evaluator({"nile": {"subtasks": {"01-equal": {"score": 6.0}}}})
        assert server._lookup_subtask_cap("ioi24", "nile", "unknown") == 0.0

    def test_swallows_evaluator_error(self) -> None:
        def _raise(*_, **__):
            raise ValueError("problem not found")

        server = _make_bare_server()
        server._evaluator = SimpleNamespace(get_problem_metadata=_raise)
        assert server._lookup_subtask_cap("ioi24", "nile", "01-equal") == 0.0

    def test_sum_tests_aggregation_uses_test_count(self) -> None:
        server = _make_server_with_evaluator(
            {"prob": {"subtasks": {"all": {"aggregation": "sum_tests", "test_names": ["t1", "t2", "t3"]}}}}
        )
        assert server._lookup_subtask_cap("comp", "prob", "all") == 3.0


class TestComputeMetrics:
    def test_empty_tasks_returns_zero_total(self, server_with_two_problems) -> None:
        metrics = server_with_two_problems.compute_metrics([])
        assert metrics["total_score"] == 0
        assert metrics["per_problem_subtask_scores"] == {}

    def test_sums_max_per_subtask_across_rollouts(self, server_with_two_problems) -> None:
        # Two rollouts for Nile pool to 6 + 13 = 19 (best subtask hit per row).
        rollout_a = _make_rollout("nile", "01-equal", {"01-equal": {"score": 6.0}, "02-permutation": {"score": 0.0}})
        rollout_b = _make_rollout(
            "nile", "02-permutation", {"01-equal": {"score": 0.0}, "02-permutation": {"score": 13.0}}
        )
        metrics = server_with_two_problems.compute_metrics([[rollout_a], [rollout_b]])
        assert metrics["total_score"] == 19
        per = metrics["per_problem_subtask_scores"]["nile"]
        assert per["total"] == {"score": 19.0, "max_score": 19.0}
        assert per["subtasks"]["01-equal"] == {"score": 6.0, "max_score": 6.0}
        assert per["subtasks"]["02-permutation"] == {"score": 13.0, "max_score": 13.0}

    def test_sums_across_problems(self, server_with_two_problems) -> None:
        nile_rollout = _make_rollout(
            "nile", "01-equal", {"01-equal": {"score": 6.0}, "02-permutation": {"score": 13.0}}
        )
        message_rollout = _make_rollout(
            "message", "01-len64", {"01-len64": {"score": 10.0}, "02-full": {"score": 0.0}}
        )
        metrics = server_with_two_problems.compute_metrics([[nile_rollout], [message_rollout]])
        # Nile 19 + Message 10 = 29
        assert metrics["total_score"] == 29
        assert set(metrics["per_problem_subtask_scores"].keys()) == {"nile", "message"}

    def test_falls_back_to_ioi_id_when_problem_id_missing(self, server_with_two_problems) -> None:
        rollout = {
            "ioi_id": "nile",
            "competition_id": "ioi24",
            "subtask": "01-equal",
            "extracted_code": "x",
            "details": {"test_case_results": {"01-equal": {"score": 6.0}}},
        }
        metrics = server_with_two_problems.compute_metrics([[rollout]])
        assert metrics["total_score"] == 6

    def test_skips_rollouts_without_problem_id(self, server_with_two_problems) -> None:
        rollout = {"details": {"test_case_results": {"01-equal": {"score": 6.0}}}}
        metrics = server_with_two_problems.compute_metrics([[rollout]])
        assert metrics["total_score"] == 0


class TestGetKeyMetrics:
    def test_promotes_total_score(self, server_with_two_problems) -> None:
        agent_metrics = {
            "total_score": 123,
            "pass@1[avg-of-2]/accuracy": 0.5,
            "pass@2/accuracy": 1.0,
        }
        key = server_with_two_problems.get_key_metrics(agent_metrics)
        assert key["total_score"] == 123

    def test_omits_total_score_when_absent(self, server_with_two_problems) -> None:
        key = server_with_two_problems.get_key_metrics({})
        assert "total_score" not in key
