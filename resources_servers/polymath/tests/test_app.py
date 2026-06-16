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
from unittest.mock import MagicMock

from pytest import approx, fixture

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.polymath.app import (
    PolyMathResourcesServer,
    PolyMathResourcesServerConfig,
    PolyMathVerifyRequest,
)


@fixture
def server() -> PolyMathResourcesServer:
    config = PolyMathResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
    )
    return PolyMathResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _msg(text: str) -> NeMoGymResponseOutputMessage:
    return NeMoGymResponseOutputMessage(
        id="m",
        content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
        role="assistant",
        status="completed",
        type="message",
    )


def _make_response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="r",
        created_at=1.0,
        model="m",
        object="response",
        output=[_msg(text)],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


class TestVerify:
    async def test_verify_passes_through_weight_and_language(self, server) -> None:
        params = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "q"}])
        resp = _make_response("The answer is \\boxed{4}.")
        body = PolyMathVerifyRequest(
            responses_create_params=params,
            response=resp,
            question="What is 2+2?",
            expected_answer="4",
            weight=4.0,
            language="en",
        )
        out = await server.verify(body)
        assert out.reward == approx(1.0)
        assert out.extracted_answer == "4"
        assert out.weight == 4.0
        assert out.language == "en"

    async def test_verify_handles_missing_metadata(self, server) -> None:
        params = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "q"}])
        resp = _make_response("\\boxed{4}")
        body = PolyMathVerifyRequest(
            responses_create_params=params,
            response=resp,
            question="What is 2+2?",
            expected_answer="4",
        )
        out = await server.verify(body)
        assert out.reward == approx(1.0)
        assert out.weight is None
        assert out.language is None


# ──────────────────────────────────────────────────────────
# Weighted aggregation
# ──────────────────────────────────────────────────────────


class TestComputeMetrics:
    """Sanity checks for the PolyMath compute_metrics override.

    The Tier 1 (unweighted, language-pooled) numbers are produced by
    the inherited ``compute_pass_majority_metrics`` and are already
    exercised in math_with_judge's test suite; here we focus on the
    weighted and per-language additions.
    """

    @staticmethod
    def _r(reward: float, ans, weight: float, language: str):
        return {
            "reward": reward,
            "library_reward": reward,
            "extracted_answer": ans,
            "weight": weight,
            "language": language,
        }

    def test_weighted_pass_at_k_matches_skills_formula(self, server) -> None:
        # Two tasks, weights 1 and 8. Task A always correct (binary 1's),
        # task B always wrong (binary 0's). Skills' weighted pass@1
        # = (1*1 + 8*0) / (1 + 8) = 1/9.
        tasks = [
            [self._r(1.0, "1", weight=1.0, language="en")] * 4,
            [self._r(0.0, "2", weight=8.0, language="en")] * 4,
        ]
        m = server.compute_metrics(tasks)
        assert m["pass@1/symbolic_accuracy_weighted"] == approx(100.0 * 1.0 / 9.0, abs=1e-3)
        # pass@4: task A gets 1.0 (any correct), task B gets 0.0.
        assert m["pass@4/symbolic_accuracy_weighted"] == approx(100.0 * 1.0 / 9.0, abs=1e-3)

    def test_weighted_pass1_avg_of_k(self, server) -> None:
        # Task A: weight 2, 2 correct out of 4 → 0.5
        # Task B: weight 4, 4 correct out of 4 → 1.0
        # Weighted mean: (2*0.5 + 4*1.0) / 6 = 5/6
        rollouts_a = [
            self._r(1.0, "1", 2.0, "en"),
            self._r(0.0, "2", 2.0, "en"),
            self._r(1.0, "1", 2.0, "en"),
            self._r(0.0, "2", 2.0, "en"),
        ]
        rollouts_b = [self._r(1.0, "3", 4.0, "en")] * 4
        tasks = [rollouts_a, rollouts_b]
        m = server.compute_metrics(tasks)
        assert m["pass@1[avg-of-4]/symbolic_accuracy_weighted"] == approx(100.0 * 5.0 / 6.0, abs=1e-3)

    def test_per_language_keys_present(self, server) -> None:
        tasks = [
            [self._r(1.0, "1", 1.0, "en")] * 2,
            [self._r(0.0, "2", 1.0, "zh")] * 2,
        ]
        m = server.compute_metrics(tasks)
        assert "en/pass@1/symbolic_accuracy" in m
        assert "zh/pass@1/symbolic_accuracy" in m
        assert m["en/pass@1/symbolic_accuracy"] == approx(100.0)
        assert m["zh/pass@1/symbolic_accuracy"] == approx(0.0)

    def test_unweighted_metrics_still_present(self, server) -> None:
        # The Tier 1 (math_with_judge-equivalent) keys must continue to
        # be emitted alongside the weighted variants.
        tasks = [
            [self._r(1.0, "1", 1.0, "en")] * 2,
            [self._r(0.0, "2", 1.0, "en")] * 2,
        ]
        m = server.compute_metrics(tasks)
        assert "pass@1/symbolic_accuracy" in m
        assert "pass@1/symbolic_accuracy_weighted" in m
        assert "pass@1[avg-of-2]/symbolic_accuracy" in m
        assert "pass@1[avg-of-2]/symbolic_accuracy_weighted" in m

    def test_weights_default_to_one(self, server) -> None:
        # Tasks without `weight` should be treated as weight=1.0
        # (matches WeightedMathMetrics._get_sample_weight).
        tasks = [
            [{"reward": 1.0, "library_reward": 1.0, "extracted_answer": "1"}] * 2,
            [{"reward": 0.0, "library_reward": 0.0, "extracted_answer": "2"}] * 2,
        ]
        m = server.compute_metrics(tasks)
        # Two tasks, equal weights: 50% pass.
        assert m["pass@1/symbolic_accuracy_weighted"] == approx(50.0, abs=1e-3)

    def test_weighted_majority_at_k(self, server) -> None:
        # Task with weight 4, two distinct extracted answers — majority
        # answer wins. With 3 of "1" (correct) and 1 of "2" (wrong) at
        # k=4, weighted majority@4 = 1.0 for that task.
        tasks = [
            [
                self._r(1.0, "1", 4.0, "en"),
                self._r(1.0, "1", 4.0, "en"),
                self._r(1.0, "1", 4.0, "en"),
                self._r(0.0, "2", 4.0, "en"),
            ],
        ]
        m = server.compute_metrics(tasks)
        assert m["majority@4/symbolic_accuracy_weighted"] == approx(100.0, abs=1e-3)

    def test_empty_tasks(self, server) -> None:
        assert server.compute_metrics([]) == {}

    def test_get_key_metrics_includes_weighted_variants(self, server) -> None:
        agent_metrics = {
            "mean/input_tokens": 100.0,
            "mean/output_tokens": 500.0,
            "pass@1[avg-of-4]/symbolic_accuracy": 50.0,
            "pass@1[avg-of-4]/symbolic_accuracy_weighted": 45.0,
            "pass@1[avg-of-4]/no_answer": 5.0,
            "pass@4/symbolic_accuracy": 70.0,
            "pass@4/symbolic_accuracy_weighted": 65.0,
            "pass@4/no_answer": 5.0,
            "majority@4/symbolic_accuracy": 60.0,
            "majority@4/symbolic_accuracy_weighted": 58.0,
            "majority@4/no_answer": 5.0,
        }
        key = server.get_key_metrics(agent_metrics)
        assert "pass@1[avg-of-4]/symbolic_accuracy" in key
        assert "pass@1[avg-of-4]/symbolic_accuracy_weighted" in key
        assert "pass@4/symbolic_accuracy" in key
        assert "pass@4/symbolic_accuracy_weighted" in key
        assert "pass@4/no_answer" not in key
        assert "majority@4/symbolic_accuracy" in key
        assert "majority@4/symbolic_accuracy_weighted" in key
