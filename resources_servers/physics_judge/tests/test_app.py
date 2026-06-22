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
"""Tests for the physics_judge subclass overrides.

Covers only the physics-specific behaviour layered on top of math_with_judge —
math-verify symbolic checking and the compute_metrics / get_key_metrics
aggregations are tested in the parent server's test suite (the per-domain
breakdown layered on top is exercised here).
"""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from pytest import approx, fixture

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.physics_judge.app import (
    PhysicsJudgeResourcesServer,
    PhysicsJudgeResourcesServerConfig,
)


class TestClassConstants:
    """Class-level overrides on top of the math_with_judge defaults."""

    def test_judge_labels(self) -> None:
        assert PhysicsJudgeResourcesServer.JUDGE_EQUAL_LABEL == "[Correct]"
        assert PhysicsJudgeResourcesServer.JUDGE_NOT_EQUAL_LABEL == "[Incorrect]"

    def test_default_judge_prompt_path(self) -> None:
        # Default points at the bundled physics judge prompt under this
        # server's prompts/ dir, resolvable from the Gym repo root.
        assert PhysicsJudgeResourcesServerConfig.model_fields["judge_prompt_path"].default == (
            "resources_servers/physics_judge/prompts/judge.yaml"
        )

    def test_judge_prompt_loaded_via_gym_prompt_system(self, server: PhysicsJudgeResourcesServer) -> None:
        # Prompt YAML loaded via load_prompt_config() at server init,
        # validated against PromptConfig (user required). The bundled
        # physics judge uses Skills-style placeholder names.
        cfg = server._judge_prompt_config
        assert cfg.user
        assert "{problem}" in cfg.user
        assert "{generation}" in cfg.user
        assert "{expected_answer}" in cfg.user
        # No system key -> the judge runs in single-user-turn mode
        # (matches Skills' physics judge config).
        assert cfg.system is None

    def test_judge_prompt_byte_identical_to_skills(self, server: PhysicsJudgeResourcesServer) -> None:
        """The user-prompt body must be a character-for-character copy of Skills'.

        Any divergence from `nemo_skills/prompt/config/judge/physics.yaml`
        breaks parity with the Skills baseline. Hard-code the Skills body
        here so the test fails loudly if the file drifts.
        """
        skills_user = (
            "You are a diligent and precise assistant tasked with evaluating the correctness of responses. "
            "You will receive a question, an output sentence, and the correct answer. "
            "Your task is to determine if the output sentence accurately answers the question based on the "
            "provided correct answer. Respond with either [Correct] or [Incorrect].\n"
            "Special considerations:\n"
            "1. Multiple Answers: If the output contains multiple answers, evaluate whether later answers "
            "modify or correct earlier ones. In such cases, compare the final answer with the correct answer. "
            "If the final answer is unclear or incorrect, respond with [Incorrect].\n"
            "2. Mathematical Problems: If the formats differ but the answers are mathematically equivalent, "
            "respond with [Correct].\n"
            "3. Explicit Options: If the question provides explicit candidate answers, the output will be "
            "considered correct if it clearly indicates the correct option’s code or the correct option’s "
            "content.\n"
            "4. No Explicit Options: If the question does not provide explicit options, the output must align "
            "with the correct answer in content and meaning to be considered [Correct].\n"
            "Question: {problem}, Output sentence: {generation}, Correct answer: {expected_answer}, Judgement:"
        )
        assert server._judge_prompt_config.user == skills_user


class TestParseVerdict:
    """Verdict parser mirrors Skills' PhysicsMetrics.is_correct_judgement."""

    def test_correct_yields_true(self) -> None:
        assert PhysicsJudgeResourcesServer._parse_verdict("Judgement: [Correct]") is True

    def test_incorrect_yields_false(self) -> None:
        assert PhysicsJudgeResourcesServer._parse_verdict("Judgement: [Incorrect]") is False

    def test_lowercase_correct(self) -> None:
        # Case-insensitive (Skills uses re.IGNORECASE).
        assert PhysicsJudgeResourcesServer._parse_verdict("[correct]") is True

    def test_uppercase_incorrect(self) -> None:
        assert PhysicsJudgeResourcesServer._parse_verdict("[INCORRECT]") is False

    def test_correct_takes_precedence_over_incorrect(self) -> None:
        # Skills checks `\[correct\]` first; if found, returns True even if
        # the text also contains [incorrect].
        assert PhysicsJudgeResourcesServer._parse_verdict("First [Incorrect], but actually [Correct]") is True

    def test_unparseable_yields_false(self) -> None:
        # Default-to-False, matching Skills' is_correct_judgement(return_none=False).
        assert PhysicsJudgeResourcesServer._parse_verdict("This looks fine to me.") is False

    def test_empty_string_yields_false(self) -> None:
        assert PhysicsJudgeResourcesServer._parse_verdict("") is False


def _make_text_response(text: str, response_id: str = "judge_resp_1") -> dict[str, Any]:
    """Wrap a plain text reply in the NeMoGymResponse envelope."""
    return NeMoGymResponse(
        id=response_id,
        created_at=1234.5,
        model="judge_model",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="msg_1",
                content=[NeMoGymResponseOutputText(text=text, annotations=[])],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    ).model_dump()


@fixture
def config() -> PhysicsJudgeResourcesServerConfig:
    return PhysicsJudgeResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge_model"),
        judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
    )


@fixture
def server(config: PhysicsJudgeResourcesServerConfig) -> PhysicsJudgeResourcesServer:
    return PhysicsJudgeResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _attach_judge_text(server: PhysicsJudgeResourcesServer, text: str) -> None:
    response_mock = AsyncMock()
    response_mock.return_value = json.dumps(_make_text_response(text))
    post_mock = MagicMock()
    post_mock.read = response_mock
    server.server_client.post = AsyncMock(return_value=post_mock)


class TestGenerateJudgeEvaluation:
    async def test_correct_verdict_yields_true(self, server: PhysicsJudgeResourcesServer) -> None:
        _attach_judge_text(server, "Judgement: [Correct]")
        equal, _ = await server._generate_judge_evaluation(
            question="Q",
            first_answer="model answer",
            second_answer="golden answer",
        )
        assert equal is True

    async def test_incorrect_verdict_yields_false(self, server: PhysicsJudgeResourcesServer) -> None:
        _attach_judge_text(server, "Reasoning... so [Incorrect].")
        equal, _ = await server._generate_judge_evaluation("Q", "a", "b")
        assert equal is False

    async def test_correct_takes_precedence(self, server: PhysicsJudgeResourcesServer) -> None:
        # Skills looks for `\[correct\]` first; if present, returns True
        # even if `\[incorrect\]` appears earlier in the text.
        _attach_judge_text(server, "Initially [Incorrect] but rethinking: [Correct].")
        equal, _ = await server._generate_judge_evaluation("Q", "a", "b")
        assert equal is True

    async def test_unparseable_yields_false(self, server: PhysicsJudgeResourcesServer) -> None:
        _attach_judge_text(server, "I think these are basically the same.")
        equal, _ = await server._generate_judge_evaluation("Q", "a", "b")
        assert equal is False

    async def test_uses_skills_placeholders_and_omits_system(self, server: PhysicsJudgeResourcesServer) -> None:
        _attach_judge_text(server, "[Correct]")
        await server._generate_judge_evaluation(
            question="What is the speed of light?",
            first_answer="3e8 m/s",
            second_answer="2.998e8 m/s",
        )
        call_kwargs = server.server_client.post.call_args.kwargs
        sent_input = call_kwargs["json"].input
        # Single user turn, no system role.
        assert len(sent_input) == 1
        assert sent_input[0].role == "user"
        # The Skills-style placeholders were filled in.
        rendered = sent_input[0].content
        assert "What is the speed of light?" in rendered
        assert "3e8 m/s" in rendered
        assert "2.998e8 m/s" in rendered

    async def test_non_text_content_yields_false(self, server: PhysicsJudgeResourcesServer) -> None:
        from nemo_gym.openai_utils import NeMoGymResponseOutputRefusal

        refusal_response = NeMoGymResponse(
            id="r",
            created_at=1.0,
            model="m",
            object="response",
            output=[
                NeMoGymResponseOutputMessage(
                    id="msg_1",
                    content=[NeMoGymResponseOutputRefusal(refusal="cannot answer")],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            ],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        ).model_dump()
        response_mock = AsyncMock()
        response_mock.return_value = json.dumps(refusal_response)
        post_mock = MagicMock()
        post_mock.read = response_mock
        server.server_client.post = AsyncMock(return_value=post_mock)
        equal, _ = await server._generate_judge_evaluation("Q", "a", "b")
        assert equal is False


class TestVerifyAnswerWithJudge:
    async def test_unidirectional_single_call(self, server: PhysicsJudgeResourcesServer) -> None:
        # Only one judge call is made — Skills' physics judge does not swap.
        _attach_judge_text(server, "[Correct]")
        reward, evaluations = await server._verify_answer_with_judge(
            question="Q",
            expected_answer="42",
            generated_answer="42",
        )
        assert reward == 1.0
        assert len(evaluations) == 1
        assert server.server_client.post.await_count == 1

    async def test_incorrect_yields_zero_reward(self, server: PhysicsJudgeResourcesServer) -> None:
        _attach_judge_text(server, "[Incorrect]")
        reward, evaluations = await server._verify_answer_with_judge("Q", "42", "wrong")
        assert reward == 0.0
        assert len(evaluations) == 1


class TestVerifyAnswer:
    async def test_symbolic_pass_skips_judge(self, server: PhysicsJudgeResourcesServer) -> None:
        # math-verify will accept "42" vs r"\boxed{42}" → no judge call needed.
        server.server_client.post = AsyncMock()
        reward, _, library_reward, evaluations = await server._verify_answer(
            question="Q",
            expected_answer="42",
            generated_answer=r"the answer is \boxed{42}",
        )
        assert reward == approx(1.0)
        assert library_reward == approx(1.0)
        assert evaluations is None
        server.server_client.post.assert_not_awaited()

    async def test_symbolic_fail_passes_full_generation_to_judge(self, server: PhysicsJudgeResourcesServer) -> None:
        # On a symbolic miss, Skills passes the full generation to the judge
        # (the judge prompt's `{generation}` placeholder is filled with the
        # raw generation, NOT an extracted answer). The judge prompt body
        # must therefore contain the full text we sent in.
        # Use a free-form physics answer (units + words) that math-verify
        # cannot symbolically match against the expected numeric answer —
        # this forces the judge fallback path.
        _attach_judge_text(server, "[Correct]")
        long_gen = (
            "Step 1: by conservation of energy, the kinetic energy at the bottom "
            "equals mgh. Therefore the speed is approximately 5 m/s."
        )
        await server._verify_answer(
            question="What is the speed at the bottom of a 1.27 m drop?",
            expected_answer=r"\sqrt{2gh}",
            generated_answer=long_gen,
        )
        sent_prompt = server.server_client.post.call_args.kwargs["json"].input[0].content
        # The judge sees the full model generation verbatim.
        assert long_gen in sent_prompt

    async def test_should_use_judge_false_skips_judge(
        self,
        config: PhysicsJudgeResourcesServerConfig,
    ) -> None:
        config.should_use_judge = False
        s = PhysicsJudgeResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        s.server_client.post = AsyncMock()
        reward, _, library_reward, evaluations = await s._verify_answer(
            question="Q",
            expected_answer="42",
            generated_answer="not even close",
        )
        assert reward == approx(library_reward)
        assert evaluations is None
        s.server_client.post.assert_not_awaited()


class TestComputeMetrics:
    """Tier-2 per-domain breakdown layered on math_with_judge's pass@k."""

    def _task(self, reward: float, library_reward: float, domain: str, extracted: str = "X") -> dict:
        return {
            "reward": reward,
            "library_reward": library_reward,
            "judge_evaluations": [{"dummy": True}] if reward != library_reward else None,
            "extracted_answer": extracted,
            "domain": domain,
        }

    def test_per_domain_breakdown(self, server: PhysicsJudgeResourcesServer) -> None:
        # 2 mechanics tasks, 1 thermo task; mix of correct/incorrect.
        tasks = [
            [self._task(1.0, 1.0, "mechanics", "a"), self._task(1.0, 1.0, "mechanics", "a")],
            [self._task(0.0, 0.0, "mechanics", "b"), self._task(1.0, 0.0, "mechanics", "b")],
            [self._task(1.0, 1.0, "thermodynamics", "c"), self._task(0.0, 0.0, "thermodynamics", "c")],
        ]
        metrics = server.compute_metrics(tasks)
        # Tier-1 keys are present (inherited from compute_pass_majority_metrics).
        assert any(k.startswith("pass@") for k in metrics)
        # Tier-2 per-domain keys are present.
        domain_keys = [k for k in metrics if k.startswith("mechanics/") or k.startswith("thermodynamics/")]
        assert domain_keys, f"Expected domain-prefixed keys, got: {sorted(metrics)}"
        assert any(k.startswith("mechanics/") for k in domain_keys)
        assert any(k.startswith("thermodynamics/") for k in domain_keys)

    def test_no_domain_field_skips_subset_metrics(self, server: PhysicsJudgeResourcesServer) -> None:
        # Tasks with no `domain` field (e.g. a non-physics dataset reusing
        # this server) should still produce Tier-1 metrics.
        tasks = [
            [{"reward": 1.0, "library_reward": 1.0, "judge_evaluations": None, "extracted_answer": "a"}],
            [{"reward": 0.0, "library_reward": 0.0, "judge_evaluations": None, "extracted_answer": "b"}],
        ]
        metrics = server.compute_metrics(tasks)
        # No domain keys (the dispatch is gated on at least one task having a `domain`).
        assert not any("/" in k and not k.startswith("pass@") and not k.startswith("majority@") for k in metrics)


class TestGetKeyMetrics:
    def test_includes_token_means_when_present(self, server: PhysicsJudgeResourcesServer) -> None:
        agent_metrics = {
            "mean/input_tokens": 100.0,
            "mean/output_tokens": 200.0,
            "pass@1[avg-of-4]/symbolic_accuracy": 0.5,
            "pass@4/symbolic_accuracy": 0.8,
            "majority@4/symbolic_accuracy": 0.7,
        }
        key = server.get_key_metrics(agent_metrics)
        assert key["mean/input_tokens"] == 100.0
        assert key["mean/output_tokens"] == 200.0
        assert "pass@1[avg-of-4]/symbolic_accuracy" in key
        assert "pass@4/symbolic_accuracy" in key
        assert "majority@4/symbolic_accuracy" in key

    def test_missing_token_means_are_omitted(self, server: PhysicsJudgeResourcesServer) -> None:
        # Tolerate metrics dicts that don't include input/output tokens.
        key = server.get_key_metrics({"pass@1[avg-of-4]/symbolic_accuracy": 0.5})
        assert "mean/input_tokens" not in key
        assert "mean/output_tokens" not in key
