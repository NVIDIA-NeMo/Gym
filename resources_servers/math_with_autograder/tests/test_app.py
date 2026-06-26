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
"""Tests for the math_with_autograder subclass overrides.

Covers only the IMO/autograder-specific behaviour layered on top of
math_with_judge — math-verify symbolic checking and the
compute_metrics / get_key_metrics aggregations are tested in the parent
server's test suite.
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
from resources_servers.math_with_autograder.app import (
    MathWithAutograderResourcesServer,
    MathWithAutograderResourcesServerConfig,
)


class TestClassConstants:
    """Class-level overrides on top of the math_with_judge defaults."""

    def test_judge_labels(self) -> None:
        assert MathWithAutograderResourcesServer.JUDGE_EQUAL_LABEL == r"\boxed{Correct}"
        assert MathWithAutograderResourcesServer.JUDGE_NOT_EQUAL_LABEL == r"\boxed{Incorrect}"

    def test_default_judge_prompt_path(self) -> None:
        # Default points at the bundled IMO autograder prompt under this server's
        # prompts/ dir, resolvable from the Gym repo root.
        assert MathWithAutograderResourcesServerConfig.model_fields["judge_prompt_path"].default == (
            "resources_servers/math_with_autograder/prompts/judge.yaml"
        )

    def test_judge_prompt_loaded_via_gym_prompt_system(self, server: MathWithAutograderResourcesServer) -> None:
        # Prompt YAML loaded via load_prompt_config() at server init,
        # validated against PromptConfig (user required). The bundled
        # autograder uses Skills-style placeholder names.
        cfg = server._judge_prompt_config
        assert cfg.user
        assert "{problem}" in cfg.user
        assert "{predicted_answer}" in cfg.user
        assert "{expected_answer}" in cfg.user
        # No system key -> the autograder runs in single-user-turn mode.
        assert cfg.system is None


class TestSearchBoxed:
    """Brace-matching extractor for the last balanced \\boxed{...}."""

    def test_returns_none_when_no_boxed(self) -> None:
        assert MathWithAutograderResourcesServer._search_boxed("no boxed here") is None

    def test_simple_boxed(self) -> None:
        assert MathWithAutograderResourcesServer._search_boxed(r"answer is \boxed{42}") == "42"

    def test_returns_last_boxed_when_multiple(self) -> None:
        text = r"first \boxed{1} then \boxed{2}"
        assert MathWithAutograderResourcesServer._search_boxed(text) == "2"

    def test_handles_nested_braces(self) -> None:
        text = r"complex \boxed{g(x)=2x^3+C \text{ or } g(x)=-2x^3+C, C\in\mathbb{R}}"
        out = MathWithAutograderResourcesServer._search_boxed(text)
        assert out == r"g(x)=2x^3+C \text{ or } g(x)=-2x^3+C, C\in\mathbb{R}"

    def test_handles_fbox(self) -> None:
        assert MathWithAutograderResourcesServer._search_boxed(r"\fbox{99}") == "99"

    def test_returns_none_on_unbalanced(self) -> None:
        assert MathWithAutograderResourcesServer._search_boxed(r"\boxed{unbalanced") is None

    def test_prefers_last_boxed_over_fbox(self) -> None:
        # rfind picks the rightmost \boxed; \fbox is the fallback only
        # when no \boxed is present at all.
        assert MathWithAutograderResourcesServer._search_boxed(r"\fbox{x} and \boxed{y}") == "y"

    def test_only_fbox_present(self) -> None:
        assert MathWithAutograderResourcesServer._search_boxed(r"only \fbox{5} here") == "5"


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
def config() -> MathWithAutograderResourcesServerConfig:
    return MathWithAutograderResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge_model"),
        judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
    )


@fixture
def server(config: MathWithAutograderResourcesServerConfig) -> MathWithAutograderResourcesServer:
    return MathWithAutograderResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _attach_judge_text(server: MathWithAutograderResourcesServer, text: str) -> None:
    response_mock = AsyncMock()
    response_mock.return_value = json.dumps(_make_text_response(text))
    post_mock = MagicMock()
    post_mock.read = response_mock
    server.server_client.post = AsyncMock(return_value=post_mock)


class TestGenerateJudgeEvaluation:
    async def test_correct_verdict_yields_true(self, server: MathWithAutograderResourcesServer) -> None:
        _attach_judge_text(server, r"My final verdict: \boxed{Correct}")
        equal, _ = await server._generate_judge_evaluation(
            question="Q",
            first_answer="model answer",
            second_answer="golden answer",
        )
        assert equal is True

    async def test_incorrect_verdict_yields_false(self, server: MathWithAutograderResourcesServer) -> None:
        _attach_judge_text(server, r"Reasoning... so \boxed{Incorrect}.")
        equal, _ = await server._generate_judge_evaluation("Q", "a", "b")
        assert equal is False

    async def test_first_label_wins(self, server: MathWithAutograderResourcesServer) -> None:
        # Both tokens present; the earlier one wins.
        _attach_judge_text(server, r"At first I thought \boxed{Incorrect}, then \boxed{Correct}.")
        equal, _ = await server._generate_judge_evaluation("Q", "a", "b")
        assert equal is False

    async def test_correct_before_incorrect(self, server: MathWithAutograderResourcesServer) -> None:
        _attach_judge_text(server, r"\boxed{Correct} (alternative \boxed{Incorrect}).")
        equal, _ = await server._generate_judge_evaluation("Q", "a", "b")
        assert equal is True

    async def test_unparseable_yields_false(self, server: MathWithAutograderResourcesServer) -> None:
        _attach_judge_text(server, "I think these are basically the same.")
        equal, _ = await server._generate_judge_evaluation("Q", "a", "b")
        assert equal is False

    async def test_uses_skills_placeholders_and_omits_system(self, server: MathWithAutograderResourcesServer) -> None:
        _attach_judge_text(server, r"\boxed{Correct}")
        await server._generate_judge_evaluation(
            question="What is 1+1?",
            first_answer="2",
            second_answer="2",
        )
        call_kwargs = server.server_client.post.call_args.kwargs
        sent_input = call_kwargs["json"].input
        # Single user turn, no system role.
        assert len(sent_input) == 1
        assert sent_input[0].role == "user"
        # The Skills-style placeholders were filled in.
        rendered = sent_input[0].content
        assert "What is 1+1?" in rendered
        assert "2" in rendered

    async def test_non_text_content_yields_false(self, server: MathWithAutograderResourcesServer) -> None:
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
    async def test_unidirectional_single_call(self, server: MathWithAutograderResourcesServer) -> None:
        # Even with positionally-biased phrasing, only one judge call is made.
        _attach_judge_text(server, r"\boxed{Correct}")
        reward, evaluations = await server._verify_answer_with_judge(
            question="Q",
            expected_answer="42",
            generated_answer="42",
        )
        assert reward == 1.0
        assert len(evaluations) == 1
        assert server.server_client.post.await_count == 1

    async def test_incorrect_yields_zero_reward(self, server: MathWithAutograderResourcesServer) -> None:
        _attach_judge_text(server, r"\boxed{Incorrect}")
        reward, evaluations = await server._verify_answer_with_judge("Q", "42", "wrong")
        assert reward == 0.0
        assert len(evaluations) == 1


class TestVerifyAnswer:
    async def test_symbolic_pass_skips_judge(self, server: MathWithAutograderResourcesServer) -> None:
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

    async def test_symbolic_fail_uses_raw_boxed_for_judge(self, server: MathWithAutograderResourcesServer) -> None:
        # math-verify will collapse a complex boxed expression into a
        # degenerate fragment; the judge should see the raw LaTeX instead.
        _attach_judge_text(server, r"\boxed{Correct}")
        complex_boxed = r"\boxed{g(x)=2x^3+C \text{ or } g(x)=-2x^3+C}"
        await server._verify_answer(
            question="Find g.",
            expected_answer=r"\pm(2x^3+C)",
            generated_answer="Step by step... " + complex_boxed,
        )
        sent_prompt = server.server_client.post.call_args.kwargs["json"].input[0].content
        # The judge should see the raw boxed contents, not the math-verify reduction.
        assert r"g(x)=2x^3+C" in sent_prompt

    async def test_no_boxed_falls_back_to_extracted_then_full(self, server: MathWithAutograderResourcesServer) -> None:
        _attach_judge_text(server, r"\boxed{Incorrect}")
        # math-verify will not extract anything from this prose-only answer,
        # so the judge should see the full generated text as the fallback.
        full_gen = "I'd rather not say."
        await server._verify_answer(
            question="Q",
            expected_answer="42",
            generated_answer=full_gen,
        )
        sent_prompt = server.server_client.post.call_args.kwargs["json"].input[0].content
        assert full_gen in sent_prompt

    async def test_should_use_judge_false_skips_judge(
        self,
        config: MathWithAutograderResourcesServerConfig,
    ) -> None:
        config.should_use_judge = False
        s = MathWithAutograderResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        s.server_client.post = AsyncMock()
        reward, _, library_reward, evaluations = await s._verify_answer(
            question="Q",
            expected_answer="42",
            generated_answer="not even close",
        )
        assert reward == approx(library_reward)
        assert evaluations is None
        s.server_client.post.assert_not_awaited()
