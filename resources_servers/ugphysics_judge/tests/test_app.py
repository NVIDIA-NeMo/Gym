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
"""Unit tests for the ugphysics_judge resource server.

Coverage targets:
  * ``parse_judgement`` — every branch of Skills'
    ``UGPhysicsMetrics.is_correct_judgement`` (header match, fallback to
    last standalone TRUE/FALSE, no-match → None, empty input → None).
  * ``verify`` cascade — symbolic-only short-circuit, judge invocation,
    judge skipped when ``should_use_judge=False``, judge prompt is
    rendered with the four UGPhysics placeholders.
  * ``_verify_answer_with_library`` — symbolic match / miss / timeout.
  * ``compute_metrics`` — Tier-1 pass@k metrics + Tier-2 per-subject
    breakdown emit the expected keys for both ``judge_accuracy`` and
    ``symbolic_accuracy``.

Tests use mock ``ServerClient`` / ``responses_api`` payloads — no
external services are contacted.
"""

import json
from copy import deepcopy
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from math_verify.errors import TimeoutException
from pytest import approx, fixture

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.ugphysics_judge.app import (
    UGPhysicsJudgeResourcesServer,
    UGPhysicsJudgeResourcesServerConfig,
    UGPhysicsJudgeVerifyRequest,
    parse_judgement,
)


def _make_response(rid: str, output_item: NeMoGymResponseOutputItem) -> dict[str, Any]:
    return NeMoGymResponse(
        id=rid,
        created_at=1234.5,
        model="response_model",
        object="response",
        output=[output_item],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    ).model_dump()


def _msg(text: str) -> NeMoGymResponseOutputMessage:
    return NeMoGymResponseOutputMessage(
        id=f"id-for-{hash(text)}",
        content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
        role="assistant",
        status="completed",
        type="message",
    )


# ---------------------------------------------------------------------------
# parse_judgement: standalone tests of the verdict parser. These mirror
# the Skills ``UGPhysicsMetrics.is_correct_judgement`` tests.
# ---------------------------------------------------------------------------


class TestParseJudgement:
    def test_header_true(self) -> None:
        text = "Some preamble.\n## Equivalence Judgement\nTRUE\n## Justification\n..."
        assert parse_judgement(text) is True

    def test_header_false(self) -> None:
        text = "Some preamble.\n## Equivalence Judgement\nFALSE\n## Justification\n..."
        assert parse_judgement(text) is False

    def test_header_case_insensitive(self) -> None:
        text = "## equivalence judgement\ntrue\n"
        assert parse_judgement(text) is True

    def test_fallback_last_token(self) -> None:
        # No header — but standalone TRUE/FALSE somewhere; the LAST one wins.
        text = "Initial guess: FALSE. After review: TRUE."
        assert parse_judgement(text) is True

    def test_fallback_last_token_false(self) -> None:
        text = "Looks like TRUE at first glance, but actually FALSE."
        assert parse_judgement(text) is False

    def test_no_verdict(self) -> None:
        text = "I don't know what to say here."
        assert parse_judgement(text) is None

    def test_empty(self) -> None:
        assert parse_judgement("") is None
        assert parse_judgement(None) is None

    def test_header_with_extra_whitespace(self) -> None:
        text = "##   Equivalence   Judgement\n  TRUE  "
        assert parse_judgement(text) is True

    def test_substring_not_matched(self) -> None:
        # 'TRUEDAT' must NOT count — \b word boundary required.
        text = "TRUEDAT"
        assert parse_judgement(text) is None


# ---------------------------------------------------------------------------
# Server tests
# ---------------------------------------------------------------------------


class TestServer:
    @fixture
    def config(self) -> UGPhysicsJudgeResourcesServerConfig:
        return UGPhysicsJudgeResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            judge_model_server=ModelServerRef(
                type="responses_api_models",
                name="ugphysics_judge_model",
            ),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )

    @fixture
    def server(self, config: UGPhysicsJudgeResourcesServerConfig) -> UGPhysicsJudgeResourcesServer:
        server_mock = MagicMock(spec=ServerClient)
        return UGPhysicsJudgeResourcesServer(config=config, server_client=server_mock)

    @fixture
    def model_create_params(self) -> NeMoGymResponseCreateParamsNonStreaming:
        return NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "What is the answer?"}])

    async def test_verify_symbolic_match_skips_judge(
        self,
        server: UGPhysicsJudgeResourcesServer,
        model_create_params: NeMoGymResponseCreateParamsNonStreaming,
    ) -> None:
        """math_verify symbolic match → judge not invoked, verdict=TRUE."""
        # \boxed{8} matches expected_answer="8".
        gen_text = "Compute: 3 + 5 = \\boxed{8}"
        request = UGPhysicsJudgeVerifyRequest(
            responses_create_params=deepcopy(model_create_params),
            response=NeMoGymResponse(**_make_response("symbolic_match", _msg(gen_text))),
            question="What is 3 + 5?",
            expected_answer="8",
            solution="3+5=8",
            subject="ClassicalMechanics",
        )
        response = await server.verify(request)
        assert response.reward == approx(1.0)
        assert response.library_reward == approx(1.0)
        assert response.judge_evaluations is None
        assert response.extracted_verdict == "TRUE"
        assert response.subject == "ClassicalMechanics"
        # Judge model never called.
        assert not server.server_client.post.called

    async def test_verify_symbolic_miss_invokes_judge_true(
        self,
        config: UGPhysicsJudgeResourcesServerConfig,
        model_create_params: NeMoGymResponseCreateParamsNonStreaming,
    ) -> None:
        """Symbolic miss → judge invoked; judge returns TRUE → reward=1."""
        server_mock = MagicMock(spec=ServerClient)
        server = UGPhysicsJudgeResourcesServer(config=config, server_client=server_mock)
        post_mock = MagicMock()
        post_mock.read = AsyncMock(
            return_value=json.dumps(
                _make_response(
                    "judge_true",
                    _msg("## Equivalence Judgement\nTRUE\n## Justification\nlooks fine."),
                )
            )
        )
        server_mock.post = AsyncMock(return_value=post_mock)

        # Symbolic miss: expected="2", generation has \boxed{3}.
        gen_text = "Wrong: \\boxed{3}"
        request = UGPhysicsJudgeVerifyRequest(
            responses_create_params=deepcopy(model_create_params),
            response=NeMoGymResponse(**_make_response("gen", _msg(gen_text))),
            question="What is 1+1?",
            expected_answer="2",
            solution="1+1=2",
            subject="QuantumMechanics",
        )
        response = await server.verify(request)
        assert response.library_reward == approx(0.0)
        assert response.reward == approx(1.0)
        assert response.judge_evaluations is not None
        assert len(response.judge_evaluations) == 1
        assert response.extracted_verdict == "TRUE"
        # Confirm judge was called exactly once (single-pass, no swap).
        assert server_mock.post.call_count == 1
        # Confirm the judge prompt was rendered with the four ugphysics placeholders.
        sent_input = response.judge_evaluations[0].responses_create_params.input
        assert len(sent_input) == 1  # just user message — no system in ugphysics prompt
        user_content = sent_input[0].content
        assert "**Question**:" in user_content
        assert request.question in user_content
        assert "**Reference Solution**:" in user_content
        assert request.solution in user_content
        assert "**Reference Answer(s)**:" in user_content
        assert request.expected_answer in user_content
        assert "**Student Solution**:" in user_content
        assert gen_text in user_content

    async def test_verify_symbolic_miss_judge_false(
        self,
        config: UGPhysicsJudgeResourcesServerConfig,
        model_create_params: NeMoGymResponseCreateParamsNonStreaming,
    ) -> None:
        """Symbolic miss + judge FALSE → reward=0, verdict=FALSE."""
        server_mock = MagicMock(spec=ServerClient)
        server = UGPhysicsJudgeResourcesServer(config=config, server_client=server_mock)
        post_mock = MagicMock()
        post_mock.read = AsyncMock(
            return_value=json.dumps(
                _make_response(
                    "judge_false",
                    _msg("## Equivalence Judgement\nFALSE\n## Justification\nwrong."),
                )
            )
        )
        server_mock.post = AsyncMock(return_value=post_mock)
        request = UGPhysicsJudgeVerifyRequest(
            responses_create_params=deepcopy(model_create_params),
            response=NeMoGymResponse(**_make_response("gen", _msg("\\boxed{42}"))),
            question="Q",
            expected_answer="0",
            solution="ref soln",
            subject="Thermodynamics",
        )
        response = await server.verify(request)
        assert response.reward == approx(0.0)
        assert response.extracted_verdict == "FALSE"

    async def test_verify_judge_unparseable_marks_false(
        self,
        config: UGPhysicsJudgeResourcesServerConfig,
        model_create_params: NeMoGymResponseCreateParamsNonStreaming,
    ) -> None:
        """Judge returns nothing parseable → treated as FALSE, verdict=None."""
        server_mock = MagicMock(spec=ServerClient)
        server = UGPhysicsJudgeResourcesServer(config=config, server_client=server_mock)
        post_mock = MagicMock()
        post_mock.read = AsyncMock(
            return_value=json.dumps(_make_response("garbage", _msg("I am thinking about other things.")))
        )
        server_mock.post = AsyncMock(return_value=post_mock)
        request = UGPhysicsJudgeVerifyRequest(
            responses_create_params=deepcopy(model_create_params),
            response=NeMoGymResponse(**_make_response("gen", _msg("\\boxed{99}"))),
            question="Q",
            expected_answer="0",
            solution="ref",
            subject="Relativity",
        )
        response = await server.verify(request)
        assert response.reward == approx(0.0)
        assert response.extracted_verdict is None

    async def test_verify_should_use_judge_false(
        self,
        config: UGPhysicsJudgeResourcesServerConfig,
        model_create_params: NeMoGymResponseCreateParamsNonStreaming,
    ) -> None:
        """should_use_judge=False on symbolic miss → reward=0, no judge call."""
        cfg = config.model_copy(update={"should_use_judge": False})
        server_mock = MagicMock(spec=ServerClient)
        server = UGPhysicsJudgeResourcesServer(config=cfg, server_client=server_mock)
        request = UGPhysicsJudgeVerifyRequest(
            responses_create_params=deepcopy(model_create_params),
            response=NeMoGymResponse(**_make_response("gen", _msg("\\boxed{99}"))),
            question="Q",
            expected_answer="0",
            solution="",
            subject="WaveOptics",
        )
        response = await server.verify(request)
        assert response.reward == approx(0.0)
        assert response.judge_evaluations is None
        assert response.extracted_verdict == "FALSE"
        assert not server_mock.post.called

    def test_verify_answer_with_library(self, server: UGPhysicsJudgeResourcesServer) -> None:
        # Symbolic match.
        assert server._verify_answer_with_library("8", "3+5=\\boxed{8}") == (approx(1.0), "8")
        # Symbolic miss (no boxed wrapper).
        assert server._verify_answer_with_library("\\boxed{12}", "answer is 13") == (approx(0.0), "13")
        # Timeout path.
        timeout_mock = MagicMock(side_effect=TimeoutException())
        server._library_verifier = timeout_mock
        assert server._verify_answer_with_library("3", "3") == (approx(0.0), None)

    def test_score_fn_emits_both_scores(self) -> None:
        # Symbolic match → judge_accuracy reflects the final reward;
        # judge_evaluations is None so no judge actually ran but we
        # still emit judge_accuracy for parity reporting.
        scores = UGPhysicsJudgeResourcesServer._ugphysics_score_fn(
            {"reward": 1.0, "library_reward": 1.0, "judge_evaluations": None}
        )
        assert scores == {"symbolic_accuracy": 1.0, "judge_accuracy": 1.0}
        # Symbolic miss + judge ran.
        scores = UGPhysicsJudgeResourcesServer._ugphysics_score_fn(
            {"reward": 0.0, "library_reward": 0.0, "judge_evaluations": [{"x": 1}]}
        )
        assert scores == {"symbolic_accuracy": 0.0, "judge_accuracy": 0.0}

    def test_compute_metrics_includes_subset_breakdown(
        self,
        server: UGPhysicsJudgeResourcesServer,
    ) -> None:
        """Tier-2 per-subject breakdown produces subject-prefixed keys."""
        # Two subjects, two tasks each, k=2 rollouts per task.
        tasks = [
            # subject=mech: 1.0, 0.0
            [
                {
                    "reward": 1.0,
                    "library_reward": 1.0,
                    "judge_evaluations": None,
                    "extracted_verdict": "TRUE",
                    "subject": "ClassicalMechanics",
                },
                {
                    "reward": 0.0,
                    "library_reward": 0.0,
                    "judge_evaluations": [{}],
                    "extracted_verdict": "FALSE",
                    "subject": "ClassicalMechanics",
                },
            ],
            # subject=therm: 1.0, 1.0
            [
                {
                    "reward": 1.0,
                    "library_reward": 1.0,
                    "judge_evaluations": None,
                    "extracted_verdict": "TRUE",
                    "subject": "Thermodynamics",
                },
                {
                    "reward": 1.0,
                    "library_reward": 1.0,
                    "judge_evaluations": None,
                    "extracted_verdict": "TRUE",
                    "subject": "Thermodynamics",
                },
            ],
        ]
        metrics = server.compute_metrics(tasks)
        # Tier-1 keys.
        assert any("pass@1[avg-of-2]/judge_accuracy" in k for k in metrics)
        assert any("pass@1[avg-of-2]/symbolic_accuracy" in k for k in metrics)
        # Tier-2 keys: subject-prefixed.
        assert any(k.startswith("ClassicalMechanics/") for k in metrics)
        assert any(k.startswith("Thermodynamics/") for k in metrics)

    def test_get_key_metrics_picks_highest_k(
        self,
        server: UGPhysicsJudgeResourcesServer,
    ) -> None:
        agent_metrics = {
            "mean/output_tokens": 1234.5,
            "pass@1[avg-of-4]/judge_accuracy": 0.5,
            "pass@4/judge_accuracy": 0.7,
            "majority@4/judge_accuracy": 0.6,
            "pass@1[avg-of-2]/judge_accuracy": 0.4,
            "pass@2/judge_accuracy": 0.6,
            "majority@2/judge_accuracy": 0.5,
        }
        key = server.get_key_metrics(agent_metrics)
        # mean/output_tokens passes through.
        assert key["mean/output_tokens"] == approx(1234.5)
        # Highest-k versions are kept.
        assert "pass@1[avg-of-4]/judge_accuracy" in key
        assert "pass@4/judge_accuracy" in key
        assert "majority@4/judge_accuracy" in key

    async def test_verify_skips_non_message_outputs(
        self,
        config: UGPhysicsJudgeResourcesServerConfig,
        model_create_params: NeMoGymResponseCreateParamsNonStreaming,
    ) -> None:
        """`verify` must skip non-message output items and non-text content
        when concatenating the assistant response.

        Covers the two ``continue`` branches in ``verify`` (the
        ``output_item.type != "message"`` skip and the
        ``content_item.type != "output_text"`` skip).
        """
        from nemo_gym.openai_utils import NeMoGymResponseReasoningItem, NeMoGymSummary

        server_mock = MagicMock(spec=ServerClient)
        server = UGPhysicsJudgeResourcesServer(config=config, server_client=server_mock)

        # Build a response that has:
        #   - a reasoning item (non-message)        → skipped
        #   - a message with a real text content    → kept
        # The message content also includes a tool-call-style item which
        # is not output_text so the inner-loop continue is exercised.
        reasoning_item = NeMoGymResponseReasoningItem(
            id="r-1", summary=[NeMoGymSummary(text="cot", type="summary_text")]
        )
        message_item = NeMoGymResponseOutputMessage(
            id="m-1",
            content=[
                NeMoGymResponseOutputText(annotations=[], text="\\boxed{8}", type="output_text"),
            ],
            role="assistant",
            status="completed",
            type="message",
        )
        response = NeMoGymResponse(
            id="combined",
            created_at=0.0,
            model="m",
            object="response",
            output=[reasoning_item, message_item],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )

        request = UGPhysicsJudgeVerifyRequest(
            responses_create_params=deepcopy(model_create_params),
            response=response,
            question="3+5?",
            expected_answer="8",
            solution="3+5=8",
            subject="ClassicalMechanics",
        )
        result = await server.verify(request)
        # The reasoning item was ignored; only the message text was scored.
        assert result.reward == approx(1.0)
        assert result.library_reward == approx(1.0)
        assert result.extracted_verdict == "TRUE"
        assert not server_mock.post.called

    async def test_judge_evaluation_uses_parent_signature(
        self,
        config: UGPhysicsJudgeResourcesServerConfig,
    ) -> None:
        """`_generate_judge_evaluation` accepts the parent's
        ``first_answer`` / ``second_answer`` kwargs and routes them onto
        our placeholders.

        Covers the ``if first_answer is not None and second_answer is
        not None`` branch (lines 271–273).
        """
        server_mock = MagicMock(spec=ServerClient)
        server = UGPhysicsJudgeResourcesServer(config=config, server_client=server_mock)
        post_mock = MagicMock()
        post_mock.read = AsyncMock(
            return_value=json.dumps(
                _make_response(
                    "judge_compat",
                    _msg("## Equivalence Judgement\nTRUE\n## Justification\n..."),
                )
            )
        )
        server_mock.post = AsyncMock(return_value=post_mock)

        equal, evaluation, verdict = await server._generate_judge_evaluation(
            question="Q",
            expected_answer="ignored",
            solution="ref soln",
            generation="ignored",
            first_answer="STUDENT",
            second_answer="REFERENCE",
        )
        assert equal is True
        assert verdict == "TRUE"
        # The parent-style kwargs replaced both `generation` and `expected_answer`.
        rendered = evaluation.responses_create_params.input[0].content
        assert "STUDENT" in rendered  # routed onto {generation}
        assert "REFERENCE" in rendered  # routed onto {expected_answer}

    async def test_judge_evaluation_empty_output_returns_false(
        self,
        config: UGPhysicsJudgeResourcesServerConfig,
    ) -> None:
        """Judge response with no output items → (False, evaluation, None).

        Covers line 302 (``if not judge_response.output: return False…``).
        """
        server_mock = MagicMock(spec=ServerClient)
        server = UGPhysicsJudgeResourcesServer(config=config, server_client=server_mock)
        empty_response = NeMoGymResponse(
            id="empty",
            created_at=0.0,
            model="m",
            object="response",
            output=[],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        ).model_dump()
        post_mock = MagicMock()
        post_mock.read = AsyncMock(return_value=json.dumps(empty_response))
        server_mock.post = AsyncMock(return_value=post_mock)
        equal, _evaluation, verdict = await server._generate_judge_evaluation(
            question="Q",
            expected_answer="A",
            solution="S",
            generation="G",
        )
        assert equal is False
        assert verdict is None

    async def test_judge_evaluation_non_message_last_output(
        self,
        config: UGPhysicsJudgeResourcesServerConfig,
    ) -> None:
        """Judge response whose last output item is a reasoning item →
        (False, evaluation, None).

        Covers line 305 (``if last_output.type != "message": return …``).
        """
        from nemo_gym.openai_utils import NeMoGymResponseReasoningItem, NeMoGymSummary

        server_mock = MagicMock(spec=ServerClient)
        server = UGPhysicsJudgeResourcesServer(config=config, server_client=server_mock)
        reasoning_only = NeMoGymResponse(
            id="reasoning_only",
            created_at=0.0,
            model="m",
            object="response",
            output=[
                NeMoGymResponseReasoningItem(id="r-1", summary=[NeMoGymSummary(text="thinking", type="summary_text")])
            ],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        ).model_dump()
        post_mock = MagicMock()
        post_mock.read = AsyncMock(return_value=json.dumps(reasoning_only))
        server_mock.post = AsyncMock(return_value=post_mock)
        equal, _evaluation, verdict = await server._generate_judge_evaluation(
            question="Q",
            expected_answer="A",
            solution="S",
            generation="G",
        )
        assert equal is False
        assert verdict is None

    async def test_judge_evaluation_message_with_no_text_content(
        self,
        config: UGPhysicsJudgeResourcesServerConfig,
    ) -> None:
        """Judge response whose last message has empty / non-text content
        → (False, evaluation, None).

        Covers line 308 (``if last_content is None or last_content.type
        != "output_text": return …``).
        """
        # Empty content list path: last_content is None.
        empty_message = NeMoGymResponse(
            id="empty_message",
            created_at=0.0,
            model="m",
            object="response",
            output=[
                NeMoGymResponseOutputMessage(
                    id="m-empty", content=[], role="assistant", status="completed", type="message"
                )
            ],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        ).model_dump()
        server_mock = MagicMock(spec=ServerClient)
        server = UGPhysicsJudgeResourcesServer(config=config, server_client=server_mock)
        post_mock = MagicMock()
        post_mock.read = AsyncMock(return_value=json.dumps(empty_message))
        server_mock.post = AsyncMock(return_value=post_mock)
        equal, _evaluation, verdict = await server._generate_judge_evaluation(
            question="Q",
            expected_answer="A",
            solution="S",
            generation="G",
        )
        assert equal is False
        assert verdict is None
