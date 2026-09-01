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
"""Tests for the stem_llm_judge server: verdict parsing, what the judge is shown,
the truncation and empty-generation short circuits, and the shipped config defaults.
"""

from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import yaml
from pytest import approx, fixture, mark

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.stem_llm_judge.app import (
    PROBLEM_EXTRACT_REGEX,
    JudgeEvaluation,
    LLMJudgeResourcesServer,
    LLMJudgeResourcesServerConfig,
    LLMJudgeVerifyRequest,
    _extract_last_assistant_text,
    _extract_question_text,
)


SERVER_DIR = Path(__file__).resolve().parents[1]

PROMPT_PREAMBLE = (
    "Answer the following problem step by step.\n"
    "Please use LaTeX format to represent the variables and formulas used in the "
    "solution process and results.\n"
    "Your response should be in the following format:\n"
    "Explanation: {your explanation for your final answer}\n"
    "Answer: {your final answer}\n"
)


def _msg(text: str) -> NeMoGymResponseOutputMessage:
    return NeMoGymResponseOutputMessage(
        id="msg_id",
        content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
        role="assistant",
        status="completed",
        type="message",
    )


def _response(output_item: NeMoGymResponseOutputItem, **kwargs) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp",
        created_at=0.0,
        model="m",
        object="response",
        output=[output_item],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
        **kwargs,
    )


@fixture
def config() -> LLMJudgeResourcesServerConfig:
    """The shipped config, loaded from configs/stem_llm_judge.yaml.

    Loading the real YAML (rather than hand-building a config) means a change to
    the shipped defaults shows up as a test failure instead of silently drifting.
    """
    with open(SERVER_DIR / "configs" / "stem_llm_judge.yaml") as f:
        shipped = yaml.safe_load(f)["stem_llm_judge"]["resources_servers"]["stem_llm_judge"]

    cfg = LLMJudgeResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        **{k: v for k, v in shipped.items() if k not in ("judge_prompt_template_fpath",)},
        # Resolve the template against the server dir: the tests do not run with
        # cwd=<server dir> the way Gym's entrypoint does.
        judge_prompt_template_fpath=str(SERVER_DIR / shipped["judge_prompt_template_fpath"]),
    )
    # Sanity: the ref is declarative in YAML, typed here.
    assert cfg.judge_model_server == ModelServerRef(type="responses_api_models", name="policy_model")
    return cfg


@fixture
def server(config: LLMJudgeResourcesServerConfig) -> LLMJudgeResourcesServer:
    return LLMJudgeResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _verdict(server: LLMJudgeResourcesServer, text: str) -> tuple[bool, str | None]:
    record = JudgeEvaluation(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=_response(_msg(text)),
    )
    equal, record = server._parse_verdict(text, record)
    return equal, record.verdict_label


class TestShippedConfig:
    def test_prompt_template_default_matches_config(self) -> None:
        assert LLMJudgeResourcesServerConfig.model_fields["judge_prompt_template_fpath"].default == (
            "prompt_templates/stem_llm_judge.txt"
        )

    def test_server_name_default(self) -> None:
        assert LLMJudgeResourcesServerConfig.model_fields["name"].default == "stem_llm_judge"

    def test_prompt_has_all_three_placeholders(self, server: LLMJudgeResourcesServer) -> None:
        template = server._judge_prompt_template
        for placeholder in ("{question}", "{expected_answer}", "{generated_answer}"):
            assert placeholder in template
        # .format() must not blow up on stray braces in the rubric's LaTeX.
        filled = template.format(question="q", expected_answer="e", generated_answer="g")
        assert "q" in filled and "e" in filled and "g" in filled

    def test_question_extraction_is_on_by_default(self, config: LLMJudgeResourcesServerConfig) -> None:
        assert LLMJudgeResourcesServerConfig.model_fields["extract_problem_from_prompt"].default is True
        assert config.extract_problem_from_prompt is True

    def test_student_answer_is_the_post_think_region(self, config: LLMJudgeResourcesServerConfig) -> None:
        assert config.response_extract_regex == "</think>(.*)"


class TestParseVerdict:
    @mark.parametrize(
        "text,expected",
        [
            ("Judgement: 'yes'", True),
            ("Judgement: 'no'", False),
            ("Judgement: yes", True),
            ("**Judgement**: **yes**", True),
            ('judgement:   "NO"', False),
        ],
    )
    def test_primary_format(self, server: LLMJudgeResourcesServer, text: str, expected: bool) -> None:
        equal, label = _verdict(server, text)
        assert equal is expected
        assert label == ("yes" if expected else "no")

    def test_last_verdict_wins(self, server: LLMJudgeResourcesServer) -> None:
        # A judge that revises its decision is scored on the final one.
        equal, label = _verdict(server, "Judgement: 'yes'\nOn reflection...\nJudgement: 'no'")
        assert equal is False
        assert label == "no"

    def test_only_post_think_region_is_parsed(self, server: LLMJudgeResourcesServer) -> None:
        # A verdict inside the judge's own scratch reasoning must not count.
        equal, _ = _verdict(server, "<think>Judgement: 'yes'</think>\nJudgement: 'no'")
        assert equal is False

    def test_fallback_only_used_when_primary_absent(self, server: LLMJudgeResourcesServer) -> None:
        # Fallback recovers judges that end with Answer/Verdict/Conclusion.
        assert _verdict(server, "Verdict: yes")[0] is True
        assert _verdict(server, "Final Judgment: no")[0] is False
        assert _verdict(server, r"Answer: \boxed{yes}")[0] is True
        # ...but a real "Judgement:" always wins over a stray "Answer:".
        equal, _ = _verdict(server, "Judgement: 'no'\nAnswer: yes")
        assert equal is False

    @mark.parametrize("text", ["", "The solution looks fine to me.", "Answer: 2/3", "Judgement: maybe"])
    def test_unparsed_scores_zero(self, server: LLMJudgeResourcesServer, text: str) -> None:
        equal, label = _verdict(server, text)
        assert equal is False
        assert label is None


class TestExtraction:
    def test_problem_extract_regex_strips_preamble(self) -> None:
        problem = "A block of mass $m$ slides down\na frictionless incline.\nFind $a$."
        params = NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "user", "content": PROMPT_PREAMBLE + problem}]
        )
        assert _extract_question_text(params, PROBLEM_EXTRACT_REGEX) == problem

    def test_problem_extract_regex_falls_back_to_full_prompt(self) -> None:
        # A prompt without the marker must not silently yield an empty question.
        params = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "Bare question?"}])
        assert _extract_question_text(params, PROBLEM_EXTRACT_REGEX) == "Bare question?"

    def test_extract_problem_from_prompt_toggle(self, config: LLMJudgeResourcesServerConfig) -> None:
        # Shipped default: on, so the judge is handed the bare problem.
        server = LLMJudgeResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        assert server._question_extract_regex() == PROBLEM_EXTRACT_REGEX

        off = config.model_copy(deep=True)
        off.extract_problem_from_prompt = False
        assert (
            LLMJudgeResourcesServer(config=off, server_client=MagicMock(spec=ServerClient))._question_extract_regex()
            is None
        )

        # An explicit question_extract_regex always wins over the toggle.
        explicit = config.model_copy(deep=True)
        explicit.question_extract_regex = "custom"
        assert (
            LLMJudgeResourcesServer(
                config=explicit, server_client=MagicMock(spec=ServerClient)
            )._question_extract_regex()
            == "custom"
        )

    def test_post_think_extraction(self) -> None:
        body = MagicMock()
        body.response = _response(_msg("<think>scratch</think>\nAnswer: 42"))
        assert _extract_last_assistant_text(body, "</think>(.*)") == "Answer: 42"
        assert "scratch" in _extract_last_assistant_text(body, None)


class TestVerify:
    def _request(
        self, generation: str, expected: str = "2", prompt: str = "Q: 1+1?", **response_kwargs
    ) -> LLMJudgeVerifyRequest:
        return LLMJudgeVerifyRequest(
            responses_create_params=deepcopy(
                NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": prompt}])
            ),
            response=_response(_msg(generation), **response_kwargs),
            expected_answer=expected,
        )

    def _server_with_judge_saying(
        self, config: LLMJudgeResourcesServerConfig, *judge_texts: str
    ) -> LLMJudgeResourcesServer:
        client = MagicMock(spec=ServerClient)
        post = MagicMock()
        post.read = AsyncMock(side_effect=[_response(_msg(t)).model_dump_json() for t in judge_texts])
        client.post = AsyncMock(return_value=post)
        return LLMJudgeResourcesServer(config=config, server_client=client)

    async def test_yes_scores_one(self, config: LLMJudgeResourcesServerConfig) -> None:
        server = self._server_with_judge_saying(config, "Reasoning...\nJudgement: 'yes'")
        res = await server.verify(self._request("<think>...</think>\nAnswer: 2"))
        assert res.reward == approx(1.0)
        assert res.expected_answer == "2"
        assert len(res.judge_evaluations) == 1
        assert res.judge_evaluations[0].verdict_label == "yes"

    async def test_no_scores_zero_after_one_judge_call(self, config: LLMJudgeResourcesServerConfig) -> None:
        # Pass/fail, no rescue pass: a "no" must cost exactly one judge call.
        server = self._server_with_judge_saying(config, "Judgement: 'no'")
        res = await server.verify(self._request("<think>...</think>\nAnswer: 3"))
        assert res.reward == approx(0.0)
        assert len(res.judge_evaluations) == 1
        assert server.server_client.post.await_count == 1

    async def test_empty_generation_skips_the_judge(self, config: LLMJudgeResourcesServerConfig) -> None:
        server = self._server_with_judge_saying(config)
        res = await server.verify(self._request("   "))
        assert res.reward == approx(0.0)
        assert res.judge_evaluations == []
        server.server_client.post.assert_not_awaited()

    async def test_regex_miss_falls_back_to_full_generation(self, config: LLMJudgeResourcesServerConfig) -> None:
        # A generation with no </think> is NOT treated as empty: _extract_last_assistant_text
        # returns the full text when the extraction regex does not match, so the judge
        # still sees the whole answer. Only a genuinely blank message skips the judge.
        server = self._server_with_judge_saying(config, "Judgement: 'yes'")
        res = await server.verify(self._request("Answer: 2 (no think tag here)"))
        assert res.reward == approx(1.0)
        sent = server.server_client.post.await_args.kwargs["json"]
        assert "no think tag here" in sent.input[-1].content

    async def test_judge_is_shown_the_bare_problem(self, config: LLMJudgeResourcesServerConfig) -> None:
        problem = "A block slides down a frictionless incline.\nFind its acceleration."
        server = self._server_with_judge_saying(config, "Judgement: 'yes'")
        await server.verify(self._request("<think>...</think>\nAnswer: 2", prompt=PROMPT_PREAMBLE + problem))

        sent = server.server_client.post.await_args.kwargs["json"].input[-1].content
        assert problem in sent
        assert "Answer the following problem step by step" not in sent

    async def test_truncation_guard(self, config: LLMJudgeResourcesServerConfig) -> None:
        truncated = {"incomplete_details": {"reason": "max_output_tokens"}}

        # Off (shipped default): the judge is still consulted.
        server = self._server_with_judge_saying(config, "Judgement: 'yes'")
        res = await server.verify(self._request("<think>...</think>\nAnswer: 2", **truncated))
        assert res.reward == approx(1.0)

        # On: score 0 and skip the judge call entirely.
        on = config.model_copy(deep=True)
        on.reward_zero_on_truncation = True
        server = self._server_with_judge_saying(on, "Judgement: 'yes'")
        res = await server.verify(self._request("<think>...</think>\nAnswer: 2", **truncated))
        assert res.reward == approx(0.0)
        assert res.judge_evaluations == []
        server.server_client.post.assert_not_awaited()

    async def test_per_record_regex_overrides_server_default(self, config: LLMJudgeResourcesServerConfig) -> None:
        server = self._server_with_judge_saying(config, "Judgement: 'yes'")
        req = self._request("<think>...</think>\nThe answer is \\boxed{2}.")
        req.template_metadata = {"output_regex": r"\\boxed\{(.*?)\}"}
        await server.verify(req)
        sent = server.server_client.post.await_args.kwargs["json"]
        assert "2" in sent.input[-1].content
        # The per-record regex won: the raw \boxed{...} wrapper is not in the prompt.
        assert "\\boxed{2}" not in sent.input[-1].content

    async def test_generation_log_written(self, config: LLMJudgeResourcesServerConfig, tmp_path) -> None:
        import json

        logged = config.model_copy(deep=True)
        logged.generation_log_dir = str(tmp_path)
        server = self._server_with_judge_saying(logged, "Judgement: 'yes'")
        await server.verify(self._request("<think>scratch</think>\nAnswer: 2"))

        files = list(tmp_path.glob("*.jsonl"))
        assert len(files) == 1
        record = json.loads(files[0].read_text().strip())
        assert record["reward"] == approx(1.0)
        assert record["verdicts"] == ["yes"]
        assert record["judged_generation"] == "Answer: 2"
        assert "scratch" in record["generation"]
