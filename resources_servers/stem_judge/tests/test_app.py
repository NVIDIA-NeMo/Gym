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
"""Tests for the stem_judge server: length guards, the boxed/free answer contract,
verdict and rubric parsing, and the shipped config."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import yaml
from pydantic import ValidationError
from pytest import approx, fixture, mark, raises

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseReasoningItem,
    NeMoGymSummary,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.stem_judge.app import (
    RubricCriterion,
    StemJudgeConfig,
    StemJudgeResourcesServer,
    StemJudgeVerifyRequest,
    final_answer,
    last_boxed,
    parse_criteria,
    parse_verdict,
    render_reference,
    render_rubric,
    rubric_fraction,
)


SERVER_DIR = Path(__file__).resolve().parents[1]
RUBRIC = [
    {"id": "r1", "points": 3, "criterion": "Final period T = 2π√(L/g)(1 + θ0²/16)"},
    {"id": "r2", "points": 2, "criterion": "Expands the period integral to second order"},
    {"id": "r3", "points": -2, "criterion": "Gives the small-angle period as the final answer"},
]


def _msg(text: str) -> NeMoGymResponseOutputMessage:
    return NeMoGymResponseOutputMessage(
        id="msg",
        content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
        role="assistant",
        status="completed",
        type="message",
    )


def _reasoning(text: str = "scratch work") -> NeMoGymResponseReasoningItem:
    return NeMoGymResponseReasoningItem(id="rs", summary=[NeMoGymSummary(text=text, type="summary_text")])


def _response(*output, truncated: bool = False) -> NeMoGymResponse:
    extra = {"incomplete_details": {"reason": "max_output_tokens"}} if truncated else {}
    return NeMoGymResponse(
        id="resp",
        created_at=0.0,
        model="m",
        object="response",
        output=list(output),
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
        **extra,
    )


def _split(answer: str, **kwargs) -> NeMoGymResponse:
    """A response as our model server returns it: reasoning item, then the answer."""
    return _response(_reasoning(), _msg(answer), **kwargs)


@fixture
def config() -> StemJudgeConfig:
    """The shipped config, so a change to its defaults shows up here."""
    with open(SERVER_DIR / "configs" / "stem_judge.yaml") as f:
        shipped = yaml.safe_load(f)["stem_judge"]["resources_servers"]["stem_judge"]
    for key in ("answer_prompt_template_fpath", "rubric_prompt_template_fpath"):
        shipped[key] = str(SERVER_DIR / shipped[key])
    return StemJudgeConfig(host="0.0.0.0", port=8080, **shipped)


def _server(config: StemJudgeConfig, *judge_texts: str) -> StemJudgeResourcesServer:
    client = MagicMock(spec=ServerClient)
    post = MagicMock()
    post.read = AsyncMock(side_effect=[_response(_msg(t)).model_dump_json() for t in judge_texts])
    client.post = AsyncMock(return_value=post)
    return StemJudgeResourcesServer(config=config, server_client=client)


def _request(response: NeMoGymResponse, expected: str = "2", **row) -> StemJudgeVerifyRequest:
    return StemJudgeVerifyRequest(
        responses_create_params=deepcopy(
            NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "Q: 1+1?"}])
        ),
        response=response,
        expected_answer=expected,
        **row,
    )


def _judge_prompt(server: StemJudgeResourcesServer) -> str:
    return server.server_client.post.await_args.kwargs["json"].input[-1].content


class TestShippedConfig:
    def test_defaults(self, config: StemJudgeConfig) -> None:
        assert config.name == "stem_judge"
        assert config.default_answer_format == "boxed"
        assert config.boxed_judge_input == "final_answer"
        assert (config.max_box_chars, config.max_answer_chars) == (None, 16000)
        assert (config.rubric_reward, config.rubric_pass_fraction) == ("fraction", 1.0)

    def test_templates_format(self, config: StemJudgeConfig) -> None:
        server = _server(config)
        answer = server._answer_template.format(question="Q", expected_answer="E", generated_answer="G")
        rubric = server._rubric_template.format(question="Q", expected_answer="E", rubric="R", generated_answer="G")
        assert answer.endswith("**Student Solution**:\nG")
        assert '{"r1": true, "r2": false, "r3": false}' in rubric
        assert rubric.endswith("**Rubric** (criterion id, points, criterion):\nR\n\n**Student Solution**:\nG")

    def test_answer_template_is_stem_v11(self) -> None:
        # Pins the stem_v11 prompt as it was measured offline.
        digest = hashlib.sha256((SERVER_DIR / "prompt_templates" / "stem_v11.txt").read_bytes()).hexdigest()
        assert digest == "9b9dd45f94b114c0ea03c7a4bf91a41b37f6f62bbcbafea5eac0b99be8598498"


class TestFinalAnswer:
    def test_the_message_is_the_answer(self) -> None:
        assert final_answer(_split("  The answer is 2.  ")) == ("The answer is 2.", "ok")
        assert final_answer(_response(_msg("The answer is 2."))) == ("The answer is 2.", "ok")

    @mark.parametrize("answer", ["", "Answer: 2"])
    def test_truncated(self, answer: str) -> None:
        assert final_answer(_split(answer, truncated=True)) == (None, "truncated")

    @mark.parametrize("response", [_split("  "), _response(_reasoning())])
    def test_empty(self, response: NeMoGymResponse) -> None:
        assert final_answer(response) == (None, "empty_final_answer")


class TestParsing:
    @mark.parametrize(
        "text, expected",
        [
            ("x \\boxed{\\frac{4}{2}} y", "\\frac{4}{2}"),
            ("\\boxed{1} then \\boxed{2}", "2"),
            ("\\boxed{1} then \\boxed{2", None),
            ("\\boxed{ }", None),
            ("no box", None),
        ],
    )
    def test_last_boxed(self, text: str, expected) -> None:
        assert last_boxed(text) == expected

    @mark.parametrize(
        "text, expected",
        [
            ("Judgement: 'yes'", "yes"),
            ("**Judgement:** 'no'", "no"),
            ("Judgement: yes\nOn reflection, Judgement: 'no'", "no"),
            ("Final Answer: \\boxed{yes}", "yes"),
            ("The student is right.", None),
        ],
    )
    def test_verdict(self, text: str, expected) -> None:
        assert parse_verdict(text) == expected

    def test_criteria(self) -> None:
        ids = ["r1", "r2"]
        assert parse_criteria('Criteria: {"r1": true, "r2": false}', ids) == {"r1": True, "r2": False}
        assert parse_criteria('```json\n{"r1": "True", "r2": "false"}\n```', ids) == {"r1": True, "r2": False}
        # The last object that marks every id wins.
        text = '{"r1": false, "r2": false} draft {"r1": true} final {"r1": true, "r2": true}'
        assert parse_criteria(text, ids) == {"r1": True, "r2": True}

    @mark.parametrize(
        "text",
        ['{"r1": true}', '{"r1": true, "r2": "maybe"}', '{"r1": 1, "r2": 0}', "r1: met, r2: not met", ""],
    )
    def test_criteria_unparsed(self, text: str) -> None:
        assert parse_criteria(text, ["r1", "r2"]) is None

    @mark.parametrize(
        "expected, rendered",
        [
            ("- (π^2/9) m_t^2 m_H^2 / m_1^2", "-(π^2/9) m_t^2 m_H^2 / m_1^2"),
            ("  - g^4 N / (4π)", "  -g^4 N / (4π)"),
            ("- \\Theta(t^2 - r^2) f\n  \\times g", "-\\Theta(t^2 - r^2) f\n  \\times g"),
            ("-3.2 eV", "-3.2 eV"),
            ("x - y", "x - y"),
            (
                "- Molecular formula: C2H6O\n- Molar mass: 46 g/mol",
                "- Molecular formula: C2H6O\n- Molar mass: 46 g/mol",
            ),
            ("", ""),
        ],
    )
    def test_render_reference(self, expected: str, rendered: str) -> None:
        assert render_reference(expected) == rendered

    def test_render_and_score(self) -> None:
        rubric = [RubricCriterion(**c) for c in RUBRIC]
        assert render_rubric(rubric).splitlines()[2] == (
            "[r3] (penalty, -2 points) Gives the small-angle period as the final answer"
        )
        assert rubric_fraction(rubric, {"r1": True, "r2": False, "r3": False}) == approx(0.6)
        assert rubric_fraction(rubric, {"r1": False, "r2": False, "r3": True}) == approx(0.0)
        assert rubric_fraction(rubric, {"r1": True, "r2": True, "r3": True}) == approx(0.6)


class TestRequestValidation:
    @mark.parametrize(
        "row",
        [
            {"rubric": []},
            {"rubric": [{"id": "r1", "points": 1, "criterion": "a"}, {"id": "r1", "points": 1, "criterion": "b"}]},
            {"rubric": [{"id": "r1", "points": 0, "criterion": "a"}]},
            {"rubric": [{"id": "r1", "points": -1, "criterion": "a"}]},
            {"answer_format": "boxd"},
        ],
    )
    def test_bad_rows_are_rejected(self, row: dict) -> None:
        with raises(ValidationError):
            _request(_split("x"), **row)


class TestVerify:
    async def test_boxed_row_judges_the_final_answer(self, config: StemJudgeConfig) -> None:
        server = _server(config, "Reasoning...\nJudgement: 'yes'")
        res = await server.verify(_request(_split("First 1+1 = 2, so \\boxed{\\frac{4}{2}}.")))
        assert (res.reward, res.grading_status, res.answer_format) == (approx(1.0), "judged", "boxed")
        assert res.judge_evaluations[0].verdict_label == "yes"
        assert "**Student Solution**:\nFirst 1+1 = 2, so \\boxed{\\frac{4}{2}}." in _judge_prompt(server)
        assert "**Question**:\nQ: 1+1?" in _judge_prompt(server)

    async def test_boxed_row_judges_only_the_box(self, config: StemJudgeConfig) -> None:
        server = _server(config.model_copy(update={"boxed_judge_input": "box"}), "Judgement: 'yes'")
        await server.verify(_request(_split("First 1+1 = 2, so \\boxed{\\frac{4}{2}}.")))
        assert _judge_prompt(server).endswith("**Student Solution**:\n\\frac{4}{2}")
        assert "First 1+1" not in _judge_prompt(server)

    @mark.parametrize(
        "response, status",
        [
            (_split("The answer is 2."), "no_boxed"),
            (_split("y" * 16001 + " \\boxed{2}"), "answer_too_long"),
            (_split("\\boxed{2}", truncated=True), "truncated"),
            (_response(_reasoning()), "empty_final_answer"),
        ],
    )
    async def test_gates_skip_the_judge(self, config: StemJudgeConfig, response, status: str) -> None:
        server = _server(config, "Judgement: 'yes'")
        res = await server.verify(_request(response))
        assert (res.reward, res.grading_status, res.judge_evaluations) == (approx(0.0), status, [])
        server.server_client.post.assert_not_awaited()

    async def test_box_cap(self, config: StemJudgeConfig) -> None:
        server = _server(config, "Judgement: 'yes'")
        res = await server.verify(_request(_split("\\boxed{" + "x" * 2001 + "}")))
        assert res.reward == approx(1.0)
        server = _server(config.model_copy(update={"max_box_chars": 2000}), "Judgement: 'yes'")
        res = await server.verify(_request(_split("\\boxed{" + "x" * 2001 + "}")))
        assert (res.reward, res.grading_status) == (approx(0.0), "box_too_long")
        server.server_client.post.assert_not_awaited()

    async def test_answer_cap_off_when_the_judge_sees_only_the_box(self, config: StemJudgeConfig) -> None:
        server = _server(config.model_copy(update={"boxed_judge_input": "box"}), "Judgement: 'yes'")
        res = await server.verify(_request(_split("y" * 16001 + " \\boxed{2}")))
        assert res.reward == approx(1.0)

    async def test_leading_minus_reference_keeps_its_sign(self, config: StemJudgeConfig) -> None:
        server = _server(config, "Judgement: 'yes'")
        res = await server.verify(_request(_split("\\boxed{-2}"), expected="- 2"))
        assert "**Reference Answer(s)**:\n-2\n" in _judge_prompt(server)
        assert res.expected_answer == "- 2"

    async def test_free_row_needs_no_box(self, config: StemJudgeConfig) -> None:
        server = _server(config, "Judgement: 'yes'")
        res = await server.verify(_request(_split("Two, because 1+1=2."), answer_format="free"))
        assert (res.reward, res.answer_format) == (approx(1.0), "free")
        assert "**Student Solution**:\nTwo, because 1+1=2." in _judge_prompt(server)

    @mark.parametrize(
        "judge, reward, status", [("Judgement: 'no'", 0.0, "judged"), ("I agree.", 0.0, "judge_unparsed")]
    )
    async def test_judge_no_or_unparsed(self, config: StemJudgeConfig, judge: str, reward: float, status: str) -> None:
        server = _server(config, judge)
        res = await server.verify(_request(_split("\\boxed{3}")))
        assert (res.reward, res.grading_status) == (approx(reward), status)
        assert server.server_client.post.await_count == 1

    async def test_rubric_fraction(self, config: StemJudgeConfig) -> None:
        server = _server(config, 'Checked each.\nCriteria: {"r1": true, "r2": false, "r3": false}')
        res = await server.verify(_request(_split("T = 2π√(L/g)(1+θ0²/16)"), answer_format="free", rubric=RUBRIC))
        assert (res.reward, res.rubric_score, res.grading_status) == (approx(0.6), approx(0.6), "judged")
        assert res.judge_evaluations[0].criteria_met == {"r1": True, "r2": False, "r3": False}
        prompt = _judge_prompt(server)
        assert "[r1] (3 points) Final period" in prompt
        assert "**Reference Answer**:\n2" in prompt

    async def test_rubric_threshold(self, config: StemJudgeConfig) -> None:
        threshold = config.model_copy(update={"rubric_reward": "threshold", "rubric_pass_fraction": 0.5})
        for marks, reward in (('"r1": true, "r2": false', 1.0), ('"r1": false, "r2": true', 0.0)):
            server = _server(threshold, '{%s, "r3": false}' % marks)
            res = await server.verify(_request(_split("T"), answer_format="free", rubric=RUBRIC))
            assert res.reward == approx(reward)

    async def test_rubric_unparsed(self, config: StemJudgeConfig) -> None:
        server = _server(config, 'Criteria: {"r1": true, "r2": false}')
        res = await server.verify(_request(_split("T"), answer_format="free", rubric=RUBRIC))
        assert (res.reward, res.grading_status, res.rubric_score) == (approx(0.0), "judge_unparsed", None)

    async def test_rubric_row_still_follows_its_answer_format(self, config: StemJudgeConfig) -> None:
        server = _server(config, '{"r1": true, "r2": true, "r3": false}')
        res = await server.verify(_request(_split("T without a box"), rubric=RUBRIC))
        assert res.grading_status == "no_boxed"

    async def test_extract_problem_from_prompt(self, config: StemJudgeConfig) -> None:
        server = _server(config.model_copy(update={"extract_problem_from_prompt": True}), "Judgement: 'yes'")
        req = _request(_split("\\boxed{2}"))
        req.responses_create_params.input[0].content = "Preamble.\nAnswer: {your final answer}\nWhat is 1+1?"
        await server.verify(req)
        assert "**Question**:\nWhat is 1+1?\n" in _judge_prompt(server)

    async def test_generation_log(self, config: StemJudgeConfig, tmp_path) -> None:
        server = _server(config.model_copy(update={"generation_log_dir": str(tmp_path)}), "Judgement: 'yes'")
        await server.verify(_request(_split("so \\boxed{2}")))
        await server.verify(_request(_split("no box")))
        records = [json.loads(line) for f in tmp_path.glob("*.jsonl") for line in f.read_text().splitlines()]
        assert [(r["grading_status"], r["reward"], r["verdicts"]) for r in records] == [
            ("judged", 1.0, ["yes"]),
            ("no_boxed", 0.0, []),
        ]
        assert records[0]["judged_generation"] == "so \\boxed{2}"
